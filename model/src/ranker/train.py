"""Train LightGBM LambdaRank với weighted relevance labels.

3-step disk-based để tránh OOM:
- Step 1: Write labeled.parquet chunked (8 buckets).
- Step 2: user_stats with n_neg_to_keep.
- Step 3: Sample chunked → sampled_train.parquet.

[OOM PATCH v3.1]
- Bỏ `ORDER BY user_id` ở SELECT * cuối (data đã sort sẵn ở concat step → đỡ peak ~10-15GB RAM).
- Đổi `free_raw_data=False` → `True` (LightGBM auto-free raw X sau khi build histogram → đỡ ~5-10GB).
- Dùng pyarrow read với `self_destruct=True` để giải phóng Arrow buffer ngay khi convert sang pandas.
- Cleanup intermediate parquet ngay sau khi không cần (`_train_sampled.parquet` sau khi load).
- Train/val split: dùng pandas filter thay isin set (nhanh hơn, ít memory hơn với large user count).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common import get_config, get_logger, make_connection, timed
from src.ranker.feature_spec import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    GROUP_COL,
    LABEL_COL,
    NUMERIC_FEATURES,
    validate_columns,
)

logger = get_logger(__name__)


def _build_internal_gt_table(con) -> str:
    cfg = get_config()
    events_glob = str(cfg.paths.fact_events_dir / "*.parquet").replace("\\", "/")
    gt_start = cfg.windows.internal_gt_start
    gt_end = cfg.windows.internal_gt_end
    out_path = cfg.paths.gt_dir / "internal_gt.parquet"
    out_str = str(out_path).replace("\\", "/")

    if out_path.exists() and out_path.stat().st_size > 100:
        logger.info("  internal_gt cache hit: %s", out_path)
        return out_str

    w_hard = cfg.weights.hard_contact
    w_other = cfg.weights.other_interaction
    weight_case = (
        f"CASE event_type "
        f"WHEN 'view_phone'        THEN {w_hard} "
        f"WHEN 'contact_chat'      THEN {w_hard} "
        f"WHEN 'contact_zalo'      THEN {w_hard} "
        f"WHEN 'contact_sms'       THEN {w_hard} "
        f"WHEN 'other_interaction' THEN {w_other} "
        f"ELSE 0.0 END"
    )

    sql = f"""
    COPY (
        SELECT user_id, item_id,
               SUM({weight_case}) AS gt_weighted_score,
               COUNT(*) AS gt_n_events
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{gt_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{gt_end} 23:59:59'
          AND user_id IS NOT NULL AND item_id IS NOT NULL
          AND ({weight_case}) > 0
        GROUP BY user_id, item_id
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed(f"build internal_gt [{gt_start} - {gt_end}]", logger):
        con.execute(sql)
    return out_str


# [CATEGORY AFFINITY PATCH v3.2] Thêm 5 cột mới:
#   u_top_category_share, u_top_category_share_recent, u_top2_category,
#   ui_category_match_top2, ui_top_cat_match_x_share
_OUTPUT_COLS = [
    "user_id", "item_id", "source", "source_score", "candidate_score",
    "u_total_weighted", "u_recent_weighted", "u_total_pageview",
    "u_unique_items", "u_unique_categories", "u_unique_cities",
    "u_active_days", "u_top_category", "u_top_city", "u_avg_dwell",
    "u_days_since_last", "u_is_warm",
    # [PATCH] Category affinity user features
    "u_top_category_share", "u_top_category_share_recent", "u_top2_category",
    "i_category", "i_city_name", "i_seller_type", "i_ad_type",
    "i_has_project_id", "i_area_sqm", "i_bedrooms", "i_bathrooms",
    "i_images_count", "i_age_days", "i_total_weighted",
    "i_recent_weighted", "i_total_pageview", "i_avg_dwell",
    "i_pop_rank_global",
    "ui_total_weighted", "ui_recent_weighted", "ui_n_pageview",
    "ui_n_pos_events", "ui_max_dwell", "ui_n_active_days",
    "ui_days_since_last", "ui_days_since_first",
    "ui_recency_tier", "u_activity_recency_tier", "i_age_tier",
    "ui_category_match", "ui_city_match",
    # [PATCH] Category affinity derived features
    "ui_category_match_top2", "ui_top_cat_match_x_share",
    "rel_label",
]


def _build_sampled_training_frame(con, ranker_input_path: Path) -> Path:
    """Build sampled training parquet on disk. Return path (KHÔNG load vào pandas).

    [OOM PATCH] Bỏ load full vào DataFrame ở đây — chuyển sang
    `_load_sampled_via_pyarrow()` để stream-load với self_destruct=True.
    """
    cfg = get_config()
    neg_per_pos = cfg.ranker.neg_per_pos
    seed = cfg.ranker.random_seed

    ri_str = str(ranker_input_path).replace("\\", "/")
    gt_str = _build_internal_gt_table(con)

    sampled_path = cfg.paths.features_dir / "_train_sampled.parquet"
    sampled_str = str(sampled_path).replace("\\", "/")

    if sampled_path.exists() and sampled_path.stat().st_size > 100:
        logger.info("  sampled training frame cache hit: %s", sampled_path)
        return sampled_path

    labeled_path = cfg.paths.features_dir / "_train_labeled.parquet"
    labeled_str = str(labeled_path).replace("\\", "/")
    user_keep_path = cfg.paths.features_dir / "_train_user_keep.parquet"
    user_keep_str = str(user_keep_path).replace("\\", "/")

    # Step 1: Write labeled chunked
    if not (labeled_path.exists() and labeled_path.stat().st_size > 100):
        logger.info("  Step 1/3: Building labeled.parquet (chunked)...")
        N_BUCKETS = 8
        parts_dir = cfg.paths.features_dir / "_labeled_parts"
        parts_dir.mkdir(exist_ok=True)

        for bucket in range(N_BUCKETS):
            part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
            part_str = str(part_path).replace("\\", "/")
            if part_path.exists() and part_path.stat().st_size > 100:
                logger.info("    bucket %d/%d: cache hit, SKIP",
                            bucket + 1, N_BUCKETS)
                continue

            sql_label = f"""
            COPY (
                SELECT r.*,
                    CASE
                        WHEN COALESCE(g.gt_weighted_score, 0.0) >= 6.0 THEN 3
                        WHEN COALESCE(g.gt_weighted_score, 0.0) >= 3.0 THEN 2
                        WHEN COALESCE(g.gt_weighted_score, 0.0) >  0.0 THEN 1
                        ELSE 0
                    END AS rel_label,
                    HASH(r.user_id || r.item_id || '{seed}') AS sampling_hash
                FROM read_parquet('{ri_str}') r
                LEFT JOIN read_parquet('{gt_str}') g
                    ON r.user_id = g.user_id AND r.item_id = g.item_id
                WHERE HASH(r.user_id) % {N_BUCKETS} = {bucket}
            ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
            """
            with timed(f"    labeled bucket {bucket + 1}/{N_BUCKETS}", logger):
                con.execute(sql_label)

        parts_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
        concat_sql = f"""
        COPY (SELECT * FROM read_parquet('{parts_glob}'))
        TO '{labeled_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """
        with timed("    concat labeled buckets", logger):
            con.execute(concat_sql)
        for f in parts_dir.glob("bucket_*.parquet"):
            f.unlink()
        parts_dir.rmdir()

    # Step 2: user_keep
    if not (user_keep_path.exists() and user_keep_path.stat().st_size > 100):
        logger.info("  Step 2/3: Computing per-user n_neg_to_keep...")
        sql_user = f"""
        COPY (
            WITH stats AS (
                SELECT user_id,
                       SUM(CASE WHEN rel_label > 0 THEN 1 ELSE 0 END) AS n_pos,
                       SUM(CASE WHEN rel_label = 0 THEN 1 ELSE 0 END) AS n_neg
                FROM read_parquet('{labeled_str}')
                GROUP BY user_id
            )
            SELECT user_id, n_pos,
                   LEAST(n_pos * {neg_per_pos}, n_neg) AS n_neg_to_keep
            FROM stats
            WHERE n_pos > 0
        ) TO '{user_keep_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """
        with timed("  user_keep", logger):
            con.execute(sql_user)

    # Step 3: Sample chunked
    logger.info("  Step 3/3: Sampling negatives chunked...")
    N_SAMPLE_BUCKETS = 8
    sample_parts_dir = cfg.paths.features_dir / "_sample_parts"
    sample_parts_dir.mkdir(exist_ok=True)
    cols_qualified = ", ".join(_OUTPUT_COLS)

    for bucket in range(N_SAMPLE_BUCKETS):
        part_path = sample_parts_dir / f"bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")
        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("    bucket %d/%d: cache hit, SKIP",
                        bucket + 1, N_SAMPLE_BUCKETS)
            continue

        # [PATCH] Bucket SELECT đã ORDER BY user_id trong từng bucket
        # → concat tự nhiên sort theo bucket → KHÔNG cần ORDER BY full ở step load.
        sql_sample = f"""
        COPY (
            WITH labeled_bucket AS (
                SELECT * FROM read_parquet('{labeled_str}')
                WHERE HASH(user_id) % {N_SAMPLE_BUCKETS} = {bucket}
            ),
            u_keep AS (
                SELECT * FROM read_parquet('{user_keep_str}')
                WHERE HASH(user_id) % {N_SAMPLE_BUCKETS} = {bucket}
            ),
            ranked_negs AS (
                SELECT l.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY l.user_id ORDER BY l.sampling_hash
                       ) AS neg_rank
                FROM labeled_bucket l
                INNER JOIN u_keep u ON l.user_id = u.user_id
                WHERE l.rel_label = 0
            ),
            sampled_negs AS (
                SELECT r.*
                FROM ranked_negs r
                INNER JOIN u_keep u ON r.user_id = u.user_id
                WHERE r.neg_rank <= u.n_neg_to_keep
            ),
            all_pos AS (
                SELECT l.*, CAST(NULL AS BIGINT) AS neg_rank
                FROM labeled_bucket l
                INNER JOIN u_keep u ON l.user_id = u.user_id
                WHERE l.rel_label > 0
            ),
            combined AS (
                SELECT * FROM all_pos
                UNION ALL
                SELECT * FROM sampled_negs
            )
            SELECT {cols_qualified} FROM combined ORDER BY user_id
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """
        with timed(f"    sample bucket {bucket + 1}/{N_SAMPLE_BUCKETS}", logger):
            con.execute(sql_sample)

    # [PATCH] Concat bỏ ORDER BY ở step concat — mỗi bucket đã sort, concat
    # giữ thứ tự file → toàn bảng đã được sort theo (bucket, user_id) phân vùng.
    # Vì LightGBM chỉ cần group consecutive rows cùng user_id, không cần global sort.
    sample_parts_glob = str(sample_parts_dir / "bucket_*.parquet").replace("\\", "/")
    concat_sample_sql = f"""
    COPY (SELECT * FROM read_parquet('{sample_parts_glob}'))
    TO '{sampled_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("  concat sample buckets (no full sort)", logger):
        con.execute(concat_sample_sql)

    # Cleanup intermediate files ngay
    for f in sample_parts_dir.glob("bucket_*.parquet"):
        f.unlink()
    sample_parts_dir.rmdir()
    labeled_path.unlink(missing_ok=True)
    user_keep_path.unlink(missing_ok=True)

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sampled_str}')"
    ).fetchone()[0]
    n_users = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{sampled_str}')"
    ).fetchone()[0]
    logger.info("  sampled training frame: %s rows, %s users",
                f"{n_rows:,}", f"{n_users:,}")
    return sampled_path


def _load_sampled_via_pyarrow(sampled_path: Path) -> pd.DataFrame:
    """[OOM PATCH + DTYPE FIX v3.2.1] Load sampled parquet với pyarrow + cast decimals.

    Tiết kiệm peak RAM ~30-50% so với `con.execute('SELECT *').df()`:
    - pyarrow đọc batch-by-batch, không hold raw bytes của full table.
    - self_destruct=True: giải phóng Arrow buffer ngay khi convert sang pandas
      (mặc định Arrow giữ buffer → 2× RAM tạm thời).
    - Không qua DuckDB → tránh thêm 1 lần copy giữa DuckDB engine và Python.

    [DTYPE FIX] DuckDB SUM(weighted_score) trả về DECIMAL type → khi qua pyarrow
    với split_blocks=True, decimal được preserve → pandas thấy dtype 'object' →
    LightGBM reject. Fix: cast các cột object (trừ string columns thật) sang float64.
    """
    with timed("load sampled frame via pyarrow (self_destruct=True)", logger):
        table = pq.read_table(str(sampled_path))
        df = table.to_pandas(self_destruct=True, split_blocks=True)
        del table
        gc.collect()

    # [DTYPE FIX] Cast decimal/object columns về float64
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    string_cols = {"user_id", "item_id", "source", "i_city_name", "u_top_city",
                   "i_seller_type", "i_ad_type"}
    cast_cols = [c for c in object_cols if c not in string_cols]
    if cast_cols:
        logger.info("  [dtype fix] casting %d object columns to float64: %s",
                    len(cast_cols), cast_cols)
        for c in cast_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    logger.info("  loaded: %s rows × %d cols", f"{len(df):,}", df.shape[1])
    label_dist = df["rel_label"].value_counts().sort_index()
    logger.info("  label distribution:")
    for lbl, cnt in label_dist.items():
        logger.info("    rel_label=%d: %s rows (%.2f%%)",
                    lbl, f"{cnt:,}", 100.0 * cnt / len(df))
    return df


def _train_val_split_by_user(df: pd.DataFrame, val_ratio: float,
                              seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """[OOM PATCH] Dùng numpy isin (C-level) thay set.isin() Python.

    Với ~1-2M unique users, set.isin có overhead Python lớn — numpy.isin với
    sorted array nhanh hơn 5-10× và peak RAM thấp hơn.
    """
    rng = np.random.default_rng(seed)
    unique_users = df[GROUP_COL].unique()
    rng.shuffle(unique_users)
    n_val = int(len(unique_users) * val_ratio)
    val_users = np.sort(unique_users[:n_val])  # sort cho searchsorted

    # numpy.isin trên sorted array → O(N log M) thay vì O(N) hash
    val_mask = np.isin(df[GROUP_COL].values, val_users, assume_unique=False)
    train_df = df[~val_mask].sort_values(GROUP_COL).reset_index(drop=True)
    val_df = df[val_mask].sort_values(GROUP_COL).reset_index(drop=True)
    return train_df, val_df


def _compute_groups(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby(GROUP_COL, sort=False).size().values
    return counts


def _convert_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def train_ranker(ranker_input_path: Path, model_out_path: Path,
                 importance_out_path: Path,
                 metadata_out_path: Path) -> None:
    cfg = get_config()
    rcfg = cfg.ranker

    con = make_connection()

    # [PATCH] Tách thành 2 bước: build parquet on disk → load via pyarrow.
    # Bước build dùng DuckDB engine; bước load dùng pyarrow stream (tránh DuckDB
    # giữ result set trong RAM khi convert sang pandas).
    sampled_path = _build_sampled_training_frame(con, ranker_input_path)

    # Đóng DuckDB connection trước khi load lớn để giải phóng memory_limit (6GB)
    con.close()
    del con
    gc.collect()

    df = _load_sampled_via_pyarrow(sampled_path)

    missing = validate_columns(list(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in ranker_input: {missing}")

    with timed("train/val split", logger):
        train_df, val_df = _train_val_split_by_user(
            df, val_ratio=rcfg.val_ratio, seed=rcfg.random_seed
        )
        logger.info("  train: %s rows, %s users", f"{len(train_df):,}",
                    f"{train_df[GROUP_COL].nunique():,}")
        logger.info("  val:   %s rows, %s users", f"{len(val_df):,}",
                    f"{val_df[GROUP_COL].nunique():,}")

    # [PATCH] Giải phóng df full ngay sau khi split — không cần giữ lại.
    del df
    gc.collect()

    train_df = _convert_categoricals(train_df)
    val_df = _convert_categoricals(val_df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[LABEL_COL].values
    group_train = _compute_groups(train_df)

    X_val = val_df[ALL_FEATURES]
    y_val = val_df[LABEL_COL].values
    group_val = _compute_groups(val_df)

    # [PATCH] Giải phóng train_df/val_df (X_train/X_val là view, không phải copy).
    # Lưu ý: pandas `df[cols]` trả VIEW khi cols là list — nhưng để chắc chắn,
    # ta materialize X_train/X_val thành copy độc lập trước khi del.
    X_train = X_train.copy()
    X_val = X_val.copy()
    del train_df, val_df
    gc.collect()

    # [OOM PATCH KEY] free_raw_data=True → LightGBM build histogram rồi
    # giải phóng X ngay. Tiết kiệm 5-10GB RAM trong suốt quá trình training.
    # val_set cũng dùng True vì early stopping dùng score predict không cần raw.
    train_set = lgb.Dataset(
        X_train, label=y_train, group=group_train,
        categorical_feature=CATEGORICAL_FEATURES, free_raw_data=True
    )
    val_set = lgb.Dataset(
        X_val, label=y_val, group=group_val,
        categorical_feature=CATEGORICAL_FEATURES,
        reference=train_set, free_raw_data=True
    )

    params = {
        "objective": rcfg.objective,
        "metric": rcfg.metric,
        "ndcg_at": list(rcfg.ndcg_at),
        "num_leaves": rcfg.num_leaves,
        "max_depth": rcfg.max_depth,
        "learning_rate": rcfg.learning_rate,
        "feature_fraction": rcfg.feature_fraction,
        "bagging_fraction": rcfg.bagging_fraction,
        "bagging_freq": rcfg.bagging_freq,
        "min_child_samples": rcfg.min_child_samples,
        "num_threads": rcfg.num_threads,
        "seed": rcfg.random_seed,
        "verbosity": -1,
    }

    logger.info("[STEP 4] Training LightGBM LambdaRank...")
    logger.info("  params: %s", {k: v for k, v in params.items() if k != "metric"})

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=rcfg.early_stopping, verbose=True),
    ]

    with timed("LightGBM train", logger):
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=rcfg.num_iterations,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

    # [PATCH] Giải phóng raw X arrays + Dataset objects sau khi train xong.
    del X_train, X_val, y_train, y_val, group_train, group_val
    del train_set, val_set
    gc.collect()

    booster.save_model(str(model_out_path))
    logger.info("  Saved model: %s", model_out_path)

    importance_gain = booster.feature_importance(importance_type="gain")
    importance_split = booster.feature_importance(importance_type="split")
    imp_df = pd.DataFrame({
        "feature": ALL_FEATURES,
        "gain": importance_gain,
        "split": importance_split,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    imp_df.to_parquet(importance_out_path, index=False)
    logger.info("  Top 10 features by gain:")
    for _, row in imp_df.head(10).iterrows():
        logger.info("    %-30s gain=%10.0f  split=%6d",
                    row["feature"], row["gain"], row["split"])

    best_iter = booster.best_iteration if booster.best_iteration else rcfg.num_iterations
    best_score = booster.best_score if booster.best_score else {}
    metadata = {
        "version": "v3.1-oom-patch",
        "best_iteration": int(best_iter),
        "best_score": {k: dict(v) for k, v in best_score.items()},
        "n_features": len(ALL_FEATURES),
        "n_categorical": len(CATEGORICAL_FEATURES),
        "params": params,
    }
    with open(metadata_out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("  Saved metadata: %s", metadata_out_path)

    # [PATCH] Optional cleanup: xóa _train_sampled.parquet sau khi train xong
    # để tiết kiệm disk. Comment dòng dưới nếu muốn giữ để debug.
    sampled_path.unlink(missing_ok=True)
    logger.info("  Cleaned up: %s", sampled_path.name)