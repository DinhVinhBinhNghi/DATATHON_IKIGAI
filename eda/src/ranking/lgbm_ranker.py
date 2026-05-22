"""LightGBM LambdaRank scaffold cho Datathon 2026 Chợ Tốt BĐS.

Pipeline:
1) Build training pairs (user, item, label) từ events_pos trong train window.
2) Build features tabular cho mỗi pair:
   - User features: lịch sử category/city, frequency, recency, device
   - Item features: category, ad_type, seller_type, price_bucket, area, age,
                    images_count, snapshot views/contacts
   - User × Item features: category match, city match, seller seen before, etc.
3) Local holdout = HOLDOUT_START_DATE..HOLDOUT_END_DATE (xem constants.py).
   Train trên events trước HOLDOUT_START_DATE.
4) Group-aware split: 1 group = 1 user trong query (cho LambdaRank).
5) Negative sampling = in-candidate (xem DECISIONS_NEEDED.md mục 8).
6) Train + serialize → outputs/models/lgb_ranker.txt
7) Eval: Recall@10 + NDCG@10 trên local holdout.

Usage:
    python scripts/08_train_ranker.py --data-root "C:/Datathon_Data"

API:
    train_ranker(con, paths, config) -> Path tới model file
    score_candidates(con, paths, model_path, k=10) -> DataFrame final recommendations
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.constants import (
    HOLDOUT_END_DATE,
    HOLDOUT_START_DATE,
    SEED,
    TRAIN_END_DATE,
    TRAIN_START_DATE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Config
# ============================================================================


@dataclass
class RankerConfig:
    """Hyperparams cho LGB LambdaRank. Có thể override qua YAML."""

    # Data
    n_negatives_per_pos: int = 10
    holdout_start: str = HOLDOUT_START_DATE
    holdout_end: str = HOLDOUT_END_DATE
    min_user_positives: int = 1  # filter users có >= N positives trong train

    # LGB params
    objective: str = "lambdarank"
    metric: str = "ndcg"
    eval_at: tuple = (10,)
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 20
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    lambda_l2: float = 1.0
    seed: int = SEED

    # Early stopping
    early_stopping_rounds: int = 30

    # Categorical features (LGB native handling)
    categorical_features: list = field(default_factory=lambda: [
        "i_category",
        "i_ad_type",
        "i_seller_type",
    ])


def load_config(yaml_path: Optional[str | Path] = None) -> RankerConfig:
    """Load config từ YAML nếu có, fallback default."""
    cfg = RankerConfig()
    if yaml_path is None:
        return cfg
    path = Path(yaml_path)
    if not path.exists():
        logger.warning("Config file not found: %s. Using defaults.", path)
        return cfg
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        ranker_data = data.get("ranker", {})
        for k, v in ranker_data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        logger.info("Loaded ranker config from %s", path)
    except ImportError:
        logger.warning("pyyaml not installed; using default config")
    except Exception as e:
        logger.warning("Failed to parse config %s: %s. Using defaults.", path, e)
    return cfg


# ============================================================================
# Feature engineering
# ============================================================================


# Feature column names (frozen for reproducibility across train/inference)
ITEM_FEATURES = [
    "i_category",
    "i_ad_type",
    "i_seller_type",
    "i_area_sqm",
    "i_bedrooms",
    "i_bathrooms",
    "i_images_count",
    "i_listing_age_days",
    "i_views_7d",
    "i_contacts_7d",
    "i_views_28d",
    "i_contacts_28d",
    "i_contact_rate_28d",
]

USER_FEATURES = [
    "u_n_pageviews",
    "u_n_positives",
    "u_n_distinct_items",
    "u_n_distinct_categories",
    "u_days_since_last_event",
    "u_top_category",
    "u_top_city",
]

CROSS_FEATURES = [
    "x_category_match",
    "x_city_match",
    "x_user_saw_item_before",
    "x_user_contacted_seller_before",
    "x_candidate_source",  # categorical: cat_city / category / city / global
]


def build_item_features(con, holdout_start: str, cache_path: Path) -> Path:
    """Tạo bảng item features. Snapshot stats tính trên window (train_start, holdout_start)."""
    if cache_path.exists():
        logger.info("Item features already cached at %s", cache_path)
        return cache_path
    logger.info("Building item features -> %s", cache_path)
    query = f"""
    COPY (
        WITH snap28 AS (
            SELECT
                item_id,
                SUM(views_24h) FILTER (WHERE date >= DATE '{holdout_start}' - INTERVAL 7 DAY) AS views_7d,
                SUM(contacts_24h) FILTER (WHERE date >= DATE '{holdout_start}' - INTERVAL 7 DAY) AS contacts_7d,
                SUM(views_24h) FILTER (WHERE date >= DATE '{holdout_start}' - INTERVAL 28 DAY) AS views_28d,
                SUM(contacts_24h) FILTER (WHERE date >= DATE '{holdout_start}' - INTERVAL 28 DAY) AS contacts_28d,
                MAX(listing_age_days) AS listing_age_days
            FROM snap_clean
            WHERE date < DATE '{holdout_start}'
            GROUP BY item_id
        )
        SELECT
            d.item_id,
            d.category AS i_category,
            d.ad_type AS i_ad_type,
            d.seller_type AS i_seller_type,
            d.seller_id AS i_seller_id,
            d.area_sqm AS i_area_sqm,
            d.bedrooms AS i_bedrooms,
            d.bathrooms AS i_bathrooms,
            d.images_count AS i_images_count,
            d.city_name AS i_city_name,
            COALESCE(s.listing_age_days, 0) AS i_listing_age_days,
            COALESCE(s.views_7d, 0) AS i_views_7d,
            COALESCE(s.contacts_7d, 0) AS i_contacts_7d,
            COALESCE(s.views_28d, 0) AS i_views_28d,
            COALESCE(s.contacts_28d, 0) AS i_contacts_28d,
            CASE
                WHEN COALESCE(s.views_28d, 0) > 0
                THEN s.contacts_28d::DOUBLE / s.views_28d
                ELSE 0.0
            END AS i_contact_rate_28d
        FROM dim_clean d
        LEFT JOIN snap28 s USING (item_id)
    ) TO '{cache_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    return cache_path


def build_user_features(con, holdout_start: str, cache_path: Path) -> Path:
    """Tạo bảng user features từ events trước holdout_start."""
    if cache_path.exists():
        logger.info("User features already cached at %s", cache_path)
        return cache_path
    logger.info("Building user features -> %s", cache_path)
    query = f"""
    COPY (
        WITH user_pos AS (
            SELECT
                user_id,
                COUNT(*) AS n_positives,
                COUNT(DISTINCT item_id) AS n_distinct_items,
                MAX(event_ts) AS last_pos_ts
            FROM events_pos
            WHERE date < DATE '{holdout_start}'
            GROUP BY user_id
        ),
        user_pos_cat AS (
            SELECT user_id, e.category AS top_category, n_evt
            FROM (
                SELECT
                    user_id,
                    category,
                    COUNT(*) AS n_evt,
                    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, category) AS rn
                FROM events_pos
                WHERE date < DATE '{holdout_start}'
                GROUP BY user_id, category
            ) e
            WHERE rn = 1
        ),
        user_pos_city AS (
            SELECT user_id, e.city_name AS top_city, n_evt
            FROM (
                SELECT
                    user_id,
                    city_name,
                    COUNT(*) AS n_evt,
                    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, city_name) AS rn
                FROM events_pos
                WHERE date < DATE '{holdout_start}'
                  AND city_name IS NOT NULL
                GROUP BY user_id, city_name
            ) e
            WHERE rn = 1
        ),
        user_cat_count AS (
            SELECT user_id, COUNT(DISTINCT category) AS n_distinct_categories
            FROM events_pos
            WHERE date < DATE '{holdout_start}'
            GROUP BY user_id
        )
        SELECT
            COALESCE(up.user_id, uc.user_id) AS user_id,
            COALESCE(up.n_positives, 0) AS u_n_positives,
            COALESCE(up.n_distinct_items, 0) AS u_n_distinct_items,
            COALESCE(uc.n_distinct_categories, 0) AS u_n_distinct_categories,
            CAST(DATE_DIFF('day', up.last_pos_ts, TIMESTAMP '{holdout_start} 00:00:00') AS DOUBLE) AS u_days_since_last_event,
            upc.top_category AS u_top_category,
            upcity.top_city AS u_top_city,
            0 AS u_n_pageviews  -- TODO: build từ events_pos hoặc pageview agg nếu có
        FROM user_pos up
        FULL OUTER JOIN user_cat_count uc USING (user_id)
        LEFT JOIN user_pos_cat upc USING (user_id)
        LEFT JOIN user_pos_city upcity USING (user_id)
    ) TO '{cache_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    return cache_path


def build_training_pairs(
    con,
    item_features_path: Path,
    user_features_path: Path,
    holdout_start: str,
    cache_path: Path,
    n_negatives_per_pos: int = 10,
    min_user_positives: int = 1,
) -> Path:
    """Build (user, item, label) pairs cho training.

    Positive = (user, item) đã có positive event trước holdout_start.
    Negative = sample từ candidate pool của user (cat_city / cat / city / global)
               nhưng user chưa có positive event với item đó.

    Output columns:
        user_id, item_id, label, plus all features.
    """
    if cache_path.exists():
        logger.info("Training pairs already cached at %s", cache_path)
        return cache_path
    logger.info("Building training pairs -> %s", cache_path)

    # Đảm bảo item_popularity + test_user_profile + candidate_scores đã sẵn sàng
    # (vì in-candidate negative cần candidate pool).
    # Pipeline assumption: build_candidates đã chạy trước.
    has_candidates = False
    try:
        con.execute("SELECT 1 FROM candidate_scores LIMIT 1")
        has_candidates = True
    except Exception:
        logger.warning("candidate_scores view không tồn tại. Sẽ dùng random negatives.")

    if has_candidates:
        query = f"""
        COPY (
            WITH positives AS (
                SELECT user_id, item_id, MAX(event_ts) AS last_pos_ts
                FROM events_pos
                WHERE date < DATE '{holdout_start}'
                GROUP BY user_id, item_id
            ),
            qualifying_users AS (
                SELECT user_id
                FROM positives
                GROUP BY user_id
                HAVING COUNT(*) >= {min_user_positives}
            ),
            pos_pairs AS (
                SELECT p.user_id, p.item_id, 1 AS label
                FROM positives p
                JOIN qualifying_users q USING (user_id)
            ),
            -- Negatives = candidate pool of qualifying users minus their positives
            neg_candidates AS (
                SELECT
                    cs.user_id,
                    cs.item_id,
                    cs.best_source,
                    cs.score AS candidate_score,
                    ROW_NUMBER() OVER (PARTITION BY cs.user_id ORDER BY RANDOM()) AS rn
                FROM candidate_scores cs
                JOIN qualifying_users q ON cs.user_id = q.user_id
                LEFT JOIN positives p ON cs.user_id = p.user_id AND cs.item_id = p.item_id
                WHERE p.user_id IS NULL  -- not in positives
            ),
            neg_pairs AS (
                SELECT user_id, item_id, 0 AS label
                FROM neg_candidates
                WHERE rn <= {n_negatives_per_pos}
            ),
            -- Also keep candidate_source for positives (if pos item is in candidate pool)
            all_pairs AS (
                SELECT pp.user_id, pp.item_id, pp.label,
                       COALESCE(cs.best_source, 'positive_outside_pool') AS x_candidate_source
                FROM pos_pairs pp
                LEFT JOIN candidate_scores cs USING (user_id, item_id)
                UNION ALL
                SELECT np.user_id, np.item_id, np.label, nc.best_source AS x_candidate_source
                FROM neg_pairs np
                JOIN neg_candidates nc ON np.user_id = nc.user_id AND np.item_id = nc.item_id
            )
            SELECT * FROM all_pairs
        ) TO '{cache_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    else:
        # Random negative fallback
        query = f"""
        COPY (
            WITH positives AS (
                SELECT user_id, item_id, 1 AS label
                FROM events_pos
                WHERE date < DATE '{holdout_start}'
                GROUP BY user_id, item_id
            ),
            popular_items AS (
                SELECT item_id, COUNT(*) AS n_pos
                FROM events_pos
                WHERE date < DATE '{holdout_start}'
                GROUP BY item_id
                ORDER BY n_pos DESC
                LIMIT 5000
            ),
            negatives AS (
                SELECT p.user_id, pi.item_id, 0 AS label
                FROM (SELECT DISTINCT user_id FROM positives) p
                CROSS JOIN popular_items pi
                LEFT JOIN positives pos
                    ON pos.user_id = p.user_id AND pos.item_id = pi.item_id
                WHERE pos.user_id IS NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY p.user_id ORDER BY RANDOM()) <= {n_negatives_per_pos}
            )
            SELECT user_id, item_id, label, 'random' AS x_candidate_source
            FROM positives
            UNION ALL
            SELECT user_id, item_id, label, 'random' AS x_candidate_source
            FROM negatives
        ) TO '{cache_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    con.execute(query)
    return cache_path


def assemble_training_frame(
    con,
    pairs_path: Path,
    item_features_path: Path,
    user_features_path: Path,
) -> pd.DataFrame:
    """Join pairs với features → final tabular frame cho LGB."""
    logger.info("Assembling training frame from %s", pairs_path)
    query = f"""
    WITH pairs AS (
        SELECT * FROM read_parquet('{pairs_path.as_posix()}')
    ),
    items AS (
        SELECT * FROM read_parquet('{item_features_path.as_posix()}')
    ),
    users AS (
        SELECT * FROM read_parquet('{user_features_path.as_posix()}')
    )
    SELECT
        p.user_id,
        p.item_id,
        p.label,
        p.x_candidate_source,
        -- Item features
        i.i_category, i.i_ad_type, i.i_seller_type, i.i_seller_id, i.i_city_name,
        i.i_area_sqm, i.i_bedrooms, i.i_bathrooms, i.i_images_count,
        i.i_listing_age_days, i.i_views_7d, i.i_contacts_7d,
        i.i_views_28d, i.i_contacts_28d, i.i_contact_rate_28d,
        -- User features
        u.u_n_positives, u.u_n_distinct_items, u.u_n_distinct_categories,
        u.u_days_since_last_event, u.u_top_category, u.u_top_city, u.u_n_pageviews,
        -- Cross features computed below in pandas
    FROM pairs p
    LEFT JOIN items i USING (item_id)
    LEFT JOIN users u USING (user_id)
    """
    df = con.execute(query).df()
    # Cross features
    df["x_category_match"] = (df["u_top_category"].astype("Int64") == df["i_category"].astype("Int64")).astype("int8")
    df["x_city_match"] = (df["u_top_city"].astype("string") == df["i_city_name"].astype("string")).astype("int8")
    # x_user_saw_item_before / x_user_contacted_seller_before: simplified placeholders
    # (full computation would need additional join, omitted for baseline speed)
    df["x_user_saw_item_before"] = 0
    df["x_user_contacted_seller_before"] = 0
    return df


# ============================================================================
# Training
# ============================================================================


def train_lgb_ranker(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    config: RankerConfig,
    model_out: Path,
) -> dict:
    """Train LGB LambdaRank.

    df must have columns: user_id, item_id, label, + all features.
    Groups là user_id; mỗi group là 1 query trong LambdaRank.
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "lightgbm chưa được cài. Thêm 'lightgbm' vào requirements.txt rồi pip install."
        ) from e

    # Sort by user_id để group đúng
    df_train = df_train.sort_values("user_id").reset_index(drop=True)
    df_valid = df_valid.sort_values("user_id").reset_index(drop=True)

    feature_cols = [c for c in ITEM_FEATURES + USER_FEATURES + CROSS_FEATURES if c in df_train.columns]
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    X_train = df_train[feature_cols]
    y_train = df_train["label"].astype(int).values
    g_train = df_train.groupby("user_id").size().values  # group sizes

    X_valid = df_valid[feature_cols]
    y_valid = df_valid["label"].astype(int).values
    g_valid = df_valid.groupby("user_id").size().values

    cat_cols = [c for c in config.categorical_features if c in feature_cols]
    # Coerce categorical columns to category dtype for LGB
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")

    train_set = lgb.Dataset(X_train, label=y_train, group=g_train, categorical_feature=cat_cols)
    valid_set = lgb.Dataset(X_valid, label=y_valid, group=g_valid, categorical_feature=cat_cols, reference=train_set)

    params = {
        "objective": config.objective,
        "metric": config.metric,
        "ndcg_eval_at": list(config.eval_at),
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "min_child_samples": config.min_child_samples,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "lambda_l2": config.lambda_l2,
        "verbose": -1,
        "seed": config.seed,
    }

    logger.info("Starting LGB training... n_train=%d, n_valid=%d", len(y_train), len(y_valid))
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=config.n_estimators,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(50),
        ],
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_out))
    logger.info("Saved model -> %s", model_out)

    # Save feature importance
    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    imp_path = model_out.parent / f"{model_out.stem}_feature_importance.csv"
    imp.to_csv(imp_path, index=False)
    logger.info("Feature importance -> %s", imp_path)

    # Save feature column list for inference reproducibility
    meta = {
        "feature_cols": feature_cols,
        "categorical_features": cat_cols,
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "best_iteration": int(booster.best_iteration or 0),
        "params": params,
    }
    meta_path = model_out.parent / f"{model_out.stem}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return meta


# ============================================================================
# Scoring & evaluation
# ============================================================================


def score_candidates(
    con,
    model_path: Path,
    item_features_path: Path,
    user_features_path: Path,
    k: int = 10,
) -> pd.DataFrame:
    """Score candidate_scores với LGB model → top-K per user.

    Output: user_id, rank, item_id, score, source='lgb_ranker'
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm chưa được cài.") from e

    meta_path = model_path.parent / f"{model_path.stem}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    cat_cols = meta.get("categorical_features", [])

    booster = lgb.Booster(model_file=str(model_path))
    logger.info("Loaded LGB ranker from %s", model_path)

    # Pull candidate set joined với features
    query = f"""
    WITH cs AS (
        SELECT user_id, item_id, best_source AS x_candidate_source, score AS pop_score
        FROM candidate_scores
    ),
    items AS (
        SELECT * FROM read_parquet('{item_features_path.as_posix()}')
    ),
    users AS (
        SELECT * FROM read_parquet('{user_features_path.as_posix()}')
    )
    SELECT
        cs.user_id, cs.item_id, cs.x_candidate_source, cs.pop_score,
        i.i_category, i.i_ad_type, i.i_seller_type, i.i_seller_id, i.i_city_name,
        i.i_area_sqm, i.i_bedrooms, i.i_bathrooms, i.i_images_count,
        i.i_listing_age_days, i.i_views_7d, i.i_contacts_7d,
        i.i_views_28d, i.i_contacts_28d, i.i_contact_rate_28d,
        u.u_n_positives, u.u_n_distinct_items, u.u_n_distinct_categories,
        u.u_days_since_last_event, u.u_top_category, u.u_top_city, u.u_n_pageviews
    FROM cs
    LEFT JOIN items i USING (item_id)
    LEFT JOIN users u USING (user_id)
    """
    df = con.execute(query).df()
    df["x_category_match"] = (df["u_top_category"].astype("Int64") == df["i_category"].astype("Int64")).astype("int8")
    df["x_city_match"] = (df["u_top_city"].astype("string") == df["i_city_name"].astype("string")).astype("int8")
    df["x_user_saw_item_before"] = 0
    df["x_user_contacted_seller_before"] = 0

    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    X = df[feature_cols]
    df["score_lgb"] = booster.predict(X)

    # Top-K per user
    df["rank"] = df.groupby("user_id")["score_lgb"].rank(method="first", ascending=False)
    out = df[df["rank"] <= k].copy()
    out["rank"] = out["rank"].astype(int)
    out["source"] = "lgb_ranker"
    return out[["user_id", "rank", "item_id", "score_lgb", "i_seller_id", "i_listing_age_days", "i_seller_type", "source"]].sort_values(["user_id", "rank"])


def evaluate_recall_ndcg(predictions: pd.DataFrame, ground_truth: pd.DataFrame, k: int = 10) -> dict:
    """Tính Recall@K và NDCG@K trên local holdout.

    predictions: cột user_id, item_id, rank (1..K)
    ground_truth: cột user_id, item_id (positive events trong holdout window)
    """
    gt = ground_truth.groupby("user_id")["item_id"].apply(set).to_dict()
    preds = predictions.sort_values(["user_id", "rank"]).groupby("user_id")["item_id"].apply(list).to_dict()

    recalls, ndcgs = [], []
    for user_id, true_items in gt.items():
        if not true_items:
            continue
        pred_items = preds.get(user_id, [])[:k]
        if not pred_items:
            recalls.append(0.0)
            ndcgs.append(0.0)
            continue
        hits = [1 if p in true_items else 0 for p in pred_items]
        recall = sum(hits) / len(true_items)
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(true_items))))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        recalls.append(recall)
        ndcgs.append(ndcg)

    return {
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_eval": len(recalls),
    }


def build_holdout_ground_truth(con, holdout_start: str, holdout_end: str) -> pd.DataFrame:
    """Lấy ground truth = positive events trong holdout window."""
    query = f"""
    SELECT user_id, item_id
    FROM events_pos
    WHERE date >= DATE '{holdout_start}'
      AND date <= DATE '{holdout_end}'
    GROUP BY user_id, item_id
    """
    return con.execute(query).df()


# ============================================================================
# Pipeline entrypoint
# ============================================================================


def run_ranker_pipeline(
    con,
    paths,
    config: Optional[RankerConfig] = None,
) -> dict:
    """Full ranker pipeline: features → pairs → train → eval.

    Returns:
        {"model_path": Path, "metrics": dict, "config": dict}
    """
    config = config or RankerConfig()
    cache_dir = paths.cache_dir / "ranker"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir = paths.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    item_features_path = cache_dir / "item_features.parquet"
    user_features_path = cache_dir / "user_features.parquet"
    pairs_path = cache_dir / "training_pairs.parquet"

    # Step 1: features
    build_item_features(con, config.holdout_start, item_features_path)
    build_user_features(con, config.holdout_start, user_features_path)

    # Step 2: pairs
    build_training_pairs(
        con,
        item_features_path,
        user_features_path,
        config.holdout_start,
        pairs_path,
        n_negatives_per_pos=config.n_negatives_per_pos,
        min_user_positives=config.min_user_positives,
    )

    # Step 3: assemble frame and split train/valid 80/20 by user
    df = assemble_training_frame(con, pairs_path, item_features_path, user_features_path)
    logger.info("Assembled frame: %d rows, %d unique users", len(df), df["user_id"].nunique())

    rng = np.random.RandomState(config.seed)
    users = df["user_id"].unique()
    rng.shuffle(users)
    n_valid = max(1, int(0.2 * len(users)))
    valid_users = set(users[:n_valid])
    df_valid = df[df["user_id"].isin(valid_users)].copy()
    df_train = df[~df["user_id"].isin(valid_users)].copy()

    # Step 4: train
    model_path = model_dir / "lgb_ranker.txt"
    meta = train_lgb_ranker(df_train, df_valid, config, model_path)

    # Step 5: eval on local holdout (positive events in holdout window)
    ground_truth = build_holdout_ground_truth(con, config.holdout_start, config.holdout_end)
    logger.info("Holdout ground truth: %d (user,item) pairs", len(ground_truth))

    predictions = score_candidates(con, model_path, item_features_path, user_features_path, k=10)
    metrics = evaluate_recall_ndcg(predictions, ground_truth, k=10)
    metrics["variant"] = "rank_only"
    logger.info("Local holdout metrics: %s", metrics)

    metrics_path = paths.table_dir / "ranker_holdout_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info("Metrics saved -> %s", metrics_path)

    return {
        "model_path": model_path,
        "metrics": metrics,
        "meta": meta,
    }
