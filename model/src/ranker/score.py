"""Score candidates với trained LightGBM model.

Workflow:
1. Load model từ cache/models/lgb_ranker.txt.
2. Load ranker_input_predict.parquet (~32M rows).
3. Predict batch (chunked nếu OOM).
4. Save scored_pool_predict.parquet với pred_score column.

Output:
- features/scored_pool_predict.parquet (user_id, item_id, pred_score, ...meta)

[DTYPE FIX v3.2.1] DuckDB SUM(weighted_score) trả về DECIMAL → pyarrow batch
preserve type → pandas thấy dtype 'object' → LightGBM reject. Fix: cast các
cột _weighted về float64 ngay sau khi convert batch → pandas trong loop.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common import get_config, get_logger, make_connection, timed
from src.ranker.feature_spec import ALL_FEATURES, CATEGORICAL_FEATURES, ID_COLS

logger = get_logger(__name__)

# [DTYPE FIX] Các cột non-string có thể bị DuckDB DECIMAL → object dtype.
# String columns thật sự (giữ nguyên type):
_STRING_COLS = {"user_id", "item_id", "source", "i_city_name", "u_top_city",
                "i_seller_type", "i_ad_type"}


def _convert_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical features sang pandas category dtype."""
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _fix_decimal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """[DTYPE FIX] Cast các cột object (decimal từ DuckDB SUM) về float64.

    Phải gọi TRƯỚC _convert_categoricals để không cast nhầm categorical cột
    (categorical đang là string/int sẽ dùng astype('category'), không cần cast).
    """
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cast_cols = [c for c in object_cols
                 if c not in _STRING_COLS and c not in CATEGORICAL_FEATURES]
    if cast_cols:
        for c in cast_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def score_candidates(ranker_input_path: Path, model_path: Path,
                      out_path: Path, chunk_size: int = 2_000_000) -> None:
    """Score candidates với LightGBM model, chunked để fit RAM.

    Args:
        ranker_input_path: ranker_input_predict.parquet.
        model_path: lgb_ranker.txt (saved model).
        out_path: scored_pool_predict.parquet.
        chunk_size: số rows per chunk (default 2M, ~1GB RAM).
    """
    logger.info("[STEP 5] Score candidates với model: %s", model_path)
    booster = lgb.Booster(model_file=str(model_path))
    logger.info("  Model: %d iterations, %d features",
                booster.num_trees(), booster.num_feature())

    ri_path = Path(ranker_input_path)
    # Đọc parquet metadata để biết tổng rows
    parquet_file = pq.ParquetFile(ri_path)
    n_total = parquet_file.metadata.num_rows
    logger.info("  Total rows: %s", f"{n_total:,}")

    # Schema cho output
    out_schema = pa.schema([
        ("user_id", pa.string()),
        ("item_id", pa.string()),
        ("pred_score", pa.float32()),
        ("source", pa.string()),
        ("source_score", pa.float64()),
        ("candidate_score", pa.float64()),
    ])

    writer = pq.ParquetWriter(str(out_path), out_schema, compression="snappy")
    n_scored = 0
    dtype_fix_logged = False

    try:
        with timed("score all candidates (chunked)", logger):
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                df = batch.to_pandas()

                # [DTYPE FIX] Cast decimal columns về float64 trước khi vào LightGBM
                if not dtype_fix_logged:
                    object_cols_before = df.select_dtypes(
                        include=["object"]
                    ).columns.tolist()
                    cast_cols = [c for c in object_cols_before
                                 if c not in _STRING_COLS
                                 and c not in CATEGORICAL_FEATURES]
                    if cast_cols:
                        logger.info(
                            "  [dtype fix] casting %d object columns to "
                            "float64: %s", len(cast_cols), cast_cols
                        )
                    dtype_fix_logged = True

                df = _fix_decimal_dtypes(df)
                df = _convert_categoricals(df)

                X = df[ALL_FEATURES]
                pred = booster.predict(X, num_iteration=booster.best_iteration)

                out_df = pd.DataFrame({
                    "user_id": df["user_id"].astype(str),
                    "item_id": df["item_id"].astype(str),
                    "pred_score": pred.astype(np.float32),
                    "source": df["source"].astype(str),
                    "source_score": df["source_score"].astype(np.float64),
                    "candidate_score": df["candidate_score"].astype(np.float64),
                })
                tbl = pa.Table.from_pandas(out_df, schema=out_schema,
                                            preserve_index=False)
                writer.write_table(tbl)

                n_scored += len(df)
                logger.info("  scored %s / %s rows",
                            f"{n_scored:,}", f"{n_total:,}")
    finally:
        writer.close()

    logger.info("  Saved scored_pool: %s (%s rows)",
                out_path, f"{n_scored:,}")

    # Quick stats
    con = make_connection()
    out_str = str(out_path).replace("\\", "/")
    stats = con.execute(f"""
        SELECT
            MIN(pred_score) AS min_score,
            MAX(pred_score) AS max_score,
            AVG(pred_score) AS avg_score,
            COUNT(DISTINCT user_id) AS n_users
        FROM read_parquet('{out_str}')
    """).fetchone()
    logger.info("  pred_score stats: min=%.4f max=%.4f avg=%.4f",
                stats[0], stats[1], stats[2])
    logger.info("  scored users: %s", f"{stats[3]:,}")