"""Temporal features: features phụ thuộc thời gian (recency tiers).

Bucketize recency thành tiers để model học non-linear pattern. LightGBM tree-based
có thể tự split continuous, nhưng tier explicit giúp model học nhanh hơn.

Output columns:
- user_id, item_id
- ui_recency_tier         (recency của user × item event cuối):
                           0=≤3d, 1=4-7d, 2=8-14d, 3=15-30d, 4=31-180d, 5=>180d/never
- u_activity_recency_tier (user last active any item):
                           same buckets
- i_age_tier              (item age):
                           0=≤3d, 1=4-7d, 2=8-14d, 3=15-30d, 4=31-180d, 5=>180d

Output: temporal_features_{mode}.parquet
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)


def build_temporal_features(cutoff_date: str, mode: str,
                              candidates_path: Path,
                              out_path: Path) -> None:
    """Build temporal features cho (user, item) pairs từ candidates.

    Args:
        cutoff_date: YYYY-MM-DD.
        mode: 'train' hoặc 'predict'.
        candidates_path: candidates_{mode}.parquet.
        out_path: output parquet.
    """
    cfg = get_config()
    user_item = str(cfg.paths.agg_dir / "user_item_daily.parquet").replace("\\", "/")
    user_daily = str(cfg.paths.agg_dir / "user_daily.parquet").replace("\\", "/")
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    cand_str = str(candidates_path).replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    # Recency tier buckets (days)
    def tier_case(days_col: str) -> str:
        return (
            f"CASE "
            f"  WHEN {days_col} <= 3   THEN 0 "
            f"  WHEN {days_col} <= 7   THEN 1 "
            f"  WHEN {days_col} <= 14  THEN 2 "
            f"  WHEN {days_col} <= 30  THEN 3 "
            f"  WHEN {days_col} <= 180 THEN 4 "
            f"  ELSE 5 "
            f"END"
        )

    sql = f"""
    COPY (
        WITH cand_pairs AS (
            SELECT DISTINCT user_id, item_id
            FROM read_parquet('{cand_str}')
        ),
        ui_recency AS (
            -- Days since last (user, item) event
            SELECT
                user_id, item_id,
                DATE_DIFF('day', MAX(date), DATE '{cutoff_date}') AS ui_days_last
            FROM read_parquet('{user_item}')
            WHERE date < DATE '{cutoff_date}'
            GROUP BY user_id, item_id
        ),
        u_recency AS (
            -- Days since user last active (any item)
            SELECT
                user_id,
                DATE_DIFF('day', MAX(date), DATE '{cutoff_date}') AS u_days_last
            FROM read_parquet('{user_daily}')
            WHERE date < DATE '{cutoff_date}'
            GROUP BY user_id
        ),
        i_age AS (
            -- Item age = cutoff - posted_date
            SELECT DISTINCT
                item_id,
                DATE_DIFF('day', posted_date, DATE '{cutoff_date}') AS i_age_days
            FROM read_parquet('{dim_glob}')
            WHERE posted_date IS NOT NULL
        )
        SELECT
            c.user_id, c.item_id,
            {tier_case('COALESCE(ur.ui_days_last, 999)')} AS ui_recency_tier,
            {tier_case('COALESCE(uda.u_days_last, 999)')} AS u_activity_recency_tier,
            {tier_case('COALESCE(ia.i_age_days, 999)')} AS i_age_tier
        FROM cand_pairs c
        LEFT JOIN ui_recency ur ON c.user_id = ur.user_id AND c.item_id = ur.item_id
        LEFT JOIN u_recency uda ON c.user_id = uda.user_id
        LEFT JOIN i_age ia      ON c.item_id = ia.item_id
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """

    con = make_connection()
    with timed(f"build temporal_features (mode={mode}, cutoff={cutoff_date})", logger):
        con.execute(sql)

    n_pairs = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  temporal_features (%s): %s pairs", mode, f"{n_pairs:,}")


