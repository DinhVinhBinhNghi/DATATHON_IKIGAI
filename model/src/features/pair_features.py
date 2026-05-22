"""User × Item interaction features (per pair).

Per (user, item) pair từ candidates, đo interaction history trước cutoff.

Output columns:
- user_id, item_id
- ui_total_weighted      (SUM weighted_score trong full history)
- ui_recent_weighted     (SUM trong recency_window_days)
- ui_n_pageview          (số pageview events)
- ui_n_pos_events        (số positive events count)
- ui_max_dwell           (max dwell time)
- ui_n_active_days       (số ngày user touch item)
- ui_days_since_last     (ngày từ event cuối tới cutoff, 999 nếu chưa interact)
- ui_days_since_first    (ngày từ event đầu tới cutoff, 999 nếu chưa interact)

NULL/missing → 0/999 (default cho items user chưa interact, dùng nhiều cho cold).
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)


def build_pair_features(cutoff_date: str, mode: str,
                         candidates_path: Path,
                         out_path: Path) -> None:
    """Build user×item features cho các (user, item) trong candidates.

    Args:
        cutoff_date: YYYY-MM-DD.
        mode: 'train' hoặc 'predict'.
        candidates_path: candidates_{mode}.parquet để biết các (user, item) cần feature.
        out_path: output parquet.
    """
    cfg = get_config()
    user_item = str(cfg.paths.agg_dir / "user_item_daily.parquet").replace("\\", "/")
    cand_str = str(candidates_path).replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    recency = cfg.candidates.recency_window_days

    sql = f"""
    COPY (
        WITH cand_pairs AS (
            -- Distinct (user, item) cần features
            SELECT DISTINCT user_id, item_id
            FROM read_parquet('{cand_str}')
        ),
        ui_agg AS (
            SELECT
                user_id, item_id,
                SUM(weighted_score) AS ui_total_weighted,
                SUM(CASE WHEN date >= DATE '{cutoff_date}' - INTERVAL {recency} DAY
                         THEN weighted_score ELSE 0.0 END) AS ui_recent_weighted,
                SUM(n_pageview) AS ui_n_pageview,
                SUM(n_pos_events) AS ui_n_pos_events,
                MAX(max_dwell) AS ui_max_dwell,
                COUNT(DISTINCT date) AS ui_n_active_days,
                MIN(date) AS first_date,
                MAX(date) AS last_date
            FROM read_parquet('{user_item}')
            WHERE date < DATE '{cutoff_date}'
            GROUP BY user_id, item_id
        )
        SELECT
            c.user_id,
            c.item_id,
            COALESCE(a.ui_total_weighted, 0.0)  AS ui_total_weighted,
            COALESCE(a.ui_recent_weighted, 0.0) AS ui_recent_weighted,
            COALESCE(a.ui_n_pageview, 0)        AS ui_n_pageview,
            COALESCE(a.ui_n_pos_events, 0)      AS ui_n_pos_events,
            COALESCE(a.ui_max_dwell, 0.0)       AS ui_max_dwell,
            COALESCE(a.ui_n_active_days, 0)     AS ui_n_active_days,
            COALESCE(
                DATE_DIFF('day', a.last_date, DATE '{cutoff_date}'),
                999
            ) AS ui_days_since_last,
            COALESCE(
                DATE_DIFF('day', a.first_date, DATE '{cutoff_date}'),
                999
            ) AS ui_days_since_first
        FROM cand_pairs c
        LEFT JOIN ui_agg a
            ON c.user_id = a.user_id AND c.item_id = a.item_id
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """

    con = make_connection()
    with timed(f"build pair_features (mode={mode}, cutoff={cutoff_date})", logger):
        con.execute(sql)

    n_pairs = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_with_history = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}') WHERE ui_total_weighted > 0"
    ).fetchone()[0]
    logger.info("  pair_features (%s): %s pairs (%s with history)",
                mode, f"{n_pairs:,}", f"{n_with_history:,}")


