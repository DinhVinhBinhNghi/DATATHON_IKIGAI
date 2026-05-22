"""Build internal ground truth: events trong [gt_start, gt_end].

Internal GT mimics test window distribution. Dùng để:
- Train ranker (đã có trong src.ranker.train._build_internal_gt_table — shared).
- Evaluate local Recall@10, NDCG@10 trước Kaggle.

Schema:
- user_id, item_id
- gt_weighted_score (SUM weighted_score events trong GT window)
- gt_n_events
"""
from __future__ import annotations

from pathlib import Path

from src.common import (
    file_exists_nonempty,
    get_config,
    get_logger,
    make_connection,
    timed,
)

logger = get_logger(__name__)


def build_internal_ground_truth(out_path: Path | None = None) -> Path:
    """Build internal GT table (idempotent — skip nếu cache).

    Args:
        out_path: nếu None, dùng default cfg.paths.gt_dir/internal_gt.parquet.

    Returns:
        Path tới GT parquet file.
    """
    cfg = get_config()
    if out_path is None:
        out_path = cfg.paths.gt_dir / "internal_gt.parquet"

    if file_exists_nonempty(out_path):
        logger.info("[GT] cache hit: %s", out_path)
        return out_path

    events_glob = str(cfg.paths.fact_events_dir / "*.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")
    gt_start = cfg.windows.internal_gt_start
    gt_end = cfg.windows.internal_gt_end
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
        SELECT
            user_id,
            item_id,
            SUM({weight_case}) AS gt_weighted_score,
            COUNT(*) AS gt_n_events
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{gt_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{gt_end} 23:59:59'
          AND user_id IS NOT NULL
          AND item_id IS NOT NULL
          AND ({weight_case}) > 0
        GROUP BY user_id, item_id
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """

    con = make_connection()
    with timed(f"build internal_gt [{gt_start} - {gt_end}]", logger):
        con.execute(sql)

    n_pairs = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_users = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_items = con.execute(
        f"SELECT COUNT(DISTINCT item_id) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  internal_gt: %s pairs, %s users, %s items",
                f"{n_pairs:,}", f"{n_users:,}", f"{n_items:,}")

    return out_path
