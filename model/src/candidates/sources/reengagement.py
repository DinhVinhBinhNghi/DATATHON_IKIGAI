"""Reengagement source: items user đã interact với (weighted_score > 0).

Đây là source RECALL CAO NHẤT theo Step 7 analysis của v2.4.0 (~21% recall_share).
Lý do: real estate users thường quay lại xem tin đã thích trước khi contact.

SQL:
    SELECT user_id, item_id, source_score
    FROM user_item_daily
    WHERE date < cutoff
      AND weighted_score > 0           -- chỉ items có positive interaction
    GROUP BY user_id, item_id
    HAVING SUM(weighted_score) > 0
    -- limit top-N per user theo SUM(weighted_score) DESC, last_event DESC

source_score = SUM(weighted_score) × time_decay
    time_decay = exp(-days_since_last * ln(2) / half_life)
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)


def build_reengagement(cutoff_date: str, top_n_per_user: int,
                       out_path: Path) -> None:
    """Build reengagement candidates per user.

    Args:
        cutoff_date: YYYY-MM-DD, chỉ dùng events < ngày này.
        top_n_per_user: limit per user (default 50).
        out_path: output parquet.

    Output columns:
        user_id, item_id, source ('reengagement'), source_score, last_event_date
    """
    cfg = get_config()
    user_item_path = str(cfg.paths.agg_dir / "user_item_daily.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    # Half-life decay: items contact gần đây có weight cao hơn
    half_life = 14.0  # days

    sql = f"""
    COPY (
        WITH base AS (
            SELECT
                user_id,
                item_id,
                SUM(weighted_score) AS total_weighted,
                MAX(date) AS last_event_date,
                DATE_DIFF('day', MAX(date), DATE '{cutoff_date}') AS days_since_last
            FROM read_parquet('{user_item_path}')
            WHERE date < DATE '{cutoff_date}'
              AND weighted_score > 0
            GROUP BY user_id, item_id
        ),
        scored AS (
            SELECT
                user_id,
                item_id,
                last_event_date,
                total_weighted * EXP(-days_since_last * LN(2) / {half_life}) AS source_score
            FROM base
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY source_score DESC, last_event_date DESC
                ) AS rn
            FROM scored
        )
        SELECT
            user_id,
            item_id,
            'reengagement' AS source,
            source_score,
            last_event_date
        FROM ranked
        WHERE rn <= {top_n_per_user}
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """

    con = make_connection()
    with timed(f"build reengagement (cutoff={cutoff_date})", logger):
        con.execute(sql)

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_users = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  reengagement: %s rows, %s users (avg %.1f items/user)",
                f"{n_rows:,}", f"{n_users:,}", n_rows / max(n_users, 1))
