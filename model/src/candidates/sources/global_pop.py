"""Global popularity source — chunked OOM-safe, no exclude."""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)

N_BUCKETS = 8


def build_global_pop_candidates(cutoff_date: str, top_n_per_user: int,
                                 out_path: Path,
                                 test_users_path: Path | None = None) -> None:
    cfg = get_config()
    item_daily = str(cfg.paths.agg_dir / "item_daily.parquet").replace("\\", "/")
    user_daily = str(cfg.paths.agg_dir / "user_daily.parquet").replace("\\", "/")
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    window = cfg.popularity.window_days
    top_n_global = cfg.popularity.top_n_global

    parts_dir = out_path.parent / "_global_pop_parts"
    parts_dir.mkdir(exist_ok=True)
    con = make_connection()

    global_pop_path = parts_dir / "_global_pop_top.parquet"
    global_pop_str = str(global_pop_path).replace("\\", "/")

    if not (global_pop_path.exists() and global_pop_path.stat().st_size > 100):
        logger.info("  [global_pop] Building top_global (shared)...")
        sql_top = f"""
        COPY (
            WITH valid_items AS (
                SELECT DISTINCT item_id FROM read_parquet('{dim_glob}')
            ),
            pop AS (
                SELECT i.item_id, SUM(i.weighted_score) AS pop_score
                FROM read_parquet('{item_daily}') i
                INNER JOIN valid_items v ON i.item_id = v.item_id
                WHERE i.date >= DATE '{cutoff_date}' - INTERVAL {window} DAY
                  AND i.date <  DATE '{cutoff_date}'
                  AND i.weighted_score > 0
                GROUP BY i.item_id
            )
            SELECT item_id, pop_score,
                   ROW_NUMBER() OVER (ORDER BY pop_score DESC) AS rank_global
            FROM pop
            QUALIFY rank_global <= {top_n_global}
        ) TO '{global_pop_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """
        with timed("[global_pop] top_global", logger):
            con.execute(sql_top)

    if test_users_path is not None:
        users_glob = str(test_users_path).replace("\\", "/")
        users_table = f"read_parquet('{users_glob}')"
    else:
        users_table = f"(SELECT DISTINCT user_id FROM read_parquet('{user_daily}'))"

    for bucket in range(N_BUCKETS):
        part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")

        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("  [global_pop] bucket %d/%d: cache hit, SKIP",
                        bucket + 1, N_BUCKETS)
            continue

        sql = f"""
        COPY (
            WITH users_pool AS (
                SELECT user_id FROM {users_table}
                WHERE HASH(user_id) % {N_BUCKETS} = {bucket}
            ),
            cross_joined AS (
                SELECT u.user_id, g.item_id, g.pop_score AS source_score, g.rank_global
                FROM users_pool u
                CROSS JOIN read_parquet('{global_pop_str}') g
            ),
            final_ranked AS (
                SELECT user_id, item_id, source_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY rank_global
                       ) AS rn
                FROM cross_joined
            )
            SELECT user_id, item_id, 'global_pop' AS source, source_score
            FROM final_ranked WHERE rn <= {top_n_per_user}
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """

        try:
            with timed(f"[global_pop] bucket {bucket + 1}/{N_BUCKETS}", logger):
                con.execute(sql)
        except Exception as e:
            logger.error("  [global_pop] bucket %d FAILED: %s",
                         bucket + 1, str(e)[:200])
            part_path.unlink(missing_ok=True)
            raise

    parts_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
    merge_sql = f"""
    COPY (
        SELECT * FROM read_parquet('{parts_glob}')
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("[global_pop] merge buckets", logger):
        con.execute(merge_sql)

    for f in parts_dir.glob("bucket_*.parquet"):
        f.unlink()
    global_pop_path.unlink(missing_ok=True)
    parts_dir.rmdir()

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  global_pop candidates: %s rows", f"{n_rows:,}")