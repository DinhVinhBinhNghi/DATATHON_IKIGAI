"""Category popularity source — chunked OOM-safe, simplified.

Excluded CTE removed (5GB join là root cause OOM). Dedup happens in merge.py.
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)

N_BUCKETS = 16


def build_category_pop_candidates(cutoff_date: str, top_n_per_user: int,
                                   out_path: Path) -> None:
    cfg = get_config()
    item_daily = str(cfg.paths.agg_dir / "item_daily.parquet").replace("\\", "/")
    user_cat = str(cfg.paths.agg_dir / "user_category_weighted.parquet").replace("\\", "/")
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    window = cfg.popularity.window_days
    top_n_per_cat = cfg.popularity.top_n_per_category

    parts_dir = out_path.parent / "_category_pop_parts"
    parts_dir.mkdir(exist_ok=True)
    con = make_connection()

    top_per_cat_path = parts_dir / "_top_per_cat.parquet"
    top_per_cat_str = str(top_per_cat_path).replace("\\", "/")

    if not (top_per_cat_path.exists() and top_per_cat_path.stat().st_size > 100):
        logger.info("  [category_pop] Building top_per_cat (shared)...")
        sql_top = f"""
        COPY (
            WITH item_cat AS (
                SELECT DISTINCT item_id, category
                FROM read_parquet('{dim_glob}')
                WHERE category IS NOT NULL
            ),
            item_pop AS (
                SELECT
                    i.item_id, ic.category,
                    SUM(i.weighted_score) AS pop_score
                FROM read_parquet('{item_daily}') i
                INNER JOIN item_cat ic ON i.item_id = ic.item_id
                WHERE i.date >= DATE '{cutoff_date}' - INTERVAL {window} DAY
                  AND i.date <  DATE '{cutoff_date}'
                  AND i.weighted_score > 0
                GROUP BY i.item_id, ic.category
            )
            SELECT item_id, category, pop_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY category ORDER BY pop_score DESC
                   ) AS rank_in_cat
            FROM item_pop
            QUALIFY rank_in_cat <= {top_n_per_cat}
        ) TO '{top_per_cat_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """
        with timed("[category_pop] top_per_cat", logger):
            con.execute(sql_top)

    for bucket in range(N_BUCKETS):
        part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")

        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("  [category_pop] bucket %d/%d: cache hit, SKIP",
                        bucket + 1, N_BUCKETS)
            continue

        sql = f"""
        COPY (
            WITH user_top_cat AS (
                SELECT user_id, category,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY weighted_score DESC
                       ) AS rn
                FROM read_parquet('{user_cat}')
                WHERE HASH(user_id) % {N_BUCKETS} = {bucket}
                QUALIFY rn = 1
            ),
            joined AS (
                SELECT u.user_id, t.item_id, t.pop_score AS source_score, t.rank_in_cat
                FROM user_top_cat u
                INNER JOIN read_parquet('{top_per_cat_str}') t ON u.category = t.category
            ),
            final_ranked AS (
                SELECT user_id, item_id, source_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY rank_in_cat
                       ) AS rn
                FROM joined
            )
            SELECT user_id, item_id, 'category_pop' AS source, source_score
            FROM final_ranked WHERE rn <= {top_n_per_user}
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """

        try:
            with timed(f"[category_pop] bucket {bucket + 1}/{N_BUCKETS}", logger):
                con.execute(sql)
        except Exception as e:
            logger.error("  [category_pop] bucket %d FAILED: %s",
                         bucket + 1, str(e)[:200])
            part_path.unlink(missing_ok=True)
            raise

    parts_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
    merge_sql = f"""
    COPY (
        SELECT * FROM read_parquet('{parts_glob}')
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("[category_pop] merge buckets", logger):
        con.execute(merge_sql)

    for f in parts_dir.glob("bucket_*.parquet"):
        f.unlink()
    top_per_cat_path.unlink(missing_ok=True)
    parts_dir.rmdir()

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  category_pop candidates: %s rows", f"{n_rows:,}")