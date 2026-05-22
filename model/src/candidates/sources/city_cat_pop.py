"""City × Category popularity source — chunked OOM-safe, no exclude."""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)

N_BUCKETS = 16


def build_city_cat_pop_candidates(cutoff_date: str, top_n_per_user: int,
                                   out_path: Path) -> None:
    cfg = get_config()
    item_daily = str(cfg.paths.agg_dir / "item_daily.parquet").replace("\\", "/")
    user_cat = str(cfg.paths.agg_dir / "user_category_weighted.parquet").replace("\\", "/")
    user_city = str(cfg.paths.agg_dir / "user_city_weighted.parquet").replace("\\", "/")
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    window = cfg.popularity.window_days
    top_n_per_cell = cfg.popularity.top_n_per_city_category

    parts_dir = out_path.parent / "_city_cat_pop_parts"
    parts_dir.mkdir(exist_ok=True)
    con = make_connection()

    top_per_cell_path = parts_dir / "_top_per_cell.parquet"
    top_per_cell_str = str(top_per_cell_path).replace("\\", "/")

    if not (top_per_cell_path.exists() and top_per_cell_path.stat().st_size > 100):
        logger.info("  [city_cat_pop] Building top_per_cell (shared)...")
        sql_top = f"""
        COPY (
            WITH item_meta AS (
                SELECT DISTINCT item_id, category, city_name
                FROM read_parquet('{dim_glob}')
                WHERE category IS NOT NULL AND city_name IS NOT NULL
            ),
            item_pop AS (
                SELECT m.item_id, m.category, m.city_name,
                       SUM(i.weighted_score) AS pop_score
                FROM read_parquet('{item_daily}') i
                INNER JOIN item_meta m ON i.item_id = m.item_id
                WHERE i.date >= DATE '{cutoff_date}' - INTERVAL {window} DAY
                  AND i.date <  DATE '{cutoff_date}'
                  AND i.weighted_score > 0
                GROUP BY m.item_id, m.category, m.city_name
            )
            SELECT item_id, category, city_name, pop_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY category, city_name ORDER BY pop_score DESC
                   ) AS rank_in_cell
            FROM item_pop
            QUALIFY rank_in_cell <= {top_n_per_cell}
        ) TO '{top_per_cell_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """
        with timed("[city_cat_pop] top_per_cell", logger):
            con.execute(sql_top)

    for bucket in range(N_BUCKETS):
        part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")

        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("  [city_cat_pop] bucket %d/%d: cache hit, SKIP",
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
            user_top_city AS (
                SELECT user_id, city_name,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY weighted_score DESC
                       ) AS rn
                FROM read_parquet('{user_city}')
                WHERE HASH(user_id) % {N_BUCKETS} = {bucket}
                QUALIFY rn = 1
            ),
            joined AS (
                SELECT u.user_id, t.item_id, t.pop_score AS source_score, t.rank_in_cell
                FROM user_top_cat u
                INNER JOIN user_top_city ci ON u.user_id = ci.user_id
                INNER JOIN read_parquet('{top_per_cell_str}') t
                    ON u.category = t.category AND ci.city_name = t.city_name
            ),
            final_ranked AS (
                SELECT user_id, item_id, source_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY rank_in_cell
                       ) AS rn
                FROM joined
            )
            SELECT user_id, item_id, 'city_cat_pop' AS source, source_score
            FROM final_ranked WHERE rn <= {top_n_per_user}
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """

        try:
            with timed(f"[city_cat_pop] bucket {bucket + 1}/{N_BUCKETS}", logger):
                con.execute(sql)
        except Exception as e:
            logger.error("  [city_cat_pop] bucket %d FAILED: %s",
                         bucket + 1, str(e)[:200])
            part_path.unlink(missing_ok=True)
            raise

    parts_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
    merge_sql = f"""
    COPY (
        SELECT * FROM read_parquet('{parts_glob}')
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("[city_cat_pop] merge buckets", logger):
        con.execute(merge_sql)

    for f in parts_dir.glob("bucket_*.parquet"):
        f.unlink()
    top_per_cell_path.unlink(missing_ok=True)
    parts_dir.rmdir()

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  city_cat_pop candidates: %s rows", f"{n_rows:,}")