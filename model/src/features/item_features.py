"""Item-level features.

Features static từ dim_listing + dynamic activity từ item_daily (trước cutoff).

Output columns:
- item_id
- i_category, i_city_name, i_district_name
- i_seller_type           (private/agent)
- i_ad_type               (sell/let)
- i_has_project_id        (1 nếu thuộc project, else 0 — proxy B2B)
- i_area_sqm, i_bedrooms, i_bathrooms
- i_images_count
- i_age_days              (cutoff - posted_date, 999 nếu thiếu)
- i_total_weighted        (lifetime weighted_score)
- i_recent_weighted       (weighted_score trong 30d window)
- i_total_pageview
- i_total_user_events     (sum of n_unique_users across days)
- i_avg_dwell             (avg dwell từ events)
- i_pop_rank_global       (rank trong global pop theo recent_weighted)
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)


def build_item_features(cutoff_date: str, mode: str, out_path: Path) -> None:
    """Build item features at cutoff.

    Args:
        cutoff_date: YYYY-MM-DD.
        mode: 'train' hoặc 'predict' (chỉ ảnh hưởng log + filename).
        out_path: output parquet.
    """
    cfg = get_config()
    item_daily = str(cfg.paths.agg_dir / "item_daily.parquet").replace("\\", "/")
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    window = cfg.popularity.window_days

    sql = f"""
    COPY (
        WITH item_dim AS (
            -- Static features từ dim_listing
            SELECT DISTINCT
                item_id,
                category AS i_category,
                city_name AS i_city_name,
                district_name AS i_district_name,
                COALESCE(seller_type, 'unknown') AS i_seller_type,
                COALESCE(ad_type, 'unknown') AS i_ad_type,
                CASE WHEN project_id IS NOT NULL AND project_id != ''
                     THEN 1 ELSE 0 END AS i_has_project_id,
                CAST(COALESCE(area_sqm, 0.0) AS DOUBLE) AS i_area_sqm,
                CAST(COALESCE(bedrooms, 0.0) AS DOUBLE) AS i_bedrooms,
                CAST(COALESCE(bathrooms, 0.0) AS DOUBLE) AS i_bathrooms,
                CAST(COALESCE(images_count, 0.0) AS DOUBLE) AS i_images_count,
                posted_date
            FROM read_parquet('{dim_glob}')
        ),
        item_act AS (
            -- Dynamic features từ item_daily (trước cutoff)
            SELECT
                item_id,
                SUM(weighted_score) AS i_total_weighted,
                SUM(CASE WHEN date >= DATE '{cutoff_date}' - INTERVAL {window} DAY
                         THEN weighted_score ELSE 0.0 END) AS i_recent_weighted,
                SUM(n_pageview) AS i_total_pageview,
                SUM(n_unique_users) AS i_total_user_events,
                AVG(avg_dwell) AS i_avg_dwell
            FROM read_parquet('{item_daily}')
            WHERE date < DATE '{cutoff_date}'
            GROUP BY item_id
        ),
        item_global_rank AS (
            -- Rank theo recent weighted (30d) — proxy popularity
            SELECT
                item_id,
                ROW_NUMBER() OVER (ORDER BY i_recent_weighted DESC) AS i_pop_rank_global
            FROM item_act
            WHERE i_recent_weighted > 0
        )
        SELECT
            d.item_id,
            d.i_category,
            d.i_city_name,
            d.i_district_name,
            d.i_seller_type,
            d.i_ad_type,
            d.i_has_project_id,
            d.i_area_sqm,
            d.i_bedrooms,
            d.i_bathrooms,
            d.i_images_count,
            COALESCE(
                DATE_DIFF('day', d.posted_date, DATE '{cutoff_date}'),
                999
            ) AS i_age_days,
            COALESCE(a.i_total_weighted, 0.0)  AS i_total_weighted,
            COALESCE(a.i_recent_weighted, 0.0) AS i_recent_weighted,
            COALESCE(a.i_total_pageview, 0)    AS i_total_pageview,
            COALESCE(a.i_total_user_events, 0) AS i_total_user_events,
            COALESCE(a.i_avg_dwell, 0.0)       AS i_avg_dwell,
            COALESCE(g.i_pop_rank_global, 999999) AS i_pop_rank_global
        FROM item_dim d
        LEFT JOIN item_act a ON d.item_id = a.item_id
        LEFT JOIN item_global_rank g ON d.item_id = g.item_id
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """

    con = make_connection()
    with timed(f"build item_features (mode={mode}, cutoff={cutoff_date})", logger):
        con.execute(sql)

    n_items = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_with_activity = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}') WHERE i_total_weighted > 0"
    ).fetchone()[0]
    logger.info("  item_features (%s): %s items (%s with activity)",
                mode, f"{n_items:,}", f"{n_with_activity:,}")

