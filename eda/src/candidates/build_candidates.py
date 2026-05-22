from __future__ import annotations

from pathlib import Path

from src.utils.constants import TRAIN_END_DATE
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_item_popularity(con, cache_dir: str | Path, overwrite: bool = False) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "item_popularity.parquet"
    if out.exists() and not overwrite:
        logger.info("Skip item_popularity.parquet because it already exists. Use --overwrite to rebuild.")
        con.execute(f"CREATE OR REPLACE VIEW item_popularity AS SELECT * FROM '{out.as_posix()}'")
        return out

    logger.info("Building item popularity table -> %s", out)
    query = f"""
    COPY (
        WITH event_item AS (
            SELECT
                item_id,
                COUNT(*) AS pos_events,
                APPROX_COUNT_DISTINCT(user_id) AS pos_users,
                COUNT(*) FILTER (WHERE date >= DATE '{TRAIN_END_DATE}' - INTERVAL 7 DAY) AS pos_events_7d,
                COUNT(*) FILTER (WHERE date >= DATE '{TRAIN_END_DATE}' - INTERVAL 28 DAY) AS pos_events_28d,
                MAX(event_ts) AS last_event_ts
            FROM events_pos
            GROUP BY item_id
        ),
        snap_item AS (
            SELECT
                item_id,
                SUM(views_24h) AS views,
                SUM(contacts_24h) AS contacts,
                MAX(date) AS last_snapshot_date,
                AVG(listing_age_days) AS avg_listing_age_days
            FROM snap_clean
            GROUP BY item_id
        )
        SELECT
            d.item_id,
            d.seller_id,
            d.category,
            d.city_name_clean AS city_name,
            d.district_name_clean AS district_name,
            d.seller_type,
            d.ad_type,
            d.posted_date,
            d.expected_expired_date,
            d.has_project,
            COALESCE(e.pos_events, 0) AS pos_events,
            COALESCE(e.pos_users, 0) AS pos_users,
            COALESCE(e.pos_events_7d, 0) AS pos_events_7d,
            COALESCE(e.pos_events_28d, 0) AS pos_events_28d,
            e.last_event_ts,
            COALESCE(s.views, 0) AS views,
            COALESCE(s.contacts, 0) AS contacts,
            s.last_snapshot_date,
            COALESCE(s.contacts / NULLIF(s.views, 0), 0) AS item_cr,
            DATE_DIFF('day', d.posted_date, DATE '{TRAIN_END_DATE}') AS age_since_posted_days,
            (
                3.0 * LOG(1 + COALESCE(e.pos_events_7d, 0)) +
                2.0 * LOG(1 + COALESCE(e.pos_events_28d, 0)) +
                1.0 * LOG(1 + COALESCE(e.pos_users, 0)) +
                1.5 * COALESCE(s.contacts / NULLIF(s.views, 0), 0) +
                0.25 * CASE WHEN d.seller_type = 'private' THEN 1 ELSE 0 END +
                0.20 * CASE WHEN DATE_DIFF('day', d.posted_date, DATE '{TRAIN_END_DATE}') BETWEEN 0 AND 30 THEN 1 ELSE 0 END
            ) AS base_score
        FROM dim_clean d
        LEFT JOIN event_item e USING (item_id)
        LEFT JOIN snap_item s USING (item_id)
        WHERE d.is_post_train_end = 0
          AND d.is_expire_before_post = 0
          AND COALESCE(d.expected_expired_date, DATE '{TRAIN_END_DATE}') >= DATE '{TRAIN_END_DATE}' - INTERVAL 60 DAY
          AND (COALESCE(e.pos_events_28d, 0) > 0 OR COALESCE(s.views, 0) > 0)
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.execute(f"CREATE OR REPLACE VIEW item_popularity AS SELECT * FROM '{out.as_posix()}'")
    return out


def build_user_profile(con, cache_dir: str | Path, overwrite: bool = False) -> Path:
    cache_dir = Path(cache_dir)
    out = cache_dir / "test_user_profile.parquet"
    if out.exists() and not overwrite:
        logger.info("Skip test_user_profile.parquet because it already exists. Use --overwrite to rebuild.")
        con.execute(f"CREATE OR REPLACE VIEW test_user_profile AS SELECT * FROM '{out.as_posix()}'")
        return out

    logger.info("Building test user profile -> %s", out)
    query = f"""
    COPY (
        WITH hist AS (
            SELECT e.*
            FROM events_pos e
            JOIN test_users_ds t ON t.user_id = e.user_id
        ),
        base AS (
            SELECT
                user_id,
                COUNT(*) AS user_pos_events,
                APPROX_COUNT_DISTINCT(item_id) AS user_unique_items,
                APPROX_COUNT_DISTINCT(date) AS user_active_days,
                MAX(event_ts) AS last_event_ts,
                ARG_MAX(category, event_ts) AS last_category,
                ARG_MAX(city_name, event_ts) AS last_city,
                ARG_MAX(device, event_ts) AS last_device
            FROM hist
            GROUP BY user_id
        ),
        cat_rank AS (
            SELECT user_id, category, COUNT(*) AS cnt,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, MAX(event_ts) DESC) AS rn
            FROM hist
            GROUP BY user_id, category
        ),
        city_rank AS (
            SELECT user_id, city_name, COUNT(*) AS cnt,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, MAX(event_ts) DESC) AS rn
            FROM hist
            WHERE city_name IS NOT NULL
            GROUP BY user_id, city_name
        )
        SELECT
            t.user_id,
            COALESCE(b.user_pos_events, 0) AS user_pos_events,
            COALESCE(b.user_unique_items, 0) AS user_unique_items,
            COALESCE(b.user_active_days, 0) AS user_active_days,
            b.last_event_ts,
            b.last_category,
            b.last_city,
            b.last_device,
            cr.category AS top_category,
            cir.city_name AS top_city,
            CASE
                WHEN COALESCE(b.user_pos_events, 0) = 0 THEN '0_cold'
                WHEN b.user_pos_events <= 5 THEN '1_low'
                WHEN b.user_pos_events <= 50 THEN '2_mid'
                ELSE '3_high'
            END AS user_activity_tier
        FROM test_users_ds t
        LEFT JOIN base b USING (user_id)
        LEFT JOIN cat_rank cr ON t.user_id = cr.user_id AND cr.rn = 1
        LEFT JOIN city_rank cir ON t.user_id = cir.user_id AND cir.rn = 1
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.execute(f"CREATE OR REPLACE VIEW test_user_profile AS SELECT * FROM '{out.as_posix()}'")
    return out


def build_candidate_scores(con, cache_dir: str | Path, overwrite: bool = False) -> Path:
    cache_dir = Path(cache_dir)
    out = cache_dir / "candidate_scores.parquet"
    if out.exists() and not overwrite:
        logger.info("Skip candidate_scores.parquet because it already exists. Use --overwrite to rebuild.")
        con.execute(f"CREATE OR REPLACE VIEW candidate_scores AS SELECT * FROM '{out.as_posix()}'")
        return out

    logger.info("Building candidate scores -> %s", out)
    query = f"""
    COPY (
        WITH
        global_pool AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY base_score DESC, pos_events_28d DESC, item_id) AS pool_rank
            FROM item_popularity
            QUALIFY pool_rank <= 100
        ),
        cat_pool AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY base_score DESC, pos_events_28d DESC, item_id) AS pool_rank
            FROM item_popularity
            QUALIFY pool_rank <= 80
        ),
        city_pool AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY city_name ORDER BY base_score DESC, pos_events_28d DESC, item_id) AS pool_rank
            FROM item_popularity
            WHERE city_name IS NOT NULL AND city_name != 'Không xác định'
            QUALIFY pool_rank <= 60
        ),
        cat_city_pool AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY category, city_name ORDER BY base_score DESC, pos_events_28d DESC, item_id) AS pool_rank
            FROM item_popularity
            WHERE city_name IS NOT NULL AND city_name != 'Không xác định'
            QUALIFY pool_rank <= 50
        ),
        candidates AS (
            SELECT p.user_id, i.item_id, 'cat_city' AS source,
                   1000.0 + i.base_score - 0.01 * i.pool_rank AS score
            FROM test_user_profile p
            JOIN cat_city_pool i
              ON i.category = COALESCE(p.top_category, p.last_category)
             AND i.city_name = COALESCE(p.top_city, p.last_city)

            UNION ALL
            SELECT p.user_id, i.item_id, 'category' AS source,
                   800.0 + i.base_score - 0.01 * i.pool_rank AS score
            FROM test_user_profile p
            JOIN cat_pool i
              ON i.category = COALESCE(p.top_category, p.last_category)

            UNION ALL
            SELECT p.user_id, i.item_id, 'city' AS source,
                   650.0 + i.base_score - 0.01 * i.pool_rank AS score
            FROM test_user_profile p
            JOIN city_pool i
              ON i.city_name = COALESCE(p.top_city, p.last_city)

            UNION ALL
            SELECT p.user_id, i.item_id, 'global' AS source,
                   500.0 + i.base_score - 0.01 * i.pool_rank AS score
            FROM test_user_profile p
            CROSS JOIN global_pool i
        ),
        dedup AS (
            SELECT user_id, item_id,
                   MAX(score) AS score,
                   ARG_MAX(source, score) AS best_source
            FROM candidates
            GROUP BY user_id, item_id
        )
        SELECT *
        FROM dedup
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.execute(f"CREATE OR REPLACE VIEW candidate_scores AS SELECT * FROM '{out.as_posix()}'")
    return out


def build_final_recommendations(con, cache_dir: str | Path, overwrite: bool = False) -> Path:
    cache_dir = Path(cache_dir)
    out = cache_dir / "final_recommendations.parquet"
    if out.exists() and not overwrite:
        logger.info("Skip final_recommendations.parquet because it already exists. Use --overwrite to rebuild.")
        con.execute(f"CREATE OR REPLACE VIEW final_recommendations AS SELECT * FROM '{out.as_posix()}'")
        return out

    logger.info("Building final top-10 recommendations -> %s", out)
    query = f"""
    COPY (
        WITH ranked AS (
            SELECT
                user_id,
                item_id,
                score,
                best_source,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC, item_id) AS rank
            FROM candidate_scores
        )
        SELECT user_id, rank, item_id, score, best_source
        FROM ranked
        WHERE rank <= 10
        ORDER BY user_id, rank
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.execute(f"CREATE OR REPLACE VIEW final_recommendations AS SELECT * FROM '{out.as_posix()}'")
    return out


def run_candidate_pipeline(con, cache_dir: str | Path, overwrite: bool = False) -> list[Path]:
    outputs = [
        build_item_popularity(con, cache_dir, overwrite=overwrite),
        build_user_profile(con, cache_dir, overwrite=overwrite),
        build_candidate_scores(con, cache_dir, overwrite=overwrite),
        build_final_recommendations(con, cache_dir, overwrite=overwrite),
    ]
    return outputs
