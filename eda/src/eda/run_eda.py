from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Saved %s (%s rows)", path, f"{len(df):,}")


def _safe_execute_to_csv(con, sql: str, out: Path) -> bool:
    """Run one EDA query safely.

    EDA tables are non-critical for producing submission.csv. If a heavy aggregate
    exceeds RAM on a laptop, we log and continue instead of stopping the full pipeline.
    """
    try:
        df = con.execute(sql).df()
        _save_df(df, out)
        return True
    except duckdb.OutOfMemoryException as e:
        logger.warning("Skip EDA table %s because DuckDB ran out of memory: %s", out.name, e)
        err = pd.DataFrame({"table": [out.name], "status": ["skipped_oom"], "error": [str(e)[:500]]})
        _save_df(err, out.with_suffix(".error.csv"))
        return False
    except Exception as e:
        logger.warning("Skip EDA table %s because query failed: %s", out.name, e)
        err = pd.DataFrame({"table": [out.name], "status": ["skipped_error"], "error": [str(e)[:500]]})
        _save_df(err, out.with_suffix(".error.csv"))
        return False


def run_eda_tables(con, table_dir: str | Path) -> dict[str, Path]:
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Use APPROX_COUNT_DISTINCT for very large event tables to avoid laptop OOM.
    queries = {
        "eda_01_category_supply.csv": """
            SELECT category, COUNT(*) AS n_listings, COUNT(DISTINCT seller_id) AS n_sellers
            FROM dim_clean GROUP BY category ORDER BY category
        """,
        "eda_02_city_supply_top20.csv": """
            SELECT city_name_clean AS city_name, COUNT(*) AS n_listings, COUNT(DISTINCT seller_id) AS n_sellers
            FROM dim_clean GROUP BY city_name_clean ORDER BY n_listings DESC LIMIT 20
        """,
        "eda_03_category_demand.csv": """
            SELECT category,
                   COUNT(*) AS pos_events,
                   APPROX_COUNT_DISTINCT(user_id) AS approx_n_users,
                   APPROX_COUNT_DISTINCT(item_id) AS approx_n_items
            FROM events_pos GROUP BY category ORDER BY category
        """,
        "eda_04_city_demand_top20.csv": """
            SELECT city_name,
                   COUNT(*) AS pos_events,
                   APPROX_COUNT_DISTINCT(user_id) AS approx_n_users,
                   APPROX_COUNT_DISTINCT(item_id) AS approx_n_items
            FROM events_pos WHERE city_name IS NOT NULL
            GROUP BY city_name ORDER BY pos_events DESC LIMIT 20
        """,
        "eda_05_weekly_events.csv": """
            SELECT DATE_TRUNC('week', event_ts)::DATE AS week,
                   COUNT(*) AS pos_events,
                   APPROX_COUNT_DISTINCT(user_id) AS approx_wau,
                   APPROX_COUNT_DISTINCT(item_id) AS approx_active_items
            FROM events_pos GROUP BY week ORDER BY week
        """,
        "eda_06_cold_start_tier.csv": """
            WITH event_users AS (
                SELECT DISTINCT e.user_id FROM events_pos e JOIN test_users_ds t USING (user_id)
            ),
            int_users AS (
                SELECT DISTINCT i.user_id FROM int_clean i JOIN test_users_ds t USING (user_id)
            ),
            user_signals AS (
                SELECT t.user_id,
                       eu.user_id IS NOT NULL AS has_positive,
                       iu.user_id IS NOT NULL AS has_interaction
                FROM test_users_ds t
                LEFT JOIN event_users eu USING (user_id)
                LEFT JOIN int_users iu USING (user_id)
            )
            SELECT has_positive, has_interaction, COUNT(*) AS n_users,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM user_signals GROUP BY has_positive, has_interaction
            ORDER BY has_positive DESC, has_interaction DESC
        """,
        "eda_07_test_user_interaction_tier.csv": """
            WITH user_signals AS (
                SELECT t.user_id, COALESCE(COUNT(i.item_id), 0) AS n_interactions
                FROM test_users_ds t LEFT JOIN int_clean i ON i.user_id = t.user_id
                GROUP BY t.user_id
            )
            SELECT CASE
                    WHEN n_interactions = 0 THEN '0_zero_history'
                    WHEN n_interactions <= 5 THEN '1_low_1_5'
                    WHEN n_interactions <= 20 THEN '2_mid_6_20'
                    WHEN n_interactions <= 100 THEN '3_high_21_100'
                    ELSE '4_heavy_100+'
                   END AS user_tier,
                   COUNT(*) AS n_users,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM user_signals GROUP BY user_tier ORDER BY user_tier
        """,
        "eda_08_event_type_breakdown.csv": """
            SELECT event_type,
                   COUNT(*) AS n_events,
                   APPROX_COUNT_DISTINCT(user_id) AS approx_n_users,
                   APPROX_COUNT_DISTINCT(item_id) AS approx_n_items,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM events_pos GROUP BY event_type ORDER BY n_events DESC
        """,
        "eda_09_session_tier.csv": """
            WITH session_stats AS (
                SELECT session_id, COUNT(*) AS events_in_session
                FROM events_pos GROUP BY session_id
            )
            SELECT CASE
                    WHEN events_in_session = 1 THEN '1_single'
                    WHEN events_in_session <= 5 THEN '2_short_2_5'
                    WHEN events_in_session <= 20 THEN '3_medium_6_20'
                    WHEN events_in_session <= 100 THEN '4_long_21_100'
                    ELSE '5_very_long_100+'
                   END AS session_tier,
                   COUNT(*) AS n_sessions,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
            FROM session_stats GROUP BY session_tier ORDER BY session_tier
        """,
        "eda_10_snapshot_weekly_marketplace.csv": """
            WITH daily AS (
                SELECT date,
                       SUM(views_24h) AS total_views,
                       SUM(contacts_24h) AS total_contacts,
                       APPROX_COUNT_DISTINCT(item_id) AS approx_active_listings
                FROM snap_clean GROUP BY date
            )
            SELECT DATE_TRUNC('week', date)::DATE AS week,
                   SUM(total_views) AS total_views,
                   SUM(total_contacts) AS total_contacts,
                   AVG(approx_active_listings) AS avg_approx_active_listings,
                   SUM(total_contacts) / NULLIF(SUM(total_views), 0) AS contact_rate
            FROM daily GROUP BY week ORDER BY week
        """,
    }

    ok_count = 0
    for name, sql in queries.items():
        out = table_dir / name
        if _safe_execute_to_csv(con, sql, out):
            outputs[name] = out
            ok_count += 1

    logger.info("EDA finished: %s/%s tables saved successfully", ok_count, len(queries))
    return outputs
