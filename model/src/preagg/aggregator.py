"""DuckDB SQL templates cho weighted aggregation.

OOM mitigation:
- Dùng approx_count_distinct() thay COUNT(DISTINCT) (HyperLogLog, ~95% accurate, 100x ít RAM)
- Drop session_id distinct
- user_item_daily được chunked merge ở pipeline.py
"""
from __future__ import annotations

from src.common import get_config


def _weight_case(alias: str | None = None) -> str:
    cfg = get_config()
    w_hard = cfg.weights.hard_contact
    w_other = cfg.weights.other_interaction
    expr = (
        "CASE event_type "
        f"WHEN 'view_phone'        THEN {w_hard} "
        f"WHEN 'contact_chat'      THEN {w_hard} "
        f"WHEN 'contact_zalo'      THEN {w_hard} "
        f"WHEN 'contact_sms'       THEN {w_hard} "
        f"WHEN 'other_interaction' THEN {w_other} "
        "ELSE 0.0 "
        "END"
    )
    if alias:
        expr = f"({expr}) AS {alias}"
    return expr


def _positive_filter() -> str:
    return (
        "event_type IN ('view_phone', 'contact_chat', 'contact_zalo', "
        "'contact_sms', 'other_interaction')"
    )


def sql_user_daily(events_glob: str, out_path: str,
                   train_start: str, train_end: str) -> str:
    w = _weight_case()
    return f"""
    COPY (
        SELECT
            user_id,
            CAST(event_ts AS DATE) AS date,
            SUM(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS n_pageview,
            SUM(CASE WHEN {_positive_filter()} THEN 1 ELSE 0 END) AS n_pos_events,
            SUM({w}) AS weighted_score,
            approx_count_distinct(item_id) AS n_unique_items
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{train_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{train_end} 23:59:59'
          AND user_id IS NOT NULL
          AND item_id IS NOT NULL
        GROUP BY user_id, CAST(event_ts AS DATE)
    ) TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """


def sql_item_daily(events_glob: str, out_path: str,
                   train_start: str, train_end: str) -> str:
    w = _weight_case()
    return f"""
    COPY (
        SELECT
            item_id,
            CAST(event_ts AS DATE) AS date,
            SUM(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS n_pageview,
            SUM(CASE WHEN {_positive_filter()} THEN 1 ELSE 0 END) AS n_pos_events,
            SUM({w}) AS weighted_score,
            approx_count_distinct(user_id) AS n_unique_users,
            AVG(CAST(dwell_time_sec AS DOUBLE)) AS avg_dwell
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{train_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{train_end} 23:59:59'
          AND item_id IS NOT NULL
        GROUP BY item_id, CAST(event_ts AS DATE)
    ) TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """


def sql_user_item_daily_chunked(events_glob: str, out_path: str,
                                  train_start: str, train_end: str,
                                  bucket: int, n_buckets: int) -> str:
    """Chunked version of user_item_daily. Bucketize theo HASH(user_id) % n_buckets."""
    w = _weight_case()
    return f"""
    COPY (
        SELECT
            user_id,
            item_id,
            CAST(event_ts AS DATE) AS date,
            SUM(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS n_pageview,
            SUM(CASE WHEN {_positive_filter()} THEN 1 ELSE 0 END) AS n_pos_events,
            SUM({w}) AS weighted_score,
            MAX(CAST(dwell_time_sec AS DOUBLE)) AS max_dwell,
            MIN(event_ts) AS first_event_ts,
            MAX(event_ts) AS last_event_ts
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{train_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{train_end} 23:59:59'
          AND user_id IS NOT NULL
          AND item_id IS NOT NULL
          AND HASH(user_id) % {n_buckets} = {bucket}
        GROUP BY user_id, item_id, CAST(event_ts AS DATE)
    ) TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """


def sql_user_item_daily(events_glob: str, out_path: str,
                        train_start: str, train_end: str) -> str:
    """Single-shot (likely OOM trên 16GB). Pipeline dùng chunked thay thế."""
    w = _weight_case()
    return f"""
    COPY (
        SELECT
            user_id,
            item_id,
            CAST(event_ts AS DATE) AS date,
            SUM(CASE WHEN event_type = 'pageview' THEN 1 ELSE 0 END) AS n_pageview,
            SUM(CASE WHEN {_positive_filter()} THEN 1 ELSE 0 END) AS n_pos_events,
            SUM({w}) AS weighted_score,
            MAX(CAST(dwell_time_sec AS DOUBLE)) AS max_dwell,
            MIN(event_ts) AS first_event_ts,
            MAX(event_ts) AS last_event_ts
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{train_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{train_end} 23:59:59'
          AND user_id IS NOT NULL
          AND item_id IS NOT NULL
        GROUP BY user_id, item_id, CAST(event_ts AS DATE)
    ) TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """


def sql_user_category(events_glob: str, out_path: str,
                      train_start: str, train_end: str) -> str:
    w = _weight_case()
    return f"""
    COPY (
        SELECT
            user_id,
            category,
            SUM(CASE WHEN {_positive_filter()} THEN 1 ELSE 0 END) AS n_pos_events,
            SUM({w}) AS weighted_score,
            approx_count_distinct(item_id) AS n_unique_items
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{train_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{train_end} 23:59:59'
          AND user_id IS NOT NULL
          AND category IS NOT NULL
        GROUP BY user_id, category
        HAVING SUM({w}) > 0
    ) TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """


def sql_user_city(events_glob: str, out_path: str,
                  train_start: str, train_end: str) -> str:
    w = _weight_case()
    return f"""
    COPY (
        SELECT
            user_id,
            city_name,
            SUM(CASE WHEN {_positive_filter()} THEN 1 ELSE 0 END) AS n_pos_events,
            SUM({w}) AS weighted_score,
            approx_count_distinct(item_id) AS n_unique_items
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{train_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{train_end} 23:59:59'
          AND user_id IS NOT NULL
          AND city_name IS NOT NULL
        GROUP BY user_id, city_name
        HAVING SUM({w}) > 0
    ) TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """


def sql_event_type_daily(events_glob: str, out_path: str,
                         train_start: str, train_end: str) -> str:
    return f"""
    COPY (
        SELECT
            CAST(event_ts AS DATE) AS date,
            event_type,
            COUNT(*) AS n_events,
            approx_count_distinct(user_id) AS n_users,
            approx_count_distinct(item_id) AS n_items
        FROM read_parquet('{events_glob}')
        WHERE is_login = 'login'
          AND event_ts >= TIMESTAMP '{train_start} 00:00:00'
          AND event_ts <  TIMESTAMP '{train_end} 23:59:59'
        GROUP BY CAST(event_ts AS DATE), event_type
        ORDER BY date, event_type
    ) TO '{out_path}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """