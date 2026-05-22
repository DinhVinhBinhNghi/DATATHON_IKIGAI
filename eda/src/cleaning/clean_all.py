from __future__ import annotations

from pathlib import Path

from src.utils.constants import POSITIVE_EVENTS, TRAIN_END_DATE, TRAIN_END_TS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sql_str_list(values: list[str]) -> str:
    return ", ".join(["'" + v.replace("'", "''") + "'" for v in values])


def clean_dim_listing(con, clean_dir: str | Path, overwrite: bool = False) -> Path:
    clean_dir = Path(clean_dir)
    out = clean_dir / "dim_listing_clean.parquet"
    if out.exists() and not overwrite:
        logger.info("Skip dim_listing_clean.parquet because it already exists. Use --overwrite to rebuild.")
        con.execute(f"CREATE OR REPLACE VIEW dim_clean AS SELECT * FROM '{out.as_posix()}'")
        return out

    logger.info("Cleaning dim_listing -> %s", out)
    query = f"""
    COPY (
        SELECT
            CAST(item_id AS VARCHAR) AS item_id,
            CAST(seller_id AS VARCHAR) AS seller_id,
            TRY_CAST(category AS INTEGER) AS category,
            title,
            seller_type,
            ad_type,
            ad_status,
            TRY_CAST(area_sqm AS DOUBLE) AS area_sqm,
            TRY_CAST(bedrooms AS DOUBLE) AS bedrooms,
            TRY_CAST(bathrooms AS DOUBLE) AS bathrooms,
            TRY_CAST(floors AS DOUBLE) AS floors,
            TRY_CAST(width_m AS DOUBLE) AS width_m,
            direction,
            legal_status,
            house_type,
            furnishing,
            city_name,
            district_name,
            ward_name,
            CAST(project_id AS VARCHAR) AS project_id,
            price_bucket,
            TRY_CAST(images_count AS DOUBLE) AS images_count,
            CAST(posted_date AS DATE) AS posted_date,
            CAST(expected_expired_date AS DATE) AS expected_expired_date,
            DATE_DIFF('day', CAST(posted_date AS DATE), CAST(expected_expired_date AS DATE)) AS listing_lifetime_days,
            CAST(CAST(posted_date AS DATE) > DATE '{TRAIN_END_DATE}' AS INTEGER) AS is_post_train_end,
            CAST(CAST(expected_expired_date AS DATE) < CAST(posted_date AS DATE) AS INTEGER) AS is_expire_before_post,
            CASE
                WHEN lower(COALESCE(direction, '')) IN ('đông bắc', 'dong bac', 'đông - bắc', 'db', 'đb') THEN 'Đông Bắc'
                WHEN lower(COALESCE(direction, '')) IN ('đông nam', 'dong nam', 'đông - nam', 'dn', 'đn') THEN 'Đông Nam'
                WHEN lower(COALESCE(direction, '')) IN ('tây bắc', 'tay bac', 'tây - bắc', 'tb') THEN 'Tây Bắc'
                WHEN lower(COALESCE(direction, '')) IN ('tây nam', 'tay nam', 'tây - nam', 'tn') THEN 'Tây Nam'
                WHEN lower(COALESCE(direction, '')) IN ('đông', 'dong') THEN 'Đông'
                WHEN lower(COALESCE(direction, '')) IN ('tây', 'tay') THEN 'Tây'
                WHEN lower(COALESCE(direction, '')) IN ('nam') THEN 'Nam'
                WHEN lower(COALESCE(direction, '')) IN ('bắc', 'bac') THEN 'Bắc'
                ELSE 'Không xác định'
            END AS direction_clean,
            COALESCE(city_name, 'Không xác định') AS city_name_clean,
            COALESCE(district_name, 'Không xác định') AS district_name_clean,
            COALESCE(ward_name, 'Không xác định') AS ward_name_clean,
            COALESCE(legal_status, 'Không xác định') AS legal_status_clean,
            COALESCE(house_type, 'Không xác định') AS house_type_clean,
            COALESCE(furnishing, 'Không xác định') AS furnishing_clean,
            COALESCE(price_bucket, 'Không xác định') AS price_bucket_clean,
            CAST(project_id IS NOT NULL AS INTEGER) AS has_project
        FROM dim_ds
        WHERE item_id IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY item_id ORDER BY CAST(posted_date AS DATE) DESC NULLS LAST) = 1
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.execute(f"CREATE OR REPLACE VIEW dim_clean AS SELECT * FROM '{out.as_posix()}'")
    return out


def clean_snapshot(con, clean_dir: str | Path, overwrite: bool = False) -> Path:
    clean_dir = Path(clean_dir)
    out = clean_dir / "snapshot_clean.parquet"
    if out.exists() and not overwrite:
        logger.info("Skip snapshot_clean.parquet because it already exists. Use --overwrite to rebuild.")
        con.execute(f"CREATE OR REPLACE VIEW snap_clean AS SELECT * FROM '{out.as_posix()}'")
        return out

    logger.info("Cleaning fact_listing_snapshot -> %s", out)
    query = f"""
    COPY (
        SELECT
            CAST(item_id AS VARCHAR) AS item_id,
            CAST(date AS DATE) AS date,
            GREATEST(COALESCE(TRY_CAST(views_24h AS DOUBLE), 0), 0) AS views_24h,
            GREATEST(COALESCE(TRY_CAST(contacts_24h AS DOUBLE), 0), 0) AS contacts_24h,
            GREATEST(COALESCE(TRY_CAST(listing_age_days AS DOUBLE), 0), 0) AS listing_age_days,
            CAST(views_24h IS NULL AS INTEGER) AS views_was_null,
            CAST(contacts_24h IS NULL AS INTEGER) AS contacts_was_null
        FROM snapshot_ds
        WHERE CAST(date AS DATE) <= DATE '{TRAIN_END_DATE}'
          AND item_id IS NOT NULL
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.execute(f"CREATE OR REPLACE VIEW snap_clean AS SELECT * FROM '{out.as_posix()}'")
    return out


def clean_interactions(con, clean_dir: str | Path, overwrite: bool = False) -> Path:
    clean_dir = Path(clean_dir)
    out = clean_dir / "interactions_clean.parquet"
    if out.exists() and not overwrite:
        logger.info("Skip interactions_clean.parquet because it already exists. Use --overwrite to rebuild.")
        con.execute(f"CREATE OR REPLACE VIEW int_clean AS SELECT * FROM '{out.as_posix()}'")
        return out

    logger.info("Cleaning fact_post_contact_interactions -> %s", out)
    query = f"""
    COPY (
        SELECT
            CAST(user_id AS VARCHAR) AS user_id,
            CAST(item_id AS VARCHAR) AS item_id,
            CAST(date AS DATE) AS date,
            TRY_CAST(category AS INTEGER) AS category,
            GREATEST(COALESCE(TRY_CAST(adview_count AS DOUBLE), 0), 0) AS adview_count,
            GREATEST(COALESCE(TRY_CAST(lead_count AS DOUBLE), 0), 0) AS lead_count,
            GREATEST(COALESCE(TRY_CAST(chat_message_count AS DOUBLE), 0), 0) AS chat_message_count,
            GREATEST(COALESCE(TRY_CAST(chat_turn_count AS DOUBLE), 0), 0) AS chat_turn_count,
            COALESCE(TRY_CAST(chat_lead AS DOUBLE), 0) AS chat_lead,
            TRY_CAST(purchased AS BOOLEAN) AS purchased
        FROM interactions_ds
        WHERE CAST(date AS DATE) <= DATE '{TRAIN_END_DATE}'
          AND user_id IS NOT NULL
          AND item_id IS NOT NULL
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.execute(f"CREATE OR REPLACE VIEW int_clean AS SELECT * FROM '{out.as_posix()}'")
    return out


def clean_events_positive(con, clean_dir: str | Path, overwrite: bool = False) -> Path:
    clean_dir = Path(clean_dir)
    out = clean_dir / "events_positive_clean.parquet"
    if out.exists() and not overwrite:
        logger.info("Skip events_positive_clean.parquet because it already exists. Use --overwrite to rebuild.")
        con.execute(f"CREATE OR REPLACE VIEW events_pos AS SELECT * FROM '{out.as_posix()}'")
        return out

    logger.info("Cleaning positive fact_user_events -> %s", out)
    pos_list = _sql_str_list(POSITIVE_EVENTS)
    query = f"""
    COPY (
        SELECT
            CAST(user_id AS VARCHAR) AS user_id,
            CAST(item_id AS VARCHAR) AS item_id,
            event_type,
            CAST(event_ts AS TIMESTAMP) AS event_ts,
            CAST(date AS DATE) AS date,
            TRY_CAST(category AS INTEGER) AS category,
            city_name,
            surface,
            device,
            TRY_CAST(position AS DOUBLE) AS position,
            LEAST(GREATEST(COALESCE(TRY_CAST(dwell_time_sec AS DOUBLE), 0), 0), 3600) AS dwell_time_sec,
            is_login,
            CAST(session_id AS VARCHAR) AS session_id
        FROM events_ds
        WHERE CAST(event_ts AS TIMESTAMP) < TIMESTAMP '{TRAIN_END_TS}'
          AND event_type IN ({pos_list})
          AND is_login = 'login'
          AND user_id IS NOT NULL
          AND item_id IS NOT NULL
    ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.execute(f"CREATE OR REPLACE VIEW events_pos AS SELECT * FROM '{out.as_posix()}'")
    return out


def run_clean_all(con, clean_dir: str | Path, overwrite: bool = False) -> list[Path]:
    clean_dir = Path(clean_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        clean_dim_listing(con, clean_dir, overwrite=overwrite),
        clean_snapshot(con, clean_dir, overwrite=overwrite),
        clean_interactions(con, clean_dir, overwrite=overwrite),
        clean_events_positive(con, clean_dir, overwrite=overwrite),
    ]
    # Anti-leak verification
    check = con.execute("SELECT MAX(event_ts) AS max_ts, COUNT(*) AS n FROM events_pos").fetchone()
    logger.info("Anti-leak check events_pos: max_ts=%s, rows=%s", check[0], f"{check[1]:,}")
    return outputs
