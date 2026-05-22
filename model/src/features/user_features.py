"""User-level features.

Tất cả features đo trong window [cutoff - lookback, cutoff). KHÔNG dùng data
sau cutoff (anti-leakage).

Output columns:
- user_id
- u_total_weighted        (SUM weighted_score toàn bộ history)
- u_recent_weighted       (SUM weighted_score trong recency_window_days)
- u_total_pageview        (total pageview events)
- u_unique_items          (số items distinct đã interact)
- u_unique_categories     (số category distinct)
- u_unique_cities         (số city distinct)
- u_active_days           (số ngày có activity)
- u_top_category          (category có max weighted_score, -1 nếu cold)
- u_top_city              (city có max weighted_score, '' nếu cold)
- u_avg_dwell             (avg dwell time tại positive events)
- u_days_since_last       (ngày từ event cuối tới cutoff, 999 nếu cold)
- u_is_warm               (1 nếu có ≥1 weighted_score>0, else 0 = cold)

[CATEGORY AFFINITY PATCH v3.2] — bám sát IKIGAI slide 5 "category affinity 86%"
- u_top_category_share        (share weighted_score của top cat / tổng, 0-1)
- u_top_category_share_recent (share top cat trong recency window, 0-1)
- u_top2_category             (category thứ 2 theo weighted_score, -1 nếu chỉ 1 cat)

Defense slide 5:
- u_top_category_share = 1.0 → user "trung thành tuyệt đối" với 1 phân khúc
- u_top_category_share < 0.6 → user multi-category (cross-segment behavior)
- u_top2_category: encode cross-segment (slide heatmap: 12% từ 1030→1020, 15% từ 1040→1020)
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)


def build_user_features(cutoff_date: str, mode: str, out_path: Path) -> None:
    """Build user features tại cutoff.

    Args:
        cutoff_date: YYYY-MM-DD.
        mode: 'train' hoặc 'predict' (chỉ ảnh hưởng log message).
        out_path: output parquet.
    """
    cfg = get_config()
    user_daily = str(cfg.paths.agg_dir / "user_daily.parquet").replace("\\", "/")
    user_item = str(cfg.paths.agg_dir / "user_item_daily.parquet").replace("\\", "/")
    user_cat = str(cfg.paths.agg_dir / "user_category_weighted.parquet").replace("\\", "/")
    user_city = str(cfg.paths.agg_dir / "user_city_weighted.parquet").replace("\\", "/")
    # [PATCH] cần item_daily + dim_listing để tính u_top_category_share_recent
    # (vì user_category_weighted không có time dimension)
    item_daily = str(cfg.paths.agg_dir / "item_daily.parquet").replace("\\", "/")
    user_item_str = str(cfg.paths.agg_dir / "user_item_daily.parquet").replace("\\", "/")
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    recency = cfg.candidates.recency_window_days

    sql = f"""
    COPY (
        WITH user_agg AS (
            -- Aggregate user_daily trước cutoff
            SELECT
                user_id,
                SUM(weighted_score) AS u_total_weighted,
                SUM(CASE WHEN date >= DATE '{cutoff_date}' - INTERVAL {recency} DAY
                         THEN weighted_score ELSE 0.0 END) AS u_recent_weighted,
                SUM(n_pageview) AS u_total_pageview,
                COUNT(DISTINCT date) AS u_active_days,
                MAX(date) AS last_active_date
            FROM read_parquet('{user_daily}')
            WHERE date < DATE '{cutoff_date}'
            GROUP BY user_id
        ),
        user_diversity AS (
            -- Distinct items + dwell
            SELECT
                user_id,
                COUNT(DISTINCT item_id) AS u_unique_items,
                AVG(max_dwell) AS u_avg_dwell
            FROM read_parquet('{user_item}')
            WHERE date < DATE '{cutoff_date}'
              AND weighted_score > 0
            GROUP BY user_id
        ),

        -- =====================================================================
        -- [CATEGORY AFFINITY PATCH] Block mới: rank category theo weighted_score
        -- và tính share của top-1 / top-2.
        -- =====================================================================
        user_cat_ranked AS (
            -- Rank tất cả categories của user theo weighted_score DESC
            SELECT
                user_id, category, weighted_score,
                SUM(weighted_score) OVER (PARTITION BY user_id) AS user_total_cat_weighted,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id ORDER BY weighted_score DESC
                ) AS cat_rank
            FROM read_parquet('{user_cat}')
            WHERE weighted_score > 0
        ),
        user_top_cat AS (
            -- Top-1 category + share của nó
            SELECT
                user_id,
                category AS u_top_category,
                CASE WHEN user_total_cat_weighted > 0
                     THEN weighted_score / user_total_cat_weighted
                     ELSE 0.0 END AS u_top_category_share
            FROM user_cat_ranked
            WHERE cat_rank = 1
        ),
        user_top2_cat AS (
            -- Top-2 category (NULL nếu user chỉ có 1 cat)
            SELECT user_id, category AS u_top2_category
            FROM user_cat_ranked
            WHERE cat_rank = 2
        ),
        user_cat_count AS (
            SELECT user_id, COUNT(DISTINCT category) AS u_unique_categories
            FROM read_parquet('{user_cat}')
            WHERE weighted_score > 0
            GROUP BY user_id
        ),

        -- =====================================================================
        -- [CATEGORY AFFINITY PATCH] Block mới: u_top_category_share_recent
        -- (top cat trong recency window, không phải lifetime).
        -- Cần join user_item_daily × dim_listing để có category cho mỗi event.
        -- Lưu ý: dùng item_id → category từ dim_listing (1 listing thuộc 1 cat).
        -- =====================================================================
        item_cat_map AS (
            SELECT DISTINCT item_id, category
            FROM read_parquet('{dim_glob}')
            WHERE category IS NOT NULL
        ),
        user_cat_recent AS (
            -- Tổng weighted_score per (user, category) trong recency window
            SELECT
                ui.user_id,
                ic.category,
                SUM(ui.weighted_score) AS recent_weighted
            FROM read_parquet('{user_item_str}') ui
            INNER JOIN item_cat_map ic ON ui.item_id = ic.item_id
            WHERE ui.date >= DATE '{cutoff_date}' - INTERVAL {recency} DAY
              AND ui.date <  DATE '{cutoff_date}'
              AND ui.weighted_score > 0
            GROUP BY ui.user_id, ic.category
        ),
        user_cat_recent_ranked AS (
            SELECT
                user_id, category, recent_weighted,
                SUM(recent_weighted) OVER (PARTITION BY user_id) AS user_recent_total,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id ORDER BY recent_weighted DESC
                ) AS recent_rank
            FROM user_cat_recent
        ),
        user_top_cat_recent AS (
            -- Share của top cat trong recency window
            SELECT
                user_id,
                CASE WHEN user_recent_total > 0
                     THEN recent_weighted / user_recent_total
                     ELSE 0.0 END AS u_top_category_share_recent
            FROM user_cat_recent_ranked
            WHERE recent_rank = 1
        ),

        user_top_city AS (
            SELECT user_id, city_name AS u_top_city,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id ORDER BY weighted_score DESC
                   ) AS rn
            FROM read_parquet('{user_city}')
            QUALIFY rn = 1
        ),
        user_city_count AS (
            SELECT user_id, COUNT(DISTINCT city_name) AS u_unique_cities
            FROM read_parquet('{user_city}')
            GROUP BY user_id
        )
        SELECT
            a.user_id,
            COALESCE(a.u_total_weighted, 0.0)   AS u_total_weighted,
            COALESCE(a.u_recent_weighted, 0.0)  AS u_recent_weighted,
            COALESCE(a.u_total_pageview, 0)     AS u_total_pageview,
            COALESCE(d.u_unique_items, 0)       AS u_unique_items,
            COALESCE(cc.u_unique_categories, 0) AS u_unique_categories,
            COALESCE(cic.u_unique_cities, 0)    AS u_unique_cities,
            COALESCE(a.u_active_days, 0)        AS u_active_days,
            COALESCE(tc.u_top_category, -1)     AS u_top_category,
            -- [PATCH] 3 features mới
            COALESCE(tc.u_top_category_share, 0.0)        AS u_top_category_share,
            COALESCE(tcr.u_top_category_share_recent, 0.0) AS u_top_category_share_recent,
            COALESCE(t2.u_top2_category, -1)              AS u_top2_category,
            COALESCE(tci.u_top_city, '')        AS u_top_city,
            COALESCE(d.u_avg_dwell, 0.0)        AS u_avg_dwell,
            COALESCE(
                DATE_DIFF('day', a.last_active_date, DATE '{cutoff_date}'),
                999
            ) AS u_days_since_last,
            CASE WHEN COALESCE(a.u_total_weighted, 0.0) > 0 THEN 1 ELSE 0 END AS u_is_warm
        FROM user_agg a
        LEFT JOIN user_diversity d  ON a.user_id = d.user_id
        LEFT JOIN user_top_cat   tc ON a.user_id = tc.user_id
        LEFT JOIN user_top2_cat  t2 ON a.user_id = t2.user_id
        LEFT JOIN user_top_cat_recent tcr ON a.user_id = tcr.user_id
        LEFT JOIN user_cat_count cc ON a.user_id = cc.user_id
        LEFT JOIN user_top_city  tci ON a.user_id = tci.user_id
        LEFT JOIN user_city_count cic ON a.user_id = cic.user_id
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """

    con = make_connection()
    with timed(f"build user_features (mode={mode}, cutoff={cutoff_date})", logger):
        con.execute(sql)

    n_users = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_warm = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}') WHERE u_is_warm = 1"
    ).fetchone()[0]
    logger.info("  user_features (%s): %s users (%s warm, %s cold)",
                mode, f"{n_users:,}", f"{n_warm:,}", f"{n_users - n_warm:,}")

    # [PATCH] Diagnostic: phân phối u_top_category_share (defense slide 5)
    diag = con.execute(f"""
        SELECT
            COUNT(*) AS n_warm,
            AVG(u_top_category_share)               AS avg_share,
            SUM(CASE WHEN u_top_category_share >= 0.9 THEN 1 ELSE 0 END) AS n_loyal,
            SUM(CASE WHEN u_top_category_share <= 0.6 THEN 1 ELSE 0 END) AS n_cross,
            SUM(CASE WHEN u_top2_category != -1 THEN 1 ELSE 0 END)       AS n_multi_cat
        FROM read_parquet('{out_str}')
        WHERE u_is_warm = 1
    """).fetchone()
    if diag and diag[0] > 0:
        n_w = diag[0]
        logger.info("  [category affinity diagnostic]:")
        logger.info("    avg u_top_category_share: %.3f", diag[1])
        logger.info("    %s users (%.1f%%) ≥0.9 share (loyal single-category)",
                    f"{diag[2]:,}", 100.0 * diag[2] / n_w)
        logger.info("    %s users (%.1f%%) ≤0.6 share (cross-category browsers)",
                    f"{diag[3]:,}", 100.0 * diag[3] / n_w)
        logger.info("    %s users (%.1f%%) có top-2 category (multi-segment)",
                    f"{diag[4]:,}", 100.0 * diag[4] / n_w)