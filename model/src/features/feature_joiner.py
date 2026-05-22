"""Feature joiner — chunked OOM-safe.

[CATEGORY AFFINITY PATCH v3.2] Thêm 4 derived features bám slide 5:
- ui_category_match            (giữ nguyên: match top-1 cat của user)
- ui_category_match_top2       (MỚI: match top-1 hoặc top-2 cat — bắt cross-segment)
- u_top_category_share         (forward từ user_features)
- u_top_category_share_recent  (forward từ user_features)
- ui_top_cat_match_x_share     (MỚI: interaction feature — match × share)

Defense slide 5:
- ui_top_cat_match_x_share cao = user trung thành cao VÀ candidate match top cat
  → tín hiệu strongest cho ranker. Tree-based có thể tự học nhưng đưa ra rõ
  ràng giúp model converge nhanh hơn + feature importance dễ defense.
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)

N_BUCKETS = 8


def build_ranker_input(mode: str,
                        candidates_path: Path,
                        user_feat_path: Path,
                        item_feat_path: Path,
                        pair_feat_path: Path,
                        temporal_feat_path: Path,
                        out_path: Path) -> None:
    """Join candidates + all features → ranker_input, chunked."""
    cand_str = str(candidates_path).replace("\\", "/")
    uf_str = str(user_feat_path).replace("\\", "/")
    if_str = str(item_feat_path).replace("\\", "/")
    pf_str = str(pair_feat_path).replace("\\", "/")
    tf_str = str(temporal_feat_path).replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    parts_dir = out_path.parent / f"_ranker_input_parts_{mode}"
    parts_dir.mkdir(exist_ok=True)
    con = make_connection()

    for bucket in range(N_BUCKETS):
        part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")

        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("  [feature_joiner] bucket %d/%d: cache hit, SKIP",
                        bucket + 1, N_BUCKETS)
            continue

        # [PATCH] Thêm 5 features mới ở cuối SELECT:
        # 1. u_top_category_share         (forward từ user_features, default 0.0)
        # 2. u_top_category_share_recent  (forward, default 0.0)
        # 3. u_top2_category              (forward categorical, default -1)
        # 4. ui_category_match_top2       (derived: match top-1 OR top-2)
        # 5. ui_top_cat_match_x_share     (derived: ui_category_match × share)
        sql = f"""
        COPY (
            SELECT
                c.user_id,
                c.item_id,
                c.source,
                c.source_score,
                c.candidate_score,
                COALESCE(u.u_total_weighted, 0.0)    AS u_total_weighted,
                COALESCE(u.u_recent_weighted, 0.0)   AS u_recent_weighted,
                COALESCE(u.u_total_pageview, 0)      AS u_total_pageview,
                COALESCE(u.u_unique_items, 0)        AS u_unique_items,
                COALESCE(u.u_unique_categories, 0)   AS u_unique_categories,
                COALESCE(u.u_unique_cities, 0)       AS u_unique_cities,
                COALESCE(u.u_active_days, 0)         AS u_active_days,
                COALESCE(u.u_top_category, -1)       AS u_top_category,
                COALESCE(u.u_top_city, '')           AS u_top_city,
                COALESCE(u.u_avg_dwell, 0.0)         AS u_avg_dwell,
                COALESCE(u.u_days_since_last, 999)   AS u_days_since_last,
                COALESCE(u.u_is_warm, 0)             AS u_is_warm,
                -- [PATCH] forward features mới từ user_features
                COALESCE(u.u_top_category_share, 0.0)        AS u_top_category_share,
                COALESCE(u.u_top_category_share_recent, 0.0) AS u_top_category_share_recent,
                COALESCE(u.u_top2_category, -1)              AS u_top2_category,
                COALESCE(i.i_category, -1)            AS i_category,
                COALESCE(i.i_city_name, '')           AS i_city_name,
                COALESCE(i.i_seller_type, 'unknown')  AS i_seller_type,
                COALESCE(i.i_ad_type, 'unknown')      AS i_ad_type,
                COALESCE(i.i_has_project_id, 0)       AS i_has_project_id,
                COALESCE(i.i_area_sqm, 0.0)           AS i_area_sqm,
                COALESCE(i.i_bedrooms, 0.0)           AS i_bedrooms,
                COALESCE(i.i_bathrooms, 0.0)          AS i_bathrooms,
                COALESCE(i.i_images_count, 0.0)       AS i_images_count,
                COALESCE(i.i_age_days, 999)           AS i_age_days,
                COALESCE(i.i_total_weighted, 0.0)     AS i_total_weighted,
                COALESCE(i.i_recent_weighted, 0.0)    AS i_recent_weighted,
                COALESCE(i.i_total_pageview, 0)       AS i_total_pageview,
                COALESCE(i.i_avg_dwell, 0.0)          AS i_avg_dwell,
                COALESCE(i.i_pop_rank_global, 999999) AS i_pop_rank_global,
                COALESCE(p.ui_total_weighted, 0.0)    AS ui_total_weighted,
                COALESCE(p.ui_recent_weighted, 0.0)   AS ui_recent_weighted,
                COALESCE(p.ui_n_pageview, 0)          AS ui_n_pageview,
                COALESCE(p.ui_n_pos_events, 0)        AS ui_n_pos_events,
                COALESCE(p.ui_max_dwell, 0.0)         AS ui_max_dwell,
                COALESCE(p.ui_n_active_days, 0)       AS ui_n_active_days,
                COALESCE(p.ui_days_since_last, 999)   AS ui_days_since_last,
                COALESCE(p.ui_days_since_first, 999)  AS ui_days_since_first,
                COALESCE(t.ui_recency_tier, 5)         AS ui_recency_tier,
                COALESCE(t.u_activity_recency_tier, 5) AS u_activity_recency_tier,
                COALESCE(t.i_age_tier, 5)              AS i_age_tier,
                -- Existing derived
                CASE WHEN COALESCE(u.u_top_category, -1) = COALESCE(i.i_category, -2)
                     THEN 1 ELSE 0 END AS ui_category_match,
                CASE WHEN COALESCE(u.u_top_city, '') = COALESCE(i.i_city_name, '')
                     THEN 1 ELSE 0 END AS ui_city_match,
                -- [PATCH] Derived mới #1: match top-1 OR top-2 cat của user
                -- → bắt cross-segment behavior (slide 5: 12% 1030→1020, 15% 1040→1020)
                CASE
                    WHEN COALESCE(u.u_top_category, -1) = COALESCE(i.i_category, -2)
                         THEN 1
                    WHEN COALESCE(u.u_top2_category, -1) = COALESCE(i.i_category, -2)
                         AND COALESCE(u.u_top2_category, -1) != -1
                         THEN 1
                    ELSE 0
                END AS ui_category_match_top2,
                -- [PATCH] Derived mới #2: interaction "match × loyalty"
                -- → tín hiệu strongest: user trung thành cao + candidate đúng cat top
                CASE WHEN COALESCE(u.u_top_category, -1) = COALESCE(i.i_category, -2)
                     THEN COALESCE(u.u_top_category_share, 0.0)
                     ELSE 0.0 END AS ui_top_cat_match_x_share
            FROM read_parquet('{cand_str}') c
            LEFT JOIN read_parquet('{uf_str}') u ON c.user_id = u.user_id
            LEFT JOIN read_parquet('{if_str}') i ON c.item_id = i.item_id
            LEFT JOIN read_parquet('{pf_str}') p
                ON c.user_id = p.user_id AND c.item_id = p.item_id
            LEFT JOIN read_parquet('{tf_str}') t
                ON c.user_id = t.user_id AND c.item_id = t.item_id
            WHERE HASH(c.user_id) % {N_BUCKETS} = {bucket}
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """

        try:
            with timed(f"[feature_joiner] bucket {bucket + 1}/{N_BUCKETS}", logger):
                con.execute(sql)
        except Exception as e:
            logger.error("  bucket %d FAILED: %s", bucket + 1, str(e)[:200])
            part_path.unlink(missing_ok=True)
            raise

        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{part_str}')"
        ).fetchone()[0]
        logger.info("    bucket %d rows: %s", bucket + 1, f"{n_rows:,}")

    parts_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
    concat_sql = f"""
    COPY (
        SELECT * FROM read_parquet('{parts_glob}')
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("[feature_joiner] concat buckets", logger):
        con.execute(concat_sql)

    for f in parts_dir.glob("bucket_*.parquet"):
        f.unlink()
    parts_dir.rmdir()

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_cols = con.execute(
        f"SELECT COUNT(*) FROM (DESCRIBE SELECT * FROM read_parquet('{out_str}'))"
    ).fetchone()[0]
    logger.info("  ranker_input (%s): %s rows × %d cols",
                mode, f"{n_rows:,}", n_cols)