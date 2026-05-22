"""Feature spec: column groups cho LightGBM.

Phân nhóm:
- ID_COLS:              không vào model (user_id, item_id)
- NUMERIC_FEATURES:     int/float, dùng trực tiếp
- CATEGORICAL_FEATURES: string/int small-cardinality, cần encode
- LABEL_COL:            target (rel_label, computed từ GT)
- GROUP_COL:            user_id để group cho LambdaRank
- SOURCE_COL:           source (debug/eval)

Centralize ở 1 file để train + score dùng cùng schema, tránh mismatch.

[CATEGORY AFFINITY PATCH v3.2] Thêm 5 features mới:
NUMERIC:
- u_top_category_share        (share top cat của user, 0-1)
- u_top_category_share_recent (share top cat trong recency window, 0-1)
- ui_category_match_top2      (match với top-1 hoặc top-2 cat của user)
- ui_top_cat_match_x_share    (match × loyalty interaction)
CATEGORICAL:
- u_top2_category             (category thứ 2 của user)

Tổng features: 36 → 41 (35 numeric + 5 categorical → 36 numeric + 5 categorical)
"""
from __future__ import annotations

# Identifier columns — KHÔNG vào model
ID_COLS = ["user_id", "item_id"]

# Bị bỏ qua trong training (chỉ debug)
META_COLS = ["source", "source_score"]

# Numeric features
NUMERIC_FEATURES = [
    # candidate score (normalized 0-1 từ merge)
    "candidate_score",
    # user features
    "u_total_weighted",
    "u_recent_weighted",
    "u_total_pageview",
    "u_unique_items",
    "u_unique_categories",
    "u_unique_cities",
    "u_active_days",
    "u_avg_dwell",
    "u_days_since_last",
    "u_is_warm",
    # [PATCH] Category affinity features — bám slide 5 "86% intra-category"
    "u_top_category_share",         # share của top cat (lifetime)
    "u_top_category_share_recent",  # share của top cat (recency window)
    # item features
    "i_has_project_id",
    "i_area_sqm",
    "i_bedrooms",
    "i_bathrooms",
    "i_images_count",
    "i_age_days",
    "i_total_weighted",
    "i_recent_weighted",
    "i_total_pageview",
    "i_avg_dwell",
    "i_pop_rank_global",
    # pair features
    "ui_total_weighted",
    "ui_recent_weighted",
    "ui_n_pageview",
    "ui_n_pos_events",
    "ui_max_dwell",
    "ui_n_active_days",
    "ui_days_since_last",
    "ui_days_since_first",
    # temporal tiers (treat as numeric, tree-based handles ordinal)
    "ui_recency_tier",
    "u_activity_recency_tier",
    "i_age_tier",
    # derived
    "ui_category_match",
    "ui_city_match",
    # [PATCH] Category affinity derived features
    "ui_category_match_top2",       # match top-1 OR top-2 cat
    "ui_top_cat_match_x_share",     # interaction: match × user loyalty
]

# Categorical features (LightGBM native categorical support)
CATEGORICAL_FEATURES = [
    "u_top_category",     # int category code (-1 = cold)
    # [PATCH] Category affinity categorical
    "u_top2_category",    # int category code thứ 2 (-1 = chỉ 1 cat hoặc cold)
    "i_category",         # int category code
    "i_seller_type",      # 'agent' / 'private' / 'unknown'
    "i_ad_type",          # 'sell' / 'let' / 'unknown'
]

# Tất cả features (cho convenience)
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Label + group
LABEL_COL = "rel_label"
GROUP_COL = "user_id"


def validate_columns(df_columns: list[str]) -> list[str]:
    """Verify input dataframe có đủ ALL_FEATURES.

    Args:
        df_columns: list column names của input DataFrame.

    Returns:
        List missing columns. Empty list = OK.
    """
    missing = [c for c in ALL_FEATURES if c not in df_columns]
    return missing