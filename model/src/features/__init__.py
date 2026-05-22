"""Feature engineering: build user/item/pair/temporal features cho ranker.

4 cấp features:
- user_features:        per user (history, recency, preferences) — 13 cols
- item_features:        per item (catalog + activity) — 18 cols
- pair_features:        per (user, item) (interaction history) — 9 cols
- temporal_features:    per (user, item) (recency tier buckets) — 3 cols

+ feature_joiner: gộp tất cả + candidates → ranker_input_{mode}.parquet (~44 cols)

Tất cả features dùng weighted_score (v3.0 fix).
"""
from src.features.user_features import build_user_features
from src.features.item_features import build_item_features
from src.features.pair_features import build_pair_features
from src.features.temporal_features import build_temporal_features
from src.features.feature_joiner import build_ranker_input

__all__ = [
    "build_user_features",
    "build_item_features",
    "build_pair_features",
    "build_temporal_features",
    "build_ranker_input",
]