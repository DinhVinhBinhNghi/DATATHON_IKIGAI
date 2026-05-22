"""5 candidate sources, mỗi source build candidates cho subset users.

Tất cả sources dùng weighted_score (v3.0 fix) thay vì raw counts.
"""
from src.candidates.sources.reengagement import build_reengagement
from src.candidates.sources.covisit import (
    build_covisit_matrix,
    expand_covisit_to_candidates,
)
from src.candidates.sources.category_pop import build_category_pop_candidates
from src.candidates.sources.city_cat_pop import build_city_cat_pop_candidates
from src.candidates.sources.global_pop import build_global_pop_candidates

__all__ = [
    "build_reengagement",
    "build_covisit_matrix",
    "expand_covisit_to_candidates",
    "build_category_pop_candidates",
    "build_city_cat_pop_candidates",
    "build_global_pop_candidates",
]


