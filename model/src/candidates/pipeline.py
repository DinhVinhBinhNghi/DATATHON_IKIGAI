"""Candidate generation pipeline orchestrator.

Function: run_build_candidates(mode) — chạy tuần tự 5 sources + merge.

Args:
    mode: 'train' hoặc 'predict'. Khác nhau ở:
        - cutoff_date
        - output filename
        - user pool (predict dùng test_users)
"""
from __future__ import annotations

from src.common import file_exists_nonempty, get_config, get_logger

from src.candidates.sources.reengagement import build_reengagement
from src.candidates.sources.covisit import (
    build_covisit_matrix,
    expand_covisit_to_candidates,
)
from src.candidates.sources.category_pop import build_category_pop_candidates
from src.candidates.sources.city_cat_pop import build_city_cat_pop_candidates
from src.candidates.sources.global_pop import build_global_pop_candidates
from src.candidates.merge import merge_candidates


logger = get_logger(__name__)


def run_build_candidates(mode: str) -> None:
    """Build candidates pool cho train hoặc predict.

    Args:
        mode: 'train' (cutoff = ranker_train_cutoff) hoặc
              'predict' (cutoff = train_end, dùng test users).
    """
    assert mode in ("train", "predict"), f"mode phải là 'train' hoặc 'predict', got {mode}"

    cfg = get_config()
    cand_dir = cfg.paths.candidates_dir

    if mode == "train":
        cutoff = cfg.windows.ranker_train_cutoff
        test_users_path = None  # dùng users có activity
        suffix = "train"
    else:
        cutoff = cfg.windows.train_end
        test_users_path = cfg.paths.test_users_file
        suffix = "predict"

    final_path = cand_dir / f"candidates_{suffix}.parquet"
    if file_exists_nonempty(final_path):
        logger.info("[STEP 2/3 mode=%s] %s đã tồn tại, SKIP.", mode, final_path.name)
        return

    logger.info("[STEP 2/3 mode=%s] cutoff=%s", mode, cutoff)

    quotas = cfg.candidates.source_quotas
    sources = {}

    # 1. Reengagement
    p = cand_dir / f"reengagement_{suffix}.parquet"
    if not file_exists_nonempty(p):
        build_reengagement(cutoff, quotas.reengagement, p)
    sources["reengagement"] = p

    # 2. Co-visit (matrix + expand)
    covisit_matrix_path = cand_dir / f"covisit_matrix_{suffix}.parquet"
    if not file_exists_nonempty(covisit_matrix_path):
        build_covisit_matrix(cutoff, covisit_matrix_path)
    p = cand_dir / f"covisit_{suffix}.parquet"
    if not file_exists_nonempty(p):
        expand_covisit_to_candidates(cutoff, covisit_matrix_path,
                                      quotas.covisit, p)
    sources["covisit"] = p

    # 3. Category pop
    p = cand_dir / f"category_pop_{suffix}.parquet"
    if not file_exists_nonempty(p):
        build_category_pop_candidates(cutoff, quotas.category_pop, p)
    sources["category_pop"] = p

    # 4. City × Cat pop
    p = cand_dir / f"city_cat_pop_{suffix}.parquet"
    if not file_exists_nonempty(p):
        build_city_cat_pop_candidates(cutoff, quotas.city_cat_pop, p)
    sources["city_cat_pop"] = p

    # 5. Global pop
    p = cand_dir / f"global_pop_{suffix}.parquet"
    if not file_exists_nonempty(p):
        build_global_pop_candidates(cutoff, quotas.global_pop, p,
                                      test_users_path=test_users_path)
    sources["global_pop"] = p

    # Merge
    logger.info("[STEP 2/3 mode=%s] Merging 5 sources...", mode)
    merge_candidates(sources, final_path)

    logger.info("[STEP 2/3 mode=%s] DONE. Output: %s", mode, final_path)
