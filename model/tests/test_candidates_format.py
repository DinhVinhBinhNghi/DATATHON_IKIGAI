"""Test candidates output format đúng schema.

KHÔNG chạy full candidate generation (quá lâu). Chỉ verify functions có
signature đúng và import được.
"""
from __future__ import annotations

import inspect

from src.candidates import run_build_candidates
from src.candidates.merge import merge_candidates
from src.candidates.sources.reengagement import build_reengagement
from src.candidates.sources.covisit import (
    build_covisit_matrix,
    expand_covisit_to_candidates,
)
from src.candidates.sources.category_pop import build_category_pop_candidates
from src.candidates.sources.city_cat_pop import build_city_cat_pop_candidates
from src.candidates.sources.global_pop import build_global_pop_candidates


class TestCandidateSourcesAPI:
    """Test 5 sources có signature đúng."""

    def test_run_build_candidates_signature(self):
        sig = inspect.signature(run_build_candidates)
        assert "mode" in sig.parameters

    def test_reengagement_signature(self):
        sig = inspect.signature(build_reengagement)
        for p in ["cutoff_date", "top_n_per_user", "out_path"]:
            assert p in sig.parameters, f"build_reengagement missing param: {p}"

    def test_covisit_signatures(self):
        sig_m = inspect.signature(build_covisit_matrix)
        assert "cutoff_date" in sig_m.parameters
        assert "out_path" in sig_m.parameters

        sig_e = inspect.signature(expand_covisit_to_candidates)
        for p in ["cutoff_date", "covisit_path", "top_n_per_user", "out_path"]:
            assert p in sig_e.parameters, f"expand_covisit missing param: {p}"

    def test_category_pop_signature(self):
        sig = inspect.signature(build_category_pop_candidates)
        for p in ["cutoff_date", "top_n_per_user", "out_path"]:
            assert p in sig.parameters

    def test_city_cat_pop_signature(self):
        sig = inspect.signature(build_city_cat_pop_candidates)
        for p in ["cutoff_date", "top_n_per_user", "out_path"]:
            assert p in sig.parameters

    def test_global_pop_signature(self):
        sig = inspect.signature(build_global_pop_candidates)
        for p in ["cutoff_date", "top_n_per_user", "out_path", "test_users_path"]:
            assert p in sig.parameters

    def test_merge_signature(self):
        sig = inspect.signature(merge_candidates)
        assert "source_files" in sig.parameters
        assert "out_path" in sig.parameters


class TestCandidateModes:
    """Test mode argument validation."""

    def test_run_build_candidates_rejects_invalid_mode(self):
        import pytest
        with pytest.raises(AssertionError):
            run_build_candidates(mode="invalid")
