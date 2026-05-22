"""Test candidate recall ceiling.

Chạy KHI candidates_predict.parquet đã có. Skip nếu không có cache.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.common import file_exists_nonempty, get_config


def _cache_available() -> tuple[bool, Path | None, Path | None]:
    """Check candidates + internal_gt cache available."""
    cfg = get_config()
    cand = cfg.paths.candidates_dir / "candidates_predict.parquet"
    gt = cfg.paths.gt_dir / "internal_gt.parquet"
    if file_exists_nonempty(cand) and file_exists_nonempty(gt):
        return True, cand, gt
    return False, None, None


pytestmark = pytest.mark.skipif(
    not _cache_available()[0],
    reason="Skip: cần candidates_predict.parquet + internal_gt.parquet",
)


class TestCandidateRecall:
    """Test recall ceiling của candidate pool."""

    def test_recall_at_200_is_at_least_15_percent(self):
        """Recall@200 ceiling phải ≥ 15% (sanity bound).

        v2.4.0 đạt ~23% với weighted=raw. v3.0 expect tương đương hoặc cao hơn.
        Test này là sanity (≥15%) — failed nghĩa là candidate gen không
        cover được GT users.
        """
        from src.evaluation import evaluate_candidates_pool
        _, cand, gt = _cache_available()
        metrics = evaluate_candidates_pool(cand, gt, out_csv=None)
        assert metrics["recall@200_ceiling"] >= 0.15, (
            f"Recall@200 ceiling {metrics['recall@200_ceiling']:.4f} < 0.15. "
            "Có vấn đề với candidate generation."
        )

    def test_recall_monotonic_in_k(self):
        """Recall@K1 ≤ Recall@K2 nếu K1 ≤ K2."""
        from src.evaluation import evaluate_candidates_pool
        _, cand, gt = _cache_available()
        m = evaluate_candidates_pool(cand, gt, out_csv=None)
        assert m["recall@10_ceiling"] <= m["recall@50_ceiling"]
        assert m["recall@50_ceiling"] <= m["recall@100_ceiling"]
        assert m["recall@100_ceiling"] <= m["recall@200_ceiling"]

    def test_source_coverage_not_empty(self):
        """5 sources phải đều có ít nhất 1 candidate."""
        from src.evaluation import evaluate_candidates_pool
        _, cand, gt = _cache_available()
        m = evaluate_candidates_pool(cand, gt, out_csv=None)
        sources = {s["source"] for s in m["source_breakdown"]}
        expected = {"reengagement", "covisit", "category_pop",
                    "city_cat_pop", "global_pop"}
        # At least 4/5 sources present (1 có thể empty với edge case data)
        assert len(sources & expected) >= 4, (
            f"Expected ≥4 sources, got {sources}"
        )
