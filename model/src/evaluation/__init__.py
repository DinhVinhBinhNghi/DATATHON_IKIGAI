"""Evaluation: local metrics for submission + candidates + marketplace health.

3 levels of evaluation:
- ground_truth.py: build internal GT từ events trong window [gt_start, gt_end]
- metrics.py: Recall@K, NDCG@K cho submission top-10
- candidate_metrics.py: Recall@N ceiling, source coverage cho candidates pool
- health_metrics.py: Gini, item/seller coverage, freshness, diversity

Internal GT KHÔNG dùng để train. Chỉ dùng để evaluate local trước khi submit Kaggle.
"""
from src.evaluation.ground_truth import build_internal_ground_truth
from src.evaluation.metrics import (
    compute_recall_at_k,
    compute_ndcg_at_k,
    evaluate_submission,
)
from src.evaluation.candidate_metrics import evaluate_candidates_pool
from src.evaluation.health_metrics import compute_marketplace_health

__all__ = [
    "build_internal_ground_truth",
    "compute_recall_at_k",
    "compute_ndcg_at_k",
    "evaluate_submission",
    "evaluate_candidates_pool",
    "compute_marketplace_health",
]

