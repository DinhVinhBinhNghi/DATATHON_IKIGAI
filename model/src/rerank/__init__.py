"""Marketplace-aware rerank (post-ranker).

Mode:
- 'raw':     không rerank, dùng pure pred_score từ LightGBM.
- 'rerank':  apply rules (diversity per seller, freshness boost).

Workflow:
1. Pick top-K candidates per user theo pred_score.
2. Apply rules theo thứ tự (diversity → freshness → fallback).
3. Output top-10 final per user.

Rules đơn giản, KHÔNG hard ràng buộc Recall (anh Minh's concern).
"""
from src.rerank.rules import (
    rule_cap_seller_diversity,
    rule_freshness_boost,
)
from src.rerank.health_reranker import run_rerank

__all__ = [
    "rule_cap_seller_diversity",
    "rule_freshness_boost",
    "run_rerank",
]

