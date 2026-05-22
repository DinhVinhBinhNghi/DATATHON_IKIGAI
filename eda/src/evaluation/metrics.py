from __future__ import annotations

import math


def recall_at_k(pred_items: list[str], true_items: set[str], k: int = 10) -> float:
    if not true_items:
        return 0.0
    return len(set(pred_items[:k]) & true_items) / len(true_items)


def ndcg_at_k(pred_items: list[str], true_items: set[str], k: int = 10) -> float:
    if not true_items:
        return 0.0
    dcg = 0.0
    for i, item in enumerate(pred_items[:k], start=1):
        if item in true_items:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(true_items))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
