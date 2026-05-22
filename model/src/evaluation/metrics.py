"""Recall@K, NDCG@K metrics cho submission evaluation.

Recall@K(u) = |R_u ∩ G_u| / |G_u|
    where R_u = predicted top-K items, G_u = ground truth items.

NDCG@K(u) = DCG@K(u) / IDCG@K(u)
    DCG@K = sum(1[R_i in G_u] / log2(i+1) for i in 1..K)
    IDCG@K = sum(1 / log2(i+1) for i in 1..min(K, |G_u|))

Average across all users in GT.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.common import get_config, get_logger, make_connection

logger = get_logger(__name__)


def compute_recall_at_k(predictions: dict[str, list[str]],
                         ground_truth: dict[str, set[str]],
                         k: int = 10) -> tuple[float, int]:
    """Compute mean Recall@K across users in ground_truth.

    Args:
        predictions: user_id → list top-K item_ids (in rank order).
        ground_truth: user_id → set of GT item_ids.
        k: top-K cap (default 10).

    Returns:
        (mean_recall, n_users_evaluated).
        n_users_evaluated = số users có in cả predictions và GT.
    """
    recalls = []
    for user_id, gt_items in ground_truth.items():
        if not gt_items:
            continue
        pred_items = predictions.get(user_id, [])[:k]
        if not pred_items:
            recalls.append(0.0)
            continue
        hits = len(set(pred_items) & gt_items)
        recalls.append(hits / len(gt_items))

    if not recalls:
        return 0.0, 0
    return sum(recalls) / len(recalls), len(recalls)


def compute_ndcg_at_k(predictions: dict[str, list[str]],
                       ground_truth: dict[str, set[str]],
                       k: int = 10) -> tuple[float, int]:
    """Compute mean NDCG@K across users in ground_truth (binary relevance)."""
    ndcgs = []
    log2 = math.log2
    for user_id, gt_items in ground_truth.items():
        if not gt_items:
            continue
        pred_items = predictions.get(user_id, [])[:k]
        if not pred_items:
            ndcgs.append(0.0)
            continue
        dcg = sum(
            1.0 / log2(i + 2) for i, item in enumerate(pred_items) if item in gt_items
        )
        ideal_hits = min(k, len(gt_items))
        idcg = sum(1.0 / log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    if not ndcgs:
        return 0.0, 0
    return sum(ndcgs) / len(ndcgs), len(ndcgs)


def evaluate_submission(submission_csv: Path, ground_truth_parquet: Path,
                         k: int = 10) -> dict:
    """Evaluate submission CSV với Recall@K, NDCG@K.

    Args:
        submission_csv: path to submission.csv.
        ground_truth_parquet: internal_gt.parquet (user_id, item_id, ...).
        k: cap (default 10).

    Returns:
        dict {
            'recall@k': float, 'ndcg@k': float,
            'n_users_evaluated': int,
            'n_users_in_gt': int, 'n_users_in_sub': int,
            'n_users_in_both': int,
        }
    """
    logger.info("[EVAL] Loading submission + GT...")

    # Load GT
    con = make_connection()
    gt_str = str(ground_truth_parquet).replace("\\", "/")
    gt_df = con.execute(
        f"SELECT user_id, item_id FROM read_parquet('{gt_str}')"
    ).df()
    ground_truth: dict[str, set[str]] = {}
    for user_id, group in gt_df.groupby("user_id"):
        ground_truth[user_id] = set(group["item_id"])
    n_users_in_gt = len(ground_truth)
    logger.info("  GT: %s users, %s pairs", f"{n_users_in_gt:,}", f"{len(gt_df):,}")

    # Load submission
    sub_df = pd.read_csv(submission_csv, dtype={"user_id": str, "item_id": str})
    sub_df = sub_df.sort_values(["user_id", "rank"])
    predictions: dict[str, list[str]] = {}
    for user_id, group in sub_df.groupby("user_id"):
        predictions[user_id] = group["item_id"].tolist()
    n_users_in_sub = len(predictions)
    logger.info("  Submission: %s users, %s rows",
                f"{n_users_in_sub:,}", f"{len(sub_df):,}")

    n_users_in_both = len(set(predictions.keys()) & set(ground_truth.keys()))
    logger.info("  Users in both: %s", f"{n_users_in_both:,}")

    # Compute metrics
    recall, n_eval = compute_recall_at_k(predictions, ground_truth, k=k)
    ndcg, _ = compute_ndcg_at_k(predictions, ground_truth, k=k)

    logger.info("  Recall@%d: %.6f (over %s users)",
                k, recall, f"{n_eval:,}")
    logger.info("  NDCG@%d:   %.6f", k, ndcg)

    return {
        f"recall@{k}": recall,
        f"ndcg@{k}": ndcg,
        "n_users_evaluated": n_eval,
        "n_users_in_gt": n_users_in_gt,
        "n_users_in_sub": n_users_in_sub,
        "n_users_in_both": n_users_in_both,
    }
