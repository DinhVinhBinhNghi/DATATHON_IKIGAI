"""Atomic rerank rules.

Mỗi rule là 1 function:
    def rule_xxx(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # adjust pred_score
        return df

Input/output:
- df có columns: user_id, item_id, pred_score, ...item_meta...
- Return df với pred_score adjusted (không drop rows).

Rules được apply theo thứ tự trong health_reranker.py.
"""
from __future__ import annotations

import pandas as pd
from src.common import get_logger

logger = get_logger(__name__)


def rule_cap_seller_diversity(df: pd.DataFrame,
                                max_per_seller: int = 2) -> pd.DataFrame:
    """Cap max items per seller trong top-10 per user.

    Logic: nếu seller có > max_per_seller items trong top-N candidates của user,
    penalize những item rank thấp.

    Args:
        df: candidates per user, có columns user_id, item_id, pred_score, seller_id.
        max_per_seller: max items same seller được lọt vào top-10.

    Returns:
        df với pred_score adjusted (items thừa của 1 seller bị giảm ~50%).
    """
    if "seller_id" not in df.columns:
        logger.warning("  rule_cap_seller_diversity: missing seller_id, skip")
        return df

    df = df.copy()
    # Per user, rank items per seller theo pred_score DESC
    df["_seller_rank"] = (
        df.groupby(["user_id", "seller_id"])["pred_score"]
        .rank(method="first", ascending=False)
    )
    # Penalize items beyond max_per_seller per (user, seller)
    df["pred_score"] = df["pred_score"].where(
        df["_seller_rank"] <= max_per_seller,
        df["pred_score"] * 0.5
    )
    df = df.drop(columns=["_seller_rank"])
    return df


def rule_freshness_boost(df: pd.DataFrame,
                          fresh_days: int = 7,
                          boost_factor: float = 1.10) -> pd.DataFrame:
    """Boost pred_score cho items posted gần đây.

    Logic: items có i_age_days <= fresh_days được boost 10%. Khuyến khích
    discovery cho new listings, support marketplace health.

    Args:
        df: candidates, có column i_age_days.
        fresh_days: threshold (mặc định 7).
        boost_factor: multiplier (1.10 = +10%).

    Returns:
        df với pred_score boosted cho fresh items.
    """
    if "i_age_days" not in df.columns:
        logger.warning("  rule_freshness_boost: missing i_age_days, skip")
        return df

    df = df.copy()
    is_fresh = df["i_age_days"] <= fresh_days
    df.loc[is_fresh, "pred_score"] = df.loc[is_fresh, "pred_score"] * boost_factor
    n_fresh = int(is_fresh.sum())
    logger.info("  rule_freshness_boost: boosted %s rows (≤%d days, ×%.2f)",
                f"{n_fresh:,}", fresh_days, boost_factor)
    return df
