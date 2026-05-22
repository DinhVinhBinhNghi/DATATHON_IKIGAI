"""Marketplace-aware re-ranker.

Lấy output của LGB ranker (top-K, K > 10) và apply các constraint:

1. **Cap items per seller**: trong top-10 final, max N items cùng seller.
   Lý do: chống over-exposure 1 seller. Insight A3.5 Gini=0.897 cho thấy
   contact tập trung mạnh — model có thể inherit pattern này.

2. **Fresh listing boost**: tin posted < D ngày nhận boost score x B.
   Lý do: insight A3.3, contact rate tin 0-3 ngày = 11.0% (cao nhất),
   cộng với phục vụ marketplace health (giữ supply tươi).

3. **Private floor**: đảm bảo X% slot top-10 cho seller_type=private.
   Lý do: insight A3.2, sell × private CR = 16.8% (cao nhất). Hard cap chỉ
   để guard, không phải tối ưu chính.

Tất cả constraints là **soft + configurable** qua YAML.

Usage:
    rerank_recommendations(scored_df, dim_listing_df, config) -> rerunked_df
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RerankConfig:
    """Constraint config. Override qua YAML."""

    # Top-K candidate đem vào rerank (sau đó cắt top-10)
    candidate_k: int = 30

    # Final output size
    final_k: int = 10

    # Cap items per seller
    enable_seller_cap: bool = True
    max_items_per_seller: int = 3

    # Fresh listing boost
    enable_fresh_boost: bool = True
    fresh_days: int = 7
    fresh_boost_factor: float = 1.05  # multiply score

    # Private seller floor
    enable_private_floor: bool = True
    private_floor_pct: float = 30.0  # at least X% of final-K slots are private


def load_rerank_config(yaml_path: Optional[str | Path] = None) -> RerankConfig:
    cfg = RerankConfig()
    if yaml_path is None:
        return cfg
    path = Path(yaml_path)
    if not path.exists():
        return cfg
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        rerank_data = data.get("rerank", {})
        for k, v in rerank_data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        logger.info("Loaded rerank config from %s", path)
    except Exception as e:
        logger.warning("Failed to parse rerank config %s: %s", path, e)
    return cfg


def _apply_fresh_boost(df: pd.DataFrame, config: RerankConfig, score_col: str) -> pd.DataFrame:
    """Multiply score by boost_factor cho items có listing_age <= fresh_days."""
    if not config.enable_fresh_boost:
        return df
    if "i_listing_age_days" not in df.columns:
        logger.warning("i_listing_age_days missing; skipping fresh boost")
        return df
    fresh_mask = df["i_listing_age_days"].fillna(999) <= config.fresh_days
    df = df.copy()
    df.loc[fresh_mask, score_col] = df.loc[fresh_mask, score_col] * config.fresh_boost_factor
    n_boosted = int(fresh_mask.sum())
    logger.info("Fresh boost applied to %d rows (factor=%.3f, threshold=%d days)",
                n_boosted, config.fresh_boost_factor, config.fresh_days)
    return df


def _enforce_seller_cap(group: pd.DataFrame, max_per_seller: int, final_k: int, score_col: str) -> pd.DataFrame:
    """Greedy selection: take highest score, but skip if seller already at cap.

    Returns top-final_k rows of group.
    """
    group = group.sort_values(score_col, ascending=False)
    selected = []
    seller_count = {}
    for _, row in group.iterrows():
        seller = row.get("i_seller_id")
        if seller is None or pd.isna(seller):
            # Treat unknown seller as own group
            seller = f"__unknown__{row['item_id']}"
        cnt = seller_count.get(seller, 0)
        if cnt >= max_per_seller:
            continue
        selected.append(row)
        seller_count[seller] = cnt + 1
        if len(selected) >= final_k:
            break
    # If we didn't fill final_k (rare: all candidates from few sellers), backfill
    # ignoring the seller cap on the lowest-rank remaining.
    if len(selected) < final_k:
        selected_items = {r["item_id"] for r in selected}
        backfill = group[~group["item_id"].isin(selected_items)].head(final_k - len(selected))
        selected.extend([row for _, row in backfill.iterrows()])
    return pd.DataFrame(selected[:final_k])


def _enforce_private_floor(top_k_df: pd.DataFrame, remaining_df: pd.DataFrame,
                            final_k: int, floor_pct: float, score_col: str) -> pd.DataFrame:
    """Nếu top-K hiện tại có ít private hơn floor, swap items thấp nhất rank với private cao điểm nhất.

    top_k_df: top-K hiện tại sau seller cap.
    remaining_df: items còn lại không vào top-K, để swap nếu cần.
    """
    required_private = int(np.ceil(final_k * floor_pct / 100.0))
    current_private = int((top_k_df.get("i_seller_type") == "private").sum())
    if current_private >= required_private:
        return top_k_df

    deficit = required_private - current_private
    if deficit <= 0 or remaining_df.empty:
        return top_k_df

    # Candidates to add: private items ranked by score desc
    private_candidates = remaining_df[remaining_df.get("i_seller_type") == "private"].sort_values(score_col, ascending=False)
    if private_candidates.empty:
        return top_k_df

    # Items to swap out: non-private in top-K, lowest score first
    non_private_in_top = top_k_df[top_k_df.get("i_seller_type") != "private"].sort_values(score_col, ascending=True)
    n_swap = min(deficit, len(private_candidates), len(non_private_in_top))
    if n_swap == 0:
        return top_k_df

    swap_in = private_candidates.head(n_swap)
    swap_out_items = set(non_private_in_top.head(n_swap)["item_id"])

    kept = top_k_df[~top_k_df["item_id"].isin(swap_out_items)]
    out = pd.concat([kept, swap_in], ignore_index=True)
    return out


def rerank_recommendations(
    scored_df: pd.DataFrame,
    config: Optional[RerankConfig] = None,
    score_col: str = "score_lgb",
) -> pd.DataFrame:
    """Apply re-ranking constraints.

    Input scored_df columns expected:
        user_id, item_id, score_lgb, i_seller_id, i_seller_type, i_listing_age_days
    Output:
        user_id, rank (1..final_k), item_id, score_final, source
    """
    config = config or RerankConfig()
    logger.info("Reranking with config: seller_cap=%s/%d, fresh=%s/%.2fx@%dd, private_floor=%s/%.0f%%",
                config.enable_seller_cap, config.max_items_per_seller,
                config.enable_fresh_boost, config.fresh_boost_factor, config.fresh_days,
                config.enable_private_floor, config.private_floor_pct)

    df = scored_df.copy()
    df = _apply_fresh_boost(df, config, score_col)

    # Per-user processing
    results = []
    for user_id, group in df.groupby("user_id"):
        # Limit candidate pool to top-candidate_k by score
        group = group.sort_values(score_col, ascending=False).head(config.candidate_k).reset_index(drop=True)
        if config.enable_seller_cap:
            top_k = _enforce_seller_cap(group, config.max_items_per_seller, config.final_k, score_col)
        else:
            top_k = group.head(config.final_k).copy()

        if config.enable_private_floor:
            remaining = group[~group["item_id"].isin(set(top_k["item_id"]))]
            top_k = _enforce_private_floor(top_k, remaining, config.final_k, config.private_floor_pct, score_col)
            # Re-sort after potential swap
            top_k = top_k.sort_values(score_col, ascending=False).head(config.final_k)

        top_k = top_k.reset_index(drop=True)
        top_k["rank"] = np.arange(1, len(top_k) + 1)
        top_k["user_id"] = user_id
        results.append(top_k)

    if not results:
        return pd.DataFrame(columns=["user_id", "rank", "item_id", "score_final", "source"])

    out = pd.concat(results, ignore_index=True)
    out["score_final"] = out[score_col]
    out["source"] = "lgb_ranker_reranked"
    return out[["user_id", "rank", "item_id", "score_final", "source"]].sort_values(["user_id", "rank"])
