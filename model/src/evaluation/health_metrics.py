"""Marketplace health metrics: Gini, coverage, freshness, diversity.

Anh Minh's framing: "Marketplace Health" là 1 trong 2 nhóm vấn đề (cùng với
Contact Rate). Metrics đo lường:

1. **Coverage**:
   - item_coverage_pct: % items có ≥1 lần recommend / tổng items active.
   - seller_coverage_pct: % sellers có ≥1 lần recommend.

2. **Gini exposure**:
   - gini_item: how concentrated recommendations are theo items.
                0 = uniform, 1 = single item gets everything.
   - gini_seller: same theo sellers.

3. **Freshness**:
   - fresh_slot_pct: % top-10 slots dùng items posted ≤7 days.

4. **Diversity**:
   - avg_unique_sellers_per_user: trung bình distinct sellers per user.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)


def _gini(values: np.ndarray) -> float:
    """Compute Gini coefficient. 0 = uniform, 1 = single value gets all."""
    if len(values) == 0:
        return 0.0
    values = np.sort(values.astype(np.float64))
    n = len(values)
    cum = np.cumsum(values)
    total = cum[-1]
    if total == 0:
        return 0.0
    # Gini = (n+1 - 2 * sum((n+1-i) * y_i) / (n * sum)) / n
    # equivalent: G = (2 * sum(i * y_i) - (n + 1) * sum) / (n * sum)
    idx = np.arange(1, n + 1)
    return (2.0 * np.sum(idx * values) / (n * total)) - (n + 1) / n


def compute_marketplace_health(submission_csv: Path,
                                 out_csv: Path | None = None) -> dict:
    """Compute marketplace health metrics cho submission.

    Args:
        submission_csv: path to submission.csv.
        out_csv: nếu set, save report CSV.

    Returns:
        dict với metrics.
    """
    cfg = get_config()
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    item_daily = str(cfg.paths.agg_dir / "item_daily.parquet").replace("\\", "/")

    con = make_connection()

    logger.info("[STEP 18] Compute marketplace health: %s", submission_csv)

    # Load submission
    sub_df = pd.read_csv(submission_csv, dtype={"user_id": str, "item_id": str})
    logger.info("  Loaded submission: %s rows, %s users",
                f"{len(sub_df):,}", f"{sub_df['user_id'].nunique():,}")

    # Get item meta (seller_id, posted_date)
    csv_str = str(submission_csv).replace("\\", "/")
    item_meta_sql = f"""
    WITH sub AS (
        SELECT DISTINCT item_id FROM read_csv('{csv_str}', AUTO_DETECT=TRUE)
    )
    SELECT DISTINCT
        d.item_id,
        d.seller_id,
        d.seller_type,
        d.posted_date,
        DATE_DIFF('day', d.posted_date, DATE '{cfg.windows.train_end}') AS i_age_days
    FROM read_parquet('{dim_glob}') d
    INNER JOIN sub s ON d.item_id = s.item_id
    """
    with timed("join submission with item meta", logger):
        meta_df = con.execute(item_meta_sql).df()
    sub_full = sub_df.merge(meta_df, on="item_id", how="left")

    # 1. Item exposure (số lần item xuất hiện trong submission)
    item_counts = sub_full.groupby("item_id").size().values
    gini_item = _gini(item_counts)

    # 2. Seller exposure
    seller_counts = sub_full.dropna(subset=["seller_id"]).groupby("seller_id").size().values
    gini_seller = _gini(seller_counts) if len(seller_counts) > 0 else 0.0

    # 3. Coverage
    n_items_in_sub = sub_full["item_id"].nunique()
    n_sellers_in_sub = sub_full.dropna(subset=["seller_id"])["seller_id"].nunique()

    # Total active items in last 30d (for normalization)
    active_items_sql = f"""
    SELECT COUNT(DISTINCT item_id) FROM read_parquet('{item_daily}')
    WHERE date >= DATE '{cfg.windows.train_end}' - INTERVAL 30 DAY
      AND date <  DATE '{cfg.windows.train_end}'
      AND weighted_score > 0
    """
    n_active_items = con.execute(active_items_sql).fetchone()[0]
    item_coverage_pct = 100.0 * n_items_in_sub / max(n_active_items, 1)

    # 4. Freshness
    fresh_mask = sub_full["i_age_days"] <= 7
    fresh_slot_pct = 100.0 * fresh_mask.sum() / len(sub_full)

    # 5. Private vs Agent slot pct
    seller_type_dist = sub_full["seller_type"].value_counts(normalize=True) * 100
    private_pct = seller_type_dist.get("private", 0.0)
    agent_pct = seller_type_dist.get("agent", 0.0)

    # 6. Diversity per user
    avg_unique_sellers = sub_full.dropna(subset=["seller_id"]).groupby("user_id")["seller_id"].nunique().mean()

    # 7. Top-1 concentration
    top1 = sub_full[sub_full["rank"] == 1]
    top1_item_pct = 100.0 * top1["item_id"].value_counts().iloc[0] / len(top1)
    top1_seller_counts = top1.dropna(subset=["seller_id"])["seller_id"].value_counts()
    top1_seller_pct = (
        100.0 * top1_seller_counts.iloc[0] / top1.dropna(subset=["seller_id"]).shape[0]
        if len(top1_seller_counts) > 0 else 0.0
    )

    metrics = {
        "n_users": int(sub_full["user_id"].nunique()),
        "n_distinct_items": int(n_items_in_sub),
        "n_distinct_sellers": int(n_sellers_in_sub),
        "n_active_items": int(n_active_items),
        "item_coverage_pct": float(item_coverage_pct),
        "gini_item_exposure": float(gini_item),
        "gini_seller_exposure": float(gini_seller),
        "fresh_slot_pct": float(fresh_slot_pct),
        "private_slot_pct": float(private_pct),
        "agent_slot_pct": float(agent_pct),
        "avg_unique_sellers_per_user": float(avg_unique_sellers),
        "top1_item_concentration_pct": float(top1_item_pct),
        "top1_seller_concentration_pct": float(top1_seller_pct),
    }

    logger.info("  n_users:                   %s", f"{metrics['n_users']:,}")
    logger.info("  n_distinct_items:          %s", f"{metrics['n_distinct_items']:,}")
    logger.info("  n_distinct_sellers:        %s", f"{metrics['n_distinct_sellers']:,}")
    logger.info("  item_coverage_pct:         %.2f%%", metrics['item_coverage_pct'])
    logger.info("  gini_item_exposure:        %.4f", metrics['gini_item_exposure'])
    logger.info("  gini_seller_exposure:      %.4f", metrics['gini_seller_exposure'])
    logger.info("  fresh_slot_pct (≤7d):      %.2f%%", metrics['fresh_slot_pct'])
    logger.info("  private_slot_pct:          %.2f%%", metrics['private_slot_pct'])
    logger.info("  agent_slot_pct:            %.2f%%", metrics['agent_slot_pct'])
    logger.info("  avg_unique_sellers/user:   %.2f", metrics['avg_unique_sellers_per_user'])
    logger.info("  top-1 item concentration:  %.2f%%", metrics['top1_item_concentration_pct'])
    logger.info("  top-1 seller concentration:%.2f%%", metrics['top1_seller_concentration_pct'])

    if out_csv is not None:
        pd.DataFrame([metrics]).to_csv(out_csv, index=False)
        logger.info("  Saved report: %s", out_csv)

    return metrics
