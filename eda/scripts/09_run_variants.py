"""So sánh 3 variant cho slide A5 marketplace health.

Variants:
    1. baseline_popularity — `build_candidates` thuần (current code).
    2. rank_only — LGB ranker trên candidate set, không re-rank.
    3. rank_rerank — LGB + marketplace-aware re-ranker.

Đo trên local holdout (2026-04-03..2026-04-09):
    - Recall@10, NDCG@10 (accuracy)
    - Coverage@10 (% items được expose)
    - Seller coverage
    - Gini coefficient của exposure (lower = better)
    - Fresh listing exposure (% items < 7 ngày)
    - Agent / Private exposure

Output: outputs/variants/comparison.csv

Usage:
    python scripts/09_run_variants.py --data-root "C:/Datathon_Data"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.candidates.build_candidates import run_candidate_pipeline
from src.io.duckdb_conn import make_connection
from src.io.local_data import load_test_users, register_clean_views, validate_local_data
from src.ranking.lgbm_ranker import (
    build_holdout_ground_truth,
    evaluate_recall_ndcg,
    load_config,
    run_ranker_pipeline,
    score_candidates,
)
from src.ranking.rerank import load_rerank_config, rerank_recommendations
from src.utils.logger import get_logger
from src.utils.paths import resolve_paths
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _gini(x: np.ndarray) -> float:
    """Gini coefficient (0=equal, 1=max inequality)."""
    if len(x) == 0:
        return 0.0
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def marketplace_metrics(preds: pd.DataFrame, dim: pd.DataFrame, train_end_date: str) -> dict:
    """Tính marketplace-health metrics cho 1 variant.

    preds: user_id, rank, item_id (top-10 per user)
    dim: dim_listing với seller_id, seller_type, posted_date
    """
    df = preds.merge(dim[["item_id", "seller_id", "seller_type", "posted_date"]], on="item_id", how="left")
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    n_sellers = df["seller_id"].nunique()

    # Coverage relative to catalog
    catalog_items = dim["item_id"].nunique()
    catalog_sellers = dim["seller_id"].nunique()

    # Exposure Gini per item (how many times each item appears in top-10)
    exposure_per_item = df.groupby("item_id").size().values
    gini_items = _gini(exposure_per_item)
    exposure_per_seller = df.groupby("seller_id").size().values
    gini_sellers = _gini(exposure_per_seller)

    # Fresh exposure (% slots where listing posted < 7 days before train_end)
    train_end = pd.Timestamp(train_end_date)
    df["age_days"] = (train_end - pd.to_datetime(df["posted_date"])).dt.days
    fresh_slot_pct = float((df["age_days"] <= 7).mean() * 100)
    private_slot_pct = float((df["seller_type"] == "private").mean() * 100)
    agent_slot_pct = float((df["seller_type"] == "agent").mean() * 100)

    return {
        "n_users_with_recs": int(n_users),
        "n_distinct_items_recommended": int(n_items),
        "n_distinct_sellers_recommended": int(n_sellers),
        "item_coverage_pct": float(100 * n_items / max(catalog_items, 1)),
        "seller_coverage_pct": float(100 * n_sellers / max(catalog_sellers, 1)),
        "gini_item_exposure": float(gini_items),
        "gini_seller_exposure": float(gini_sellers),
        "fresh_slot_pct": fresh_slot_pct,
        "private_slot_pct": private_slot_pct,
        "agent_slot_pct": agent_slot_pct,
    }


def get_popularity_preds(con, k: int = 10) -> pd.DataFrame:
    """Variant 1: pure popularity from build_candidates."""
    return con.execute(f"""
        SELECT user_id, rank, item_id
        FROM final_recommendations
        WHERE rank <= {k}
    """).df()


def get_rank_only_preds(con, model_path: Path, item_feat: Path, user_feat: Path, k: int = 10) -> pd.DataFrame:
    """Variant 2: LGB ranking without rerank."""
    scored = score_candidates(con, model_path, item_feat, user_feat, k=k)
    return scored[["user_id", "rank", "item_id"]]


def get_rank_rerank_preds(con, model_path: Path, item_feat: Path, user_feat: Path,
                            rerank_cfg, candidate_k: int = 30) -> pd.DataFrame:
    """Variant 3: LGB + rerank."""
    scored = score_candidates(con, model_path, item_feat, user_feat, k=candidate_k)
    reranked = rerank_recommendations(scored, config=rerank_cfg)
    return reranked[["user_id", "rank", "item_id"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-variant comparison")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default=None, help="Optional ranker YAML")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="10GB")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, reuse existing model")
    args = parser.parse_args()

    set_seed(42)
    paths = resolve_paths(args.data_root, ROOT)
    validate_local_data(paths.data_root)
    con = make_connection(paths.duckdb_tmp, threads=args.threads, memory_limit=args.memory_limit)

    df_test = load_test_users(paths.test_path)
    con.register("test_users_ds", df_test)
    register_clean_views(con, paths.clean_dir, required=True)
    run_candidate_pipeline(con, paths.cache_dir, overwrite=False)

    # Config
    config = load_config(args.config or (ROOT / "config" / "ranker.yaml"))
    rerank_cfg = load_rerank_config(args.config or (ROOT / "config" / "ranker.yaml"))

    # Train ranker (or load existing)
    model_path = paths.model_dir / "lgb_ranker.txt"
    if args.skip_train and model_path.exists():
        logger.info("Skip training, reuse existing model at %s", model_path)
        item_feat = paths.cache_dir / "ranker" / "item_features.parquet"
        user_feat = paths.cache_dir / "ranker" / "user_features.parquet"
        if not item_feat.exists() or not user_feat.exists():
            logger.warning("Feature caches missing despite --skip-train; rebuilding via run_ranker_pipeline")
            run_ranker_pipeline(con, paths, config)
    else:
        result = run_ranker_pipeline(con, paths, config)
        model_path = result["model_path"]

    item_feat = paths.cache_dir / "ranker" / "item_features.parquet"
    user_feat = paths.cache_dir / "ranker" / "user_features.parquet"

    # Ground truth on local holdout
    gt = build_holdout_ground_truth(con, config.holdout_start, config.holdout_end)
    logger.info("Local holdout GT: %d (user,item) pairs", len(gt))

    # Pull dim for marketplace metrics
    dim = con.execute("SELECT item_id, seller_id, seller_type, posted_date FROM dim_clean").df()

    # ---- Run 3 variants ----
    rows = []

    logger.info("===== Variant 1: baseline_popularity =====")
    p1 = get_popularity_preds(con, k=10)
    m1 = evaluate_recall_ndcg(p1, gt, k=10)
    mh1 = marketplace_metrics(p1, dim, train_end_date="2026-04-09")
    rows.append({"variant": "baseline_popularity", **m1, **mh1})

    logger.info("===== Variant 2: rank_only =====")
    p2 = get_rank_only_preds(con, model_path, item_feat, user_feat, k=10)
    m2 = evaluate_recall_ndcg(p2, gt, k=10)
    mh2 = marketplace_metrics(p2, dim, train_end_date="2026-04-09")
    rows.append({"variant": "rank_only", **m2, **mh2})

    logger.info("===== Variant 3: rank_rerank =====")
    p3 = get_rank_rerank_preds(con, model_path, item_feat, user_feat, rerank_cfg,
                                candidate_k=rerank_cfg.candidate_k)
    m3 = evaluate_recall_ndcg(p3, gt, k=10)
    mh3 = marketplace_metrics(p3, dim, train_end_date="2026-04-09")
    rows.append({"variant": "rank_rerank", **m3, **mh3})

    # ---- Save comparison table ----
    out_dir = paths.project_root / "outputs" / "variants"
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / "comparison.csv"
    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(comparison_path, index=False)
    logger.info("==================== COMPARISON DONE ====================")
    logger.info("Saved -> %s", comparison_path)
    print("\n", df_comp.to_string(index=False), "\n")

    # ---- Print A5 narrative numbers ----
    if len(rows) == 3:
        r2 = rows[1]
        r3 = rows[2]
        delta_recall = r3["recall_at_10"] - r2["recall_at_10"]
        delta_gini_item = r2["gini_item_exposure"] - r3["gini_item_exposure"]
        delta_fresh = r3["fresh_slot_pct"] - r2["fresh_slot_pct"]
        print(f"\nA5 narrative (rank_rerank vs rank_only):")
        print(f"  Δ Recall@10: {delta_recall:+.4f}")
        print(f"  Δ Gini(item exposure): {-delta_gini_item:+.4f}  (negative = more equal, better)")
        print(f"  Δ Fresh slot %: {delta_fresh:+.2f} pp")


if __name__ == "__main__":
    main()
