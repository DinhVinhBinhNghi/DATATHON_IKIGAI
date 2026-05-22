"""Step 6: Rerank + build Kaggle submission CSV.

Workflow:
1. Apply rerank rules (mode='raw' hoặc 'rerank').
2. Pick top-10 per user.
3. Build submission CSV với fallback global pop.
4. Validate format.

Output: submissions/submission_{mode}_{timestamp}.csv

Run:
    python scripts\\15_rerank_and_submit.py
    python scripts\\15_rerank_and_submit.py --mode raw
    python scripts\\15_rerank_and_submit.py --mode rerank
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_exists_nonempty, get_config, get_logger
from src.rerank import run_rerank
from src.submission import build_submission_csv, validate_submission

logger = get_logger("15_rerank_and_submit")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="raw",
                        choices=["raw", "rerank", "all"],
                        help="Rerank mode (default: raw)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip submission validation")
    args = parser.parse_args()

    cfg = get_config()
    feat_dir = cfg.paths.features_dir
    sub_dir = cfg.paths.submissions_root
    scored_path = feat_dir / "scored_pool_predict.parquet"
    item_feat = feat_dir / "item_features_predict.parquet"

    if not file_exists_nonempty(scored_path):
        logger.error("Missing scored_pool_predict.parquet — chạy scripts\\14 trước.")
        return 1
    if not file_exists_nonempty(item_feat):
        logger.error("Missing item_features_predict.parquet — chạy scripts\\14 trước.")
        return 1

    modes = ["raw", "rerank"] if args.mode == "all" else [args.mode]
    ts = time.strftime("%Y%m%d_%H%M")

    for mode in modes:
        logger.info("=" * 70)
        logger.info("STEP 6 mode=%s: Rerank + Build submission", mode)
        logger.info("=" * 70)

        topk_path = feat_dir / f"final_topk_{mode}.parquet"
        if file_exists_nonempty(topk_path):
            logger.info("  final_topk_%s.parquet đã có, SKIP rerank.", mode)
        else:
            try:
                run_rerank(scored_path, item_feat, mode, topk_path)
            except Exception as e:
                logger.exception("Rerank failed (mode=%s): %s", mode, e)
                return 1

        out_csv = sub_dir / f"submission_{mode}_{ts}.csv"
        try:
            build_submission_csv(topk_path, out_csv)
        except Exception as e:
            logger.exception("Build submission failed (mode=%s): %s", mode, e)
            return 1

        if not args.no_validate:
            try:
                ok = validate_submission(out_csv, raise_on_error=False)
                if not ok:
                    logger.error("Validation FAILED cho %s", out_csv)
            except Exception as e:
                logger.exception("Validate failed: %s", e)

    logger.info("=" * 70)
    logger.info("STEP 6 ✓ DONE")
    logger.info("=" * 70)
    logger.info("Submission files trong: %s", sub_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
