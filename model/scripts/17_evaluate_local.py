"""Step 7b: Evaluate submission local — Recall@10, NDCG@10.

Run trên internal GT để biết submission tốt cỡ nào trước Kaggle.

Args:
    --submission: path to submission.csv (default: latest in submissions/).

Output:
- Log Recall@10, NDCG@10 trên terminal
- outputs/tables/local_eval_{filename}.csv (single-row metrics)

Run:
    python scripts\\17_evaluate_local.py
    python scripts\\17_evaluate_local.py --submission submissions\\submission_raw_xxx.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.common import get_config, get_logger
from src.evaluation import build_internal_ground_truth, evaluate_submission

logger = get_logger("17_evaluate_local")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default=None,
                        help="Path to submission.csv (default: latest in submissions/)")
    args = parser.parse_args()

    cfg = get_config()
    sub_dir = cfg.paths.submissions_root
    tables_dir = cfg.paths.tables_dir

    if args.submission:
        sub_path = Path(args.submission)
    else:
        # Latest submission
        candidates = sorted(sub_dir.glob("submission_*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            logger.error("Không tìm thấy submission file trong %s", sub_dir)
            return 1
        sub_path = candidates[0]
        logger.info("Auto-select latest submission: %s", sub_path)

    if not sub_path.exists():
        logger.error("File không tồn tại: %s", sub_path)
        return 1

    logger.info("=" * 70)
    logger.info("STEP 7b: Evaluate submission local")
    logger.info("=" * 70)

    gt_path = build_internal_ground_truth()
    try:
        metrics = evaluate_submission(sub_path, gt_path, k=10)
    except Exception as e:
        logger.exception("Eval submission failed: %s", e)
        return 1

    # Save CSV
    out_csv = tables_dir / f"local_eval_{sub_path.stem}.csv"
    pd.DataFrame([{**metrics, "submission": sub_path.name}]).to_csv(out_csv, index=False)
    logger.info("Saved: %s", out_csv)

    logger.info("=" * 70)
    logger.info("STEP 7b ✓ DONE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
