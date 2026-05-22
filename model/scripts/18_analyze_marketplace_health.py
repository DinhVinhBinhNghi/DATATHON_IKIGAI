"""Step 7c: Analyze marketplace health của submission.

Compute Gini, coverage, freshness, diversity. Dùng cho slide phần "trade-off
giữa Recall và Marketplace Health".

Output:
- outputs/tables/health_{filename}.csv

Run:
    python scripts\\18_analyze_marketplace_health.py
    python scripts\\18_analyze_marketplace_health.py --submission submissions\\submission_raw_xxx.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import get_config, get_logger
from src.evaluation import compute_marketplace_health

logger = get_logger("18_marketplace_health")


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
    logger.info("STEP 7c: Marketplace health analysis")
    logger.info("=" * 70)

    out_csv = tables_dir / f"health_{sub_path.stem}.csv"
    try:
        compute_marketplace_health(sub_path, out_csv=out_csv)
    except Exception as e:
        logger.exception("Health analysis failed: %s", e)
        return 1

    logger.info("=" * 70)
    logger.info("STEP 7c ✓ DONE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
