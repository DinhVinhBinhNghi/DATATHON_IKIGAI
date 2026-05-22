"""Step 3: Build candidate pool cho PREDICT (Kaggle submission).

Cutoff = train_end (2026-04-09).
User pool = test_users.parquet (161,568 users).
Output: cache/candidates/candidates_predict.parquet (~32M rows).

Run:
    python scripts\\12_build_candidates_predict.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import get_logger
from src.candidates import run_build_candidates

logger = get_logger("12_candidates_predict")


def main() -> int:
    logger.info("=" * 70)
    logger.info("STEP 3: Build candidates (mode=predict)")
    logger.info("=" * 70)
    try:
        run_build_candidates(mode="predict")
        logger.info("=" * 70)
        logger.info("STEP 3 ✓ DONE")
        logger.info("=" * 70)
        return 0
    except Exception as e:
        logger.exception("STEP 3 FAILED: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
