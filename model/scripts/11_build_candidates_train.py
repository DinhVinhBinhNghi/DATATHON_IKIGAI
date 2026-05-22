"""Step 2: Build candidate pool cho RANKER TRAINING.

Cutoff = ranker_train_cutoff (2026-03-12, 28 ngày trước train_end).
Output: cache/candidates/candidates_train.parquet (~36M rows)
        + 5 source files + covisit matrix.

Run:
    python scripts\\11_build_candidates_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import get_logger
from src.candidates import run_build_candidates

logger = get_logger("11_candidates_train")


def main() -> int:
    logger.info("=" * 70)
    logger.info("STEP 2: Build candidates (mode=train)")
    logger.info("=" * 70)
    try:
        run_build_candidates(mode="train")
        logger.info("=" * 70)
        logger.info("STEP 2 ✓ DONE")
        logger.info("=" * 70)
        return 0
    except Exception as e:
        logger.exception("STEP 2 FAILED: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
