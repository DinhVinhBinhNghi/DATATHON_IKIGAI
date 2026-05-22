"""Step 1: Preaggregate events thành user/item/user_item daily stats với weighted_score.

Đây là chỗ FIX BUG của v2.4.0 — mọi aggregation dùng weighted_score thay vì raw count.

Run:
    python scripts\\10_preaggregate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import get_logger
from src.preagg import run_preaggregate

logger = get_logger("10_preaggregate")


def main() -> int:
    logger.info("=" * 70)
    logger.info("STEP 1: Preaggregate (weighted_score)")
    logger.info("=" * 70)
    try:
        run_preaggregate()
        logger.info("=" * 70)
        logger.info("STEP 1 ✓ DONE")
        logger.info("=" * 70)
        return 0
    except Exception as e:
        logger.exception("STEP 1 FAILED: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
