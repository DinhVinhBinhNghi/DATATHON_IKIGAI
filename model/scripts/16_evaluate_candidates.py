"""Step 7a: Evaluate candidate pool quality (Recall@N ceiling + source coverage).

Đây là evaluation TRƯỚC RANKER. Trả lời câu: "Candidate gen có cover được
ground truth không?". Recall@200 ceiling = max Recall ranker có thể đạt được.

Compare với Recall@10 final → biết bottleneck:
- ceiling cao, final thấp → ranker yếu, cần tune
- ceiling thấp → candidate gen yếu, cần thêm sources

Output:
- outputs/tables/candidate_pool_eval.csv

Run:
    python scripts\\16_evaluate_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_exists_nonempty, get_config, get_logger
from src.evaluation import build_internal_ground_truth, evaluate_candidates_pool

logger = get_logger("16_evaluate_candidates")


def main() -> int:
    cfg = get_config()
    cand_dir = cfg.paths.candidates_dir
    tables_dir = cfg.paths.tables_dir

    candidates_path = cand_dir / "candidates_predict.parquet"
    if not file_exists_nonempty(candidates_path):
        logger.error("Missing candidates_predict.parquet — chạy scripts\\12 trước.")
        return 1

    logger.info("=" * 70)
    logger.info("STEP 7a: Evaluate candidate pool")
    logger.info("=" * 70)

    gt_path = build_internal_ground_truth()
    out_csv = tables_dir / "candidate_pool_eval.csv"

    try:
        evaluate_candidates_pool(candidates_path, gt_path, out_csv=out_csv)
    except Exception as e:
        logger.exception("Eval candidates failed: %s", e)
        return 1

    logger.info("=" * 70)
    logger.info("STEP 7a ✓ DONE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
