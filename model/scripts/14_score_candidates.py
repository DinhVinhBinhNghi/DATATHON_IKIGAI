"""Step 5: Build features cho predict + Score candidates với LightGBM model.

Workflow:
1. Build features_predict (cutoff = train_end).
2. Join features → ranker_input_predict.parquet.
3. Load model → predict pred_score.
4. Save scored_pool_predict.parquet.

Run:
    python scripts\\14_score_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import file_exists_nonempty, get_config, get_logger
from src.features import (
    build_user_features,
    build_item_features,
    build_pair_features,
    build_temporal_features,
    build_ranker_input,
)
from src.ranker import score_candidates

logger = get_logger("14_score_candidates")


def main() -> int:
    cfg = get_config()
    mode = "predict"
    cutoff = cfg.windows.train_end
    feat_dir = cfg.paths.features_dir
    cand_dir = cfg.paths.candidates_dir
    model_dir = cfg.paths.models_dir

    candidates_path = cand_dir / "candidates_predict.parquet"
    model_path = model_dir / "lgb_ranker.txt"
    if not file_exists_nonempty(candidates_path):
        logger.error("Missing candidates_predict.parquet — chạy scripts\\12 trước.")
        return 1
    if not file_exists_nonempty(model_path):
        logger.error("Missing lgb_ranker.txt — chạy scripts\\13 trước.")
        return 1

    logger.info("=" * 70)
    logger.info("STEP 5: Build features + Score (mode=predict, cutoff=%s)", cutoff)
    logger.info("=" * 70)

    # 1. User features
    user_feat = feat_dir / "user_features_predict.parquet"
    if not file_exists_nonempty(user_feat):
        build_user_features(cutoff, mode, user_feat)
    else:
        logger.info("  user_features_predict.parquet đã có, SKIP")

    # 2. Item features
    item_feat = feat_dir / "item_features_predict.parquet"
    if not file_exists_nonempty(item_feat):
        build_item_features(cutoff, mode, item_feat)
    else:
        logger.info("  item_features_predict.parquet đã có, SKIP")

    # 3. Pair features
    pair_feat = feat_dir / "pair_features_predict.parquet"
    if not file_exists_nonempty(pair_feat):
        build_pair_features(cutoff, mode, candidates_path, pair_feat)
    else:
        logger.info("  pair_features_predict.parquet đã có, SKIP")

    # 4. Temporal features
    temp_feat = feat_dir / "temporal_features_predict.parquet"
    if not file_exists_nonempty(temp_feat):
        build_temporal_features(cutoff, mode, candidates_path, temp_feat)
    else:
        logger.info("  temporal_features_predict.parquet đã có, SKIP")

    # 5. Join → ranker_input
    ranker_input = feat_dir / "ranker_input_predict.parquet"
    if not file_exists_nonempty(ranker_input):
        build_ranker_input(
            mode=mode,
            candidates_path=candidates_path,
            user_feat_path=user_feat,
            item_feat_path=item_feat,
            pair_feat_path=pair_feat,
            temporal_feat_path=temp_feat,
            out_path=ranker_input,
        )
    else:
        logger.info("  ranker_input_predict.parquet đã có, SKIP")

    # 6. Score
    scored_path = feat_dir / "scored_pool_predict.parquet"
    if file_exists_nonempty(scored_path):
        logger.info("  scored_pool_predict.parquet đã có, SKIP")
    else:
        try:
            score_candidates(ranker_input, model_path, scored_path)
        except Exception as e:
            logger.exception("Score failed: %s", e)
            return 1

    logger.info("=" * 70)
    logger.info("STEP 5 ✓ DONE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
