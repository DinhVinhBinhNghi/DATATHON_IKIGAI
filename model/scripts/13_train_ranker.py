"""Step 4: Build features cho train + Train LightGBM LambdaRank.

Workflow:
1. Build features_train (user, item, pair, temporal) tại ranker_train_cutoff.
2. Join features → ranker_input_train.parquet.
3. Train LightGBM LambdaRank với label graded 0-3.
4. Save model + feature importance + metadata.

Run:
    python scripts\\13_train_ranker.py
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
from src.ranker import train_ranker

logger = get_logger("13_train_ranker")


def main() -> int:
    cfg = get_config()
    mode = "train"
    cutoff = cfg.windows.ranker_train_cutoff
    feat_dir = cfg.paths.features_dir
    cand_dir = cfg.paths.candidates_dir
    model_dir = cfg.paths.models_dir

    candidates_path = cand_dir / "candidates_train.parquet"
    if not file_exists_nonempty(candidates_path):
        logger.error("Missing candidates_train.parquet — chạy scripts\\11 trước.")
        return 1

    logger.info("=" * 70)
    logger.info("STEP 4: Build features + Train ranker (mode=train, cutoff=%s)", cutoff)
    logger.info("=" * 70)

    # 1. User features
    user_feat = feat_dir / "user_features_train.parquet"
    if not file_exists_nonempty(user_feat):
        build_user_features(cutoff, mode, user_feat)
    else:
        logger.info("  user_features_train.parquet đã có, SKIP")

    # 2. Item features
    item_feat = feat_dir / "item_features_train.parquet"
    if not file_exists_nonempty(item_feat):
        build_item_features(cutoff, mode, item_feat)
    else:
        logger.info("  item_features_train.parquet đã có, SKIP")

    # 3. Pair features (cần candidates)
    pair_feat = feat_dir / "pair_features_train.parquet"
    if not file_exists_nonempty(pair_feat):
        build_pair_features(cutoff, mode, candidates_path, pair_feat)
    else:
        logger.info("  pair_features_train.parquet đã có, SKIP")

    # 4. Temporal features
    temp_feat = feat_dir / "temporal_features_train.parquet"
    if not file_exists_nonempty(temp_feat):
        build_temporal_features(cutoff, mode, candidates_path, temp_feat)
    else:
        logger.info("  temporal_features_train.parquet đã có, SKIP")

    # 5. Join all → ranker_input
    ranker_input = feat_dir / "ranker_input_train.parquet"
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
        logger.info("  ranker_input_train.parquet đã có, SKIP")

    # 6. Train ranker
    model_path = model_dir / "lgb_ranker.txt"
    imp_path = model_dir / "lgb_feature_importance.parquet"
    meta_path = model_dir / "lgb_metadata.json"
    if file_exists_nonempty(model_path):
        logger.info("  lgb_ranker.txt đã có, SKIP train.")
        logger.info("  → Để retrain, xóa: %s", model_path)
    else:
        try:
            train_ranker(ranker_input, model_path, imp_path, meta_path)
        except Exception as e:
            logger.exception("Train failed: %s", e)
            return 1

    logger.info("=" * 70)
    logger.info("STEP 4 ✓ DONE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
