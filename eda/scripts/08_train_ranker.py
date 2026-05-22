"""Train LGB LambdaRank ranker trên local data.

Prerequisites:
    1) `python scripts/01_clean_all.py --data-root <root>` đã chạy xong.
    2) `python scripts/03_build_candidates.py --data-root <root>` đã chạy xong
       (để có candidate_scores cho in-candidate negative sampling).

Usage:
    python scripts/08_train_ranker.py --data-root "C:/Datathon_Data"
    python scripts/08_train_ranker.py --data-root "C:/Datathon_Data" --config config/ranker.yaml

Output:
    outputs/models/lgb_ranker.txt
    outputs/models/lgb_ranker_meta.json
    outputs/models/lgb_ranker_feature_importance.csv
    outputs/tables/ranker_holdout_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.candidates.build_candidates import run_candidate_pipeline
from src.io.duckdb_conn import make_connection
from src.io.local_data import load_test_users, register_clean_views, validate_local_data
from src.ranking.lgbm_ranker import load_config, run_ranker_pipeline
from src.utils.logger import get_logger
from src.utils.paths import resolve_paths
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LGB LambdaRank ranker")
    parser.add_argument("--data-root", required=True, help="Folder containing train/ and test/")
    parser.add_argument("--config", default=None, help="Optional path to ranker YAML config")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="10GB")
    parser.add_argument("--overwrite-candidates", action="store_true",
                        help="Rebuild candidate_scores trước khi train")
    args = parser.parse_args()

    set_seed(42)
    paths = resolve_paths(args.data_root, ROOT)
    validate_local_data(paths.data_root)
    con = make_connection(paths.duckdb_tmp, threads=args.threads, memory_limit=args.memory_limit)

    df_test = load_test_users(paths.test_path)
    con.register("test_users_ds", df_test)
    register_clean_views(con, paths.clean_dir, required=True)

    # Make sure candidate_scores view is available (for in-candidate negative sampling)
    logger.info("Ensuring candidate_scores is available...")
    run_candidate_pipeline(con, paths.cache_dir, overwrite=args.overwrite_candidates)

    config = load_config(args.config) if args.config else load_config()
    if args.config is None:
        # Try default location
        default_yaml = ROOT / "config" / "ranker.yaml"
        if default_yaml.exists():
            logger.info("Found default config at %s", default_yaml)
            config = load_config(default_yaml)

    result = run_ranker_pipeline(con, paths, config)
    logger.info("==================== RANKER DONE ====================")
    logger.info("Model: %s", result["model_path"])
    logger.info("Local holdout metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
