from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root
PROJECT_ROOT = bootstrap_project_root()

from src.candidates.build_candidates import run_candidate_pipeline
from src.evaluation.marketplace_health import save_recommendation_health
from src.io.duckdb_conn import make_connection
from src.io.local_data import load_test_users, register_clean_views
from src.utils.logger import get_logger
from src.utils.paths import resolve_paths

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="10GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = resolve_paths(args.data_root, PROJECT_ROOT)
    con = make_connection(paths.duckdb_tmp, threads=args.threads, memory_limit=args.memory_limit)
    df_test = load_test_users(paths.test_path)
    con.register("test_users_ds", df_test)
    register_clean_views(con, paths.clean_dir, required=True)
    outputs = run_candidate_pipeline(con, paths.cache_dir, overwrite=args.overwrite)
    health_path = save_recommendation_health(con, paths.table_dir / "recommendation_health.csv")
    logger.info("Candidate pipeline done. Outputs:")
    for path in outputs:
        logger.info("  %s", path)
    logger.info("Health summary: %s", health_path)


if __name__ == "__main__":
    main()
