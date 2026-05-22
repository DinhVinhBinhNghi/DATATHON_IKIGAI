from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root
PROJECT_ROOT = bootstrap_project_root()

from src.candidates.build_candidates import run_candidate_pipeline
from src.cleaning.clean_all import run_clean_all
from src.eda.run_eda import run_eda_tables
from src.evaluation.marketplace_health import save_recommendation_health
from src.io.duckdb_conn import make_connection
from src.io.local_data import drop_raw_views, load_test_users, register_clean_views, register_raw_views, validate_local_data
from src.submission.make_submission import make_submission_csv
from src.submission.validate_submission import validate_submission_csv
from src.utils.logger import get_logger
from src.utils.paths import resolve_paths
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full local Datathon 2026 recommender pipeline")
    parser.add_argument("--data-root", required=True, help="Folder containing train/ and test/")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="10GB")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing clean/cache files")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    paths = resolve_paths(args.data_root, PROJECT_ROOT)
    validate_local_data(paths.data_root)
    con = make_connection(paths.duckdb_tmp, threads=args.threads, memory_limit=args.memory_limit)

    df_test = load_test_users(paths.test_path)
    con.register("test_users_ds", df_test)

    if not args.skip_clean:
        logger.info("STEP 1/5: register raw local data")
        register_raw_views(con, paths.data_root, df_test=df_test)
        logger.info("STEP 2/5: clean all tables")
        run_clean_all(con, paths.clean_dir, overwrite=args.overwrite)
        drop_raw_views(con)
    else:
        logger.info("Skip cleaning. Restoring clean views.")
        register_clean_views(con, paths.clean_dir, required=True)

    # Make sure clean views exist after either path.
    register_clean_views(con, paths.clean_dir, required=True)

    if not args.skip_eda:
        logger.info("STEP 3/5: run EDA aggregate tables")
        run_eda_tables(con, paths.table_dir)

    logger.info("STEP 4/5: build candidates and final recommendations")
    run_candidate_pipeline(con, paths.cache_dir, overwrite=args.overwrite)
    save_recommendation_health(con, paths.table_dir / "recommendation_health.csv")

    logger.info("STEP 5/5: make and validate submission")
    out = make_submission_csv(con, paths.submission_dir, filename="submission.csv")
    validate_submission_csv(con, out)
    logger.info("DONE. Submission ready: %s", out)


if __name__ == "__main__":
    main()
