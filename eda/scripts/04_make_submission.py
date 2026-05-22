from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root
PROJECT_ROOT = bootstrap_project_root()

from src.io.duckdb_conn import make_connection
from src.io.local_data import load_test_users, register_clean_views
from src.submission.make_submission import make_submission_csv
from src.submission.validate_submission import validate_submission_csv
from src.utils.logger import get_logger
from src.utils.paths import resolve_paths

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--filename", default="submission.csv")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()

    paths = resolve_paths(args.data_root, PROJECT_ROOT)
    con = make_connection(paths.duckdb_tmp, threads=args.threads, memory_limit=args.memory_limit)
    df_test = load_test_users(paths.test_path)
    con.register("test_users_ds", df_test)
    register_clean_views(con, paths.clean_dir, required=True)
    con.execute(f"CREATE OR REPLACE VIEW final_recommendations AS SELECT * FROM '{(paths.cache_dir / 'final_recommendations.parquet').as_posix()}'")
    out = make_submission_csv(con, paths.submission_dir, filename=args.filename)
    validate_submission_csv(con, out)
    logger.info("Submission ready: %s", out)


if __name__ == "__main__":
    main()
