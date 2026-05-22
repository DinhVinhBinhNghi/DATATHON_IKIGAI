from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_project_root
PROJECT_ROOT = bootstrap_project_root()

from src.io.duckdb_conn import make_connection
from src.io.local_data import load_test_users, register_clean_views
from src.submission.validate_submission import validate_submission_csv
from src.utils.paths import resolve_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--submission", default="submissions/submission.csv")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="8GB")
    args = parser.parse_args()

    paths = resolve_paths(args.data_root, PROJECT_ROOT)
    con = make_connection(paths.duckdb_tmp, threads=args.threads, memory_limit=args.memory_limit)
    df_test = load_test_users(paths.test_path)
    con.register("test_users_ds", df_test)
    register_clean_views(con, paths.clean_dir, required=True)
    validate_submission_csv(con, Path(args.submission))
    print("✓ Submission validation passed")


if __name__ == "__main__":
    main()
