from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root
PROJECT_ROOT = bootstrap_project_root()

from src.eda.run_eda import run_eda_tables
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
    args = parser.parse_args()

    paths = resolve_paths(args.data_root, PROJECT_ROOT)
    con = make_connection(paths.duckdb_tmp, threads=args.threads, memory_limit=args.memory_limit)
    df_test = load_test_users(paths.test_path)
    con.register("test_users_ds", df_test)
    register_clean_views(con, paths.clean_dir, required=True)
    outputs = run_eda_tables(con, paths.table_dir)
    logger.info("EDA done. %s tables saved to %s", len(outputs), paths.table_dir)


if __name__ == "__main__":
    main()
