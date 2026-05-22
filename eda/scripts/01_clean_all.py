from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root
PROJECT_ROOT = bootstrap_project_root()

from src.cleaning.clean_all import run_clean_all
from src.io.duckdb_conn import make_connection
from src.io.local_data import drop_raw_views, register_raw_views, validate_local_data
from src.utils.logger import get_logger
from src.utils.paths import resolve_paths
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="10GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    paths = resolve_paths(args.data_root, PROJECT_ROOT)
    validate_local_data(paths.data_root)
    con = make_connection(paths.duckdb_tmp, threads=args.threads, memory_limit=args.memory_limit)
    register_raw_views(con, paths.data_root)
    outputs = run_clean_all(con, paths.clean_dir, overwrite=args.overwrite)
    drop_raw_views(con)
    logger.info("Clean outputs:")
    for path in outputs:
        logger.info("  %s", path)


if __name__ == "__main__":
    main()
