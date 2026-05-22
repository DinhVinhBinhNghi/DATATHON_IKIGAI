"""Common utilities: config, logging, DuckDB, IO helpers.

Public API:
- get_config()                 — load + cache config từ config/local.yaml
- get_logger(name)             — logger với format thống nhất
- make_connection()            — DuckDB connection với memory limit + temp dir
- file_exists_nonempty(path)   — check resume-safe
- read_parquet_dataset(path)   — đọc dir hoặc file parquet
- timed(label, logger)         — context manager đo time
"""
from src.common.config import get_config
from src.common.logging_utils import get_logger, timed
from src.common.db import make_connection
from src.common.io import file_exists_nonempty, read_parquet_dataset

__all__ = [
    "get_config",
    "get_logger",
    "timed",
    "make_connection",
    "file_exists_nonempty",
    "read_parquet_dataset",
]