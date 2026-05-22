"""DuckDB connection factory.

Pattern:
    from src.common import make_connection
    con = make_connection()
    con.execute("SELECT 1").fetchone()
    # con tự đóng khi out of scope (DuckDB Python binding handles GC)

Khuyến nghị: tạo 1 connection per script. Connection có cost setup
(memory_limit, threads), nhưng query nhanh.

Memory limit + temp dir đọc từ cfg.duckdb. Spill tới disk khi vượt limit.
"""
from __future__ import annotations

import duckdb

from src.common.config import get_config
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def make_connection(memory_limit: str | None = None,
                    threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """Tạo DuckDB connection với memory + threads từ config.

    Args:
        memory_limit: override config (vd "4GB"). Default = cfg.duckdb.memory_limit.
        threads: override config. Default = cfg.duckdb.threads.

    Returns:
        DuckDB connection in-memory (không persist tới file).
    """
    cfg = get_config()
    db = cfg.duckdb

    mem = memory_limit or getattr(db, "memory_limit", "6GB")
    n_threads = threads if threads is not None else getattr(db, "threads", 4)

    con = duckdb.connect(database=":memory:", read_only=False)
    con.execute(f"SET memory_limit='{mem}'")
    con.execute(f"SET threads TO {n_threads}")

    # Temp directory cho spill (khi query vượt memory_limit)
    temp_dir = getattr(db, "temp_directory", None)
    if temp_dir:
        con.execute(f"SET temp_directory='{temp_dir}'")

    # Enable PRAGMAs hữu ích
    con.execute("SET preserve_insertion_order=false")  # nhanh hơn cho bulk insert

    logger.debug("DuckDB connection: memory=%s, threads=%d", mem, n_threads)
    return con

