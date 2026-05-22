from __future__ import annotations

from pathlib import Path

import duckdb


def make_connection(temp_directory: str | Path, threads: int = 2, memory_limit: str = "10GB") -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection tuned for large local Parquet scans.

    Notes
    -----
    - Lower thread count often reduces peak RAM on Windows laptops.
    - preserve_insertion_order=false lets DuckDB use less memory for large aggregations.
    - temp_directory enables spilling when possible.
    """
    tmp = Path(temp_directory)
    tmp.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"SET threads TO {int(threads)};")
    con.execute(f"SET memory_limit='{memory_limit}';")
    con.execute(f"SET temp_directory='{tmp.as_posix()}';")
    con.execute("SET preserve_insertion_order=false;")
    return con
