"""IO helpers: parquet read/write, file existence check.

Functions:
- file_exists_nonempty(path): True nếu file/dir tồn tại VÀ có content.
  Dùng cho resume-safe checks ở đầu mỗi step.
- read_parquet_dataset(path): đọc 1 file hoặc cả dir, trả pyarrow Table.
- read_parquet_columns(path, columns): chỉ đọc subset columns (tiết kiệm RAM).
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as ds
import pyarrow as pa


def file_exists_nonempty(path: Path | str) -> bool:
    """Resume-safe check: file/dir tồn tại + có size > 0.

    Cho file: kiểm size > 0 (parquet rỗng vẫn ≥ vài bytes nên check >100B).
    Cho dir: kiểm có ít nhất 1 file con.
    """
    p = Path(path)
    if not p.exists():
        return False
    if p.is_file():
        return p.stat().st_size > 100  # parquet rỗng ~ 4 bytes magic
    if p.is_dir():
        # Có ít nhất 1 file con (recursive)
        try:
            next(p.rglob("*"))
            return True
        except StopIteration:
            return False
    return False


def read_parquet_dataset(path: Path | str,
                         columns: list[str] | None = None,
                         filter_expr=None) -> pa.Table:
    """Đọc 1 file parquet hoặc cả dir thành pyarrow Table.

    Args:
        path: đường dẫn file hoặc dir.
        columns: list columns cần đọc (None = đọc hết).
        filter_expr: pyarrow.dataset filter expression (push-down).

    Returns:
        pyarrow.Table.
    """
    p = Path(path)
    dataset = ds.dataset(str(p), format="parquet")
    return dataset.to_table(columns=columns, filter=filter_expr)


def parquet_row_count(path: Path | str) -> int:
    """Đếm rows trong file/dir parquet KHÔNG load vào memory."""
    p = Path(path)
    dataset = ds.dataset(str(p), format="parquet")
    return dataset.count_rows()




