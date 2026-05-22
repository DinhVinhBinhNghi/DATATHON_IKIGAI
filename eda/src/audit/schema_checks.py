from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from src.utils.constants import EXPECTED_TRAIN_FILES


def audit_local_structure(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root).expanduser().resolve()
    rows = []
    for table, expected in EXPECTED_TRAIN_FILES.items():
        folder = root / "train" / table
        if folder.exists():
            dataset = ds.dataset(str(folder), format="parquet")
            n_files = len(dataset.files)
            ok = n_files == expected
        else:
            n_files = 0
            ok = False
        rows.append({"table": table, "n_files": n_files, "expected_files": expected, "ok": ok})
    test_file = root / "test" / "test_users.parquet"
    rows.append({"table": "test_users", "n_files": int(test_file.exists()), "expected_files": 1, "ok": test_file.exists()})
    return pd.DataFrame(rows)
