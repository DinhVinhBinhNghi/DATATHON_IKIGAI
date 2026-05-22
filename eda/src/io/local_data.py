from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from src.utils.constants import EXPECTED_TRAIN_FILES


def validate_local_data(data_root: str | Path) -> dict[str, int]:
    root = Path(data_root).expanduser().resolve()
    train = root / "train"
    test = root / "test"
    if not root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
    if not train.exists():
        raise FileNotFoundError(f"Missing train/: {train}")
    if not test.exists():
        raise FileNotFoundError(f"Missing test/: {test}")
    if not (test / "test_users.parquet").exists():
        raise FileNotFoundError(f"Missing test_users.parquet: {test / 'test_users.parquet'}")

    counts = {}
    for name in EXPECTED_TRAIN_FILES:
        folder = train / name
        if not folder.exists():
            raise FileNotFoundError(f"Missing train/{name}/: {folder}")
        dataset = ds.dataset(str(folder), format="parquet")
        counts[name] = len(dataset.files)
    return counts


def load_test_users(test_path: str | Path) -> pd.DataFrame:
    df_test = pd.read_parquet(Path(test_path) / "test_users.parquet")
    if "user_id" not in df_test.columns:
        raise ValueError("test_users.parquet missing user_id")
    return df_test


def register_raw_views(con, data_root: str | Path, df_test: pd.DataFrame | None = None) -> pd.DataFrame:
    root = Path(data_root).expanduser().resolve()
    train = root / "train"
    test = root / "test"

    datasets = {
        "dim_ds": ds.dataset(str(train / "dim_listing"), format="parquet"),
        "snapshot_ds": ds.dataset(str(train / "fact_listing_snapshot"), format="parquet"),
        "interactions_ds": ds.dataset(str(train / "fact_post_contact_interactions"), format="parquet"),
        "events_ds": ds.dataset(str(train / "fact_user_events"), format="parquet"),
    }
    for view, dataset in datasets.items():
        con.register(view, dataset)

    if df_test is None:
        df_test = load_test_users(test)
    con.register("test_users_ds", df_test)
    return df_test


def register_clean_views(con, clean_dir: str | Path, required: bool = True) -> list[str]:
    clean_dir = Path(clean_dir)
    mapping = {
        "dim_clean": clean_dir / "dim_listing_clean.parquet",
        "snap_clean": clean_dir / "snapshot_clean.parquet",
        "int_clean": clean_dir / "interactions_clean.parquet",
        "events_pos": clean_dir / "events_positive_clean.parquet",
    }
    registered = []
    for view, path in mapping.items():
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing clean file for {view}: {path}")
            continue
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM '{path.as_posix()}'")
        registered.append(view)
    return registered


def drop_raw_views(con) -> None:
    for view in ["events_ds", "dim_ds", "snapshot_ds", "interactions_ds"]:
        con.execute(f"DROP VIEW IF EXISTS {view}")
