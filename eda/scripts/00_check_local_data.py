from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root
bootstrap_project_root()

from src.audit.schema_checks import audit_local_structure
from src.io.local_data import load_test_users, validate_local_data
from src.utils.constants import EXPECTED_TRAIN_FILES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Folder containing train/ and test/")
    args = parser.parse_args()

    counts = validate_local_data(args.data_root)
    df_test = load_test_users(f"{args.data_root}/test")
    print(f"✓ DATA_ROOT: {args.data_root}")
    print(f"✓ Test users: {len(df_test):,} rows, {df_test['user_id'].nunique():,} unique")
    print("\nTrain file counts:")
    for name, expected in EXPECTED_TRAIN_FILES.items():
        n_files = counts[name]
        flag = "✓" if n_files == expected else "⚠"
        print(f"{flag} {name:35s}: {n_files:3d} files expected {expected}")
    print("\nAudit table:")
    print(audit_local_structure(args.data_root).to_string(index=False))
    print("\n✓ Local data structure looks ready.")


if __name__ == "__main__":
    main()
