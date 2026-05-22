from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_submission_csv(con, submission_path: str | Path) -> dict[str, int | bool]:
    path = Path(submission_path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Read with pandas for quick column checks.
    head = pd.read_csv(path, nrows=5)
    expected_cols = ["ID", "user_id", "rank", "item_id"]
    if list(head.columns) != expected_cols:
        raise ValueError(f"Invalid columns: {list(head.columns)}. Expected {expected_cols}")

    con.execute(f"CREATE OR REPLACE TEMP VIEW sub AS SELECT * FROM read_csv_auto('{path.as_posix()}', HEADER=TRUE)")

    checks = {}
    checks["n_rows"] = con.execute("SELECT COUNT(*) FROM sub").fetchone()[0]
    checks["n_users"] = con.execute("SELECT COUNT(DISTINCT user_id) FROM sub").fetchone()[0]
    checks["n_test_users"] = con.execute("SELECT COUNT(DISTINCT user_id) FROM test_users_ds").fetchone()[0]
    checks["duplicate_user_rank"] = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT user_id, rank, COUNT(*) AS n FROM sub GROUP BY user_id, rank HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    checks["duplicate_user_item"] = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT user_id, item_id, COUNT(*) AS n FROM sub GROUP BY user_id, item_id HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    checks["bad_rank"] = con.execute("SELECT COUNT(*) FROM sub WHERE rank < 1 OR rank > 10").fetchone()[0]
    checks["unknown_users"] = con.execute("""
        SELECT COUNT(*) FROM (SELECT DISTINCT user_id FROM sub EXCEPT SELECT user_id FROM test_users_ds)
    """).fetchone()[0]
    checks["missing_users"] = con.execute("""
        SELECT COUNT(*) FROM (SELECT user_id FROM test_users_ds EXCEPT SELECT DISTINCT user_id FROM sub)
    """).fetchone()[0]
    checks["unknown_items"] = con.execute("""
        SELECT COUNT(*) FROM (SELECT DISTINCT item_id FROM sub EXCEPT SELECT item_id FROM dim_clean)
    """).fetchone()[0]
    checks["max_rows_per_user"] = con.execute("""
        SELECT MAX(n) FROM (SELECT user_id, COUNT(*) AS n FROM sub GROUP BY user_id)
    """).fetchone()[0]

    checks["is_valid"] = all([
        checks["duplicate_user_rank"] == 0,
        checks["duplicate_user_item"] == 0,
        checks["bad_rank"] == 0,
        checks["unknown_users"] == 0,
        checks["unknown_items"] == 0,
        checks["max_rows_per_user"] <= 10,
    ])

    for key, value in checks.items():
        logger.info("validation %-22s = %s", key, value)

    if not checks["is_valid"]:
        raise ValueError("Submission validation failed. See logs above.")
    return checks
