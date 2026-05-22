"""Validate submission CSV trước khi upload Kaggle.

Checks:
1. Header: "ID,user_id,rank,item_id"
2. Row count: = n_test_users × 10
3. ID unique và bắt đầu từ 1
4. user_id all in test_users.parquet
5. rank in [1, 10] per user
6. (user_id, rank) unique
7. (user_id, item_id) unique per user
8. item_id all in dim_listing (warning nếu không)
9. No NULL/empty
10. UTF-8 encoding, no BOM
11. File size: warning nếu > 100 MB (cần ZIP manual cho Kaggle)

[v3.2.3] Bỏ check .csv.gz. File CSV > 100 MB chỉ warning (không fail), vì:
- Kaggle accept .zip thay vì .gz nên gzip auto vô ích
- User tự ZIP bằng tay khi cần upload Kaggle
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import get_config, get_logger, make_connection

logger = get_logger(__name__)


def validate_submission(csv_path: Path, raise_on_error: bool = True) -> bool:
    """Validate submission CSV.

    Args:
        csv_path: path to submission.csv.
        raise_on_error: if True, raise AssertionError on first failure.

    Returns:
        True nếu tất cả checks pass, else False.
    """
    cfg = get_config()
    test_users_path = cfg.paths.test_users_file
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")

    logger.info("[VALIDATE] %s", csv_path)
    all_ok = True

    def fail(msg: str) -> None:
        nonlocal all_ok
        all_ok = False
        if raise_on_error:
            raise AssertionError(f"Submission validation FAIL: {msg}")
        else:
            logger.error("  FAIL: %s", msg)

    # 1. Check file exists + UTF-8
    if not csv_path.exists():
        fail(f"File không tồn tại: {csv_path}")
        return False
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            first_bytes = f.read(3)
        if first_bytes.startswith("\ufeff"):
            fail("CSV có BOM (BTC yêu cầu UTF-8 không BOM)")
    except UnicodeDecodeError as e:
        fail(f"Encoding không phải UTF-8: {e}")
        return False

    # 2. Header
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
    expected = "ID,user_id,rank,item_id"
    if header != expected:
        fail(f"Header sai: '{header}' != '{expected}'")

    # 3. Read CSV
    df = pd.read_csv(csv_path, dtype={"user_id": str, "item_id": str})
    logger.info("  Rows: %s", f"{len(df):,}")

    # 4. Row count
    con = make_connection()
    n_test = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{str(test_users_path).replace(chr(92), '/')}')"
    ).fetchone()[0]
    expected_rows = n_test * 10
    if len(df) != expected_rows:
        fail(f"Row count {len(df):,} != {expected_rows:,} (= {n_test:,} users × 10)")
    else:
        logger.info("  Row count OK: %s = %s users × 10",
                    f"{len(df):,}", f"{n_test:,}")

    # 5. ID unique + sequential
    if df["ID"].is_unique:
        logger.info("  ID unique: OK")
    else:
        fail("ID không unique")
    if df["ID"].min() != 1:
        fail(f"ID start = {df['ID'].min()}, expected 1")

    # 6. rank in [1, 10]
    rank_min, rank_max = df["rank"].min(), df["rank"].max()
    if not (rank_min == 1 and rank_max == 10):
        fail(f"rank không trong [1, 10]: min={rank_min}, max={rank_max}")
    else:
        logger.info("  rank in [1, 10]: OK")

    # 7. (user_id, rank) unique
    if df.duplicated(["user_id", "rank"]).any():
        n_dup = df.duplicated(["user_id", "rank"]).sum()
        fail(f"{n_dup} cặp (user_id, rank) bị duplicate")
    else:
        logger.info("  (user_id, rank) unique: OK")

    # 8. (user_id, item_id) unique per user
    dup_per_user = df.duplicated(["user_id", "item_id"]).sum()
    if dup_per_user > 0:
        fail(f"{dup_per_user} (user_id, item_id) duplicated trong cùng user")
    else:
        logger.info("  (user_id, item_id) unique within user: OK")

    # 9. Each user has exactly 10 rows
    rows_per_user = df.groupby("user_id").size()
    not_10 = (rows_per_user != 10).sum()
    if not_10 > 0:
        fail(f"{not_10} users không có đúng 10 rows")
    else:
        logger.info("  Each user has exactly 10 rows: OK")

    # 10. user_id all in test_users
    test_str = str(test_users_path).replace("\\", "/")
    test_ids = set(con.execute(
        f"SELECT user_id FROM read_parquet('{test_str}')"
    ).df()["user_id"])
    sub_users = set(df["user_id"])
    not_in_test = sub_users - test_ids
    if not_in_test:
        fail(f"{len(not_in_test)} user_id không có trong test_users")
    else:
        logger.info("  All user_id in test_users: OK")
    missing_in_sub = test_ids - sub_users
    if missing_in_sub:
        fail(f"{len(missing_in_sub)} test user_id missing trong submission")
    else:
        logger.info("  All test users covered: OK")

    # 11. item_id all in dim_listing (warning only)
    sub_items = set(df["item_id"])
    dim_items = set(con.execute(
        f"SELECT DISTINCT item_id FROM read_parquet('{dim_glob}')"
    ).df()["item_id"])
    invalid_items = sub_items - dim_items
    if invalid_items:
        logger.warning("  WARN: %d item_ids không có trong dim_listing (BTC sẽ drop)",
                       len(invalid_items))
    else:
        logger.info("  All item_id in dim_listing: OK")

    # 12. No NULL
    n_null = df.isnull().sum().sum()
    if n_null > 0:
        fail(f"{n_null} cells có NULL")
    else:
        logger.info("  No NULL: OK")

    # 13. [v3.2.3] File size — warning only, không fail nếu > 100 MB
    size_mb = csv_path.stat().st_size / 1e6
    if size_mb > 100:
        logger.warning("  File size: %.1f MB > 100 MB (Kaggle limit)", size_mb)
        logger.warning("  -> Cần ZIP manual trước khi upload Kaggle:")
        logger.warning("     PowerShell: Compress-Archive -Path '%s' -DestinationPath '%s.zip'",
                       csv_path, csv_path.stem)
    else:
        logger.info("  File size: %.1f MB (<= 100 MB OK)", size_mb)

    if all_ok:
        logger.info("[VALIDATE] ALL CHECKS PASSED")
    else:
        logger.error("[VALIDATE] SOME CHECKS FAILED")

    return all_ok