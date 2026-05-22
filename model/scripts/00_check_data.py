"""Step 0: Verify raw data structure trước khi chạy pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import get_config, get_logger, make_connection

logger = get_logger("00_check_data")


def main() -> int:
    cfg = get_config()
    p = cfg.paths
    all_ok = True

    def check(label: str, path: Path, is_dir: bool = False) -> None:
        nonlocal all_ok
        if not path.exists():
            logger.error("  X %s MISSING: %s", label, path)
            all_ok = False
            return
        if is_dir and not path.is_dir():
            logger.error("  X %s khong phai dir: %s", label, path)
            all_ok = False
            return
        logger.info("  OK %s: %s", label, path)

    logger.info("[STEP 0] Check raw data structure")
    logger.info("  raw_root: %s", p.raw_root)
    check("raw_root", p.raw_root, is_dir=True)
    check("dim_listing dir", p.dim_listing_dir, is_dir=True)
    check("fact_user_events dir", p.fact_events_dir, is_dir=True)
    check("fact_listing_snapshot dir", p.fact_snapshot_dir, is_dir=True)
    check("fact_post_contact_interactions dir", p.fact_interactions_dir, is_dir=True)
    check("test_users.parquet", p.test_users_file, is_dir=False)

    if not all_ok:
        logger.error("[STEP 0] FAIL - fix paths in config/local.yaml")
        return 1

    logger.info("[STEP 0] Read sample data...")
    con = make_connection()

    test_str = str(p.test_users_file).replace("\\", "/")
    n_test = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{test_str}')"
    ).fetchone()[0]
    logger.info("  test_users: %s users", f"{n_test:,}")

    dim_glob = str(p.dim_listing_dir / "*.parquet").replace("\\", "/")
    n_items = con.execute(
        f"SELECT COUNT(DISTINCT item_id) FROM read_parquet('{dim_glob}')"
    ).fetchone()[0]
    logger.info("  dim_listing: %s unique items", f"{n_items:,}")

    import glob
    event_files = sorted(glob.glob(str(p.fact_events_dir / "*.parquet")))
    logger.info("  fact_user_events: %d files total", len(event_files))
    if event_files:
        sample_file = event_files[0].replace("\\", "/")
        sample = con.execute(f"""
            SELECT
                COUNT(*) AS n_events,
                MIN(event_ts) AS min_ts,
                MAX(event_ts) AS max_ts,
                COUNT(DISTINCT user_id) AS n_users,
                COUNT(DISTINCT item_id) AS n_items
            FROM read_parquet('{sample_file}')
        """).fetchone()
        logger.info("    [Sample from 1/%d files]", len(event_files))
        logger.info("    events:        %s", f"{sample[0]:,}")
        logger.info("    time range:    %s -> %s", sample[1], sample[2])
        logger.info("    unique users:  %s", f"{sample[3]:,}")
        logger.info("    unique items:  %s", f"{sample[4]:,}")

    logger.info("[STEP 0] OK Data check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())