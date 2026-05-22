"""Preaggregate pipeline orchestrator.

OOM mitigation: user_item_daily là file lớn nhất, chunked 16 buckets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.common import (
    file_exists_nonempty,
    get_config,
    get_logger,
    make_connection,
    timed,
)
from src.preagg import aggregator as agg

logger = get_logger(__name__)

MARKER_FILENAME = "_marker_v3.json"
USER_ITEM_BUCKETS = 16


def _make_marker(cfg) -> dict:
    payload = {
        "version": "v3.0",
        "weights": {
            "hard_contact": cfg.weights.hard_contact,
            "other_interaction": cfg.weights.other_interaction,
            "pageview": cfg.weights.pageview,
        },
        "windows": {
            "train_start": cfg.windows.train_start,
            "train_end": cfg.windows.train_end,
        },
    }
    payload["hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def _marker_matches(marker_path: Path, expected: dict) -> bool:
    if not marker_path.exists():
        return False
    try:
        with marker_path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
        return actual.get("hash") == expected["hash"]
    except (json.JSONDecodeError, KeyError):
        return False


def _events_glob(cfg) -> str:
    return str(cfg.paths.fact_events_dir / "*.parquet").replace("\\", "/")


def _build_user_item_chunked(con, events_glob: str, out_path: Path,
                              train_start: str, train_end: str) -> None:
    """Chunked fallback cho user_item_daily."""
    cfg = get_config()
    parts_dir = out_path.parent / "_user_item_parts"
    parts_dir.mkdir(exist_ok=True)
    part_files = []

    for bucket in range(USER_ITEM_BUCKETS):
        part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
        if file_exists_nonempty(part_path):
            logger.info("  bucket %d/%d: cache hit, SKIP",
                        bucket + 1, USER_ITEM_BUCKETS)
            part_files.append(str(part_path).replace("\\", "/"))
            continue

        part_path_str = str(part_path).replace("\\", "/")
        sql = agg.sql_user_item_daily_chunked(
            events_glob, part_path_str, train_start, train_end,
            bucket, USER_ITEM_BUCKETS
        )
        with timed(f"bucket {bucket + 1}/{USER_ITEM_BUCKETS}", logger):
            con.execute(sql)
        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{part_path_str}')"
        ).fetchone()[0]
        logger.info("    bucket %d rows: %s", bucket + 1, f"{n_rows:,}")
        part_files.append(part_path_str)

    parts_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")
    merge_sql = f"""
    COPY (
        SELECT * FROM read_parquet('{parts_glob}')
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("merge user_item_daily buckets", logger):
        con.execute(merge_sql)

    for f in parts_dir.glob("bucket_*.parquet"):
        f.unlink()
    parts_dir.rmdir()


def run_preaggregate() -> None:
    cfg = get_config()
    agg_dir = cfg.paths.agg_dir
    marker_path = agg_dir / MARKER_FILENAME
    expected_marker = _make_marker(cfg)

    outputs = {
        "user_daily": agg_dir / "user_daily.parquet",
        "item_daily": agg_dir / "item_daily.parquet",
        "user_item_daily": agg_dir / "user_item_daily.parquet",
        "user_category": agg_dir / "user_category_weighted.parquet",
        "user_city": agg_dir / "user_city_weighted.parquet",
        "event_type_daily": agg_dir / "event_type_daily.parquet",
    }

    marker_ok = _marker_matches(marker_path, expected_marker)
    if not marker_ok:
        logger.info("[STEP 1] Marker không khớp config hiện tại → rebuild hết.")
        for p in outputs.values():
            p.unlink(missing_ok=True)
        parts_dir = agg_dir / "_user_item_parts"
        if parts_dir.exists():
            for f in parts_dir.glob("*.parquet"):
                f.unlink()
            parts_dir.rmdir()
        marker_path.unlink(missing_ok=True)
    else:
        existing = [k for k, p in outputs.items() if file_exists_nonempty(p)]
        if len(existing) == len(outputs):
            logger.info("[STEP 1] Cache đầy đủ + marker khớp → SKIP toàn bộ.")
            return
        if existing:
            logger.info("[STEP 1] Cache có sẵn: %s. Sẽ rebuild các phần còn lại.",
                        existing)

    events_glob = _events_glob(cfg)
    train_start = cfg.windows.train_start
    train_end = cfg.windows.train_end

    con = make_connection()

    order = ["event_type_daily", "user_daily", "item_daily",
             "user_category", "user_city", "user_item_daily"]

    builders = {
        "event_type_daily": agg.sql_event_type_daily,
        "user_daily": agg.sql_user_daily,
        "item_daily": agg.sql_item_daily,
        "user_category": agg.sql_user_category,
        "user_city": agg.sql_user_city,
        "user_item_daily": agg.sql_user_item_daily,
    }

    for name in order:
        out_path = outputs[name]
        if file_exists_nonempty(out_path):
            logger.info("[STEP 1] %s: cache có sẵn, SKIP.", name)
            continue

        out_path_posix = str(out_path).replace("\\", "/")
        logger.info("[STEP 1] Building %s...", name)

        if name == "user_item_daily":
            try:
                _build_user_item_chunked(con, events_glob, out_path,
                                          train_start, train_end)
            except Exception as e:
                logger.error("  FAILED user_item_daily chunked: %s", str(e)[:200])
                out_path.unlink(missing_ok=True)
                raise
        else:
            sql = builders[name](events_glob, out_path_posix, train_start, train_end)
            try:
                with timed(f"build {name}", logger):
                    con.execute(sql)
            except Exception as e:
                logger.error("  FAILED %s: %s", name, str(e)[:200])
                out_path.unlink(missing_ok=True)
                raise

        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out_path_posix}')"
        ).fetchone()[0]
        logger.info("  %s: %s rows", name, f"{n_rows:,}")

    with marker_path.open("w", encoding="utf-8") as f:
        json.dump(expected_marker, f, indent=2)
    logger.info("[STEP 1] DONE. Marker saved: %s", marker_path)

    _log_sanity_stats(con, outputs["event_type_daily"])


def _log_sanity_stats(con, event_type_path: Path) -> None:
    p = str(event_type_path).replace("\\", "/")
    cfg = get_config()

    rows = con.execute(f"""
        SELECT
            event_type,
            SUM(n_events) AS total_events,
            ROUND(100.0 * SUM(n_events) / SUM(SUM(n_events)) OVER (), 2) AS pct
        FROM read_parquet('{p}')
        GROUP BY event_type
        ORDER BY total_events DESC
    """).fetchall()

    logger.info("[STEP 1] Event type distribution (login-only):")
    logger.info("  weights: hard_contact=%.1f, other_interaction=%.1f, pageview=0.0",
                cfg.weights.hard_contact, cfg.weights.other_interaction)
    logger.info("  %-20s %15s %8s", "event_type", "n_events", "pct")
    for row in rows:
        logger.info("  %-20s %15s %7.2f%%",
                    row[0], f"{row[1]:,}", row[2])