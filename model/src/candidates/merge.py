"""Merge 5 sources thành candidate pool.

[OOM PATCH v3.2-merge] Refactor để tránh 3-layer window function trên full UNION ALL.

Vấn đề cũ: 1 SQL với 3 lớp window:
  - MAX(source_score) OVER (PARTITION BY source)            ← cần materialize full
  - ROW_NUMBER OVER (PARTITION BY user_id, item_id ORDER BY)
  - ROW_NUMBER OVER (PARTITION BY user_id ORDER BY)
→ DuckDB phải giữ toàn bộ ~300-500M rows trong RAM → OOM trên máy 16GB.

Giải pháp mới: 3 phases ghi disk + chunked theo HASH(user_id) % 8:

Phase 1: Compute max_per_source (tiny table, ~5 rows) — KHÔNG dùng window
Phase 2: Per bucket — dedup theo (user, item) + chọn source tốt nhất
Phase 3: Per bucket — rank top-K per user → final concat

Mỗi bucket xử lý 1/8 user space → peak RAM giảm ~8 lần.

Logic giữ NGUYÊN:
- Normalize source_score / max_per_source → candidate_score 0-1
- Per (user, item) duplicate: keep source với candidate_score cao nhất
- Top-K per user theo candidate_score

Output schema giữ NGUYÊN: user_id, item_id, source, source_score, candidate_score
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)

# Bucket user space để chunked. 8 buckets = peak RAM ~1/8 so với single-shot.
MERGE_BUCKETS = 8


def merge_candidates(source_files: dict[str, Path], out_path: Path) -> None:
    """Merge 5 source files thành 1 candidates pool, chunked OOM-safe.

    Args:
        source_files: dict {source_name: path}.
        out_path: output parquet.
    """
    cfg = get_config()
    top_k = cfg.candidates.top_k_per_user
    out_str = str(out_path).replace("\\", "/")

    # Build UNION ALL fragments (1 lần, dùng lại)
    union_parts = []
    for src_name, src_path in source_files.items():
        path_str = str(src_path).replace("\\", "/")
        union_parts.append(
            f"SELECT user_id, item_id, source, source_score "
            f"FROM read_parquet('{path_str}')"
        )
    union_sql = "\nUNION ALL\n".join(union_parts)

    con = make_connection()

    parts_dir = out_path.parent / "_merge_parts"
    parts_dir.mkdir(exist_ok=True)

    # =========================================================================
    # PHASE 1: max_per_source — tiny table (5 rows), GROUP BY thay vì window
    # =========================================================================
    max_per_source_path = parts_dir / "_max_per_source.parquet"
    max_per_source_str = str(max_per_source_path).replace("\\", "/")

    if not (max_per_source_path.exists() and max_per_source_path.stat().st_size > 100):
        logger.info("  [merge phase 1] Compute max_per_source (5 rows)...")
        # Compute MAX(source_score) per source — GROUP BY là streaming, không cần window
        # Để tránh full union materialize, query từng source riêng rồi union kết quả
        max_per_src_unions = []
        for src_name, src_path in source_files.items():
            path_str = str(src_path).replace("\\", "/")
            max_per_src_unions.append(
                f"SELECT '{src_name}' AS source, "
                f"MAX(source_score) AS max_score, "
                f"COUNT(*) AS n_rows "
                f"FROM read_parquet('{path_str}')"
            )
        max_sql = f"""
        COPY (
            {' UNION ALL '.join(max_per_src_unions)}
        ) TO '{max_per_source_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """
        with timed("  [merge phase 1] max_per_source", logger):
            con.execute(max_sql)

        # Log max values + row count để debug
        stats = con.execute(
            f"SELECT source, max_score, n_rows "
            f"FROM read_parquet('{max_per_source_str}') ORDER BY n_rows DESC"
        ).fetchall()
        logger.info("  [merge phase 1] source stats:")
        for src, mx, n in stats:
            logger.info("    %-15s max_score=%12.4f  n_rows=%15s",
                        src, mx, f"{n:,}")

    # =========================================================================
    # PHASE 2: Dedup per (user, item) — chunked theo HASH(user_id) % 8
    # Mỗi bucket: load 1/8 union → normalize → dedup → ghi disk.
    # =========================================================================
    deduped_parts = []
    logger.info("  [merge phase 2] Dedup per (user, item) — %d buckets...",
                MERGE_BUCKETS)

    for bucket in range(MERGE_BUCKETS):
        part_path = parts_dir / f"_dedup_bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")

        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("    bucket %d/%d: cache hit, SKIP",
                        bucket + 1, MERGE_BUCKETS)
            deduped_parts.append(part_str)
            continue

        # Build union với HASH filter — DuckDB push down filter vào từng read_parquet
        bucket_union_parts = []
        for src_name, src_path in source_files.items():
            path_str = str(src_path).replace("\\", "/")
            bucket_union_parts.append(
                f"SELECT user_id, item_id, source, source_score "
                f"FROM read_parquet('{path_str}') "
                f"WHERE HASH(user_id) % {MERGE_BUCKETS} = {bucket}"
            )
        bucket_union_sql = "\nUNION ALL\n".join(bucket_union_parts)

        # SQL: normalize bằng JOIN với max_per_source (tiny table)
        # → tránh window function, dùng equi-join nhanh + ít RAM.
        sql = f"""
        COPY (
            WITH all_sources AS (
                {bucket_union_sql}
            ),
            normalized AS (
                SELECT
                    a.user_id, a.item_id, a.source, a.source_score,
                    CASE
                        WHEN m.max_score > 0 THEN a.source_score / m.max_score
                        ELSE 0.0
                    END AS candidate_score
                FROM all_sources a
                INNER JOIN read_parquet('{max_per_source_str}') m
                    ON a.source = m.source
            ),
            deduped AS (
                SELECT user_id, item_id, source, source_score, candidate_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id, item_id
                           ORDER BY candidate_score DESC, source_score DESC
                       ) AS dup_rn
                FROM normalized
            )
            SELECT user_id, item_id, source, source_score, candidate_score
            FROM deduped WHERE dup_rn = 1
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """

        try:
            with timed(f"    [merge phase 2] bucket {bucket + 1}/{MERGE_BUCKETS}", logger):
                con.execute(sql)
        except Exception as e:
            logger.error("    bucket %d FAILED: %s", bucket + 1, str(e)[:200])
            part_path.unlink(missing_ok=True)
            raise

        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{part_str}')"
        ).fetchone()[0]
        logger.info("      bucket %d deduped rows: %s",
                    bucket + 1, f"{n_rows:,}")
        deduped_parts.append(part_str)

    # =========================================================================
    # PHASE 3: Top-K per user — chunked, mỗi bucket chỉ rank trong bucket đó.
    # Bucket theo HASH(user_id) → mọi rows của 1 user nằm cùng bucket → top-K
    # tính được local đúng kết quả global (không cần merge phase 4).
    # =========================================================================
    topk_parts = []
    logger.info("  [merge phase 3] Top-K per user (top_k=%d) — %d buckets...",
                top_k, MERGE_BUCKETS)

    for bucket in range(MERGE_BUCKETS):
        dedup_str = deduped_parts[bucket]
        part_path = parts_dir / f"_topk_bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")

        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("    bucket %d/%d: cache hit, SKIP",
                        bucket + 1, MERGE_BUCKETS)
            topk_parts.append(part_str)
            continue

        sql = f"""
        COPY (
            WITH ranked AS (
                SELECT user_id, item_id, source, source_score, candidate_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY candidate_score DESC
                       ) AS rn
                FROM read_parquet('{dedup_str}')
            )
            SELECT user_id, item_id, source, source_score, candidate_score
            FROM ranked WHERE rn <= {top_k}
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """

        try:
            with timed(f"    [merge phase 3] bucket {bucket + 1}/{MERGE_BUCKETS}", logger):
                con.execute(sql)
        except Exception as e:
            logger.error("    bucket %d FAILED: %s", bucket + 1, str(e)[:200])
            part_path.unlink(missing_ok=True)
            raise

        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{part_str}')"
        ).fetchone()[0]
        logger.info("      bucket %d top-K rows: %s",
                    bucket + 1, f"{n_rows:,}")
        topk_parts.append(part_str)

    # =========================================================================
    # PHASE 4: Concat 8 bucket files → final candidates parquet
    # =========================================================================
    logger.info("  [merge phase 4] Concat %d buckets → final...", MERGE_BUCKETS)
    topk_glob = str(parts_dir / "_topk_bucket_*.parquet").replace("\\", "/")
    concat_sql = f"""
    COPY (
        SELECT * FROM read_parquet('{topk_glob}')
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("  [merge phase 4] concat buckets", logger):
        con.execute(concat_sql)

    # Cleanup intermediate parts
    for f in parts_dir.glob("_dedup_bucket_*.parquet"):
        f.unlink()
    for f in parts_dir.glob("_topk_bucket_*.parquet"):
        f.unlink()
    max_per_source_path.unlink(missing_ok=True)
    try:
        parts_dir.rmdir()
    except OSError:
        pass  # parts_dir có file khác → để lại

    # Stats
    stats = con.execute(f"""
        SELECT
            source,
            COUNT(*) AS n_rows,
            COUNT(DISTINCT user_id) AS n_users
        FROM read_parquet('{out_str}')
        GROUP BY source ORDER BY n_rows DESC
    """).fetchall()

    total_rows = sum(r[1] for r in stats)
    total_users = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  Merged: %s rows, %s unique users",
                f"{total_rows:,}", f"{total_users:,}")
    logger.info("  Source breakdown (post-merge):")
    for src, n_rows, n_users in stats:
        logger.info("    %-15s %15s rows %15s users",
                    src, f"{n_rows:,}", f"{n_users:,}")