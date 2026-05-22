"""Build Kaggle submission CSV từ final_topk_{mode}.parquet.

Format yêu cầu (từ Đề thi):
    ID,user_id,rank,item_id
    1,abc...,1,xyz...
    2,abc...,2,uvw...
    ...

Constraints:
- Mỗi user có đúng 10 rows (rank 1..10).
- (user_id, rank) unique.
- (user_id, item_id) unique per user.
- item_id phải có trong train/dim_listing/ (else BTC drop).
- Encoding UTF-8 không BOM.
- Dung lượng ≤ 100 MB (Kaggle accept .csv hoặc .zip).

Fallback cho users < 10 items: dùng global top-10 weighted-pop items.

[OOM PATCH v3.1]
- Push QUALIFY top-10 + dim_listing join xuống DuckDB SQL.
- Stream test_users qua iterator thay vì load full DataFrame.

[v3.2.3] BỎ logic gzip — chỉ output .csv thuần. Nếu file > 100 MB, user
phải tự ZIP bằng tay (Compress-Archive trong PowerShell, hoặc 7-Zip).
Lý do: Kaggle accept .zip nhưng KHÔNG accept .gz, nên gzip auto là vô ích.
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)


def _build_global_fallback(con) -> list[str]:
    """Build top-10 global weighted-pop items as fallback cho users thiếu candidates.

    Window: 30 ngày cuối train (sát test window).

    Returns:
        list 10 item_ids, sorted by weighted_score DESC.
    """
    cfg = get_config()
    item_daily = str(cfg.paths.agg_dir / "item_daily.parquet").replace("\\", "/")
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    cutoff = cfg.windows.train_end
    window = cfg.popularity.window_days

    sql = f"""
    WITH valid_items AS (
        SELECT DISTINCT item_id FROM read_parquet('{dim_glob}')
    ),
    pop AS (
        SELECT i.item_id, SUM(i.weighted_score) AS pop_score
        FROM read_parquet('{item_daily}') i
        INNER JOIN valid_items v ON i.item_id = v.item_id
        WHERE i.date >= DATE '{cutoff}' - INTERVAL {window} DAY
          AND i.date <  DATE '{cutoff}'
          AND i.weighted_score > 0
        GROUP BY i.item_id
        ORDER BY pop_score DESC
        LIMIT 10
    )
    SELECT item_id FROM pop
    """
    rows = con.execute(sql).fetchall()
    return [r[0] for r in rows]


def build_submission_csv(final_topk_path: Path, out_csv_path: Path) -> None:
    """Convert final_topk_{mode}.parquet thành submission CSV.

    Args:
        final_topk_path: parquet với (user_id, item_id, rank, pred_score, source).
        out_csv_path: output CSV file.

    [OOM PATCH] Tối ưu peak RAM:
    1. Filter dim_listing + top-10 ngay trong DuckDB SQL (giảm rows ~20×).
    2. Materialize df_top10 chỉ với 3 cột cần thiết.
    3. Stream test_users qua DuckDB cursor batch.
    """
    cfg = get_config()
    test_users_path = cfg.paths.test_users_file
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    final_str = str(final_topk_path).replace("\\", "/")
    test_str = str(test_users_path).replace("\\", "/")

    con = make_connection()

    # [PATCH] Step 1: Filter dim_listing + QUALIFY top-10 ngay trong SQL.
    filtered_top10_sql = f"""
    WITH valid_items AS (
        SELECT DISTINCT item_id FROM read_parquet('{dim_glob}')
    ),
    valid_topk AS (
        SELECT t.user_id, t.item_id, t.pred_score
        FROM read_parquet('{final_str}') t
        INNER JOIN valid_items v ON t.item_id = v.item_id
    ),
    top10 AS (
        SELECT user_id, item_id, pred_score,
               ROW_NUMBER() OVER (
                   PARTITION BY user_id ORDER BY pred_score DESC
               ) AS rank_per_user
        FROM valid_topk
    )
    SELECT user_id, item_id, rank_per_user
    FROM top10 WHERE rank_per_user <= 10
    ORDER BY user_id, rank_per_user
    """
    with timed("filter top-10 + dim_listing join (DuckDB QUALIFY)", logger):
        df_top10 = con.execute(filtered_top10_sql).df()

    n_pairs = len(df_top10)
    n_users_have = df_top10["user_id"].nunique()
    logger.info("  After dim_listing filter + top-10: %s pairs, %s users",
                f"{n_pairs:,}", f"{n_users_have:,}")

    # [PATCH] Build user_to_items dict bằng pandas groupby native (vectorized).
    logger.info("  Building user_to_items mapping...")
    with timed("groupby user_id -> items list", logger):
        user_to_items_series = df_top10.groupby(
            "user_id", sort=False
        )["item_id"].apply(list)
        user_to_items: dict[str, list[str]] = user_to_items_series.to_dict()
        del user_to_items_series, df_top10

    # Step 2: Load test users (stream qua DuckDB, không qua pandas)
    n_test = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{test_str}')"
    ).fetchone()[0]
    logger.info("  Test users: %s", f"{n_test:,}")

    # Step 3: Build global fallback
    with timed("build global fallback", logger):
        fallback = _build_global_fallback(con)
    logger.info("  Fallback items: %d (first: %s...)", len(fallback),
                fallback[0][:16] if fallback else "")

    if not fallback:
        logger.warning("  Fallback rỗng! Users thiếu items sẽ submit < 10 rows.")

    # [PATCH] Step 4: Stream test users qua DuckDB cursor -> ghi CSV trực tiếp.
    logger.info("  Streaming submission rows to CSV (UTF-8 no BOM)...")

    rid = 1
    n_filled_from_fallback = 0
    n_users_full_fallback = 0
    n_users_written = 0

    test_cursor = con.execute(
        f"SELECT user_id FROM read_parquet('{test_str}')"
    )

    with timed("write submission CSV (streaming)", logger):
        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            f.write("ID,user_id,rank,item_id\n")

            BATCH_SIZE = 50_000
            while True:
                batch = test_cursor.fetchmany(BATCH_SIZE)
                if not batch:
                    break

                for (user_id,) in batch:
                    items = list(user_to_items.get(user_id, []))
                    existing = set(items)

                    # Fill từ fallback nếu thiếu
                    if len(items) < 10:
                        if len(items) == 0:
                            n_users_full_fallback += 1
                        for fb in fallback:
                            if fb not in existing:
                                items.append(fb)
                                existing.add(fb)
                                n_filled_from_fallback += 1
                            if len(items) >= 10:
                                break

                    # Trim to 10 (safety)
                    items = items[:10]

                    # Ghi rows ngay (không buffer)
                    for rank_i, item in enumerate(items, start=1):
                        f.write(f"{rid},{user_id},{rank_i},{item}\n")
                        rid += 1
                    n_users_written += 1

    n_total_rows = rid - 1
    logger.info("  Filled %s slots from fallback (%s users fully fallback)",
                f"{n_filled_from_fallback:,}", f"{n_users_full_fallback:,}")
    logger.info("  Written: %s rows for %s users",
                f"{n_total_rows:,}", f"{n_users_written:,}")

    if n_users_written != n_test:
        logger.warning(
            "  Mismatch: wrote %s users, expected %s test users",
            f"{n_users_written:,}", f"{n_test:,}"
        )

    size_mb = out_csv_path.stat().st_size / 1e6
    logger.info("  Saved CSV: %s (%.1f MB)", out_csv_path, size_mb)
    if size_mb > 100:
        logger.warning("  File > 100 MB. Compress to .zip manually for Kaggle:")
        logger.warning("    PowerShell: Compress-Archive -Path '%s' -DestinationPath '%s.zip'",
                       out_csv_path, out_csv_path.stem)