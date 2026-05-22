"""Co-visit source: item-item graph từ user_item_daily (OOM-safe).

[OOM PATCH v3.1]
- Bỏ COUNT(DISTINCT user_id) → dùng COUNT(*) (vì user_items đã GROUP BY (user, item)
  trong subquery, mỗi cặp (item_a, item_b) trong pairs có user_id unique theo
  cách construct → COUNT(*) tương đương COUNT(DISTINCT) nhưng nhanh ~10× và RAM thấp hơn).
- Thêm pre-filter `w_item > weight_threshold` để giảm pairs explosion (self-join O(N²)
  trên users có quá nhiều items có thể explode).
- Thêm cap `n_seeds_per_user` để chặn power users tạo quá nhiều pairs.

Trước khi sửa: 1 power user xem 1000 items → tạo 1000×999/2 ≈ 500K pairs trong 1 bucket.
Sau khi sửa: cap 100 seeds/user → tối đa 100×99/2 ≈ 5K pairs/user. Giảm 100×.
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)

COVISIT_BUCKETS = 16

# [PATCH] Pre-filter để chặn pairs explosion:
# - MAX_SEEDS_PER_USER: cap số items mỗi user contribute vào pairs.
#   Power users xem 1000+ items → cap 100 top items theo weighted_score.
# - MIN_USER_ITEM_WEIGHT: chỉ giữ (user, item) có signal đủ mạnh.
MAX_SEEDS_PER_USER = 100
MIN_USER_ITEM_WEIGHT = 1.0  # weighted_score >= 1.0 (≥ 1 other_interaction hoặc ≥ 0.33 hard contact)


def build_covisit_matrix(cutoff_date: str, out_path: Path) -> None:
    """Build item-item co-visit edges weighted, OOM-safe via chunking.

    [PATCH] Bỏ COUNT(DISTINCT user_id) trong bucket-level query.
    Vì user_items đã GROUP BY user_id, item_id → trong pairs join, mỗi cặp
    (user_id, item_a, item_b) là unique theo construction. Nên COUNT(*) trong
    GROUP BY (item_a, item_b) đếm đúng số distinct users (sau khi UNION ALL
    để symmetric, chia 2 — nhưng đơn giản hơn: chỉ symmetrize ở merge step).
    """
    cfg = get_config()
    user_item_path = str(cfg.paths.agg_dir / "user_item_daily.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    min_pair = cfg.covisit.min_pair_count
    top_k = cfg.covisit.top_k_neighbors
    lookback = cfg.covisit.lookback_days

    parts_dir = out_path.parent / "_covisit_parts"
    parts_dir.mkdir(exist_ok=True)
    part_files = []

    logger.info("  [covisit] Building chunked (N=%d buckets, "
                "max_seeds_per_user=%d, min_weight=%.1f)...",
                COVISIT_BUCKETS, MAX_SEEDS_PER_USER, MIN_USER_ITEM_WEIGHT)
    con = make_connection()

    for bucket in range(COVISIT_BUCKETS):
        part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")

        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("  [covisit] bucket %d/%d: cache hit, SKIP",
                        bucket + 1, COVISIT_BUCKETS)
            part_files.append(part_str)
            continue

        # [PATCH] Logic mới:
        # 1. user_items: GROUP BY (user_id, item_id) + filter weight + cap top-N per user
        #    → pre-filter giảm pairs explosion.
        # 2. pairs: self-join với item_a < item_b (chỉ giữ 1 chiều, symmetrize ở merge).
        # 3. agg: COUNT(*) thay COUNT(DISTINCT user_id) (đã đúng do user_items unique theo user+item).
        # 4. Không UNION ALL symmetric trong bucket — symmetrize ở merge step để
        #    giảm 50% rows trung gian.
        sql = f"""
        COPY (
            WITH user_items_raw AS (
                SELECT user_id, item_id, SUM(weighted_score) AS w_item
                FROM read_parquet('{user_item_path}')
                WHERE date >= DATE '{cutoff_date}' - INTERVAL {lookback} DAY
                  AND date <  DATE '{cutoff_date}'
                  AND weighted_score > 0
                  AND HASH(user_id) % {COVISIT_BUCKETS} = {bucket}
                GROUP BY user_id, item_id
                HAVING SUM(weighted_score) >= {MIN_USER_ITEM_WEIGHT}
            ),
            user_items AS (
                -- Cap top-N seeds per user để chặn power-user explosion
                SELECT user_id, item_id, w_item
                FROM (
                    SELECT user_id, item_id, w_item,
                           ROW_NUMBER() OVER (
                               PARTITION BY user_id ORDER BY w_item DESC
                           ) AS seed_rn
                    FROM user_items_raw
                ) WHERE seed_rn <= {MAX_SEEDS_PER_USER}
            ),
            pairs AS (
                -- Self-join: chỉ giữ item_a < item_b (1 chiều, không symmetric)
                -- Mỗi row trong pairs có user_id unique theo (user, item_a, item_b).
                SELECT u1.item_id AS item_a, u2.item_id AS item_b,
                       u1.user_id,
                       u1.w_item * u2.w_item AS pair_weight
                FROM user_items u1
                INNER JOIN user_items u2
                    ON u1.user_id = u2.user_id
                    AND u1.item_id < u2.item_id
            )
            -- [PATCH] COUNT(*) thay COUNT(DISTINCT user_id):
            -- Trong pairs, mỗi (item_a, item_b, user_id) là unique theo construction
            -- (user_items đã dedupe theo (user, item) ở CTE trên).
            -- → COUNT(*) per (item_a, item_b) chính là số distinct users co-view.
            SELECT
                item_a, item_b,
                SUM(pair_weight) AS edge_weight,
                COUNT(*) AS n_co_users
            FROM pairs
            GROUP BY item_a, item_b
            HAVING COUNT(*) >= {min_pair}
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """

        try:
            with timed(f"[covisit] bucket {bucket + 1}/{COVISIT_BUCKETS}", logger):
                con.execute(sql)
        except Exception as e:
            logger.error("  [covisit] bucket %d FAILED: %s",
                         bucket + 1, str(e)[:200])
            part_path.unlink(missing_ok=True)
            raise

        n_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{part_str}')"
        ).fetchone()[0]
        logger.info("    bucket %d edges: %s", bucket + 1, f"{n_rows:,}")
        part_files.append(part_str)

    # [PATCH] Merge: symmetrize (item_a <-> item_b) + top-K per item_a.
    # Symmetrize ở merge thay vì bucket → giảm 50% rows trung gian trong bucket files.
    parts_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
    merge_sql = f"""
    COPY (
        WITH all_edges AS (
            SELECT item_a, item_b, edge_weight, n_co_users
            FROM read_parquet('{parts_glob}')
            UNION ALL
            -- Symmetrize: edge (a→b) cũng có chiều ngược (b→a)
            SELECT item_b AS item_a, item_a AS item_b, edge_weight, n_co_users
            FROM read_parquet('{parts_glob}')
        ),
        merged AS (
            SELECT item_a, item_b,
                   SUM(edge_weight) AS edge_weight,
                   SUM(n_co_users)  AS n_co_users
            FROM all_edges
            GROUP BY item_a, item_b
            HAVING SUM(n_co_users) >= {min_pair}
        ),
        ranked AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY item_a
                    ORDER BY edge_weight DESC, n_co_users DESC
                ) AS rn
            FROM merged
        )
        SELECT item_a, item_b, edge_weight, n_co_users
        FROM ranked WHERE rn <= {top_k}
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("[covisit] merge buckets + symmetrize + top-K", logger):
        con.execute(merge_sql)

    for f in parts_dir.glob("bucket_*.parquet"):
        f.unlink()
    parts_dir.rmdir()

    n_edges = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_items_a = con.execute(
        f"SELECT COUNT(DISTINCT item_a) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  covisit matrix: %s edges, %s source items",
                f"{n_edges:,}", f"{n_items_a:,}")


def expand_covisit_to_candidates(cutoff_date: str, covisit_path: Path,
                                  top_n_per_user: int, out_path: Path) -> None:
    """Expand covisit matrix → user-level candidates. NO exclude (dedup ở merge.py).

    [PATCH] Cap seeds per user (giữ giống logic build_covisit) để consistent
    và đỡ explosion trong bucket. Tăng từ 20 → MAX_SEEDS_PER_USER (100) để
    candidate coverage tốt hơn — phần lớn rows được filter ở top_n_per_user cuối.
    """
    cfg = get_config()
    user_item_path = str(cfg.paths.agg_dir / "user_item_daily.parquet").replace("\\", "/")
    covisit_str = str(covisit_path).replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    n_buckets = 16
    parts_dir = out_path.parent / "_covisit_expand_parts"
    parts_dir.mkdir(exist_ok=True)

    # [PATCH] Cap seeds per user nhất quán với build_covisit
    seeds_per_user = MAX_SEEDS_PER_USER  # 100 — giữ giống build_covisit

    con = make_connection()
    logger.info("  [expand_covisit] Building chunked (N=%d, seeds_per_user=%d)...",
                n_buckets, seeds_per_user)

    for bucket in range(n_buckets):
        part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
        part_str = str(part_path).replace("\\", "/")

        if part_path.exists() and part_path.stat().st_size > 100:
            logger.info("    bucket %d/%d: cache hit, SKIP", bucket + 1, n_buckets)
            continue

        sql = f"""
        COPY (
            WITH user_seeds AS (
                SELECT user_id, item_id AS seed_item,
                       SUM(weighted_score) AS seed_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id
                           ORDER BY SUM(weighted_score) DESC
                       ) AS seed_rn
                FROM read_parquet('{user_item_path}')
                WHERE date < DATE '{cutoff_date}'
                  AND weighted_score > 0
                  AND HASH(user_id) % {n_buckets} = {bucket}
                GROUP BY user_id, item_id
                QUALIFY seed_rn <= {seeds_per_user}
            ),
            expanded AS (
                SELECT
                    s.user_id,
                    c.item_b AS item_id,
                    SUM(s.seed_score * c.edge_weight) AS source_score
                FROM user_seeds s
                INNER JOIN read_parquet('{covisit_str}') c
                    ON s.seed_item = c.item_a
                GROUP BY s.user_id, c.item_b
            ),
            ranked AS (
                SELECT user_id, item_id, source_score,
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY source_score DESC
                       ) AS rn
                FROM expanded
            )
            SELECT user_id, item_id, 'covisit' AS source, source_score
            FROM ranked WHERE rn <= {top_n_per_user}
        ) TO '{part_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
        """

        try:
            with timed(f"[expand_covisit] bucket {bucket + 1}/{n_buckets}", logger):
                con.execute(sql)
        except Exception as e:
            logger.error("  bucket %d FAILED: %s", bucket + 1, str(e)[:200])
            part_path.unlink(missing_ok=True)
            raise

    parts_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
    merge_sql = f"""
    COPY (
        SELECT * FROM read_parquet('{parts_glob}')
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """
    with timed("[expand_covisit] merge buckets", logger):
        con.execute(merge_sql)

    for f in parts_dir.glob("bucket_*.parquet"):
        f.unlink()
    parts_dir.rmdir()

    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    logger.info("  covisit candidates: %s rows", f"{n_rows:,}")