"""Rerank orchestrator: apply rules -> pick top-K final per user.

[OOM PATCH v3.2.2-bucket] Mode 'rerank' chia thành 2 phase chunked để tránh
OOM với memory_limit=5GB:

Phase 1: Compute seller_id + i_age_days lookup tables (small, 1 lan).
Phase 2: Per user bucket (8 buckets) - apply rules + top-K, ghi parquet.
Phase 3: Concat buckets -> final_topk.

Peak RAM giam tu ~5GB single-shot xuong ~1-2GB per bucket.

Mode 'raw' giu nguyen (don gian, da chay xong OK).

Output schema giu NGUYEN: user_id, item_id, rank, pred_score, source
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)

RERANK_BUCKETS = 8


def _build_lookup_tables(con, dim_glob: str, item_feat_str: str,
                         parts_dir: Path) -> tuple[str, str]:
    """Phase 1: Build small lookup tables for seller_id + i_age_days.

    Returns:
        (seller_path_str, age_path_str)
    """
    seller_path = parts_dir / "_item_seller.parquet"
    age_path = parts_dir / "_item_age.parquet"
    seller_str = str(seller_path).replace("\\", "/")
    age_str = str(age_path).replace("\\", "/")

    if not (seller_path.exists() and seller_path.stat().st_size > 100):
        with timed("[rerank phase 1a] build item_seller lookup", logger):
            con.execute(f"""
                COPY (
                    SELECT item_id, ANY_VALUE(seller_id) AS seller_id
                    FROM read_parquet('{dim_glob}')
                    WHERE seller_id IS NOT NULL
                    GROUP BY item_id
                ) TO '{seller_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
            """)

    if not (age_path.exists() and age_path.stat().st_size > 100):
        with timed("[rerank phase 1b] build item_age lookup", logger):
            con.execute(f"""
                COPY (
                    SELECT item_id, i_age_days
                    FROM read_parquet('{item_feat_str}')
                ) TO '{age_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
            """)

    return seller_str, age_str


def _build_rerank_sql_raw(scored_str: str, out_str: str, top_k: int) -> str:
    """Mode raw: chi top-K, khong apply rules."""
    return f"""
    COPY (
        WITH ranked AS (
            SELECT user_id, item_id, source, pred_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id ORDER BY pred_score DESC
                   ) AS rank
            FROM read_parquet('{scored_str}')
        )
        SELECT user_id, item_id, rank,
               CAST(pred_score AS DOUBLE) AS pred_score,
               source
        FROM ranked WHERE rank <= {top_k}
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """


def _build_rerank_sql_bucket(scored_str: str, seller_str: str, age_str: str,
                              out_str: str, bucket: int, n_buckets: int,
                              top_k: int, fresh_days: int, fresh_boost: float,
                              max_per_seller: int, seller_penalty: float) -> str:
    """Mode rerank bucketed: 1 bucket users at a time."""
    return f"""
    COPY (
        WITH scored_bucket AS (
            SELECT * FROM read_parquet('{scored_str}')
            WHERE HASH(user_id) % {n_buckets} = {bucket}
        ),
        enriched AS (
            SELECT
                s.user_id, s.item_id, s.source, s.pred_score,
                COALESCE(se.seller_id, '') AS seller_id,
                COALESCE(ia.i_age_days, 999) AS i_age_days
            FROM scored_bucket s
            LEFT JOIN read_parquet('{seller_str}') se ON s.item_id = se.item_id
            LEFT JOIN read_parquet('{age_str}')    ia ON s.item_id = ia.item_id
        ),
        boosted AS (
            SELECT
                user_id, item_id, source, seller_id,
                CASE WHEN i_age_days <= {fresh_days}
                     THEN pred_score * {fresh_boost}
                     ELSE pred_score END AS boosted_score
            FROM enriched
        ),
        seller_ranked AS (
            SELECT
                user_id, item_id, source, boosted_score,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, seller_id
                    ORDER BY boosted_score DESC
                ) AS seller_rank
            FROM boosted
        ),
        final_scored AS (
            SELECT
                user_id, item_id, source,
                CASE WHEN seller_rank <= {max_per_seller}
                     THEN boosted_score
                     ELSE boosted_score * {seller_penalty} END AS pred_score
            FROM seller_ranked
        ),
        ranked AS (
            SELECT user_id, item_id, source, pred_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id ORDER BY pred_score DESC
                   ) AS rank
            FROM final_scored
        )
        SELECT user_id, item_id, rank,
               CAST(pred_score AS DOUBLE) AS pred_score,
               source
        FROM ranked WHERE rank <= {top_k}
    ) TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """


def run_rerank(scored_path: Path, item_feat_path: Path,
                mode: str, out_path: Path) -> None:
    """Apply rerank rules + pick top-K final."""
    if mode not in ("raw", "rerank"):
        raise ValueError(f"Unknown mode: {mode}")

    cfg = get_config()
    top_k = cfg.submission.top_k_final
    max_per_seller = cfg.rerank.max_items_per_seller
    fresh_days = cfg.rerank.fresh_boost_days
    fresh_boost = cfg.rerank.fresh_boost_factor
    seller_penalty = 0.5

    scored_str = str(scored_path).replace("\\", "/")
    if_str = str(item_feat_path).replace("\\", "/")
    dim_glob = str(cfg.paths.dim_listing_dir / "*.parquet").replace("\\", "/")
    out_str = str(out_path).replace("\\", "/")

    con = make_connection()

    logger.info("[STEP 6 mode=%s] Rerank via DuckDB SQL (no pandas)...", mode)

    if mode == "raw":
        logger.info("  no rules applied (raw pred_score)")
        sql = _build_rerank_sql_raw(scored_str, out_str, top_k)
        with timed(f"rerank mode={mode} (DuckDB streaming)", logger):
            con.execute(sql)
    else:
        # Mode rerank: bucketed
        logger.info("  rules: freshness_boost (<=%dd x%.2f), "
                    "seller_cap (max=%d, penalty=x%.2f)",
                    fresh_days, fresh_boost, max_per_seller, seller_penalty)

        parts_dir = out_path.parent / "_rerank_parts"
        parts_dir.mkdir(exist_ok=True)

        # Phase 1: lookup tables
        seller_lookup_str, age_lookup_str = _build_lookup_tables(
            con, dim_glob, if_str, parts_dir
        )

        # Phase 2: bucketed rerank
        logger.info("  [rerank phase 2] Bucketed rerank (%d buckets)...",
                    RERANK_BUCKETS)
        bucket_files = []
        for bucket in range(RERANK_BUCKETS):
            part_path = parts_dir / f"bucket_{bucket:02d}.parquet"
            part_str = str(part_path).replace("\\", "/")

            if part_path.exists() and part_path.stat().st_size > 100:
                logger.info("    bucket %d/%d: cache hit, SKIP",
                            bucket + 1, RERANK_BUCKETS)
                bucket_files.append(part_str)
                continue

            sql = _build_rerank_sql_bucket(
                scored_str, seller_lookup_str, age_lookup_str,
                part_str, bucket, RERANK_BUCKETS, top_k,
                fresh_days, fresh_boost, max_per_seller, seller_penalty
            )
            with timed(f"    [rerank] bucket {bucket + 1}/{RERANK_BUCKETS}", logger):
                con.execute(sql)
            n_rows = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{part_str}')"
            ).fetchone()[0]
            logger.info("      bucket %d rows: %s", bucket + 1, f"{n_rows:,}")
            bucket_files.append(part_str)

        # Phase 3: concat
        bucket_glob = str(parts_dir / "bucket_*.parquet").replace("\\", "/")
        with timed("  [rerank phase 3] concat buckets", logger):
            con.execute(f"""
                COPY (SELECT * FROM read_parquet('{bucket_glob}'))
                TO '{out_str}' (FORMAT 'parquet', COMPRESSION 'snappy')
            """)

        # Cleanup
        for f in parts_dir.glob("bucket_*.parquet"):
            f.unlink()
        # Keep lookup tables for resume (small files)
        try:
            (parts_dir / "_item_seller.parquet").unlink(missing_ok=True)
            (parts_dir / "_item_age.parquet").unlink(missing_ok=True)
            parts_dir.rmdir()
        except OSError:
            pass

    # Stats
    n_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    n_users = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{out_str}')"
    ).fetchone()[0]
    avg_per_user = n_rows / n_users if n_users > 0 else 0
    logger.info("  Saved final_topk: %s (%s rows, %s users, avg %.1f items/user)",
                out_path, f"{n_rows:,}", f"{n_users:,}", avg_per_user)