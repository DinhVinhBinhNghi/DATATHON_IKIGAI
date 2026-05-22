"""Candidate pool metrics: Recall@N ceiling, source coverage.

Khác với metrics.py (đánh giá submission final):
- candidate_metrics đánh giá CANDIDATE POOL (200 items/user trước khi ranker).
- Recall@200 = ceiling tốt nhất ranker có thể đạt.
- Source coverage: source nào đóng góp recall nhiều nhất.

Output:
- Recall@200 (ceiling), Recall@100, Recall@50, Recall@10
- Per-source: n_candidates, n_users_covered, recall_share
"""
from __future__ import annotations

from pathlib import Path

from src.common import get_config, get_logger, make_connection, timed

logger = get_logger(__name__)


def evaluate_candidates_pool(candidates_path: Path,
                               ground_truth_path: Path,
                               out_csv: Path | None = None) -> dict:
    """Compute Recall@N ceiling + source breakdown.

    Args:
        candidates_path: candidates_predict.parquet (output of merge.py).
        ground_truth_path: internal_gt.parquet.
        out_csv: nếu set, save report CSV.

    Returns:
        dict với metrics.
    """
    con = make_connection()
    cand_str = str(candidates_path).replace("\\", "/")
    gt_str = str(ground_truth_path).replace("\\", "/")

    # Compute Recall@N at different cutoffs
    sql_recall = f"""
    WITH cand_ranked AS (
        SELECT
            user_id, item_id, source,
            ROW_NUMBER() OVER (
                PARTITION BY user_id ORDER BY candidate_score DESC
            ) AS rn
        FROM read_parquet('{cand_str}')
    ),
    gt AS (
        SELECT user_id, item_id FROM read_parquet('{gt_str}')
    ),
    gt_with_topn AS (
        SELECT
            gt.user_id, gt.item_id,
            COALESCE(MIN(cand_ranked.rn), 99999) AS first_rank
        FROM gt
        LEFT JOIN cand_ranked
            ON gt.user_id = cand_ranked.user_id
           AND gt.item_id = cand_ranked.item_id
        GROUP BY gt.user_id, gt.item_id
    )
    SELECT
        AVG(CASE WHEN first_rank <= 10 THEN 1.0 ELSE 0.0 END)
            AS hit_rate_at_10,
        AVG(CASE WHEN first_rank <= 50 THEN 1.0 ELSE 0.0 END)
            AS hit_rate_at_50,
        AVG(CASE WHEN first_rank <= 100 THEN 1.0 ELSE 0.0 END)
            AS hit_rate_at_100,
        AVG(CASE WHEN first_rank <= 200 THEN 1.0 ELSE 0.0 END)
            AS hit_rate_at_200,
        COUNT(*) AS n_gt_pairs,
        COUNT(DISTINCT user_id) AS n_users_in_gt
    FROM gt_with_topn
    """
    with timed("compute recall@N ceiling", logger):
        row = con.execute(sql_recall).fetchone()

    hit_10, hit_50, hit_100, hit_200, n_pairs, n_users = row
    logger.info("[STEP 16] Candidate pool recall:")
    logger.info("  Recall@10 ceiling:  %.4f", hit_10)
    logger.info("  Recall@50 ceiling:  %.4f", hit_50)
    logger.info("  Recall@100 ceiling: %.4f", hit_100)
    logger.info("  Recall@200 ceiling: %.4f", hit_200)

    # Source breakdown
    sql_source = f"""
    WITH gt AS (
        SELECT user_id, item_id FROM read_parquet('{gt_str}')
    ),
    cand AS (
        SELECT user_id, item_id, source FROM read_parquet('{cand_str}')
    ),
    cand_in_gt AS (
        SELECT c.source, COUNT(*) AS n_hits
        FROM cand c
        INNER JOIN gt ON c.user_id = gt.user_id AND c.item_id = gt.item_id
        GROUP BY c.source
    ),
    cand_stats AS (
        SELECT
            source,
            COUNT(*) AS n_candidates,
            COUNT(DISTINCT user_id) AS n_users_covered
        FROM cand
        GROUP BY source
    )
    SELECT
        s.source,
        s.n_candidates,
        s.n_users_covered,
        COALESCE(h.n_hits, 0) AS n_hits_in_gt,
        ROUND(100.0 * COALESCE(h.n_hits, 0) / {n_pairs}, 4) AS recall_share_pct
    FROM cand_stats s
    LEFT JOIN cand_in_gt h ON s.source = h.source
    ORDER BY n_hits_in_gt DESC
    """
    src_rows = con.execute(sql_source).fetchall()
    logger.info("  Source breakdown (recall_share = % of GT pairs covered):")
    logger.info("    %-18s %15s %15s %15s %12s",
                "source", "n_candidates", "n_users", "n_hits_in_gt", "recall%")
    for src, n_cands, n_users_c, n_hits, recall_pct in src_rows:
        logger.info("    %-18s %15s %15s %15s %11.4f%%",
                    src, f"{n_cands:,}", f"{n_users_c:,}",
                    f"{n_hits:,}", recall_pct)

    if out_csv is not None:
        import pandas as pd
        df = pd.DataFrame(src_rows, columns=[
            "source", "n_candidates", "n_users_covered",
            "n_hits_in_gt", "recall_share_pct"
        ])
        df.to_csv(out_csv, index=False)
        logger.info("  Saved report: %s", out_csv)

    return {
        "recall@10_ceiling": hit_10,
        "recall@50_ceiling": hit_50,
        "recall@100_ceiling": hit_100,
        "recall@200_ceiling": hit_200,
        "n_gt_pairs": n_pairs,
        "n_users_in_gt": n_users,
        "source_breakdown": [
            {"source": r[0], "n_candidates": r[1],
             "n_users_covered": r[2], "n_hits_in_gt": r[3],
             "recall_share_pct": r[4]}
            for r in src_rows
        ],
    }
