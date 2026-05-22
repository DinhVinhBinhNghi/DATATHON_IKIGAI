from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_recommendation_health(con) -> pd.DataFrame:
    return con.execute("""
        SELECT
            COUNT(*) AS n_recommendation_rows,
            COUNT(DISTINCT r.user_id) AS n_users,
            COUNT(DISTINCT r.item_id) AS n_items_exposed,
            COUNT(DISTINCT d.seller_id) AS n_sellers_exposed,
            ROUND(100.0 * AVG(CASE WHEN d.seller_type = 'private' THEN 1 ELSE 0 END), 2) AS pct_private_exposure,
            ROUND(100.0 * AVG(CASE WHEN DATE_DIFF('day', d.posted_date, DATE '2026-04-09') BETWEEN 0 AND 30 THEN 1 ELSE 0 END), 2) AS pct_fresh_30d_exposure
        FROM final_recommendations r
        JOIN dim_clean d USING (item_id)
    """).df()


def save_recommendation_health(con, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = summarize_recommendation_health(con)
    df.to_csv(out, index=False, encoding="utf-8")
    return out
