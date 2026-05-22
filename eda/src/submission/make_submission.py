from __future__ import annotations

from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_submission_csv(con, submission_dir: str | Path, filename: str = "submission.csv") -> Path:
    submission_dir = Path(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    out = submission_dir / filename
    logger.info("Writing submission -> %s", out)
    query = f"""
    COPY (
        SELECT
            ROW_NUMBER() OVER (ORDER BY user_id, rank) AS ID,
            user_id,
            rank,
            item_id
        FROM final_recommendations
        ORDER BY user_id, rank
    ) TO '{out.as_posix()}' (HEADER, DELIMITER ',')
    """
    con.execute(query)
    return out
