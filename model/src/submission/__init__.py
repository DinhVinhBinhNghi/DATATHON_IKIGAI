"""Submission building + validation.

Workflow:
1. build_submission_csv(final_topk_path, out_csv):
   - Filter items có trong dim_listing (BTC drop invalid items)
   - Fallback fill cho users < 10 items: dùng global pop
   - Output CSV với format: ID, user_id, rank, item_id
2. validate_submission(csv_path):
   - Check header
   - Check row count = n_test_users × 10
   - Check rank in [1, 10] per user
   - Check (user_id, rank) unique
   - Check item_id all in dim_listing
"""
from src.submission.builder import build_submission_csv
from src.submission.validator import validate_submission

__all__ = ["build_submission_csv", "validate_submission"]
