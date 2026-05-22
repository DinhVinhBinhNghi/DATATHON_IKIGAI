"""Candidate generation: build candidate pool per user từ 5 sources.

5 sources với quota (config `candidates.source_quotas`):
- reengagement   (default 50): items user đã interact (weighted_score > 0)
- covisit        (default 80): item-item từ user history qua co-visit graph
- category_pop   (default 30): top weighted-pop trong user's top category
- city_cat_pop   (default 20): top weighted-pop trong (user_city × category)
- global_pop     (default 20): top weighted-pop fallback (cold users)

Total: 200 candidates/user (config `candidates.top_k_per_user`).

Output:
- candidates_train.parquet     (cutoff = ranker_train_cutoff)
- candidates_predict.parquet   (cutoff = train_end)

Schema:
- user_id, item_id, source, source_score (raw score per source, sẽ normalize)

Mỗi source CHỈ dùng data trước cutoff → no leakage.
"""
from src.candidates.pipeline import run_build_candidates

__all__ = ["run_build_candidates"]
