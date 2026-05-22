"""Preaggregation: aggregate raw events thành user/item/user_item daily stats.

CORE: mọi cột "score" đều là `weighted_score`:
    weighted_score = SUM(weight per event_type)
    where weight =
        3.0  for view_phone, contact_chat, contact_zalo, contact_sms
        1.0  for other_interaction
        0.0  for pageview

So với v2.4.0 (đếm n_positive bằng nhau), v3.0 weighted_score giảm tầm ảnh hưởng
của other_interaction (94% positive) và amplify hard contacts (lead-gen signal).

Output files trong cache/agg/:
- user_daily.parquet           (user_id, date, n_pageview, n_pos_events, weighted_score)
- item_daily.parquet           (item_id, date, ...)
- user_item_daily.parquet      (user_id, item_id, date, n_pageview, weighted_score)
- user_category_weighted.parquet
- user_city_weighted.parquet
- event_type_daily.parquet     (cho debug/validation)
- _marker_v3.json              (marker file, dùng để verify cache valid)
"""
from src.preagg.pipeline import run_preaggregate

__all__ = ["run_preaggregate"]
