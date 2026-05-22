from __future__ import annotations

TRAIN_START_DATE = "2025-11-09"
TRAIN_END_DATE = "2026-04-09"
TRAIN_END_TS = "2026-04-10 00:00:00"
PREDICT_START = "2026-04-10"
PREDICT_END = "2026-05-07"
SEED = 42

# Canonical funnel A1 threshold. See DECISIONS_NEEDED.md mục 1.
# Override here will propagate to EDA + slides if you rerun the pipeline.
DWELL_THRESHOLD_SEC = 30

# Local holdout for ranker training, see DECISIONS_NEEDED.md mục 7.
HOLDOUT_START_DATE = "2026-04-03"
HOLDOUT_END_DATE = "2026-04-09"

POSITIVE_EVENTS = [
    "view_phone",
    "contact_chat",
    "other_interaction",
    "contact_zalo",
    "contact_sms",
]

EXPECTED_TRAIN_FILES = {
    "dim_listing": 40,
    "fact_listing_snapshot": 62,
    "fact_post_contact_interactions": 147,
    "fact_user_events": 500,
}

CATEGORY_NAMES = {
    1010: "Phòng trọ/thuê",
    1020: "Căn hộ/CC",
    1030: "Nhà ở",
    1040: "Đất nền/TM",
    1050: "Dự án mới",
}
