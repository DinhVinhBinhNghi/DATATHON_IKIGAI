# Model Pipeline Runbook

> Hướng dẫn chạy pipeline model (`model/`) từ đầu để reproduce submission Kaggle.

## Prerequisites

- Python ≥3.10
- ≥16GB RAM, ~100GB free disk
- Raw data từ BTC ở thư mục local (xem `docs/eda/DATA_COMPLIANCE.md`)

## Setup

```powershell
# Từ repo root
cd model

# Copy config template, sửa raw_root trỏ tới data
copy config\local.example.yaml config\local.yaml
notepad config\local.yaml   # sửa "raw_root: C:/Datathon_Data"
```

## Pipeline 10 steps

Mỗi script idempotent — chạy lại sẽ skip nếu cache đã có. Để force rerun: xóa file trong `cache/`.

### Step 0: Kiểm tra raw data

```powershell
python scripts\00_check_data.py
```

Verify schema của 4 bảng BTC + `test_users.parquet`. Báo lỗi nếu thiếu file hoặc cột.

### Step 1: Pre-aggregate (`10_preaggregate.py`)

```powershell
python scripts\10_preaggregate.py
```

Build 7 aggregate tables từ `fact_user_events`:
- `user_daily.parquet`, `item_daily.parquet`
- `user_item_daily.parquet` (chunked theo HASH(user_id))
- `user_category_weighted.parquet`, `user_city_weighted.parquet`
- `event_type_daily.parquet`

**Time**: ~45 phút.
**Output**: `cache/agg/*.parquet`.

### Step 2-3: Candidate generation (`11_*`, `12_*`)

```powershell
python scripts\11_build_candidates_train.py    # cutoff = 2026-03-12
python scripts\12_build_candidates_predict.py  # cutoff = 2026-04-09
```

5 sources song song mỗi mode:
- `reengagement` (50 items/user)
- `covisit` (80 items/user) — item-item co-occurrence
- `category_pop` (30) — pop trong user top category
- `city_cat_pop` (20) — pop trong (city × category)
- `global_pop` (20) — fallback cold-start

Merge 5 sources → top 200/user → `candidates_{train,predict}.parquet`.

**Time**: ~30 phút mỗi mode.

### Step 4: Train ranker (`13_train_ranker.py`)

```powershell
python scripts\13_train_ranker.py
```

Pipeline trong script:
1. Build user/item/pair/temporal features tại `ranker_train_cutoff = 2026-03-12`.
2. Join features → `ranker_input_train.parquet` (~50 cols).
3. Build labels từ internal GT (13/03 - 09/04), graded 0-3.
4. Sample negatives 10:1 ratio.
5. Train LightGBM LambdaRank, early stopping NDCG@10.
6. Save `lgb_ranker.txt` + `lgb_metadata.json` + feature importance.

**Time**: ~60-90 phút.
**Best iteration**: thường ~iteration 350-450/1000.

### Step 5: Score predict pool (`14_score_candidates.py`)

```powershell
python scripts\14_score_candidates.py
```

Build features tại `predict_cutoff = 2026-04-09` (mới — KHÔNG reuse từ train mode).
Join → `ranker_input_predict.parquet` (~32M rows).
Score chunked (2M rows/chunk) → `scored_pool_predict.parquet`.

**Time**: ~15 phút.

### Step 6: Rerank + submit (`15_rerank_and_submit.py`)

```powershell
python scripts\15_rerank_and_submit.py
```

2 modes:
- `raw`: chỉ pick top-10 theo pred_score (không apply rule).
- `rerank`: freshness boost 1.10× cho items ≤7d + seller diversity cap (max 2/seller).

Build `submission_<mode>_<timestamp>.csv` cho cả 2 modes.

**Time**: ~15 phút (cả 2 modes).
**Output**: `submissions/submission_rerank_*.csv` (upload Kaggle).

### Step 7-9: Evaluation (optional)

```powershell
python scripts\16_evaluate_candidates.py    # Recall@K của candidate sources
python scripts\17_evaluate_local.py         # NDCG@10 trên internal val
python scripts\18_analyze_marketplace_health.py  # Gini, coverage, freshness
```

---

## Cấu trúc cache

```
cache/
├── agg/                      # Pre-aggregates (Step 1)
├── candidates/               # 5 sources × 2 modes + merged (Step 2-3)
├── features/                 # user, item, pair, temporal features + ranker_input (Step 4-5)
├── ground_truth/             # internal_gt.parquet (Step 4)
└── models/                   # lgb_ranker.txt + metadata + feature_importance (Step 4)
```

Total cache size: ~30-40GB sau full run.

---

## Smoke test trên data nhỏ

Để test pipeline chạy được trước khi run full:

```powershell
# Hardcode tạm trong scripts: for bucket in [0]:  (chỉ 1/8 buckets)
# Hoặc dùng test_users.parquet subset 100 users
```

---

## Troubleshooting

### OOM ở step 13 train

Pipeline đã có 6 OOM patches. Nếu vẫn OOM:
- Tăng `N_SAMPLE_BUCKETS` trong `train.py` từ 8 lên 16.
- Giảm `neg_per_pos` trong `config/local.yaml` từ 10 xuống 5.
- Giảm `top_k_per_user` trong candidates từ 200 xuống 150.

### DTYPE error `object dtype` trong LightGBM

Đã patch ở v3.2.1 (`score.py` + `train.py`). Nếu xuất hiện ở feature mới: thêm CAST AS DOUBLE trong SQL aggregation.

### File CSV > 100 MB

Item_id SHA-256 64-char → CSV ~225 MB. Workaround: ZIP thủ công trước upload Kaggle:

```powershell
Compress-Archive -Path submissions\submission_rerank_*.csv -DestinationPath submissions\submission.zip -Force
```

Kaggle accept `.zip` (không accept `.gz`).

### Pipeline crash giữa chừng

Resume safe — chạy lại cùng script, sẽ tiếp từ step chưa xong.

---

## Reproduce verify

Sau khi pipeline xong, verify reproducibility:

```powershell
# Check val NDCG khớp với report
python -c "import json; m=json.load(open('cache/models/lgb_metadata.json')); print(f'Val NDCG@10: {m[\"best_score\"][\"valid_0\"][\"ndcg@10\"]:.4f}')"

# Run validator trên submission
python -m src.submission.validator submissions\submission_rerank_*.csv
```

Expected: NDCG@10 val ~0.6927 ± 0.005 (DuckDB thread non-determinism).
