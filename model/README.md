# Model Pipeline — Datathon 2026 IKIGAI

> Phần 2 của bài thi: Hệ Khuyến Nghị (Recommender System).
> LightGBM LambdaRank với marketplace-aware rerank.

## Quick start

```powershell
cd model
copy config\local.example.yaml config\local.yaml
# Sửa "raw_root" trong local.yaml trỏ tới raw data BTC

# Run full pipeline (4-5h tổng)
python scripts\00_check_data.py              # 30s
python scripts\10_preaggregate.py            # 45min
python scripts\11_build_candidates_train.py  # 30min
python scripts\12_build_candidates_predict.py # 30min
python scripts\13_train_ranker.py            # 60-90min
python scripts\14_score_candidates.py        # 15min
python scripts\15_rerank_and_submit.py       # 15min

# Output: submissions/submission_rerank_<timestamp>.csv
```

## Kết quả

| Metric | Value |
|---|---|
| **Val NDCG@10** | 0.6927 |
| **Val window** | 28 ngày (13/03 - 09/04) mimic test |
| **Total features** | 44 (39 numeric + 5 categorical) |
| **Total candidate sources** | 5 (reengagement, covisit, category_pop, city_cat_pop, global_pop) |
| **Training samples** | ~32M (user, item) pairs, neg sampling 10:1 |

## Triết lý core (v3.0+)

### Event weighting

Bug v2: mọi positive event đếm bằng 1. `other_interaction` chiếm 94% positives nhưng là engagement noise → ranker học theo noise → Recall@10 Kaggle public 0.0063.

v3 fix: weight asymmetric.

```python
hard_contact (view_phone, contact_chat, contact_zalo, contact_sms) = 3.0
other_interaction                                                   = 1.0
pageview                                                            = 0.0
```

Apply ở mọi tầng: aggregation (`weighted_score`), popularity sorting, co-visit edge weights, ranker label.

### Graded label 0-3

```python
gt_weighted_score >= 6.0 → rel_label = 3  (≥2 hard contacts)
gt_weighted_score >= 3.0 → rel_label = 2  (1 hard contact)
gt_weighted_score >  0.0 → rel_label = 1  (only other_interaction)
ELSE                     → rel_label = 0
```

LambdaRank optimize ordering theo graded label → model học rank thay vì memorize.

## Cấu trúc 10 scripts

| # | Script | Tác vụ | Time |
|---|---|---|---|
| 00 | `00_check_data.py` | Verify raw data BTC | 30s |
| 10 | `10_preaggregate.py` | 7 daily aggregates (chunked) | 45min |
| 11 | `11_build_candidates_train.py` | Candidates @ ranker_train_cutoff = 2026-03-12 | 30min |
| 12 | `12_build_candidates_predict.py` | Candidates @ predict_cutoff = 2026-04-09 | 30min |
| 13 | `13_train_ranker.py` | Build features + train LightGBM LambdaRank | 60-90min |
| 14 | `14_score_candidates.py` | Score predict pool (chunked) | 15min |
| 15 | `15_rerank_and_submit.py` | Rerank `raw` + `rerank` modes, build submission | 15min |
| 16 | `16_evaluate_candidates.py` | Recall@K per source | 5min |
| 17 | `17_evaluate_local.py` | NDCG@10 trên internal val | 5min |
| 18 | `18_analyze_marketplace_health.py` | Gini, coverage, freshness analysis | 10min |

## Time windows

```
2025-11-09 ──────────────── 2026-03-12 ─────── 2026-04-09 ─── 2026-04-10 ─── 2026-05-07
   train_start              ranker_train_     train_end       gt_start         gt_end
                            cutoff            predict_cutoff
                            (mimic test       (predict actual
                             window 28d)       test window 28d)
```

- **Ranker train**: data ≤ 2026-03-12, internal GT [13/03 - 09/04].
- **Predict**: data ≤ 2026-04-09, hidden GT [10/04 - 07/05] (BTC giữ kín).
- Window size mimic nhau (28 ngày) → model learn distribution đúng.

## 5 candidate sources

| Source | Quota | Pattern bắt |
|---|---|---|
| `reengagement` | 50 | Users quay lại tin đã xem trước khi contact |
| `covisit` | 80 | Users xem tin tương tự có behavior tương tự |
| `category_pop` | 30 | Personalized popularity trong user top category |
| `city_cat_pop` | 20 | Local popularity ở (city × category) |
| `global_pop` | 20 | Fallback cho cold-start users (~60% test set) |

Total 200/user. Merge với hash-based bucketing → top-200 per user.

## Marketplace rerank

Mode `rerank` áp dụng 2 rule:
- **Freshness boost**: items posted ≤ 7d được boost ×1.10
- **Seller diversity cap**: max 2 items/seller/user, beyond bị penalty ×0.5

Trade-off chủ động: NDCG giảm ~3% nhưng:
- Gini seller: 0.73 → 0.61 (-16%)
- Coverage items: 68% → 79% (+11pp)

Defense slide 6: rerank không chọn phe (môi giới vs chính chủ), chỉ cap concentration.

## OOM patches (cho máy 16GB RAM)

Pipeline ban đầu OOM trên máy 16GB. 6 patches áp dụng:

1. `covisit`: cap MAX_SEEDS_PER_USER=100 (giảm pair count 100×)
2. `merge.py`: chunked 4-phase, 8 buckets HASH(user_id)
3. `train.py`: 3-step disk-based (labeled → user_keep → sampled)
4. `health_reranker.py`: pure DuckDB SQL, bỏ pandas (10-15GB → 1-2GB)
5. `builder.py`: push QUALIFY top-10 + dim_listing join xuống SQL
6. DTYPE fix: DuckDB DECIMAL → float64 cast trong pyarrow batch loop

Mỗi patch có comment `[OOM PATCH vX.Y]` trong code.

## Tests

Yêu cầu install deps trước (`duckdb`, `pyarrow`, `lightgbm`, `pytest`):

```powershell
# Từ repo root, install deps
python -m pip install -r requirements.txt

# Sau đó chạy tests
cd model
python -m pytest tests/ -v
```

Coverage:
- `test_no_leakage.py`: anti-leakage at config level
- `test_candidate_recall.py`: candidate quality
- `test_candidates_format.py`: candidate schema
- `test_ranker_features.py`: feature schema
- `test_submission_format.py`: submission validator
- `test_weighted_score.py`: weighted aggregation correctness

## Reproducibility

- `random_seed = 42` ở `config/local.yaml`, forward khắp pipeline
- Mọi step có cache check `file_exists_nonempty()` — resume-safe
- `cache/models/lgb_metadata.json` save best_iteration + params sau train
- DuckDB threads = 4 (configurable). Threading có thể tạo small non-determinism (sai số NDCG < 0.005)

## Future work

Đã identify nhưng chưa integrate vì time constraint:

1. **`fact_listing_snapshot` features**: views_24h, contacts_24h từ BTC. Estimate +0.02-0.03 NDCG.
2. **`fact_post_contact_interactions` features**: lead_count, chat_lead. Estimate +0.03-0.05 NDCG nhưng có leak risk.
3. **`category_pop` top-2 cat recall**: hiện chỉ recall top-1, miss ~13% cross-category users.
4. **Hyperparameter tuning**: LightGBM dùng default. Optuna có thể tăng ~0.5-1%.
5. **Content-based features**: NLP từ `title` (Vietnamese phobert) cho cold items.
6. **Ensemble** raw + rerank blend.

Chi tiết: [`../docs/model/ARCHITECTURE.md`](../docs/model/ARCHITECTURE.md).

## Documentation

- [`../docs/model/RUNBOOK.md`](../docs/model/RUNBOOK.md) — Run instructions chi tiết
- [`../docs/model/ARCHITECTURE.md`](../docs/model/ARCHITECTURE.md) — Design decisions
- [`../docs/eda/MODEL_CARD.md`](../docs/eda/MODEL_CARD.md) — Model card overview
