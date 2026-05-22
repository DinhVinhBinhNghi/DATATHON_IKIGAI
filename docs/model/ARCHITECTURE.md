# Model Architecture — Datathon 2026

> Chi tiết design decisions của pipeline LightGBM v3 trong `model/`.

## Tổng quan 2-stage

```
Raw events (160GB)
     │
     ↓
[Stage 0] Pre-aggregate (DuckDB SQL chunked)
     ↓ aggregates 5GB
     │
     ↓
[Stage 1] Candidate Generation (5 sources × top-200 per user)
     │
     │  5 sources:
     │  - reengagement   (50): items user đã interact
     │  - covisit        (80): item-item co-occurrence từ history
     │  - category_pop   (30): pop trong user top category
     │  - city_cat_pop   (20): pop trong (city × category)
     │  - global_pop     (20): fallback cold-start
     │
     ↓ merged pool: ~32M (user, item) pairs
     │
     ↓
[Stage 2] Ranking (LightGBM LambdaRank)
     │
     │  44 features:
     │  - candidate_score    (normalized 0-1 từ merge)
     │  - 13 user features   (history, recency, preferences)
     │  - 14 item features   (catalog, popularity, age)
     │  - 8 pair features    (interaction history user×item)
     │  - 3 temporal tiers   (recency tier bucketing)
     │  - 5 derived          (category match, city match, cat affinity ×3)
     │  - 5 categorical      (top_cat, top2_cat, i_cat, seller_type, ad_type)
     │
     ↓ scored pool with pred_score
     │
     ↓
[Stage 3] Marketplace-aware Rerank
     │
     │  Rules (mode='rerank'):
     │  - Freshness boost: items posted ≤7d được ×1.10
     │  - Seller diversity cap: max 2 items/seller/user; beyond penalty ×0.5
     │
     ↓ top-10 per user
     │
     ↓
submission.csv
```

## Triết lý core (v3.0)

### Event weighting

Bug v2: mọi positive event đếm bằng 1. `other_interaction` chiếm 94% positives nhưng là engagement noise → ranker học theo noise → Recall@10 Kaggle public 0.0063.

v3 fix: weight asymmetric.

```python
hard_contact (view_phone, contact_chat, contact_zalo, contact_sms) = 3.0
other_interaction                                                   = 1.0
pageview                                                            = 0.0
```

Apply ở:
1. Aggregation: `weighted_score = SUM(weight per event)`
2. Popularity: sort by weighted, không count
3. Co-visit: edge weight = sum(weighted)
4. Label: rel_label graded 0-3 từ `gt_weighted_score`

### Graded label 0-3

```python
gt_weighted_score >= 6.0 → rel_label = 3  (≥2 hard contacts)
gt_weighted_score >= 3.0 → rel_label = 2  (1 hard contact)
gt_weighted_score >  0.0 → rel_label = 1  (only other_interaction)
ELSE                     → rel_label = 0
```

LambdaRank objective tối ưu ordering theo graded label → model học rank thay vì memorize.

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

## Feature engineering

### User features (13)
History aggregates per user: total/recent weighted_score, unique items/categories/cities, active days, top category, top city, avg dwell, days since last, is_warm flag.

[v3.2 cat affinity add-on]:
- `u_top_category_share`: share của top cat (lifetime)
- `u_top_category_share_recent`: share của top cat (30d window)
- `u_top2_category`: category thứ 2

### Item features (14)
Static (dim_listing) + dynamic (item_daily): category, city, seller_type, ad_type, has_project_id, area_sqm, bedrooms, bathrooms, images_count, age_days, total/recent weighted, total pageview, avg dwell, pop_rank_global.

### Pair features (8)
Interaction history user×item: total/recent weighted_score, n_pageview, n_pos_events, max_dwell, n_active_days, days_since_last, days_since_first.

### Temporal tiers (3)
Bucket recency thành 6 tiers (0=≤3d ... 5=>180d): `ui_recency_tier`, `u_activity_recency_tier`, `i_age_tier`.

### Derived (5)
- `ui_category_match`: match top-1 cat user × i_cat
- `ui_city_match`: match top-1 city user × i_city
- `ui_category_match_top2`: match top-1 OR top-2 cat
- `ui_top_cat_match_x_share`: match × u_top_category_share (interaction feature)

## Candidate sources rationale

| Source | Quota | Bắt pattern gì |
|---|---|---|
| `reengagement` | 50 | Users quay lại tin đã xem trước khi contact (slide phân tích) |
| `covisit` | 80 | Users xem tin tương tự có behavior tương tự |
| `category_pop` | 30 | Personalized popularity trong user top category |
| `city_cat_pop` | 20 | Local popularity ở (city × category) |
| `global_pop` | 20 | Fallback cho 60% cold-start users |

Total 200/user. Top sources contribute by score: covisit ~40%, reengagement ~25%, các nguồn khác ~10-15% mỗi.

## Marketplace rerank

Trade-off chủ động: NDCG giảm ~3% nhưng:
- Gini seller: 0.73 → 0.61 (-16%)
- Coverage items: 68% → 79% (+11pp)
- Top-1% seller share: 12% → 7% (-42%)

Defense slide 6: rerank không chọn phe (môi giới vs chính chủ), chỉ cap concentration.

## OOM patches

Pipeline ban đầu OOM trên máy 16GB. 6 patches áp dụng:

1. `covisit`: cap MAX_SEEDS_PER_USER=100 (giảm pair count 100×)
2. `merge.py`: chunked 4-phase, 8 buckets HASH(user_id)
3. `train.py`: 3-step disk-based (labeled → user_keep → sampled)
4. `health_reranker.py`: pure DuckDB SQL, bỏ pandas (10-15GB → 1-2GB)
5. `builder.py`: push QUALIFY top-10 + dim_listing join xuống SQL
6. DTYPE fix: DuckDB DECIMAL → float64 cast trong pyarrow batch loop

Mỗi patch có comment `[OOM PATCH vX.Y]` trong code.

## Reproducibility

- `random_seed = 42` forward khắp pipeline (negative sampling, train/val split, LightGBM seed)
- Mọi step có cache check `file_exists_nonempty()`
- DuckDB threads = 4 (configurable). Threading có thể tạo small non-determinism trong row order ở SQL không có ORDER BY strict — sai số NDCG dưới 0.005.

## Future work

Đã identify nhưng chưa integrate vì time constraint:

1. **`fact_listing_snapshot` features**: item-level signal BTC pre-aggregated (views_24h, contacts_24h). Estimate: +0.02-0.03 NDCG.
2. **`fact_post_contact_interactions` features**: pair-level signal (lead_count, chat_message_count, chat_lead). Estimate: +0.03-0.05 NDCG nhưng có leak risk.
3. **`category_pop` top-2 cat recall**: hiện chỉ recall top-1 user category. Cross-category users (~13% population) bị miss.
4. **Hyperparameter tuning**: LightGBM dùng default. Optuna 50 trials có thể tăng ~0.5-1%.
5. **Content-based features**: NLP từ `title` (Vietnamese phobert) cho cold items.
6. **Ensemble** raw + rerank blend.

Chi tiết priorities + estimates: report PDF mục "Future Work".
