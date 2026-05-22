# Datathon 2026 — Team IKIGAI

> **VinUniversity Datathon 2026 — Vòng Chung Kết**
> **Đề bài**: Recommender system cho Chợ Tốt BĐS (top-10 items per user)
> **Team**: Đinh Vĩnh Bình Nghi · Nguyễn Thanh Ngân

---

## Tổng quan repo

Repo này chứa toàn bộ pipeline của team IKIGAI cho Vòng Chung Kết Datathon 2026. Pipeline được tổ chức thành **2 phần độc lập** phục vụ 2 yêu cầu của đề bài:

```
datathon2026_final/
├── eda/                    # Phần 1: Trực quan hoá và Phân tích Dữ liệu
│   ├── scripts/            # 12 scripts: clean → EDA → storyline → slide charts
│   ├── src/                # Modules: cleaning, eda, ranking, submission, utils
│   ├── audit/              # Data quality + audit scripts (B2B/C2C, D30, OI)
│   ├── config/             # Config example
│   ├── outputs/            # Figures, tables (gitignored, generated)
│   └── README.md
│
├── model/                  # Phần 2: Mô hình Gợi ý (Recommender System)
│   ├── scripts/            # 10 scripts: preagg → candidates → ranker → submit
│   ├── src/                # Modules: candidates, features, ranker, rerank
│   ├── config/             # Config example
│   ├── tests/              # Unit tests + anti-leakage tests
│   ├── cache/              # Intermediate artifacts (gitignored)
│   ├── submissions/        # Final CSV (gitignored)
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md
│
├── docs/
│   ├── eda/                # EDA docs: storyline runbook, model card, compliance
│   └── model/              # Model docs: architecture, runbook
│
├── data/sample/            # Sample data nhỏ (gitignored)
├── requirements.txt        # Dependencies chung
├── pyproject.toml          # Project metadata
└── .gitignore              # Loại bỏ raw data, cache, submissions
```

---

## Cách reproduce submission cuối cùng

> **Yêu cầu**: Python 3.10+, Windows hoặc Linux, ≥16GB RAM, ~100GB free disk.

### Bước 1 — Setup môi trường

```bash
# Clone repo
git clone <repo_url> datathon2026_final
cd datathon2026_final

# Tạo venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install deps
pip install -r requirements.txt
```

### Bước 2 — Setup data path

Raw data từ BTC **không nằm trong repo** (xem `docs/eda/DATA_COMPLIANCE.md`). Đặt data ở thư mục local riêng và config:

```bash
# Phần model:
cp model/config/local.example.yaml model/config/local.yaml
# Sửa "raw_root" trỏ tới thư mục chứa raw data BTC

# Phần EDA:
cp eda/config/config.example.yaml eda/config/local.yaml
# Sửa "data_root" trỏ tới thư mục chứa raw data BTC
```

Structure raw data cần có:

```
C:/Datathon_Data/    (hoặc path khác)
├── train/
│   ├── dim_listing/
│   ├── fact_user_events/
│   ├── fact_listing_snapshot/
│   └── fact_post_contact_interactions/
└── test/
    └── test_users.parquet
```

### Bước 3 — Chạy pipeline model (sinh submission Kaggle)

```bash
cd model

# Pipeline 7 steps chính, resume-safe — mỗi step skip nếu cache đã có:
python scripts/00_check_data.py              # 30s — verify raw data
python scripts/10_preaggregate.py            # 45min — daily aggregates
python scripts/11_build_candidates_train.py  # 30min — candidates @ train cutoff
python scripts/12_build_candidates_predict.py # 30min — candidates @ predict cutoff
python scripts/13_train_ranker.py            # 60-90min — train LightGBM
python scripts/14_score_candidates.py        # 15min — score predict pool
python scripts/15_rerank_and_submit.py       # 15min — rerank + write CSV

# Output: submissions/submission_rerank_<timestamp>.csv
```

**Tổng thời gian**: ~4-5 giờ trên máy local Windows i7/16GB.

### Bước 4 — (Optional) Chạy pipeline EDA (sinh slide visuals + reports)

```bash
cd eda
python scripts/10_run_full_eda_storyline.py --data-root "C:/Datathon_Data" --threads 2 --memory-limit "10GB"
python scripts/11_build_slide_visuals.py
```

Output: `eda/outputs/figures/{main,appendix,slides}/`.

---

## Kết quả

| Metric | Value | Note |
|---|---|---|
| **Internal val NDCG@10** | 0.6927 | 28-day holdout (13/03 - 09/04) mimic test window |
| **Kaggle Public LB** | 0.1450 | Tính trên 5% ground truth |
| **Kaggle Private LB** | 0.1451 | Final ranking |

---

## Kiến trúc giải pháp

```
                    ┌──────────────────────┐
                    │   Raw 4 BTC tables   │
                    │  (fact_user_events,  │
                    │   dim_listing, ...)  │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                ↓                             ↓
        ┌───────────────┐            ┌─────────────────┐
        │  EDA Pipeline │            │  Model Pipeline │
        │  (eda/)       │            │  (model/)       │
        └───────┬───────┘            └────────┬────────┘
                │                              │
                ↓                              ↓
        Slide visuals,                 ┌──────────────┐
        marketplace                    │  Candidate   │
        health analysis                │  Generation  │
                                       │  (5 sources) │
                                       └──────┬───────┘
                                              ↓
                                       ┌──────────────┐
                                       │   LightGBM   │
                                       │  LambdaRank  │
                                       │  44 features │
                                       └──────┬───────┘
                                              ↓
                                       ┌──────────────┐
                                       │  Marketplace │
                                       │   Rerank     │
                                       └──────┬───────┘
                                              ↓
                                       submission.csv
```

**Triết lý**: 2-stage standard cho recsys industrial (candidate gen → ranking). Plus marketplace-aware rerank để balance accuracy với marketplace health.

Chi tiết design decisions: [`docs/model/ARCHITECTURE.md`](docs/model/ARCHITECTURE.md).

---

## Cấu trúc submission

Theo Đề bài mục 3.3.5:

```csv
ID,user_id,rank,item_id
1,abc123...,1,xyz456...
2,abc123...,2,uvw789...
...
```

- UTF-8 không BOM.
- Mỗi user đúng 10 rows (rank 1..10).
- (user_id, rank) và (user_id, item_id) unique.
- Dung lượng ≤ 100 MB (nếu vượt, ZIP thủ công trước upload).

Validator có sẵn ở `model/src/submission/validator.py`.

---

## Documentation chi tiết

| Doc | Chứa gì |
|---|---|
| [`eda/README.md`](eda/README.md) | Quick start EDA pipeline |
| [`model/README.md`](model/README.md) | Chi tiết pipeline LightGBM v3 |
| [`docs/eda/RUNBOOK.md`](docs/eda/RUNBOOK.md) | Steps chạy EDA |
| [`docs/eda/EDA_STORYLINE_RUNBOOK.md`](docs/eda/EDA_STORYLINE_RUNBOOK.md) | Storyline 5-nhánh A1-A5 |
| [`docs/eda/MODEL_CARD.md`](docs/eda/MODEL_CARD.md) | Model spec + assumptions |
| [`docs/eda/REPRODUCIBILITY.md`](docs/eda/REPRODUCIBILITY.md) | Hướng dẫn reproduce |
| [`docs/eda/DATA_COMPLIANCE.md`](docs/eda/DATA_COMPLIANCE.md) | Quy định dữ liệu Chợ Tốt |
| [`docs/model/RUNBOOK.md`](docs/model/RUNBOOK.md) | Steps chạy model pipeline |
| [`docs/model/ARCHITECTURE.md`](docs/model/ARCHITECTURE.md) | Design decisions của model |

---

## Team

**Đinh Vĩnh Bình Nghi** — lead EDA + storyline + report writing.
**Nguyễn Thanh Ngân** — lead pipeline engineering (candidate generation + ranker + rerank).

---

## License & Compliance

Dữ liệu Chợ Tốt được sử dụng độc quyền cho Datathon 2026. Repo này tuân thủ:

1. KHÔNG commit raw data, cache, hoặc submission có định danh thật (xem `.gitignore`).
2. KHÔNG sử dụng dữ liệu trong ground truth window (10/04 - 07/05/2026).
3. KHÔNG cố giải ẩn danh user/item/seller IDs.

Chi tiết: [`docs/eda/DATA_COMPLIANCE.md`](docs/eda/DATA_COMPLIANCE.md).
