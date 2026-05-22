# EDA Pipeline — Datathon 2026 IKIGAI

> Phần 1 của bài thi: Trực quan hoá và Phân tích Dữ liệu.

## Quick start

```powershell
# Setup
cd eda
copy config\config.example.yaml config\local.yaml
# Sửa "data_root" trong local.yaml trỏ tới raw data BTC

# Smoke test với 10 file sample
python scripts\10_run_full_eda_storyline.py --data-root "C:/Datathon_Data" --sample-files 10 --force-events --threads 1 --memory-limit "8GB"

# Full run với 500 files
python scripts\10_run_full_eda_storyline.py --data-root "C:/Datathon_Data" --force-events --threads 2 --memory-limit "10GB"

# Build slide visuals
python scripts\11_build_slide_visuals.py

# Build real slide charts (data-driven version)
python scripts\query_and_make_real_slide_charts_v3.py
```

## Output

```
outputs/
├── agg/             # Pre-aggregated parquet (gitignored)
├── cache/           # DuckDB intermediates (gitignored)
├── figures/
│   ├── main/        # Figures cho main slide
│   ├── appendix/    # Figures cho appendix
│   └── slides/      # Composite figures cho slide deck
├── slide_charts_real/  # Real data charts
├── slide_data_real/    # Underlying data tables
└── tables/          # Summary tables (CSV)
```

## Storyline 5 nhánh

Pipeline build storyline cho slide V3:

- **A1** — Contact là tín hiệu giá trị quan sát được
- **A2** — Người mua: phân khúc theo mức độ rõ ràng của nhu cầu
- **A3** — Người bán/listing: lead đang được phân phối cho ai
- **A4** — Kiến trúc model 3 tầng
- **A5** — Cân bằng hiệu quả ngắn hạn và sức khỏe marketplace dài hạn

Chi tiết: [`../docs/eda/EDA_STORYLINE_RUNBOOK.md`](../docs/eda/EDA_STORYLINE_RUNBOOK.md).

## Scripts overview

| Script | Tác vụ | Time |
|---|---|---|
| `00_check_local_data.py` | Verify data BTC | 30s |
| `01_clean_all.py` | Clean raw → parquet sạch | 30min |
| `02_run_eda.py` | EDA aggregate | 15min |
| `03_build_candidates.py` | Baseline candidate gen (alt pipeline) | 20min |
| `04_make_submission.py` | Build submission (alt pipeline) | 10min |
| `05_validate_submission.py` | Validator | 1min |
| `08_train_ranker.py` | LightGBM ranker (alt experiment) | 30min |
| `09_run_variants.py` | A/B test variants | 30min |
| `10_run_full_eda_storyline.py` | **Full EDA pipeline** | 45-60min |
| `11_build_slide_visuals.py` | **Build slide figures** | 10min |
| `query_and_make_real_slide_charts_v3.py` | Real-data slide charts | 15min |
| `run_full_pipeline.py` | One-shot orchestrator | 90min |

## Audit scripts (`audit/`)

Data quality + audit scripts có giá trị defense slide:

| Script | Tác vụ |
|---|---|
| `audit_other_interaction.py` | Audit `other_interaction` event — verify không phải tracking noise |
| `audit_seller_type_diagnostic.py` | Diagnostic seller_type (môi giới vs chính chủ) |
| `verify_b2b_c2c_split.py` | Verify B2B/C2C split không leak |
| `verify_d30_consistency.py` | Verify D30 contact rate consistency với pre-aggregate |
| `check_schema.py` | Quick schema check raw data |

Chạy độc lập:
```powershell
python audit\audit_other_interaction.py
python audit\verify_d30_consistency.py
```

## Relationship với `model/`

`eda/` và `model/` là **2 codebases độc lập**:

- `eda/`: focus phân tích + slide visuals + baseline alt pipeline. Output: figures cho slide V3.
- `model/`: focus production-quality LightGBM ranker. Output: `submission.csv` cho Kaggle.

Cả 2 đọc cùng raw data BTC nhưng KHÔNG gọi nhau. Lý do: 2 phần đề bài độc lập, mỗi phần có constraint riêng (EDA cần flexibility cho exploration, Model cần reproducibility cứng cho submission).

## Documentation

- [`../docs/eda/RUNBOOK.md`](../docs/eda/RUNBOOK.md) — Steps chạy chi tiết
- [`../docs/eda/EDA_STORYLINE_RUNBOOK.md`](../docs/eda/EDA_STORYLINE_RUNBOOK.md) — 5-nhánh storyline
- [`../docs/eda/MODEL_CARD.md`](../docs/eda/MODEL_CARD.md) — Baseline model spec
- [`../docs/eda/REPRODUCIBILITY.md`](../docs/eda/REPRODUCIBILITY.md) — Reproduce instructions
- [`../docs/eda/DATA_COMPLIANCE.md`](../docs/eda/DATA_COMPLIANCE.md) — Quy định dữ liệu
