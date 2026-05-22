# Reproducibility

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Full run

```bash
python scripts/run_full_pipeline.py --data-root "C:/Datathon_Data"
```

## Faster rerun after cleaning

```bash
python scripts/run_full_pipeline.py --data-root "C:/Datathon_Data" --skip-clean --skip-eda
```

## Main outputs

```text
data/clean/*.parquet                 # local only
outputs/cache/*.parquet              # local only
outputs/tables/*.csv                 # aggregate tables
submissions/submission.csv           # local only, submit to Kaggle
```

## Random seed

Seed mặc định: `42`, set trong `src/utils/constants.py` và `src/utils/seed.py`.
