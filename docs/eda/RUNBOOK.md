# Runbook

## 1. Kiểm tra data

```bash
python scripts/00_check_local_data.py --data-root "C:/Datathon_Data"
```

## 2. Clean dữ liệu

```bash
python scripts/01_clean_all.py --data-root "C:/Datathon_Data"
```

## 3. EDA aggregate

```bash
python scripts/02_run_eda.py --data-root "C:/Datathon_Data"
```

## 4. Candidate + recommendation

```bash
python scripts/03_build_candidates.py --data-root "C:/Datathon_Data"
```

## 5. Submission

```bash
python scripts/04_make_submission.py --data-root "C:/Datathon_Data"
```

## 6. Validate

```bash
python scripts/05_validate_submission.py --data-root "C:/Datathon_Data" --submission submissions/submission.csv
```
