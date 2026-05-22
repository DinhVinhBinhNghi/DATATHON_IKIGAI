# Data Compliance

Repo này được thiết kế để phù hợp với quy định dữ liệu của Datathon 2026 Final.

## Nguyên tắc

1. Dữ liệu Chợ Tốt chỉ được dùng cho mục đích tham gia cuộc thi.
2. Không commit/push raw parquet, clean parquet, cache, model artifact, submission thật hoặc file có định danh user/item/seller lên GitHub public.
3. Không chia sẻ/tái phân phối dữ liệu cho bên thứ ba.
4. Không reverse-engineer định danh người dùng, người bán hoặc tin đăng.
5. Không dùng dữ liệu sau boundary train:
   - `date <= 2026-04-09`
   - `event_ts < 2026-04-10 00:00:00`

## Local-only paths

Dữ liệu thật nên nằm ngoài repo, ví dụ:

```text
C:/Datathon_Data/
├── train/
└── test/
```

Các folder sau bị `.gitignore`:

```text
data/raw/
data/clean/
outputs/cache/
outputs/models/
submissions/*.csv
```

## Kiểm tra trước khi push

```bash
git status
```

Không được thấy file `.parquet`, `.duckdb`, `.pkl`, `.joblib`, `submission.csv` thật trong danh sách chuẩn bị commit.
