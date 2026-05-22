# EDA Storyline Runbook — Datathon 2026 Chợ Tốt BĐS

## Mục tiêu

Pipeline này tạo full EDA cho 5 nhánh:

- A1 — Contact là tín hiệu giá trị quan sát được
- A2 — Người mua: phân khúc theo mức độ rõ ràng của nhu cầu
- A3 — Người bán/listing: lead đang được phân phối cho ai
- A4 — Kiến trúc model 3 tầng
- A5 — Cân bằng hiệu quả ngắn hạn và sức khỏe marketplace dài hạn

## Chạy sample trước

```powershell
python scripts/10_run_full_eda_storyline.py --data-root "C:/Datathon_Data" --sample-files 10 --force-events --threads 1 --memory-limit "8GB"
```

Sample mode chỉ để test code/schema. Không dùng số liệu sample cho slide.

## Chạy full 500 files

```powershell
python scripts/10_run_full_eda_storyline.py --data-root "C:/Datathon_Data" --force-events --threads 2 --memory-limit "10GB"
```

## Rerun nhanh sau khi đã có aggregation cache

```powershell
python scripts/10_run_full_eda_storyline.py --data-root "C:/Datathon_Data" --skip-event-agg --threads 2 --memory-limit "10GB"
```

## Output

```text
outputs/agg/                         # bảng pre-aggregated parquet, local only
outputs/tables/                      # bảng summary CSV nhỏ để đọc/paste slide
outputs/figures/main/                # PNG DPI >= 200
notebooks/02_full_eda_storyline.ipynb
```

## Lưu ý compliance

- Không commit `outputs/agg/` vì có user_id/item_id/seller_id đã ẩn danh nhưng vẫn là private contest data.
- Không dùng event_ts >= 2026-04-10.
- User-level metrics chỉ dùng login users; non-login user_id thay đổi theo session nên không dùng để kết luận hành vi dài hạn của user.
- `other_interaction` được giữ trong official positive target theo đề. Khi trình bày lead/contact economics, có thể song song báo cáo direct contacts gồm `view_phone`, `contact_chat`, `contact_zalo`, `contact_sms`.
