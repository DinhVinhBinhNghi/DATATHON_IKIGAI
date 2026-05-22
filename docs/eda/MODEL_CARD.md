# Model Card — Baseline Recommender

## Problem

Dự đoán top-10 `item_id` cho mỗi `user_id` trong test set, tối ưu tương tác tích cực trong cửa sổ 10/04/2026–07/05/2026.

## Positive interactions

```text
view_phone, contact_chat, other_interaction, contact_zalo, contact_sms
```

## Current baseline

Baseline hiện tại là recommender theo popularity có điều kiện:

- Global popular active items.
- Category-specific popular items.
- City-specific popular items.
- Category × city popular items.
- User profile lấy từ lịch sử positive events trước boundary train.

## Marketplace health additions

Score có cộng nhẹ cho:

- tin chính chủ (`seller_type = private`),
- tin mới trong 30 ngày,
- item có contact rate tốt từ snapshot.

## Known limitations

- Chưa có learning-to-rank supervised offline split.
- Chưa dùng graph/co-visitation sâu.
- Chưa tối ưu diversity theo mỗi user bằng constraint rõ ràng.
- Baseline ưu tiên độ ổn định và reproducibility.
