"""Two-tower neural ranker — EXPERIMENTAL STUB.

KHÔNG được dùng trong baseline. File này là placeholder để team có thể
explore sau khi đã có LGB baseline. Lý do không chọn two-tower làm baseline:

1. 64% test users là cold-start (no_history). Two-tower user embedding cho 64%
   users sẽ chỉ là cold-start fallback embedding (vô dụng), và pipeline vẫn
   cần content/popularity branch riêng.

2. Two-tower cần train embedding (vài giờ GPU) + ANN index (FAISS/ScaNN) ở
   inference. Cost cao hơn LGB nhiều.

3. Defensibility trước BGK: LGB feature importance dễ defend hơn embedding
   layer ("tôi dùng feature X vì insight Y").

Khi nào nên train two-tower:
- Sau khi LGB baseline đã có Recall@10 trên Kaggle LB.
- Nếu LGB recall đã saturate và team có thời gian thử user/item embedding.
- Nếu team muốn unify candidate generation + ranking (two-tower có thể làm cả 2).

API dự kiến (chưa implement):
    train_twotower(con, paths, config) -> embeddings + index
    retrieve_topk(user_id, embedding, index, k=100) -> candidate item_ids
"""
from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_twotower(*args, **kwargs):
    raise NotImplementedError(
        "Two-tower training chưa được implement. Đây là stub để team explore sau. "
        "Xem docstring module để biết khi nào nên train."
    )


def retrieve_topk(*args, **kwargs):
    raise NotImplementedError(
        "Two-tower retrieval chưa được implement."
    )
