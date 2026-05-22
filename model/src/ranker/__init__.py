"""LightGBM LambdaRank ranker.

Train: dùng ranker_input_train.parquet với label từ internal ground truth.
Label: 1 nếu (user, item) có positive event trong GT window, else 0.
Group: per user (LambdaRank cần group ID để biết ranking context).

Score: load trained model → predict pred_score cho ranker_input_predict.parquet.
"""
from src.ranker.feature_spec import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    ID_COLS,
)
from src.ranker.train import train_ranker
from src.ranker.score import score_candidates

__all__ = [
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "ALL_FEATURES",
    "ID_COLS",
    "train_ranker",
    "score_candidates",
]
