from src.ranker import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, ID_COLS,
    train_ranker, score_candidates,
)
from src.rerank import (
    rule_cap_seller_diversity, rule_freshness_boost, run_rerank,
)
print("Phase 6 import OK")
print(f"  numeric features: {len(NUMERIC_FEATURES)}")
print(f"  categorical features: {len(CATEGORICAL_FEATURES)}")
print(f"  total features: {len(ALL_FEATURES)}")
print()
print("Functions:")
for f in [train_ranker, score_candidates,
          rule_cap_seller_diversity, rule_freshness_boost, run_rerank]:
    print(f"  {f.__name__}")
