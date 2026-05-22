"""Test ranker feature spec đầy đủ + không trùng cột.

Verify ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES, không trùng.
Verify validate_columns() phát hiện missing.
"""
from __future__ import annotations

from src.ranker.feature_spec import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLS,
    NUMERIC_FEATURES,
    LABEL_COL,
    GROUP_COL,
    validate_columns,
)


class TestFeatureSpec:
    def test_all_features_union(self):
        """ALL_FEATURES = NUMERIC + CATEGORICAL."""
        expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        assert ALL_FEATURES == expected

    def test_no_feature_overlap(self):
        """NUMERIC + CATEGORICAL phải disjoint (1 col không vào 2 list)."""
        overlap = set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
        assert not overlap, f"Features overlap: {overlap}"

    def test_no_id_in_features(self):
        """user_id, item_id KHÔNG được trong features."""
        for id_col in ID_COLS:
            assert id_col not in ALL_FEATURES, (
                f"ID col '{id_col}' không được trong ALL_FEATURES"
            )

    def test_label_col_not_in_features(self):
        """rel_label KHÔNG được trong features."""
        assert LABEL_COL not in ALL_FEATURES

    def test_group_col_is_user_id(self):
        """GROUP_COL phải là user_id (LambdaRank groups by user)."""
        assert GROUP_COL == "user_id"

    def test_minimum_feature_count(self):
        """Phải có ≥ 30 features cho ranker đủ mạnh."""
        assert len(ALL_FEATURES) >= 30, (
            f"Chỉ có {len(ALL_FEATURES)} features, quá ít"
        )


class TestFeatureValidation:
    def test_validate_columns_pass(self):
        """validate_columns trả empty list khi đủ cột."""
        df_cols = ID_COLS + ALL_FEATURES + [LABEL_COL]
        missing = validate_columns(df_cols)
        assert missing == []

    def test_validate_columns_detects_missing(self):
        """validate_columns phát hiện missing columns."""
        df_cols = ID_COLS + ALL_FEATURES[:5]  # cắt bớt
        missing = validate_columns(df_cols)
        assert len(missing) > 0

    def test_required_critical_features(self):
        """Critical features phải present."""
        critical = [
            "candidate_score",       # score từ candidate gen
            "u_total_weighted",      # user history
            "i_recent_weighted",     # item recent pop
            "ui_total_weighted",     # pair interaction
        ]
        for c in critical:
            assert c in ALL_FEATURES, f"Critical feature missing: {c}"

    def test_categorical_features_are_int_or_str(self):
        """CATEGORICAL_FEATURES phải là cols có cardinality thấp (int/string)."""
        # Check names — không test value runtime, chỉ verify list không rỗng
        assert len(CATEGORICAL_FEATURES) >= 2, "Cần ít nhất 2 categorical"
        # Common categorical patterns
        cat_set = set(CATEGORICAL_FEATURES)
        # Phải có ít nhất 1 trong: top_category, seller_type, ad_type
        common = {"u_top_category", "i_category", "i_seller_type", "i_ad_type"}
        assert cat_set & common, (
            f"Categorical features không có column quen thuộc: {CATEGORICAL_FEATURES}"
        )
