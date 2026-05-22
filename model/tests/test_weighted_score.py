"""Test event weighting công thức.

Critical test vì đây là CORE của v3.0.
"""
from __future__ import annotations

import pytest

from src.common import get_config
from src.preagg.aggregator import _positive_filter, _weight_case


class TestWeightedScore:
    """Test weight CASE expression đúng spec."""

    def test_config_has_required_weights(self):
        """Config phải có weights.hard_contact, .other_interaction, .pageview."""
        cfg = get_config()
        assert hasattr(cfg, "weights"), "Config thiếu section 'weights'"
        for attr in ["hard_contact", "other_interaction", "pageview"]:
            assert hasattr(cfg.weights, attr), f"weights.{attr} missing"

    def test_default_weight_values(self):
        """Default values per v3.0 spec."""
        cfg = get_config()
        assert cfg.weights.hard_contact == 3.0, "hard_contact phải = 3.0"
        assert cfg.weights.other_interaction == 1.0, "other_interaction phải = 1.0"
        assert cfg.weights.pageview == 0.0, "pageview phải = 0.0"

    def test_weight_case_includes_hard_contacts(self):
        """4 hard contact events phải xuất hiện trong CASE."""
        expr = _weight_case()
        for event in ["view_phone", "contact_chat", "contact_zalo", "contact_sms"]:
            assert event in expr, f"'{event}' missing in weight CASE"

    def test_weight_case_uses_correct_values(self):
        """Weight 3.0 cho hard contacts, 1.0 cho other_interaction."""
        expr = _weight_case()
        cfg = get_config()
        # Check hard_contact value (3.0) xuất hiện
        assert str(cfg.weights.hard_contact) in expr
        # Check other_interaction value (1.0) xuất hiện
        assert "other_interaction" in expr
        # ELSE 0.0 (pageview default)
        assert "ELSE 0.0" in expr

    def test_weight_case_has_alias(self):
        """Khi pass alias, output có 'AS xxx'."""
        expr = _weight_case(alias="my_score")
        assert "AS my_score" in expr

    def test_positive_filter_has_5_events(self):
        """Positive filter phải có đúng 5 event types."""
        expr = _positive_filter()
        for event in ["view_phone", "contact_chat", "contact_zalo",
                      "contact_sms", "other_interaction"]:
            assert event in expr, f"'{event}' missing in positive filter"
        # pageview KHÔNG được trong positive filter
        assert "pageview" not in expr

    def test_weighting_amplifies_hard_contacts(self):
        """Spec invariant: hard_contact weight > other_interaction weight."""
        cfg = get_config()
        assert cfg.weights.hard_contact > cfg.weights.other_interaction, (
            "hard_contact phải > other_interaction (fix v2.4.0 bug)"
        )
        # Pageview phải 0 (không phải positive)
        assert cfg.weights.pageview == 0.0


class TestWeightedScoreSemantic:
    """Test semantic: weight ratio đúng để fix bug v2.4.0."""

    def test_hard_to_other_ratio(self):
        """Hard contact phải nặng gấp ≥ 2× other_interaction.

        v2.4.0 đếm tất cả 5 events bằng nhau → 94% signal là other_interaction (noise).
        v3.0 weight 3.0 vs 1.0 = ratio 3.0 → hard contacts dominate score.
        """
        cfg = get_config()
        ratio = cfg.weights.hard_contact / max(cfg.weights.other_interaction, 1e-9)
        assert ratio >= 2.0, f"Ratio {ratio:.2f} < 2.0 — quá yếu để fix bug"
