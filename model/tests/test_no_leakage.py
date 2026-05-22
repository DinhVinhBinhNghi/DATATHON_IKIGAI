"""Test anti-leakage: pipeline KHÔNG dùng data sau cutoff.

Critical cho competition rule: "Không được train trên data ≥ 10/04/2026".
"""
from __future__ import annotations

from src.common import get_config


class TestWindowsConfig:
    """Test windows cấu hình đúng spec BTC."""

    def test_train_window_within_btc_period(self):
        """Train window phải nằm trong [2025-11-09, 2026-04-09]."""
        cfg = get_config()
        assert cfg.windows.train_start >= "2025-11-09", (
            f"train_start {cfg.windows.train_start} sớm hơn BTC spec"
        )
        assert cfg.windows.train_end <= "2026-04-09", (
            f"train_end {cfg.windows.train_end} muộn hơn BTC spec — LEAK!"
        )

    def test_ranker_train_cutoff_before_train_end(self):
        """ranker_train_cutoff phải trước train_end (để có GT window từ giữa)."""
        cfg = get_config()
        assert cfg.windows.ranker_train_cutoff < cfg.windows.train_end, (
            "ranker_train_cutoff phải < train_end"
        )

    def test_internal_gt_within_train_window(self):
        """Internal GT window phải nằm trong train_window.

        BTC giữ kín test window 10/04 - 07/05. Internal GT mimic test bằng cách
        chia train window. Nếu internal_gt_end > train_end → LEAK.
        """
        cfg = get_config()
        assert cfg.windows.internal_gt_start > cfg.windows.ranker_train_cutoff, (
            "internal_gt_start phải > ranker_train_cutoff (tránh overlap với ranker training)"
        )
        assert cfg.windows.internal_gt_end <= cfg.windows.train_end, (
            f"internal_gt_end {cfg.windows.internal_gt_end} > train_end "
            f"{cfg.windows.train_end} — LEAK!"
        )

    def test_no_future_test_window(self):
        """Train end KHÔNG được ≥ 10/04/2026 (BTC ground truth start)."""
        cfg = get_config()
        gt_btc_start = "2026-04-10"
        assert cfg.windows.train_end < gt_btc_start, (
            f"train_end {cfg.windows.train_end} ≥ BTC GT start {gt_btc_start} — LEAK!"
        )


class TestCutoffSemantic:
    """Test semantic: cutoffs làm đúng những gì SQL filter expect."""

    def test_cutoff_format_iso(self):
        """Cutoffs phải format YYYY-MM-DD."""
        cfg = get_config()
        import re
        iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for attr in ["train_start", "train_end",
                     "ranker_train_cutoff",
                     "internal_gt_start", "internal_gt_end"]:
            val = getattr(cfg.windows, attr)
            assert iso.match(val), f"windows.{attr} sai format: {val}"
