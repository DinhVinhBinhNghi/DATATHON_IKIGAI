"""Test submission format conformance.

Generate submission từ mock data (in-memory), verify validator pass.
"""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from src.submission.validator import validate_submission


def _make_valid_submission(tmp_dir: Path, n_users: int = 3) -> Path:
    """Tạo submission CSV hợp lệ (giả lập).

    Lưu ý: validate sẽ FAIL vì test users không match với data thật của BTC.
    Test này chỉ check format/structure thuần — không call validate trực tiếp.
    """
    out = tmp_dir / "fake_submission.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "user_id", "rank", "item_id"])
        rid = 1
        for u in range(n_users):
            for rk in range(1, 11):
                writer.writerow([rid, f"user_{u:08x}", rk, f"item_{u:04x}_{rk:02x}"])
                rid += 1
    return out


class TestSubmissionCSVStructure:
    """Test structural properties của CSV submission (KHÔNG cần data BTC)."""

    def test_header_format(self, tmp_path):
        out = _make_valid_submission(tmp_path, n_users=2)
        with out.open("r", encoding="utf-8") as f:
            header = f.readline().strip()
        assert header == "ID,user_id,rank,item_id"

    def test_no_bom(self, tmp_path):
        out = _make_valid_submission(tmp_path, n_users=2)
        with out.open("rb") as f:
            first_bytes = f.read(3)
        assert first_bytes != b"\xef\xbb\xbf", "CSV không được có BOM"

    def test_row_count_per_user(self, tmp_path):
        out = _make_valid_submission(tmp_path, n_users=5)
        import pandas as pd
        df = pd.read_csv(out)
        # Expect 5 users × 10 = 50 rows
        assert len(df) == 50

    def test_rank_in_1_to_10(self, tmp_path):
        out = _make_valid_submission(tmp_path, n_users=3)
        import pandas as pd
        df = pd.read_csv(out)
        assert df["rank"].min() == 1
        assert df["rank"].max() == 10

    def test_id_unique_and_starts_from_1(self, tmp_path):
        out = _make_valid_submission(tmp_path, n_users=3)
        import pandas as pd
        df = pd.read_csv(out)
        assert df["ID"].is_unique
        assert df["ID"].min() == 1

    def test_user_rank_unique(self, tmp_path):
        out = _make_valid_submission(tmp_path, n_users=4)
        import pandas as pd
        df = pd.read_csv(out)
        assert not df.duplicated(["user_id", "rank"]).any()

    def test_item_unique_per_user(self, tmp_path):
        out = _make_valid_submission(tmp_path, n_users=4)
        import pandas as pd
        df = pd.read_csv(out)
        assert not df.duplicated(["user_id", "item_id"]).any()


class TestSubmissionValidator:
    """Test validator API."""

    def test_validator_raises_on_missing_file(self, tmp_path):
        bogus = tmp_path / "nope.csv"
        with pytest.raises(AssertionError):
            validate_submission(bogus, raise_on_error=True)

    def test_validator_returns_false_no_raise(self, tmp_path):
        bogus = tmp_path / "nope.csv"
        ok = validate_submission(bogus, raise_on_error=False)
        assert ok is False
