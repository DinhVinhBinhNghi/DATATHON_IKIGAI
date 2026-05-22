"""
verify_b2b_c2c_split.py
=======================
Sanity checks for B2B vs C2C segment split (item #2 trong feedback mentor v2).

Mục đích: sau khi chạy pipeline với patch item #2, verify:

1. AUDIT HỢP LÝ: tổng B2B + C2C = tổng listings; tin có project_id mà KHÔNG ở
   cat 1050 = đo lường "B2B ẩn" mà mentor nghi.

2. CONSISTENCY VỚI ITEM #1: B2B + C2C qualified_pairs sum lên ≈ tổng qualified
   pairs (từ funnel A1 N2). Weighted avg CR D30 across segments khớp CR D30
   overall.

3. GINI D30 vs GINI SNAPSHOT: info table. Nếu lệch nhiều → universe matter
   (cần defend khi thuyết trình). Nếu agree → concentration là property thật.

4. RANGE: mọi CR ∈ [0, 100]%, Gini ∈ [0, 1], volume share ∈ [0, 1].

5. VOLUME SANITY: B2B và C2C phải đều có data (> 0 qualified pairs). Nếu 1 segment
   rỗng → bug ở proxy hoặc data thực tế lệch hẳn về 1 phía.

Cách dùng:
    python verify_b2b_c2c_split.py --tables-dir outputs/tables

Exit code:
    0 = pass
    1 = fail
    2 = missing file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ABS_PP_TOLERANCE = 0.1


REQUIRED_FILES = {
    "audit":          "eda_A3_b2b_c2c_proxy_audit.csv",
    "overall":        "eda_A3_b2b_c2c_overall_d30.csv",
    "ad_seller":      "eda_A3_b2b_c2c_ad_seller_d30.csv",
    "category":       "eda_A3_b2b_c2c_category_d30.csv",
    "gini_d30":       "eda_A3_b2b_c2c_gini_d30.csv",
    "gini_snapshot":  "eda_A3_b2b_c2c_gini_snapshot.csv",
    "comparison":     "eda_A3_b2b_vs_c2c_comparison.csv",
    "funnel":         "eda_A1_funnel_summary.csv",
    "summary":        "eda_summary_metrics_for_slides.csv",
}


def load_required(tables_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    missing = []
    for key, fname in REQUIRED_FILES.items():
        path = tables_dir / fname
        if not path.exists():
            missing.append(fname)
            continue
        out[key] = pd.read_csv(path)
    if missing:
        print("\n✗ THIẾU FILE — chạy lại pipeline trước:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(2)
    return out


def check_audit(data: dict[str, pd.DataFrame]) -> list[str]:
    errors = []
    audit = data["audit"]

    total_listings = int(audit["n_listings"].sum())
    sum_b2b = int(audit["n_b2b"].sum())
    sum_c2c = int(audit["n_c2c"].sum())
    if sum_b2b + sum_c2c != total_listings:
        errors.append(f"[FAIL] B2B + C2C = {sum_b2b + sum_c2c:,} != total {total_listings:,}")

    # B2B "ẩn": tin có project_id mà KHÔNG phải cat 1050
    hidden_b2b_rows = audit[audit["category"] != 1050]
    hidden_b2b_count = int(hidden_b2b_rows["n_with_project_id"].sum())
    hidden_b2b_pct = 100 * hidden_b2b_count / max(total_listings, 1)

    print(f"\n  Tổng listings: {total_listings:,}")
    print(f"  B2B (cat 1050 OR project_id NOT NULL): {sum_b2b:,} ({100*sum_b2b/total_listings:.2f}%)")
    print(f"  C2C: {sum_c2c:,} ({100*sum_c2c/total_listings:.2f}%)")
    print(f"\n  B2B 'ẩn' (project_id NOT NULL nhưng category != 1050): "
          f"{hidden_b2b_count:,} ({hidden_b2b_pct:.3f}% tổng listings)")
    if hidden_b2b_pct > 1.0:
        print(f"  → MENTOR ĐÚNG: có {hidden_b2b_pct:.1f}% tin dự án ẩn trong cat 1010-1040.")
        print(f"     Proxy P3 bắt được nhóm này; nếu chỉ dùng cat==1050 sẽ miss.")
    else:
        print(f"  → P3 ≈ P1: rất ít B2B ẩn ngoài cat 1050. Có thể defend dùng cat==1050 cho gọn.")

    print(f"\n  Chi tiết per-category:")
    audit_show = audit[["category", "category_name", "n_listings",
                        "n_with_project_id", "pct_with_project_id", "pct_b2b"]].copy()
    print(audit_show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return errors


def check_consistency_with_item1(data: dict[str, pd.DataFrame]) -> list[str]:
    """B2B + C2C qualified_pairs sum vs item #1 universe."""
    errors = []
    overall = data["overall"]
    funnel = data["funnel"].iloc[0]

    total_q_seg = int(overall["qualified_pairs"].sum())
    total_qp_seg = int(overall["qualified_pos_pairs"].sum())
    N2_funnel = int(funnel["N2_serious_pageview_pairs"])
    N3_funnel = int(funnel["N3_positive_pairs_inside_serious_pv"])

    # B2B/C2C union là pair_dim_glob WHERE segment IN ('B2B','C2C')
    # tức là pair có item_id JOIN được dim_listing. Lệch <3% là expected (item ngoài dim).
    pct_drop_q = 100 * (N2_funnel - total_q_seg) / max(N2_funnel, 1)
    pct_drop_qp = 100 * (N3_funnel - total_qp_seg) / max(N3_funnel, 1)

    print(f"\n  Universe consistency: B2B+C2C sum vs funnel A1")
    print(f"    qualified_pairs:     B2B+C2C={total_q_seg:,} / funnel N2={N2_funnel:,} "
          f"({100-pct_drop_q:.2f}% coverage)")
    print(f"    qualified_pos_pairs: B2B+C2C={total_qp_seg:,} / funnel N3={N3_funnel:,} "
          f"({100-pct_drop_qp:.2f}% coverage)")

    if pct_drop_q > 5.0:
        errors.append(f"[FAIL] B2B+C2C qualified_pairs drop {pct_drop_q:.2f}% so với funnel — quá cao")

    # Weighted-avg CR D30 across segments vs funnel CR D30 overall
    cr_overall_funnel = float(funnel["P2_positive_over_serious_pct"])
    if total_q_seg > 0:
        weighted_cr = 100 * (overall["cr_d30"] * overall["qualified_pairs"]).sum() / total_q_seg
        diff = cr_overall_funnel - weighted_cr
        print(f"\n  CR D30 weighted-avg (B2B + C2C): {weighted_cr:.3f}%")
        print(f"  CR D30 overall (funnel):         {cr_overall_funnel:.3f}%")
        print(f"  Diff: {diff:+.4f}pp (tolerance: ±{ABS_PP_TOLERANCE}pp + coverage loss)")
        # Tolerance lớn hơn item #1 vì có universe loss
        if abs(diff) > ABS_PP_TOLERANCE * 5:
            errors.append(f"[FAIL] CR D30 weighted-avg lệch funnel quá nhiều: {diff:+.4f}pp")

    return errors


def check_segment_volumes(data: dict[str, pd.DataFrame]) -> list[str]:
    """Cả B2B và C2C đều phải có volume > 0."""
    errors = []
    overall = data["overall"]
    for seg in ["B2B", "C2C"]:
        sub = overall[overall["segment"] == seg]
        if sub.empty:
            errors.append(f"[FAIL] Segment {seg} không xuất hiện trong overall — bug ở segment column?")
            continue
        q = int(sub.iloc[0]["qualified_pairs"])
        if q == 0:
            errors.append(f"[FAIL] Segment {seg} có 0 qualified pairs — không thể split")
    return errors


def check_ranges_and_health(data: dict[str, pd.DataFrame]) -> list[str]:
    """CR ∈ [0,1], Gini ∈ [0,1], volume_share ∈ [0,1]."""
    errors = []

    for r in data["overall"].itertuples(index=False):
        if not (0 <= r.cr_d30 <= 1):
            errors.append(f"[FAIL] overall {r.segment}: cr_d30={r.cr_d30} ngoài [0,1]")

    for r in data["gini_d30"].itertuples(index=False):
        if not (0 <= r.gini_d30 <= 1):
            errors.append(f"[FAIL] gini_d30 {r.segment}: {r.gini_d30} ngoài [0,1]")

    for r in data["ad_seller"].itertuples(index=False):
        if not (0 <= r.cr_d30 <= 1):
            errors.append(f"[FAIL] ad_seller {r.segment}/{r.ad_type}/{r.seller_type}: cr_d30={r.cr_d30}")
        if not (0 <= r.volume_share_within_segment <= 1.0001):
            errors.append(f"[FAIL] ad_seller {r.segment}/{r.ad_type}/{r.seller_type}: "
                          f"volume_share={r.volume_share_within_segment} ngoài [0,1]")
    return errors


def print_gini_comparison(data: dict[str, pd.DataFrame]) -> None:
    """Side-by-side D30 vs snapshot Gini — info only, không fail."""
    print("\n" + "=" * 72)
    print("GINI D30 vs GINI SNAPSHOT — INFO ONLY")
    print("=" * 72)

    d30 = data["gini_d30"][["segment", "gini_d30", "n_sellers"]].rename(columns={"n_sellers": "n_sellers_d30"})
    snap = data["gini_snapshot"][["segment", "gini_snapshot", "n_sellers_with_contact"]].rename(
        columns={"n_sellers_with_contact": "n_sellers_snap"}
    )
    merged = d30.merge(snap, on="segment", how="outer")
    merged["delta"] = merged["gini_d30"] - merged["gini_snapshot"]
    print(merged.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n  Diễn giải:")
    print("    - Gini D30: phân phối qualified_pos_pairs theo seller (pair-level clickstream).")
    print("    - Gini snapshot: phân phối total_contacts theo seller (view-level legacy).")
    print("    - Nếu delta nhỏ (< 0.05): concentration là property marketplace, không phải artifact universe.")
    print("    - Nếu delta lớn: universe matter — defend lý do chọn D30 cho slide chính.")


def print_marketplace_health(data: dict[str, pd.DataFrame]) -> None:
    """Top-K share table cho slide 6 marketplace health."""
    print("\n" + "=" * 72)
    print("MARKETPLACE HEALTH — Top-K seller share trên D30 (cho slide 6)")
    print("=" * 72)
    show = data["gini_d30"].copy()
    show = show[show["segment"] != "ALL"]
    cols = ["segment", "n_sellers", "gini_d30", "top10_seller_share",
            "top1pct_seller_share", "top10pct_seller_share"]
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_ad_seller_per_segment(data: dict[str, pd.DataFrame]) -> None:
    """Test mentor hypothesis: agent dominate trong B2B."""
    print("\n" + "=" * 72)
    print("ad×seller PER SEGMENT — Test 'agent dominate B2B' hypothesis")
    print("=" * 72)
    df = data["ad_seller"].copy()
    df["cr_d30_pct"] = df["cr_d30"] * 100
    df["volume_share_pct"] = df["volume_share_within_segment"] * 100
    show = df[["segment", "ad_type", "seller_type", "qualified_pairs", "cr_d30_pct", "volume_share_pct"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main():
    parser = argparse.ArgumentParser(description="Verify B2B vs C2C split consistency")
    parser.add_argument("--tables-dir", type=Path, required=True)
    args = parser.parse_args()

    tables_dir = args.tables_dir.expanduser().resolve()
    if not tables_dir.is_dir():
        print(f"✗ {tables_dir} không phải directory")
        sys.exit(2)

    print(f"→ Verify B2B/C2C tại: {tables_dir}\n")
    data = load_required(tables_dir)
    print("✓ Đọc đầy đủ file CSV cần thiết\n")

    all_errors: list[str] = []

    print("─" * 72)
    print("CHECK 1 — AUDIT cross-tab cat × project_id")
    print("─" * 72)
    all_errors.extend(check_audit(data))

    print("\n" + "─" * 72)
    print("CHECK 2 — CONSISTENCY VỚI ITEM #1 (B2B+C2C sum vs funnel A1)")
    print("─" * 72)
    all_errors.extend(check_consistency_with_item1(data))

    print("\n" + "─" * 72)
    print("CHECK 3 — VOLUME SANITY")
    print("─" * 72)
    seg_errors = check_segment_volumes(data)
    all_errors.extend(seg_errors)
    if not seg_errors:
        print("  [OK] Cả B2B và C2C có volume > 0")

    print("\n" + "─" * 72)
    print("CHECK 4 — RANGE")
    print("─" * 72)
    range_errors = check_ranges_and_health(data)
    all_errors.extend(range_errors)
    if not range_errors:
        print("  [OK] CR, Gini, volume_share đều trong dải hợp lý")

    print_gini_comparison(data)
    print_marketplace_health(data)
    print_ad_seller_per_segment(data)

    print("\n" + "=" * 72)
    if all_errors:
        print(f"✗ {len(all_errors)} CHECK FAILED:")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("✓ TẤT CẢ CHECK PASS — B2B/C2C split nhất quán.")
        print("  Sẵn sàng dùng cho slide 6 (Marketplace Health với segment split).")
        sys.exit(0)


if __name__ == "__main__":
    main()
