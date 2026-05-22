from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb
import pandas as pd


def fmt_int(x):
    return f"{int(x):,}"


def fmt_pct(x, digits=2):
    return f"{float(x):.{digits}f}%"


def check_required_columns(con, parquet_path: str, required: list[str]) -> list[str]:
    cols_df = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
    ).fetchdf()
    cols = cols_df["column_name"].tolist()

    missing = [c for c in required if c not in cols]
    if missing:
        print("[ERROR] File thiếu cột bắt buộc:", missing)
        print("Các cột có sẵn:", cols)
        sys.exit(1)

    return cols


def compute_q1(con, parquet_path: str) -> pd.DataFrame:
    sql = f"""
    WITH base AS (
        SELECT
            user_id,
            item_id,
            seller_type,
            CAST(is_qualified_pair AS INTEGER) AS is_qualified,
            CAST(is_qualified_pos_pair AS INTEGER) AS is_positive
        FROM read_parquet('{parquet_path}')
        WHERE seller_type IN ('private', 'agent')
    ),
    agg AS (
        SELECT
            seller_type,
            COUNT(DISTINCT item_id) AS n_listings,
            COUNT(*) AS n_pageview_pairs,
            SUM(is_qualified) AS n_qualified_pairs,
            SUM(is_positive) AS n_positive_pairs,
            COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT item_id), 0) AS pageview_per_listing,
            SUM(is_qualified) * 1.0 / NULLIF(COUNT(DISTINCT item_id), 0) AS qualified_per_listing,
            SUM(is_qualified) * 100.0 / NULLIF(COUNT(*), 0) AS qualifying_rate_pct,
            SUM(is_positive) * 100.0 / NULLIF(SUM(is_qualified), 0) AS CR_D30_pct
        FROM base
        GROUP BY seller_type
    )
    SELECT *
    FROM agg
    ORDER BY CASE WHEN seller_type = 'private' THEN 1 ELSE 2 END
    """
    raw = con.execute(sql).fetchdf()

    out = {}
    for _, row in raw.iterrows():
        stype = row["seller_type"]
        out[stype] = {
            "n_listings": int(row["n_listings"]),
            "n_pageview_pairs": int(row["n_pageview_pairs"]),
            "n_qualified_pairs": int(row["n_qualified_pairs"]),
            "n_positive_pairs": int(row["n_positive_pairs"]),
            "pageview_per_listing": round(float(row["pageview_per_listing"]), 2),
            "qualified_per_listing": round(float(row["qualified_per_listing"]), 2),
            "qualifying_rate_pct": round(float(row["qualifying_rate_pct"]), 2),
            "CR_D30_pct": round(float(row["CR_D30_pct"]), 2),
        }

    diag = pd.DataFrame(out)
    diag = diag[["private", "agent"]]

    ratios = []
    for idx in diag.index:
        p = diag.loc[idx, "private"]
        a = diag.loc[idx, "agent"]
        ratios.append("N/A" if p == 0 else f"{a / p:.2f}x")
    diag["ratio_agent_over_private"] = ratios

    return diag


def compute_q2(con, parquet_path: str) -> pd.DataFrame:
    sql = f"""
    WITH listing_dim AS (
        SELECT DISTINCT
            item_id,
            segment,
            seller_type
        FROM read_parquet('{parquet_path}')
        WHERE segment IN ('B2B', 'C2C')
          AND seller_type IN ('private', 'agent')
    ),
    counts AS (
        SELECT
            segment,
            seller_type,
            COUNT(DISTINCT item_id) AS n_listings
        FROM listing_dim
        GROUP BY segment, seller_type
    ),
    pivoted AS (
        SELECT
            segment,
            SUM(CASE WHEN seller_type = 'agent' THEN n_listings ELSE 0 END) AS agent,
            SUM(CASE WHEN seller_type = 'private' THEN n_listings ELSE 0 END) AS private
        FROM counts
        GROUP BY segment
    ),
    with_total AS (
        SELECT
            segment,
            agent,
            private,
            agent + private AS Total
        FROM pivoted

        UNION ALL

        SELECT
            'Total' AS segment,
            SUM(agent) AS agent,
            SUM(private) AS private,
            SUM(agent + private) AS Total
        FROM pivoted
    )
    SELECT *
    FROM with_total
    ORDER BY CASE
        WHEN segment = 'B2B' THEN 1
        WHEN segment = 'C2C' THEN 2
        ELSE 3
    END
    """
    ct = con.execute(sql).fetchdf()
    ct = ct.set_index("segment")

    total = float(ct.loc["Total", "Total"])
    ct["agent_pct"] = (ct["agent"] / total * 100).round(2)
    ct["private_pct"] = (ct["private"] / total * 100).round(2)
    ct["Total_pct"] = (ct["Total"] / total * 100).round(2)

    return ct


def build_markdown(diag: pd.DataFrame, ct: pd.DataFrame) -> str:
    p = diag["private"]
    a = diag["agent"]

    pv_priv = float(p["pageview_per_listing"])
    pv_agent = float(a["pageview_per_listing"])
    cr_priv = float(p["CR_D30_pct"])
    cr_agent = float(a["CR_D30_pct"])
    gap = cr_priv - cr_agent

    if pv_priv > pv_agent * 1.10:
        h1 = (
            f"Có cơ sở ủng hộ một phần: chính chủ có pageview/listing cao hơn "
            f"môi giới ({pv_priv:.1f} vs {pv_agent:.1f})."
        )
    elif pv_agent > pv_priv * 1.10:
        h1 = (
            f"Bác bỏ H1: môi giới có pageview/listing cao hơn chính chủ "
            f"({pv_agent:.1f} vs {pv_priv:.1f}), nên CR thấp không phải do thiếu exposure."
        )
    else:
        h1 = (
            f"Bác bỏ H1: pageview/listing hai nhóm gần ngang nhau "
            f"({pv_priv:.1f} vs {pv_agent:.1f})."
        )

    if gap > 1.0:
        h2 = (
            f"Ủng hộ H2: trong nhóm đã xem nghiêm túc D30, CR chính chủ "
            f"{cr_priv:.1f}% cao hơn môi giới {cr_agent:.1f}% "
            f"(chênh {gap:.1f} điểm phần trăm)."
        )
    elif gap < -1.0:
        h2 = (
            f"Đảo ngược kỳ vọng: CR môi giới {cr_agent:.1f}% cao hơn chính chủ "
            f"{cr_priv:.1f}% trong nhóm D30."
        )
    else:
        h2 = (
            f"H2 yếu: CR D30 hai nhóm gần ngang nhau "
            f"({cr_priv:.1f}% vs {cr_agent:.1f}%)."
        )

    try:
        b2b_agent = int(ct.loc["B2B", "agent"])
        b2b_private = int(ct.loc["B2B", "private"])
        c2c_agent = int(ct.loc["C2C", "agent"])
        c2c_private = int(ct.loc["C2C", "private"])

        b2b_total = b2b_agent + b2b_private
        c2c_total = c2c_agent + c2c_private

        b2b_private_pct = b2b_private / b2b_total * 100 if b2b_total else 0
        c2c_agent_pct = c2c_agent / c2c_total * 100 if c2c_total else 0

        reconcile = (
            f"- Trong nhóm B2B theo project proxy, có {b2b_private_pct:.1f}% là private seller "
            f"({fmt_int(b2b_private)}/{fmt_int(b2b_total)} listing).\n"
            f"- Trong nhóm C2C theo project proxy, có {c2c_agent_pct:.1f}% là agent "
            f"({fmt_int(c2c_agent)}/{fmt_int(c2c_total)} listing).\n"
            "- Vì vậy, project proxy và seller_type là hai lát cắt khác nhau: "
            "một bên nói về loại sản phẩm/dự án, một bên nói về người đăng tin."
        )
    except Exception as e:
        reconcile = f"Không đọc được crosstab: {e}"

    paste = f"""
## Mở rộng — Vì sao môi giới CR thấp hơn chính chủ?

Sau câu hỏi của anh, team kiểm tra thêm 2 khả năng:

- H1 — Có phải môi giới CR thấp vì Chợ Tốt serve ít tin môi giới hơn?
  {h1}

- H2 — Có phải user vẫn ưu tiên chính chủ hơn sau khi đã xem kỹ?
  {h2}

Team chưa có dữ liệu trực tiếp về việc user nhìn thấy seller_type trên giao diện như thế nào,
nên không claim quan hệ nhân quả quá mạnh. Trong slide V3, team nên gọi đây là
một dấu hiệu hành vi quan sát được: ranker cần học được khác biệt giữa chính chủ và môi giới,
thay vì nói cứng rằng “người dùng ghét môi giới”.

## Reconcile Gini — Vì sao 2 cách tách ra số khác nhau?

{reconcile}

Cách đọc nên dùng trong V3:
- Project proxy trả lời câu hỏi: “Tin dự án có làm lệch marketplace không?”
- Seller_type trả lời câu hỏi: “Môi giới hay chính chủ tạo phân phối lead lệch hơn?”
- Hai câu hỏi khác nhau, nên số Gini khác nhau là hợp lý, không phải mâu thuẫn.
"""

    md = []
    md.append("# Audit bổ sung — Seller type diagnostic\n\n")
    md.append("## 1. Bảng số chính\n\n")
    md.append(diag.to_markdown())
    md.append("\n\n")
    md.append("## 2. Crosstab segment × seller_type\n\n")
    md.append(ct.to_markdown())
    md.append("\n\n")
    md.append("## 3. Verdict\n\n")
    md.append(f"### H1 — Algorithm bias\n\n{h1}\n\n")
    md.append(f"### H2 — User behavioral\n\n{h2}\n\n")
    md.append("## 4. Wording gợi ý để paste vào V2 / speaker note\n\n")
    md.append(paste)
    md.append("\n")
    return "".join(md)


def print_summary(diag: pd.DataFrame, ct: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY — Seller type diagnostic")
    print("=" * 72)

    print("\n[Q1] Pair-level exposure vs CR (D30 universe):")
    print(diag.to_string())

    p = diag["private"]
    a = diag["agent"]

    pv_priv = float(p["pageview_per_listing"])
    pv_agent = float(a["pageview_per_listing"])
    cr_priv = float(p["CR_D30_pct"])
    cr_agent = float(a["CR_D30_pct"])
    gap = cr_priv - cr_agent

    print("\nĐọc nhanh:")
    print(f"  - pageview per listing: chính chủ {pv_priv:.2f} vs môi giới {pv_agent:.2f}")
    if pv_priv > pv_agent * 1.10:
        print("    → Chính chủ được exposure cao hơn → H1 có cơ sở.")
    elif pv_agent > pv_priv * 1.10:
        print("    → Môi giới được exposure cao hơn → H1 bị bác.")
    else:
        print("    → Exposure gần ngang nhau → H1 yếu/bị bác.")

    print(f"  - CR D30: chính chủ {cr_priv:.2f}% vs môi giới {cr_agent:.2f}%")
    if gap > 1.0:
        print(f"    → Chính chủ cao hơn {gap:.1f}pp trong nhóm đã xem kỹ → H2 được ủng hộ.")
    elif gap < -1.0:
        print("    → Môi giới CR cao hơn → H2 đảo ngược.")
    else:
        print("    → Gap nhỏ → H2 yếu.")

    print("\n[Q2] Crosstab segment (project proxy) × seller_type:")
    print(ct.to_string())

    print("\n" + "=" * 72)
    print("Đã ghi 3 file:")
    print("  - outputs/audit/seller_type_exposure_vs_cr.csv")
    print("  - outputs/audit/segment_seller_type_crosstab.csv")
    print("  - outputs/audit/seller_type_diagnostic_summary.md")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intermediates",
        type=Path,
        default=Path("outputs/agg/reduced"),
        help="Folder chứa pair_flags_with_dim.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/audit"),
        help="Folder output",
    )
    args = parser.parse_args()

    pair_path = args.intermediates / "pair_flags_with_dim.parquet"
    if not pair_path.exists():
        print(f"[ERROR] Không tìm thấy file: {pair_path}")
        sys.exit(1)

    parquet_path = pair_path.as_posix().replace("'", "''")

    print(f"[INFO] Đọc metadata/parquet bằng DuckDB: {pair_path}")
    print("[INFO] Bản này không load toàn bộ parquet vào RAM.")

    con = duckdb.connect()

    required = [
        "user_id",
        "item_id",
        "seller_type",
        "segment",
        "is_qualified_pair",
        "is_qualified_pos_pair",
    ]
    check_required_columns(con, parquet_path, required)

    print("[INFO] Schema mapping:")
    print("  - qualified = is_qualified_pair")
    print("  - positive D30 = is_qualified_pos_pair")
    print("  - pageview pair = COUNT(*) pair-level")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Compute Q1 — exposure vs CR ...")
    diag = compute_q1(con, parquet_path)

    print("[INFO] Compute Q2 — segment × seller_type crosstab ...")
    ct = compute_q2(con, parquet_path)

    diag_csv = args.out_dir / "seller_type_exposure_vs_cr.csv"
    ct_csv = args.out_dir / "segment_seller_type_crosstab.csv"
    md_path = args.out_dir / "seller_type_diagnostic_summary.md"

    diag.to_csv(diag_csv, encoding="utf-8-sig")
    ct.to_csv(ct_csv, encoding="utf-8-sig")
    md_path.write_text(build_markdown(diag, ct), encoding="utf-8")

    print(f"[OK] Ghi {diag_csv}")
    print(f"[OK] Ghi {ct_csv}")
    print(f"[OK] Ghi {md_path}")

    print_summary(diag, ct)


if __name__ == "__main__":
    main()
