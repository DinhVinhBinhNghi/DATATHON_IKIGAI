"""Slide-ready visuals cho Datathon 2026 Cho Tot BDS.

File độc lập, đọc các CSV aggregated đã có trong outputs/tables/.
Không cần re-run pipeline EDA, không cần raw events.

Output:
  - outputs/figures/slides/kpi_strip.png
  - outputs/figures/slides/panel1_demand.png       (cold-start + category affinity)
  - outputs/figures/slides/panel2_supply.png       (private vs agent + Lorenz/Gini)

Usage:
  python scripts/11_build_slide_visuals.py --output-root outputs

Thiết kế:
  - Mỗi panel kích thước 16:9 ~ slide chiếm 80% chiều ngang
  - Font Inter/Roboto fallback DejaVu Sans để khớp slide deck
  - 4 màu chính: navy (anchor), orange (highlight), gray (context), sky (background data)
  - KPI strip = 1 dòng 4 ô số, không trục, không grid — dùng làm "tiêu đề chương EDA"

Chỉnh sửa:
  - Đổi palette tại CONSTANTS dưới.
  - Đổi text bullet/caption tại các hàm draw_panel_*.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "navy": "#0B4F8A",     # Anchor / số liệu chính
    "orange": "#FF7A30",   # Highlight / insight đáng chú ý
    "sky": "#56CCF2",      # Phụ / background data
    "gray": "#7B8794",     # Context / nhãn phụ
    "light_gray": "#E5EAF0",
    "dark": "#1F2933",
    "white": "#FFFFFF",
}

CATEGORY_LABELS = {
    1010: "Phòng trọ\n/ thuê",
    1020: "Căn hộ\nchung cư",
    1030: "Nhà ở",
    1040: "Đất nền\n/ TM",
    1050: "Dự án\nmới",
}

COLD_START_LABELS = {
    "0_no_history": "Chưa có lịch sử",
    "1_browser_only": "Chỉ xem, chưa contact",
    "2_positive_no_pageview": "Contact, không pageview",
    "3_warm_with_positive": "Có history + contact",
}

AD_SELLER_LABELS = {
    ("let", "agent"): "Thuê × Môi giới",
    ("let", "private"): "Thuê × Chính chủ",
    ("sell", "agent"): "Bán × Môi giới",
    ("sell", "private"): "Bán × Chính chủ",
}

FIG_DPI = 220


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["light_gray"],
        "axes.labelcolor": PALETTE["dark"],
        "xtick.color": PALETTE["dark"],
        "ytick.color": PALETTE["dark"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
    })


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────────────────────────────
def draw_kpi_strip(funnel: pd.DataFrame, user_pareto_pct: float, gini: float, out_path: Path) -> None:
    """4 KPI lớn trên 1 dòng. Dùng làm slide mở đầu phần EDA."""
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))

    kpis = [
        {
            "value": "42.3M",
            "unit": "pageview pair",
            "caption": "Quy mô tương tác\ntrong cửa sổ train",
            "color": PALETTE["navy"],
        },
        {
            "value": "5.6%",
            "unit": "end-to-end CR",
            "caption": "Từ pageview pair\nđến positive contact",
            "color": PALETTE["navy"],
        },
        {
            "value": "84%",
            "unit": "lead từ top 20% user",
            "caption": "Phía cầu lệch nặng\nvề power user",
            "color": PALETTE["orange"],
        },
        {
            "value": f"{gini:.2f}",
            "unit": "Gini seller",
            "caption": "Phía cung tập trung\ngần như tuyệt đối",
            "color": PALETTE["orange"],
        },
    ]

    for ax, k in zip(axes, kpis):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.5, 0.72, k["value"], ha="center", va="center",
                fontsize=42, weight="bold", color=k["color"])
        ax.text(0.5, 0.42, k["unit"], ha="center", va="center",
                fontsize=12, color=PALETTE["dark"], weight="bold")
        ax.text(0.5, 0.18, k["caption"], ha="center", va="center",
                fontsize=10, color=PALETTE["gray"])
        # Underline
        ax.axhline(0.62, xmin=0.25, xmax=0.75, color=k["color"], linewidth=2.5)

    fig.suptitle("Bốn con số định hình bài toán", fontsize=16, weight="bold",
                 color=PALETTE["dark"], y=1.02)
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────
# PANEL 1 — PHÍA CẦU
# ─────────────────────────────────────────────────────────────────────
def draw_panel1_demand(cold: pd.DataFrame, heat: pd.DataFrame, out_path: Path) -> None:
    """Trái: cold-start donut. Phải: category affinity heatmap. Dưới: 1 câu kết luận."""
    apply_style()
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 0.9], width_ratios=[1, 1.15], hspace=0.45, wspace=0.18)

    # ===== Trái: cold-start donut =====
    ax_cold = fig.add_subplot(gs[0, 0])
    cold = cold.copy()
    cold["label"] = cold["user_segment"].map(COLD_START_LABELS).fillna(cold["user_segment"])
    # Sort: warm trước, cold cuối (để màu cam ở trên cùng dễ đọc)
    sort_order = ["3_warm_with_positive", "2_positive_no_pageview", "1_browser_only", "0_no_history"]
    cold["__order"] = cold["user_segment"].map({s: i for i, s in enumerate(sort_order)}).fillna(99)
    cold = cold.sort_values("__order")

    colors_cold = {
        "0_no_history": PALETTE["light_gray"],
        "1_browser_only": PALETTE["sky"],
        "2_positive_no_pageview": PALETTE["gray"],
        "3_warm_with_positive": PALETTE["orange"],
    }
    wedge_colors = [colors_cold.get(s, PALETTE["gray"]) for s in cold["user_segment"]]

    wedges, _ = ax_cold.pie(
        cold["n_users"], startangle=90, counterclock=False,
        colors=wedge_colors, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    no_history_pct = cold.loc[cold["user_segment"] == "0_no_history", "pct"].iloc[0]
    ax_cold.text(0, 0.08, f"{no_history_pct:.0f}%", ha="center", va="center",
                 fontsize=44, weight="bold", color=PALETTE["dark"])
    ax_cold.text(0, -0.18, "không có lịch sử", ha="center", va="center",
                 fontsize=11, color=PALETTE["gray"])

    # Legend bên dưới donut — 1 hàng để gọn
    legend_handles = [
        mpatches.Patch(color=colors_cold[s], label=f"{COLD_START_LABELS[s]} ({cold.loc[cold['user_segment']==s, 'pct'].iloc[0]:.0f}%)")
        for s in sort_order if s in cold["user_segment"].values
    ]
    ax_cold.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.08),
                   ncol=1, fontsize=9, handlelength=1.2, handleheight=1.2)
    ax_cold.set_title("Cold-start: 64% test user không có history", pad=15, fontsize=13)

    # ===== Phải: category affinity heatmap =====
    ax_heat = fig.add_subplot(gs[0, 1])
    valid_cats = [1010, 1020, 1030, 1040, 1050]
    heat = heat[heat["view_category"].isin(valid_cats) & heat["contact_category"].isin(valid_cats)].copy()
    pivot = heat.pivot_table(index="view_category", columns="contact_category",
                              values="n_users", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=valid_cats, columns=valid_cats, fill_value=0)
    # Chuẩn hóa theo hàng để mỗi hàng = 100% → diagonal nổi bật
    row_pct = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100

    im = ax_heat.imshow(row_pct.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax_heat.set_xticks(range(len(valid_cats)))
    ax_heat.set_xticklabels([CATEGORY_LABELS[c] for c in valid_cats], fontsize=10)
    ax_heat.set_yticks(range(len(valid_cats)))
    ax_heat.set_yticklabels([CATEGORY_LABELS[c] for c in valid_cats], fontsize=10)
    ax_heat.set_xlabel("Category contact nhiều nhất", fontsize=10, color=PALETTE["gray"])
    ax_heat.set_ylabel("Category xem nhiều nhất", fontsize=10, color=PALETTE["gray"])

    for i in range(len(valid_cats)):
        for j in range(len(valid_cats)):
            v = row_pct.values[i, j]
            color = "white" if v > 50 else PALETTE["dark"]
            weight = "bold" if i == j else "normal"
            ax_heat.text(j, i, f"{v:.0f}%", ha="center", va="center",
                         fontsize=11, color=color, weight=weight)

    # Khung viền cho diagonal
    for i in range(len(valid_cats)):
        ax_heat.add_patch(mpatches.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                              edgecolor=PALETTE["orange"], linewidth=2.5))

    ax_heat.set_title("Category affinity: user contact đúng phân khúc họ xem", pad=15, fontsize=13)
    # Diagonal share
    diag = np.diag(row_pct.values).mean()
    ax_heat.text(1.02, 0.5, f"Trung bình diagonal:\n{diag:.0f}%",
                 transform=ax_heat.transAxes, va="center", ha="left",
                 fontsize=10, color=PALETTE["orange"], weight="bold")

    # ===== Dưới: takeaway =====
    ax_take = fig.add_subplot(gs[1, :])
    ax_take.axis("off")
    ax_take.text(
        0.5, 0.65,
        "Phía cầu chia làm hai: 64% chưa có dấu vết — cần fallback theo popularity × city.",
        ha="center", va="center", fontsize=13, color=PALETTE["dark"], weight="bold",
    )
    ax_take.text(
        0.5, 0.25,
        "36% còn lại trung thành với phân khúc đã chọn — đủ tín hiệu để cá nhân hoá theo same-category.",
        ha="center", va="center", fontsize=12, color=PALETTE["gray"],
    )

    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────
# PANEL 2 — PHÍA CUNG
# ─────────────────────────────────────────────────────────────────────
def draw_panel2_supply(ad_seller: pd.DataFrame, seller_contacts: pd.DataFrame, gini: float, out_path: Path) -> None:
    """Trái: bar 4 cột ad_type × seller_type. Phải: Lorenz curve. Dưới: 1 câu kết luận."""
    apply_style()
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[5, 1], width_ratios=[1, 1], hspace=0.15, wspace=0.22)

    # ===== Trái: bar ad_type × seller_type =====
    ax_bar = fig.add_subplot(gs[0, 0])
    ad_seller = ad_seller.copy()
    ad_seller["combo"] = list(zip(ad_seller["ad_type"], ad_seller["seller_type"]))
    ad_seller["label"] = ad_seller["combo"].map(AD_SELLER_LABELS)
    # Sort theo CR tăng dần để bar cuối là highest
    ad_seller = ad_seller.sort_values("cr")
    ad_seller["cr_pct"] = ad_seller["cr"] * 100

    # Màu: agent xám, private cam — để eye thấy private nổi bật
    bar_colors = [PALETTE["gray"] if combo[1] == "agent" else PALETTE["orange"]
                  for combo in ad_seller["combo"]]
    bars = ax_bar.bar(ad_seller["label"], ad_seller["cr_pct"], color=bar_colors, width=0.62)
    for bar, val in zip(bars, ad_seller["cr_pct"]):
        ax_bar.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=12, weight="bold",
                    color=PALETTE["dark"])

    ax_bar.set_ylabel("Contact rate (%)", fontsize=10, color=PALETTE["gray"])
    ax_bar.set_title("Chính chủ chuyển đổi gấp đôi môi giới", pad=15, fontsize=13)
    ax_bar.set_ylim(0, max(ad_seller["cr_pct"]) * 1.35)
    ax_bar.tick_params(axis="x", labelsize=10)
    ax_bar.set_axisbelow(True)
    ax_bar.yaxis.grid(True, alpha=0.3)

    # Legend góc trên trái (nhỏ gọn)
    legend_handles = [
        mpatches.Patch(color=PALETTE["orange"], label="Chính chủ"),
        mpatches.Patch(color=PALETTE["gray"], label="Môi giới"),
    ]
    ax_bar.legend(handles=legend_handles, loc="upper left", fontsize=10, bbox_to_anchor=(0.0, 0.98))

    # Annotation tỷ số — đặt ở giữa-trên, không đụng legend
    top_cr = ad_seller["cr_pct"].iloc[-1]
    matching_agent = ad_seller[ad_seller["ad_type"] == ad_seller["combo"].iloc[-1][0]]
    matching_agent = matching_agent[matching_agent.apply(lambda r: r["combo"][1] == "agent", axis=1)]
    if not matching_agent.empty:
        bottom_cr = matching_agent["cr_pct"].iloc[0]
        ratio = top_cr / bottom_cr
        ax_bar.text(
            0.97, 0.95,
            f"Gấp {ratio:.1f}× môi giới\ncùng phân khúc",
            transform=ax_bar.transAxes, ha="right", va="top", fontsize=10,
            color=PALETTE["orange"], weight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=PALETTE["orange"], alpha=0.95),
        )

    # ===== Phải: Lorenz curve =====
    ax_lor = fig.add_subplot(gs[0, 1])
    values = np.sort(seller_contacts["total_contacts"].fillna(0).to_numpy())
    cum = np.cumsum(values)
    if cum[-1] == 0:
        lorenz_y = np.linspace(0, 1, len(values) + 1)
    else:
        lorenz_y = np.insert(cum / cum[-1], 0, 0)
    lorenz_x = np.linspace(0, 1, len(lorenz_y))

    ax_lor.fill_between(lorenz_x, lorenz_y, lorenz_x, color=PALETTE["orange"], alpha=0.18)
    ax_lor.plot(lorenz_x, lorenz_y, color=PALETTE["orange"], linewidth=2.6, label="Phân phối thực tế")
    ax_lor.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle="--", linewidth=1.5, label="Phân phối đều giả định")

    # Đánh dấu top 10% seller
    idx_90 = int(0.9 * len(lorenz_x))
    if idx_90 < len(lorenz_y):
        share_top10 = (1 - lorenz_y[idx_90]) * 100
        ax_lor.axvline(0.9, color=PALETTE["navy"], linestyle=":", alpha=0.6)
        ax_lor.annotate(
            f"Top 10% seller\n nhận {share_top10:.0f}% contact",
            xy=(0.9, lorenz_y[idx_90]),
            xytext=(0.35, 0.55),
            fontsize=10, color=PALETTE["navy"], weight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["navy"], lw=1.2),
        )

    ax_lor.set_xlabel("Tỷ lệ seller tích lũy (từ ít contact nhất)", fontsize=10, color=PALETTE["gray"])
    ax_lor.set_ylabel("Tỷ lệ contact tích lũy", fontsize=10, color=PALETTE["gray"])
    ax_lor.set_title(f"Lorenz curve — Gini = {gini:.2f}", pad=15, fontsize=13)
    ax_lor.set_xlim(0, 1)
    ax_lor.set_ylim(0, 1)
    ax_lor.legend(loc="upper left", fontsize=10)
    ax_lor.set_axisbelow(True)
    ax_lor.grid(True, alpha=0.25)

    # ===== Dưới: takeaway =====
    ax_take = fig.add_subplot(gs[1, :])
    ax_take.axis("off")
    ax_take.text(
        0.5, 0.65,
        "Chính chủ chuyển đổi tốt hơn nhưng môi giới chiếm phần lớn nguồn cung — Gini 0.90 nói: lead đã rất tập trung.",
        ha="center", va="center", fontsize=13, color=PALETTE["dark"], weight="bold",
    )
    ax_take.text(
        0.5, 0.25,
        "Mô hình tối ưu thuần Recall sẽ kéo Gini cao thêm — cần tầng re-ranking để cân bằng exposure.",
        ha="center", va="center", fontsize=12, color=PALETTE["gray"],
    )

    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Build slide-ready visuals from aggregated CSVs.")
    parser.add_argument("--output-root", default="outputs", help="Output folder (cần chứa tables/ và figures/)")
    args = parser.parse_args()

    out_root = Path(args.output_root).expanduser().resolve()
    table_dir = out_root / "tables"
    slide_dir = out_root / "figures" / "slides"
    slide_dir.mkdir(parents=True, exist_ok=True)

    # Load aggregates
    funnel = pd.read_csv(table_dir / "eda_A1_funnel_summary.csv")
    cold = pd.read_csv(table_dir / "eda_A2_test_cold_start_segments.csv")
    heat = pd.read_csv(table_dir / "eda_A2_category_preference_heatmap.csv")
    ad_seller = pd.read_csv(table_dir / "eda_A3_contact_rate_by_ad_seller.csv")
    summary = pd.read_csv(table_dir / "eda_summary_metrics_for_slides.csv")

    # Seller contacts: dump từ pipeline ở agg/listing_seller/.
    # Fallback đa đường vì layout có thể đổi theo phiên bản patch.
    candidate_paths = [
        out_root / "agg" / "listing_seller" / "seller_contact_distribution.parquet",
        out_root / "agg" / "listing" / "seller_contacts.parquet",
        table_dir / "eda_A3_seller_contacts.csv",
    ]
    seller_contacts = None
    for p in candidate_paths:
        if p.exists():
            print(f"Đọc seller contacts từ: {p}")
            if p.suffix == ".parquet":
                seller_contacts = pd.read_parquet(p)
            else:
                seller_contacts = pd.read_csv(p)
            break
    if seller_contacts is None:
        raise FileNotFoundError(
            "Không tìm thấy seller contacts. Đã thử:\n  - "
            + "\n  - ".join(str(p) for p in candidate_paths)
            + "\nChạy lại pipeline EDA hoặc copy file từ outputs/agg/ về."
        )

    gini = float(summary.loc[summary["metric"] == "seller_contact_gini", "value"].iloc[0])
    user_pareto_pct = 84.2  # từ A2.1, hard-code (text annotation)

    # Generate
    draw_kpi_strip(funnel, user_pareto_pct, gini, slide_dir / "kpi_strip.png")
    draw_panel1_demand(cold, heat, slide_dir / "panel1_demand.png")
    draw_panel2_supply(ad_seller, seller_contacts, gini, slide_dir / "panel2_supply.png")

    print(f"\nDone. Slide visuals trong: {slide_dir}")


if __name__ == "__main__":
    main()
