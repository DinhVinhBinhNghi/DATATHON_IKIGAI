#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
query_and_make_real_slide_charts_v3.py

All-in-one script:
1) Query real slide metrics from pipeline outputs WITHOUT loading the full parquet into RAM.
2) Export small CSV summaries for Slide 4/5/6.
3) Draw real charts for Canva.

Designed to be run from the EDA project root (eda/)

Main fix vs V2:
- Uses DuckDB parquet_scan instead of pandas.read_parquet for pair_flags_with_dim.parquet.
- Recognizes the real columns:
  n_positive_events, n_direct_contact_events, max_dwell_sec,
  is_qualified_pair, is_qualified_pos_pair.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError as e:
    raise SystemExit(
        "Missing dependency: duckdb\n"
        "Run: pip install duckdb pandas numpy matplotlib pyarrow"
    ) from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "Missing dependency: matplotlib\n"
        "Run: pip install matplotlib"
    ) from e


# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------
NAVY = "#0F2D5C"
ORANGE = "#F97316"
YELLOW = "#FBBF24"
TEAL = "#14B8A6"
GREEN = "#22C55E"
GRAY = "#64748B"
LIGHT = "#F8FAFC"
BORDER = "#E2E8F0"
RED = "#EF4444"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": BORDER,
    "axes.labelcolor": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def line(char: str = "=", n: int = 72) -> None:
    print(char * n)


def section(title: str) -> None:
    print()
    line("=")
    print(title)
    line("=")


def esc(path: Path) -> str:
    """DuckDB-safe path string."""
    return path.as_posix().replace("'", "''")


def pct(x: float | int | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "NA"
    return f"{100 * float(x):.{digits}f}%"


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved: {path.relative_to(ROOT)}")


def find_first(candidates: list[Path], pattern: str | None = None) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    if pattern:
        hits = sorted(ROOT.glob(pattern))
        if hits:
            return hits[0]
    return None


def bool_expr(col: str) -> str:
    """Convert bool/int/string-like flags to 0/1 robustly in DuckDB."""
    return (
        f"CASE "
        f"WHEN {col} IS NULL THEN 0 "
        f"WHEN TRY_CAST({col} AS BOOLEAN) THEN 1 "
        f"WHEN TRY_CAST({col} AS INTEGER) = 1 THEN 1 "
        f"ELSE 0 END"
    )


def build_pair_view(con: duckdb.DuckDBPyConnection, pair_path: Path) -> list[str]:
    # Read only metadata first.
    cols_df = con.execute(f"DESCRIBE SELECT * FROM parquet_scan('{esc(pair_path)}')").df()
    cols = cols_df["column_name"].tolist()
    print("Columns detected:")
    print(", ".join(cols))

    required = [
        "user_id", "item_id", "seller_id", "category", "seller_type", "ad_type",
        "n_pageview", "n_positive_events", "n_direct_contact_events",
        "max_dwell_sec", "is_qualified_pair", "is_qualified_pos_pair",
    ]
    missing = [c for c in required if c not in cols]
    if missing:
        print("\nWARNING: missing expected columns:", missing)
        print("The script will try to continue where possible.")

    # Use only columns needed for slide metrics.
    select_parts = []
    def has(c: str) -> bool:
        return c in cols

    # IDs / dimensions.
    select_parts.append("user_id" if has("user_id") else "NULL::VARCHAR AS user_id")
    select_parts.append("item_id" if has("item_id") else "NULL::VARCHAR AS item_id")
    select_parts.append("seller_id" if has("seller_id") else "NULL::VARCHAR AS seller_id")
    select_parts.append("category" if has("category") else "NULL AS category")
    select_parts.append(
        "COALESCE(NULLIF(CAST(seller_type AS VARCHAR), ''), 'unknown') AS seller_type"
        if has("seller_type") else "'unknown' AS seller_type"
    )
    select_parts.append(
        "COALESCE(NULLIF(CAST(ad_type AS VARCHAR), ''), 'unknown') AS ad_type"
        if has("ad_type") else "'unknown' AS ad_type"
    )

    # Numeric behavior columns.
    select_parts.append(
        "COALESCE(TRY_CAST(n_pageview AS DOUBLE), 0.0) AS n_pageview"
        if has("n_pageview") else "0.0 AS n_pageview"
    )
    select_parts.append(
        "COALESCE(TRY_CAST(n_positive_events AS DOUBLE), 0.0) AS n_positive_events"
        if has("n_positive_events") else "0.0 AS n_positive_events"
    )
    select_parts.append(
        "COALESCE(TRY_CAST(n_direct_contact_events AS DOUBLE), 0.0) AS n_direct_contact_events"
        if has("n_direct_contact_events") else "0.0 AS n_direct_contact_events"
    )
    select_parts.append(
        "COALESCE(TRY_CAST(max_dwell_sec AS DOUBLE), 0.0) AS max_dwell_sec"
        if has("max_dwell_sec") else "0.0 AS max_dwell_sec"
    )

    # Flags.
    if has("is_qualified_pair"):
        select_parts.append(f"{bool_expr('is_qualified_pair')} AS q_pair")
    else:
        select_parts.append("CASE WHEN COALESCE(TRY_CAST(max_dwell_sec AS DOUBLE),0) >= 30 OR COALESCE(TRY_CAST(n_positive_events AS DOUBLE),0) > 0 THEN 1 ELSE 0 END AS q_pair")

    if has("is_qualified_pos_pair"):
        select_parts.append(f"{bool_expr('is_qualified_pos_pair')} AS q_pos_pair")
    else:
        select_parts.append("CASE WHEN COALESCE(TRY_CAST(n_positive_events AS DOUBLE),0) > 0 THEN 1 ELSE 0 END AS q_pos_pair")

    con.execute(f"""
        CREATE OR REPLACE VIEW pf AS
        SELECT
            {", ".join(select_parts)}
        FROM parquet_scan('{esc(pair_path)}')
    """)

    return cols


def category_label(x) -> str:
    mapping = {
        1010: "1010\nRent room",
        1020: "1020\nApartment",
        1030: "1030\nHouse",
        1040: "1040\nLand",
        1050: "1050\nProject",
        "1010": "1010\nRent room",
        "1020": "1020\nApartment",
        "1030": "1030\nHouse",
        "1040": "1040\nLand",
        "1050": "1050\nProject",
    }
    return mapping.get(x, str(x))


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def add_card_title(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=NAVY, pad=16)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10.5, color=GRAY)


def plot_donut(cold_norm: pd.DataFrame, out_path: Path) -> None:
    labels_order = ["no_history", "warm_positive", "browser_only", "positive_no_pageview"]
    color_map = {
        "no_history": ORANGE,
        "warm_positive": NAVY,
        "browser_only": YELLOW,
        "positive_no_pageview": TEAL,
    }
    label_map = {
        "no_history": "No history",
        "warm_positive": "History + contact",
        "browser_only": "Browsing only",
        "positive_no_pageview": "Contact no PV",
    }

    df = cold_norm.set_index("tier").reindex(labels_order).fillna(0).reset_index()
    sizes = df["pct"].to_numpy(dtype=float)
    colors = [color_map[t] for t in df["tier"]]

    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.36, edgecolor="white", linewidth=3),
    )
    ax.text(0, 0.08, f"{sizes[0]:.0f}%", ha="center", va="center",
            fontsize=35, fontweight="bold", color=ORANGE)
    ax.text(0, -0.17, "cold-start\nusers", ha="center", va="center",
            fontsize=12, color=NAVY, linespacing=1.2)

    legend_labels = [
        f"{label_map[t]}  {p:.1f}%"
        for t, p in zip(df["tier"], sizes)
    ]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(0.95, 0.5),
              frameon=False, fontsize=10.5)

    ax.set_title("Cold-start dominates the test users", loc="left",
                 fontsize=16, fontweight="bold", color=NAVY, pad=16)
    ax.text(-1.28, -1.25, "Source: real pipeline output - eda_06_cold_start_tier.csv",
            fontsize=8.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(ROOT)}")


def plot_heatmap(row_pct: pd.DataFrame, out_path: Path) -> None:
    # row_pct index=view_category, columns=contact_category, values percent in [0,100]
    row_pct = row_pct.sort_index().sort_index(axis=1)
    labels_y = [category_label(x) for x in row_pct.index]
    labels_x = [category_label(x) for x in row_pct.columns]
    vals = row_pct.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.7, 5.7), dpi=180)
    im = ax.imshow(vals, vmin=0, vmax=max(100, np.nanmax(vals)), cmap="YlOrBr")

    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_yticklabels(labels_y, fontsize=9)
    ax.set_xlabel("Contact category", fontsize=10, color=NAVY, labelpad=8)
    ax.set_ylabel("Viewed category", fontsize=10, color=NAVY, labelpad=8)
    ax.set_title("Users usually contact within the same category", loc="left",
                 fontsize=15.5, fontweight="bold", color=NAVY, pad=16)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                color = "white" if v >= 50 else NAVY
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    # Grid lines.
    ax.set_xticks(np.arange(-.5, len(labels_x), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels_y), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.025)
    cbar.ax.tick_params(labelsize=8, colors=NAVY)
    cbar.set_label("Row %", fontsize=9, color=NAVY)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(ROOT)}")


def plot_cr_by_seller(cr_df: pd.DataFrame, out_path: Path) -> None:
    df = cr_df.copy()
    df["label"] = df["ad_type"].astype(str) + " x " + df["seller_type"].astype(str)
    df = df.sort_values("cr_d30", ascending=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    bars = ax.barh(df["label"], df["cr_d30"] * 100, color=[ORANGE if "private" in x else NAVY for x in df["label"]])
    ax.set_xlim(0, max(5, df["cr_d30"].max() * 100 * 1.25))
    ax.set_xlabel("D30 contact rate (%)", fontsize=10)
    ax.set_title("Contact rate differs by listing side", loc="left",
                 fontsize=15.5, fontweight="bold", color=NAVY, pad=14)
    ax.grid(axis="x", alpha=0.18)

    for b in bars:
        w = b.get_width()
        ax.text(w + 0.6, b.get_y() + b.get_height() / 2, f"{w:.1f}%",
                va="center", ha="left", fontsize=10.5, fontweight="bold", color=NAVY)

    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(ROOT)}")


def plot_top10_share(conc_df: pd.DataFrame, out_path: Path) -> None:
    df = conc_df[conc_df["seller_group"].isin(["B2B_agent", "C2C_private"])].copy()
    if df.empty:
        df = conc_df.copy()
    df = df.sort_values("top10_share", ascending=True)

    fig, ax = plt.subplots(figsize=(6.7, 4.5), dpi=180)
    labels = df["seller_group"].replace({"B2B_agent": "B2B / agent", "C2C_private": "C2C / private"})
    vals = df["top10_share"] * 100
    colors = [NAVY if "B2B" in lab else ORANGE for lab in labels]
    bars = ax.barh(labels, vals, color=colors, height=0.52)

    ax.set_xlim(0, max(100, vals.max() * 1.12))
    ax.set_xlabel("Share of positive pairs captured by top 10% sellers (%)", fontsize=9.5)
    ax.set_title("Concentration appears on both sides", loc="left",
                 fontsize=15.5, fontweight="bold", color=NAVY, pad=14)
    ax.grid(axis="x", alpha=0.18)

    for b in bars:
        w = b.get_width()
        ax.text(w + 1.0, b.get_y() + b.get_height() / 2, f"{w:.1f}%",
                va="center", ha="left", fontsize=12, fontweight="bold", color=NAVY)

    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(ROOT)}")


def lorenz_points(values: np.ndarray, max_points: int = 800) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if len(values) == 0 or values.sum() <= 0:
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    values = np.sort(values)
    cum = np.cumsum(values)
    y = np.concatenate([[0.0], cum / cum[-1]])
    x = np.linspace(0, 1, len(y))

    # Gini = 1 - 2 * area under Lorenz curve.
    area = np.trapezoid(y, x)
    gini = 1 - 2 * area

    if len(x) > max_points:
        idx = np.unique(np.linspace(0, len(x) - 1, max_points).astype(int))
        x = x[idx]
        y = y[idx]
    return x, y, gini


def plot_lorenz(seller_df: pd.DataFrame, out_path: Path, data_out_dir: Path) -> pd.DataFrame:
    groups = {
        "B2B / agent": seller_df.loc[seller_df["seller_group"].eq("B2B_agent"), "lead_pairs"].to_numpy(),
        "C2C / private": seller_df.loc[seller_df["seller_group"].eq("C2C_private"), "lead_pairs"].to_numpy(),
    }

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=180)
    ax.plot([0, 1], [0, 1], linestyle="--", color=BORDER, linewidth=2, label="Equal distribution")

    colors = {"B2B / agent": NAVY, "C2C / private": ORANGE}
    rows = []
    for name, vals in groups.items():
        if vals.size == 0:
            continue
        x, y, gini = lorenz_points(vals)
        ax.plot(x, y, linewidth=3, color=colors[name], label=f"{name} - Gini {gini:.3f}")
        rows.append({"seller_group": name, "gini": gini, "n_sellers": len(vals), "total_lead_pairs": float(vals.sum())})
        pd.DataFrame({"seller_cum_share": x, "lead_cum_share": y}).to_csv(
            data_out_dir / f"slide6_lorenz_{name.replace(' / ', '_').replace(' ', '_').lower()}_real.csv",
            index=False, encoding="utf-8-sig"
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative seller share", fontsize=10)
    ax.set_ylabel("Cumulative positive-pair share", fontsize=10)
    ax.set_title("Lorenz curve: lead opportunity is concentrated", loc="left",
                 fontsize=15.0, fontweight="bold", color=NAVY, pad=14)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(ROOT)}")
    return pd.DataFrame(rows)


def plot_dwell_hist(hist_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=180)
    ax.bar(hist_df["bin_label"], hist_df["pairs"], color=NAVY, alpha=0.92)
    ax.set_title("D30 filter removes short-dwell browsing noise", loc="left",
                 fontsize=15.0, fontweight="bold", color=NAVY, pad=14)
    ax.set_xlabel("Max dwell time per user-item pair", fontsize=10)
    ax.set_ylabel("Pairs", fontsize=10)
    ax.grid(axis="y", alpha=0.18)
    ax.tick_params(axis="x", rotation=30)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
ROOT = Path.cwd()
OUT_DATA = ROOT / "outputs" / "slide_data_real"
OUT_CHARTS = ROOT / "outputs" / "slide_charts_real"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_CHARTS.mkdir(parents=True, exist_ok=True)

line("=")
print("QUERY + MAKE REAL SLIDE CHARTS V3")
line("=")
print(f"Project root: {ROOT}")
print(f"Data out    : {OUT_DATA.relative_to(ROOT)}")
print(f"Chart out   : {OUT_CHARTS.relative_to(ROOT)}")

pair_path = find_first(
    [
        ROOT / "outputs" / "agg" / "reduced" / "pair_flags_with_dim.parquet",
        ROOT / "outputs" / "intermediates" / "pair_flags_with_dim.parquet",
        ROOT / "outputs" / "agg" / "pair_flags_with_dim.parquet",
    ],
    "outputs/**/pair_flags_with_dim.parquet"
)

if pair_path is None:
    raise SystemExit(
        "\nERROR: Cannot find pair_flags_with_dim.parquet.\n"
        "Run this to locate it:\n"
        "Get-ChildItem outputs -Recurse -File | Where-Object { $_.Name -match 'pair_flags_with_dim' } | Select FullName"
    )

section("0. LOAD pair_flags_with_dim via DuckDB")
print(f"Found pair flags: {pair_path.relative_to(ROOT)}")

con = duckdb.connect(database=":memory:")
con.execute("PRAGMA threads=4")
# Keep memory moderate; DuckDB will stream/aggregate instead of pandas-loading the full file.
try:
    con.execute("PRAGMA memory_limit='8GB'")
except Exception:
    pass

cols = build_pair_view(con, pair_path)

# ---------------------------------------------------------------------
# Slide 4 + core metrics
# ---------------------------------------------------------------------
section("1. CORE METRICS FROM pair_flags")

basic_sql = """
SELECT
    COUNT(*)::DOUBLE AS observed_pairs,
    SUM(CASE WHEN n_pageview > 0 THEN 1 ELSE 0 END)::DOUBLE AS pageview_pairs,
    SUM(q_pair)::DOUBLE AS d30_engaged_pairs,
    SUM(q_pos_pair)::DOUBLE AS d30_positive_pairs,
    SUM(CASE WHEN n_positive_events > 0 THEN 1 ELSE 0 END)::DOUBLE AS raw_positive_pairs,
    SUM(n_positive_events)::DOUBLE AS raw_positive_events,
    COUNT(DISTINCT user_id)::DOUBLE AS observed_users,
    COUNT(DISTINCT CASE WHEN q_pos_pair = 1 THEN user_id END)::DOUBLE AS positive_users,
    COUNT(DISTINCT item_id)::DOUBLE AS observed_items,
    COUNT(DISTINCT CASE WHEN q_pos_pair = 1 THEN item_id END)::DOUBLE AS contacted_items,
    COUNT(DISTINCT seller_id)::DOUBLE AS observed_sellers,
    COUNT(DISTINCT CASE WHEN q_pos_pair = 1 THEN seller_id END)::DOUBLE AS contacted_sellers
FROM pf
"""
basic = con.execute(basic_sql).df().iloc[0].to_dict()

computed = {
    "d30_engagement_rate_over_pageview": safe_div(basic["d30_engaged_pairs"], basic["pageview_pairs"]),
    "d30_contact_rate_over_pageview": safe_div(basic["d30_positive_pairs"], basic["pageview_pairs"]),
    "d30_contact_rate_over_engaged": safe_div(basic["d30_positive_pairs"], basic["d30_engaged_pairs"]),
    "listing_coverage_observed_universe": safe_div(basic["contacted_items"], basic["observed_items"]),
    "seller_coverage_observed_universe": safe_div(basic["contacted_sellers"], basic["observed_sellers"]),
    "user_positive_coverage_observed_universe": safe_div(basic["positive_users"], basic["observed_users"]),
}
basic.update(computed)

metrics_rows = []
for k, v in basic.items():
    metrics_rows.append({"metric": k, "value": v})
metrics_df = pd.DataFrame(metrics_rows)
save_df(metrics_df, OUT_DATA / "real_slide_metrics_summary_v3.csv")

for k in [
    "pageview_pairs", "d30_engaged_pairs", "d30_positive_pairs",
    "d30_engagement_rate_over_pageview", "d30_contact_rate_over_pageview",
    "d30_contact_rate_over_engaged", "listing_coverage_observed_universe"
]:
    v = basic.get(k)
    if "rate" in k or "coverage" in k:
        print(f"  {k}: {pct(v)}")
    else:
        print(f"  {k}: {v:,.0f}")

# ---------------------------------------------------------------------
# Slide 6: CR by ad_type x seller_type
# ---------------------------------------------------------------------
section("2. SLIDE 6 CR BY ad_type x seller_type")

cr_sql = """
SELECT
    ad_type,
    seller_type,
    COUNT(*)::DOUBLE AS observed_pairs,
    SUM(q_pair)::DOUBLE AS d30_engaged_pairs,
    SUM(q_pos_pair)::DOUBLE AS d30_positive_pairs,
    COUNT(DISTINCT item_id)::DOUBLE AS observed_items,
    COUNT(DISTINCT CASE WHEN q_pos_pair = 1 THEN item_id END)::DOUBLE AS contacted_items,
    CASE WHEN SUM(q_pair) > 0 THEN SUM(q_pos_pair)::DOUBLE / SUM(q_pair) ELSE NULL END AS cr_d30,
    CASE WHEN COUNT(*) > 0 THEN SUM(q_pos_pair)::DOUBLE / COUNT(*) ELSE NULL END AS cr_observed
FROM pf
GROUP BY 1, 2
ORDER BY ad_type, seller_type
"""
cr_df = con.execute(cr_sql).df()
save_df(cr_df, OUT_DATA / "slide6_cr_by_seller_real.csv")
print(cr_df.to_string(index=False))
plot_cr_by_seller(cr_df, OUT_CHARTS / "slide6_cr_by_seller_real.png")

# ---------------------------------------------------------------------
# Slide 6: seller concentration + Lorenz
# ---------------------------------------------------------------------
section("3. SLIDE 6 SELLER CONCENTRATION + LORENZ")

seller_sql = """
SELECT
    seller_id,
    CASE
        WHEN seller_type = 'agent' THEN 'B2B_agent'
        WHEN seller_type = 'private' THEN 'C2C_private'
        ELSE 'unknown'
    END AS seller_group,
    ANY_VALUE(seller_type) AS seller_type,
    COUNT(*)::DOUBLE AS observed_pairs,
    SUM(q_pair)::DOUBLE AS d30_engaged_pairs,
    SUM(q_pos_pair)::DOUBLE AS lead_pairs,
    COUNT(DISTINCT item_id)::DOUBLE AS observed_items,
    COUNT(DISTINCT CASE WHEN q_pos_pair = 1 THEN item_id END)::DOUBLE AS contacted_items
FROM pf
WHERE seller_id IS NOT NULL
GROUP BY 1, 2
HAVING SUM(q_pos_pair) > 0
ORDER BY lead_pairs DESC
"""
seller_df = con.execute(seller_sql).df()
save_df(seller_df, OUT_DATA / "slide6_seller_lead_distribution_real.csv")

conc_rows = []
for group, g in seller_df.groupby("seller_group", dropna=False):
    vals = g["lead_pairs"].astype(float).sort_values(ascending=False).to_numpy()
    n = len(vals)
    top_n = max(1, int(math.ceil(0.10 * n)))
    top10_share = float(vals[:top_n].sum() / vals.sum()) if vals.sum() > 0 else float("nan")
    _, _, gini = lorenz_points(vals)
    conc_rows.append({
        "seller_group": group,
        "n_sellers_with_positive": n,
        "total_lead_pairs": float(vals.sum()),
        "top10_n_sellers": top_n,
        "top10_share": top10_share,
        "gini": gini,
    })
conc_df = pd.DataFrame(conc_rows).sort_values("seller_group")
save_df(conc_df, OUT_DATA / "slide6_concentration_summary_real.csv")
print(conc_df.to_string(index=False))

plot_top10_share(conc_df, OUT_CHARTS / "slide6_top10_share_real.png")
lorenz_summary = plot_lorenz(seller_df, OUT_CHARTS / "slide6_lorenz_real.png", OUT_DATA)
save_df(lorenz_summary, OUT_DATA / "slide6_lorenz_summary_real.csv")

# ---------------------------------------------------------------------
# Slide 5: cold-start from CSV
# ---------------------------------------------------------------------
section("4. SLIDE 5 COLD-START DONUT")

cold_path = find_first(
    [
        ROOT / "outputs" / "tables" / "eda_06_cold_start_tier.csv",
        ROOT / "outputs" / "tables" / "cold_start_tier.csv",
    ],
    "outputs/tables/*cold*start*.csv"
)

if cold_path is None:
    print("  SKIP: cold-start CSV not found.")
else:
    print(f"  loaded: {cold_path.relative_to(ROOT)}")
    cold_raw = pd.read_csv(cold_path)
    save_df(cold_raw, OUT_DATA / "slide5_cold_start_raw.csv")

    # Expected columns from your log: has_positive, has_interaction, n_users, pct.
    required = {"has_positive", "has_interaction", "n_users"}
    if required.issubset(cold_raw.columns):
        def to_bool_series(s: pd.Series) -> pd.Series:
            if s.dtype == bool:
                return s
            return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])

        hp = to_bool_series(cold_raw["has_positive"])
        hi = to_bool_series(cold_raw["has_interaction"])
        tiers = np.where(
            hp & hi, "warm_positive",
            np.where(hp & ~hi, "positive_no_pageview",
                     np.where(~hp & hi, "browser_only", "no_history"))
        )
        cold_norm = cold_raw.assign(tier=tiers).groupby("tier", as_index=False)["n_users"].sum()
        total_users = cold_norm["n_users"].sum()
        cold_norm["pct"] = cold_norm["n_users"] / total_users * 100
        save_df(cold_norm, OUT_DATA / "slide5_cold_start_normalized.csv")
        print(cold_norm.to_string(index=False))
        plot_donut(cold_norm, OUT_CHARTS / "slide5_donut_real.png")
    else:
        print(f"  SKIP: cannot normalize cold-start CSV. Columns={list(cold_raw.columns)}")

# ---------------------------------------------------------------------
# Slide 5: category affinity heatmap from CSV
# ---------------------------------------------------------------------
section("5. SLIDE 5 CATEGORY AFFINITY HEATMAP")

heatmap_path = find_first(
    [
        ROOT / "outputs" / "tables" / "eda_A2_category_preference_heatmap.csv",
        ROOT / "outputs" / "tables" / "category_preference_heatmap.csv",
    ],
    "outputs/tables/*category*heatmap*.csv"
)

if heatmap_path is None:
    print("  SKIP: category heatmap CSV not found.")
else:
    print(f"  loaded: {heatmap_path.relative_to(ROOT)}")
    hm_raw = pd.read_csv(heatmap_path)
    save_df(hm_raw, OUT_DATA / "slide5_heatmap_raw.csv")
    print("  columns:", ", ".join(hm_raw.columns.astype(str)))

    if {"view_category", "contact_category", "n_users"}.issubset(hm_raw.columns):
        mat = hm_raw.pivot_table(
            index="view_category",
            columns="contact_category",
            values="n_users",
            aggfunc="sum",
            fill_value=0
        )
        row_pct = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0) * 100
        row_pct.to_csv(OUT_DATA / "slide5_heatmap_row_pct_real.csv", encoding="utf-8-sig")
        print("  saved:", (OUT_DATA / "slide5_heatmap_row_pct_real.csv").relative_to(ROOT))
        diag_vals = []
        for c in row_pct.index:
            if c in row_pct.columns:
                diag_vals.append(row_pct.loc[c, c])
        if diag_vals:
            print(f"  mean diagonal row %: {np.nanmean(diag_vals):.2f}%")
        plot_heatmap(row_pct, OUT_CHARTS / "slide5_heatmap_real.png")
    else:
        print("  SKIP: Cannot infer affinity matrix format from CSV.")

# ---------------------------------------------------------------------
# Appendix: dwell distribution histogram
# ---------------------------------------------------------------------
section("6. APPENDIX DWELL DISTRIBUTION")

hist_sql = """
SELECT
    CASE
        WHEN max_dwell_sec < 5 THEN '0-5s'
        WHEN max_dwell_sec < 10 THEN '5-10s'
        WHEN max_dwell_sec < 30 THEN '10-30s'
        WHEN max_dwell_sec < 60 THEN '30-60s'
        WHEN max_dwell_sec < 180 THEN '1-3m'
        WHEN max_dwell_sec < 600 THEN '3-10m'
        ELSE '10m+'
    END AS bin_label,
    CASE
        WHEN max_dwell_sec < 5 THEN 1
        WHEN max_dwell_sec < 10 THEN 2
        WHEN max_dwell_sec < 30 THEN 3
        WHEN max_dwell_sec < 60 THEN 4
        WHEN max_dwell_sec < 180 THEN 5
        WHEN max_dwell_sec < 600 THEN 6
        ELSE 7
    END AS bin_order,
    COUNT(*)::DOUBLE AS pairs,
    SUM(q_pos_pair)::DOUBLE AS positive_pairs,
    CASE WHEN COUNT(*) > 0 THEN SUM(q_pos_pair)::DOUBLE / COUNT(*) ELSE NULL END AS positive_rate
FROM pf
GROUP BY 1, 2
ORDER BY bin_order
"""
hist_df = con.execute(hist_sql).df()
save_df(hist_df, OUT_DATA / "appendix_dwell_histogram_real.csv")
print(hist_df.to_string(index=False))
plot_dwell_hist(hist_df, OUT_CHARTS / "appendix_A4_2_dwell_defend_real.png")

# ---------------------------------------------------------------------
# Final inventory
# ---------------------------------------------------------------------
section("DONE - GENERATED FILES")
print("\nData CSVs:")
for p in sorted(OUT_DATA.glob("*")):
    print(f"  {p.relative_to(ROOT)}")

print("\nCharts:")
for p in sorted(OUT_CHARTS.glob("*.png")):
    print(f"  {p.relative_to(ROOT)}")

print("\nNext:")
print("1) Send this terminal log back to ChatGPT.")
print("2) Import *_real.png from outputs\\slide_charts_real into Canva.")


