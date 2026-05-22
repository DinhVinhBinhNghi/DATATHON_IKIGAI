"""
audit_other_interaction.py
==========================
Item #5 — investigate what `other_interaction` event_type actually is.

CONTEXT:
- A1.3 donut chart shows other_interaction = 94.3% of all positive events.
- Funnel CR D30 = 37% relies heavily on other_interaction being a "real" engagement.
- If other_interaction is noise/tracking/passive event, the entire v2 story collapses.
- Mentor and BGK will ask. Need defendable answer.

AUDIT STRATEGY (4 angles):
  A. Attribute distribution: surface, device, position, query, is_contact
  B. Time/session pattern: dwell_time_sec, frequency per session
  C. Direct-contact-only CR backup: compute CR using only 4 unambiguous events
  D. is_contact flag cross-check: does BTC's own flag agree with event_type?

SAMPLING:
  Default 30/500 files (~6%). Adequate for qualitative audit (finding patterns).
  If sample reveals ambiguity, re-run with --n-files 500 for full scan.

USAGE:
  python audit_other_interaction.py \\
      --data-root C:/Datathon_Data \\
      --output-dir outputs/audit \\
      --n-files 30

OUTPUT (in --output-dir):
  audit_other_interaction_summary.csv      — top-line findings (one row per question)
  audit_oi_by_surface.csv                  — Angle A1: % per surface
  audit_oi_by_device.csv                   — Angle A2: % per device
  audit_oi_by_position.csv                 — Angle A3: distribution of position
  audit_oi_is_contact_crosstab.csv         — Angle D: event_type × is_contact flag
  audit_oi_dwell_distribution.csv          — Angle B1: dwell stats
  audit_oi_dc_vs_oi_cr_d30.csv             — Angle C: DC-only CR D30 vs full CR D30
  audit_oi_session_pattern.csv             — Angle B2: events per session distribution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


POSITIVE_EVENTS = [
    "view_phone",
    "contact_chat",
    "contact_zalo",
    "contact_sms",
    "other_interaction",
]
DIRECT_CONTACT_EVENTS = ["view_phone", "contact_chat", "contact_zalo", "contact_sms"]

# Columns needed for audit. Loading subset reduces I/O significantly.
AUDIT_COLUMNS = [
    "user_id", "session_id", "item_id", "event_type",
    "surface", "device", "position", "is_contact", "dwell_time_sec",
    "event_ts", "query",
]


def select_sample_files(events_dir: Path, n_files: int) -> list[Path]:
    """Pick N files uniformly across the directory listing.

    Files in fact_user_events are named with sequential offset suffixes
    (datathon_fact_user_events-NNNNNNNN.parquet). Uniform sampling across
    the sorted list avoids time-window bias if files happen to be partitioned
    chronologically.
    """
    all_files = sorted(events_dir.glob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No .parquet files in {events_dir}")
    if n_files >= len(all_files):
        return all_files
    indices = np.linspace(0, len(all_files) - 1, n_files, dtype=int)
    return [all_files[i] for i in indices]


def load_audit_sample(files: list[Path]) -> pd.DataFrame:
    """Load only positive events from sample files, with audit columns only."""
    print(f"  Loading {len(files)} files with push-down filter event_type IN positive events...")
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    available_cols = set(dataset.schema.names)
    cols_to_load = [c for c in AUDIT_COLUMNS if c in available_cols]
    missing = set(AUDIT_COLUMNS) - available_cols
    if missing:
        print(f"  [WARN] columns not in schema, skipping: {missing}")

    # Push-down filter: only positive events
    filt = pc.is_in(pc.field("event_type"), value_set=pa.array(POSITIVE_EVENTS))
    table = dataset.to_table(columns=cols_to_load, filter=filt)
    df = table.to_pandas()
    print(f"  Loaded {len(df):,} positive event rows.")
    return df


def angle_a_attribute_distribution(df: pd.DataFrame, out_dir: Path) -> dict:
    """Angle A: how does other_interaction distribute across surface/device/position?

    Output: side-by-side share of each attribute value for other_interaction
    vs direct-contact events. Large discrepancies hint at "OI is a different
    kind of thing" (e.g., concentrated on a few surfaces = a specific feature).
    """
    print("\n  [Angle A] Attribute distribution...")
    findings = {}

    # A1 — Surface
    if "surface" in df.columns:
        # Split into 2 groups: OI vs DC
        df["event_group"] = np.where(
            df["event_type"] == "other_interaction", "other_interaction", "direct_contact"
        )
        surface_dist = (
            df.groupby(["event_group", "surface"])
              .size()
              .unstack(fill_value=0)
        )
        # % within group
        surface_pct = surface_dist.div(surface_dist.sum(axis=1), axis=0) * 100
        # Long format for CSV
        rows = []
        for group in surface_pct.index:
            for surf in surface_pct.columns:
                rows.append({
                    "event_group": group,
                    "surface": surf,
                    "n_events": int(surface_dist.loc[group, surf]),
                    "pct_within_group": float(surface_pct.loc[group, surf]),
                })
        df_surf = pd.DataFrame(rows).sort_values(["event_group", "pct_within_group"], ascending=[True, False])
        df_surf.to_csv(out_dir / "audit_oi_by_surface.csv", index=False)

        # Compute top-3 surface for OI to surface concentration
        oi_surf = df_surf[df_surf["event_group"] == "other_interaction"].head(3)
        findings["A1_oi_top3_surfaces"] = ", ".join(
            f"{r.surface} ({r.pct_within_group:.1f}%)" for r in oi_surf.itertuples(index=False)
        )
        findings["A1_oi_top1_surface_share"] = float(oi_surf.iloc[0]["pct_within_group"]) if len(oi_surf) else np.nan

    # A2 — Device
    if "device" in df.columns:
        dev_dist = df.groupby(["event_group", "device"]).size().unstack(fill_value=0)
        dev_pct = dev_dist.div(dev_dist.sum(axis=1), axis=0) * 100
        rows = []
        for group in dev_pct.index:
            for dev in dev_pct.columns:
                rows.append({
                    "event_group": group, "device": dev,
                    "n_events": int(dev_dist.loc[group, dev]),
                    "pct_within_group": float(dev_pct.loc[group, dev]),
                })
        pd.DataFrame(rows).sort_values(["event_group", "pct_within_group"], ascending=[True, False]).to_csv(
            out_dir / "audit_oi_by_device.csv", index=False
        )

    # A3 — Position (where in feed/search results)
    if "position" in df.columns:
        # Position is often sparse, use bucketing
        df["position_bucket"] = pd.cut(
            df["position"].fillna(-1),
            bins=[-2, -0.5, 0.5, 5, 10, 20, 50, np.inf],
            labels=["null", "0", "1-5", "6-10", "11-20", "21-50", "50+"],
            include_lowest=True,
        )
        pos_dist = df.groupby(["event_group", "position_bucket"], observed=True).size().unstack(fill_value=0)
        pos_pct = pos_dist.div(pos_dist.sum(axis=1), axis=0) * 100
        rows = []
        for group in pos_pct.index:
            for pos in pos_pct.columns:
                rows.append({
                    "event_group": group, "position_bucket": str(pos),
                    "n_events": int(pos_dist.loc[group, pos]),
                    "pct_within_group": float(pos_pct.loc[group, pos]),
                })
        pd.DataFrame(rows).to_csv(out_dir / "audit_oi_by_position.csv", index=False)

    # A4 — Query (only meaningful for search-driven events)
    if "query" in df.columns:
        df["has_query"] = df["query"].notna() & (df["query"].astype(str).str.len() > 0)
        q_pct = df.groupby("event_group")["has_query"].mean() * 100
        findings["A4_oi_pct_with_query"] = float(q_pct.get("other_interaction", np.nan))
        findings["A4_dc_pct_with_query"] = float(q_pct.get("direct_contact", np.nan))

    return findings


def angle_b_time_pattern(df: pd.DataFrame, out_dir: Path) -> dict:
    """Angle B: dwell time and session-level frequency.

    Logic:
    - If OI has dwell_time_sec >> 0, it's an event AFTER reading content (real engagement).
    - If OI has dwell_time_sec = 0 mostly, it's a passive/tracking event.
    - High freq per session for OI (vs DC ~1-2/session) hints at noise.
    """
    print("\n  [Angle B] Time/session pattern...")
    findings = {}

    # B1 — Dwell distribution per event_group
    if "dwell_time_sec" in df.columns:
        # Clip to 1hr cap (timer artifact)
        df["dwell_clipped"] = df["dwell_time_sec"].clip(lower=0, upper=3600)
        dwell_stats_rows = []
        for group, g in df.groupby("event_group"):
            d = g["dwell_clipped"].dropna()
            dwell_stats_rows.append({
                "event_group": group,
                "n_events": len(g),
                "n_with_dwell": d.notna().sum(),
                "pct_dwell_zero": float((d == 0).mean() * 100) if len(d) else np.nan,
                "pct_dwell_under_5s": float((d < 5).mean() * 100) if len(d) else np.nan,
                "pct_dwell_under_30s": float((d < 30).mean() * 100) if len(d) else np.nan,
                "median_dwell_sec": float(d.median()) if len(d) else np.nan,
                "p75_dwell_sec": float(d.quantile(0.75)) if len(d) else np.nan,
                "p95_dwell_sec": float(d.quantile(0.95)) if len(d) else np.nan,
            })
        pd.DataFrame(dwell_stats_rows).to_csv(out_dir / "audit_oi_dwell_distribution.csv", index=False)

        # Pull a key finding for summary
        oi_row = next((r for r in dwell_stats_rows if r["event_group"] == "other_interaction"), None)
        if oi_row:
            findings["B1_oi_median_dwell"] = oi_row["median_dwell_sec"]
            findings["B1_oi_pct_dwell_zero"] = oi_row["pct_dwell_zero"]
            findings["B1_oi_pct_dwell_under_5s"] = oi_row["pct_dwell_under_5s"]
        dc_row = next((r for r in dwell_stats_rows if r["event_group"] == "direct_contact"), None)
        if dc_row:
            findings["B1_dc_median_dwell"] = dc_row["median_dwell_sec"]
            findings["B1_dc_pct_dwell_zero"] = dc_row["pct_dwell_zero"]

    # B2 — Events per session per group
    if "session_id" in df.columns:
        sess_freq = (
            df.groupby(["session_id", "event_group"])
              .size()
              .reset_index(name="n_events")
        )
        freq_stats_rows = []
        for group, g in sess_freq.groupby("event_group"):
            counts = g["n_events"]
            freq_stats_rows.append({
                "event_group": group,
                "n_sessions_with_event": len(counts),
                "median_events_per_session": float(counts.median()),
                "mean_events_per_session": float(counts.mean()),
                "p95_events_per_session": float(counts.quantile(0.95)),
                "max_events_per_session": int(counts.max()),
            })
        pd.DataFrame(freq_stats_rows).to_csv(out_dir / "audit_oi_session_pattern.csv", index=False)

        oi_sess = next((r for r in freq_stats_rows if r["event_group"] == "other_interaction"), None)
        if oi_sess:
            findings["B2_oi_median_per_session"] = oi_sess["median_events_per_session"]
            findings["B2_oi_max_per_session"] = oi_sess["max_events_per_session"]
        dc_sess = next((r for r in freq_stats_rows if r["event_group"] == "direct_contact"), None)
        if dc_sess:
            findings["B2_dc_median_per_session"] = dc_sess["median_events_per_session"]

    return findings


def angle_c_dc_only_cr(events_dir: Path, files: list[Path], out_dir: Path) -> dict:
    """Angle C: backup CR D30 using DIRECT_CONTACT_EVENTS only.

    Strategy: re-scan pageview + direct contact events at pair-level, compute
    qualified pair → DC-positive pair conversion. Compare against full CR D30
    (37%) to show how much story depends on other_interaction.

    Note: this re-aggregates from raw events on the sample. Numbers will be
    sample-level, not directly comparable to the 37% from full pipeline.
    Workaround: compute BOTH metrics on the same sample for fair ratio.
    """
    print("\n  [Angle C] Direct-contact-only CR D30 backup metric...")
    findings = {}

    # Need pageview + positive events for this pair-level computation
    dataset = ds.dataset([str(f) for f in files], format="parquet")
    cols = ["user_id", "item_id", "event_type", "dwell_time_sec"]
    available = set(dataset.schema.names)
    cols = [c for c in cols if c in available]

    # Two scans, one for pageview, one for positive
    pv_filt = pc.equal(pc.field("event_type"), "pageview")
    pv_table = dataset.to_table(columns=cols, filter=pv_filt)
    pv_df = pv_table.to_pandas()
    pv_df["dwell_clipped"] = pv_df["dwell_time_sec"].clip(lower=0, upper=3600 - 0.001)
    pos_filt = pc.is_in(pc.field("event_type"), value_set=pa.array(POSITIVE_EVENTS))
    pos_df = dataset.to_table(columns=["user_id", "item_id", "event_type"], filter=pos_filt).to_pandas()

    # Pair-level: max_dwell from pageview
    pair_pv = pv_df.groupby(["user_id", "item_id"], dropna=False).agg(
        max_dwell=("dwell_clipped", "max"),
        n_pageview=("dwell_clipped", "size"),
    ).reset_index()
    pair_pv["is_qualified_pair"] = (pair_pv["max_dwell"] >= 30) & (pair_pv["max_dwell"] < 3600)

    # Pair-level: has_any_positive vs has_dc_positive
    pos_df["is_dc"] = pos_df["event_type"].isin(DIRECT_CONTACT_EVENTS)
    pair_pos = pos_df.groupby(["user_id", "item_id"], dropna=False).agg(
        has_any_positive=("event_type", "size"),
        has_dc_positive=("is_dc", "any"),
    ).reset_index()
    pair_pos["has_any_positive"] = pair_pos["has_any_positive"] > 0

    # Merge
    pair = pair_pv.merge(pair_pos, on=["user_id", "item_id"], how="left")
    pair["has_any_positive"] = pair["has_any_positive"].fillna(False).astype(bool)
    pair["has_dc_positive"] = pair["has_dc_positive"].fillna(False).astype(bool)

    qualified = pair[pair["is_qualified_pair"]]
    n_qualified = len(qualified)
    n_qual_any_pos = int(qualified["has_any_positive"].sum())
    n_qual_dc_pos = int(qualified["has_dc_positive"].sum())

    cr_full_sample = 100 * n_qual_any_pos / max(n_qualified, 1)
    cr_dc_sample = 100 * n_qual_dc_pos / max(n_qualified, 1)
    cr_ratio = cr_dc_sample / cr_full_sample if cr_full_sample > 0 else np.nan

    rows = [
        {"metric": "qualified_pairs_in_sample", "value": n_qualified, "interpretation": "Denominator for both CR variants"},
        {"metric": "qualified_pos_pairs_FULL_5_events", "value": n_qual_any_pos, "interpretation": "Includes other_interaction"},
        {"metric": "qualified_pos_pairs_DC_ONLY_4_events", "value": n_qual_dc_pos, "interpretation": "Excludes other_interaction"},
        {"metric": "CR_D30_FULL_5_events_pct", "value": cr_full_sample, "interpretation": "What slide 4 v2 currently shows (sample)"},
        {"metric": "CR_D30_DC_ONLY_4_events_pct", "value": cr_dc_sample, "interpretation": "Backup metric if OI is noise"},
        {"metric": "DC_to_FULL_ratio", "value": cr_ratio, "interpretation": "If close to 1.0: OI doesn't matter much. If small: story depends on OI."},
        {"metric": "drop_pct_when_excluding_OI", "value": (1 - cr_ratio) * 100 if not np.isnan(cr_ratio) else np.nan,
         "interpretation": "% of qualified-pos signal lost by excluding other_interaction"},
    ]
    pd.DataFrame(rows).to_csv(out_dir / "audit_oi_dc_vs_oi_cr_d30.csv", index=False)

    findings["C_cr_full_sample_pct"] = cr_full_sample
    findings["C_cr_dc_only_sample_pct"] = cr_dc_sample
    findings["C_dc_to_full_ratio"] = cr_ratio
    findings["C_drop_pct_excluding_oi"] = (1 - cr_ratio) * 100 if not np.isnan(cr_ratio) else np.nan

    return findings


def angle_d_is_contact_flag(df: pd.DataFrame, out_dir: Path) -> dict:
    """Angle D: cross-check event_type with is_contact flag.

    BTC's own is_contact flag: should be 1 for "events that are contact".
    - If OI has is_contact=1 mostly: BTC officially treats OI as contact → defendable.
    - If OI has is_contact=0 mostly: discrepancy in BTC's own labeling → bomb.
    """
    print("\n  [Angle D] is_contact flag cross-check...")
    findings = {}

    if "is_contact" not in df.columns:
        return findings

    ct = df.groupby(["event_type", "is_contact"], dropna=False).size().unstack(fill_value=0)
    ct.to_csv(out_dir / "audit_oi_is_contact_crosstab.csv")

    # Compute % is_contact=1 per event_type
    rows = []
    for et in df["event_type"].unique():
        sub = df[df["event_type"] == et]
        pct1 = (sub["is_contact"] == 1).mean() * 100
        pct0 = (sub["is_contact"] == 0).mean() * 100
        pct_null = sub["is_contact"].isna().mean() * 100
        rows.append({
            "event_type": et,
            "n_events": len(sub),
            "pct_is_contact_1": float(pct1),
            "pct_is_contact_0": float(pct0),
            "pct_is_contact_null": float(pct_null),
        })
    df_d = pd.DataFrame(rows).sort_values("n_events", ascending=False)

    oi_row = df_d[df_d["event_type"] == "other_interaction"]
    if not oi_row.empty:
        findings["D_oi_pct_is_contact_1"] = float(oi_row.iloc[0]["pct_is_contact_1"])
    # Compare with view_phone (canonical "contact" event)
    vp_row = df_d[df_d["event_type"] == "view_phone"]
    if not vp_row.empty:
        findings["D_view_phone_pct_is_contact_1"] = float(vp_row.iloc[0]["pct_is_contact_1"])

    return findings


def build_summary(findings: dict, out_dir: Path) -> None:
    """Final summary CSV with one-line answers per question."""
    summary_rows = []

    # A1 — Surface concentration
    surf_share = findings.get("A1_oi_top1_surface_share", np.nan)
    if not np.isnan(surf_share):
        verdict = "concentrated" if surf_share > 50 else ("partial" if surf_share > 25 else "spread")
        summary_rows.append({
            "angle": "A1_surface",
            "question": "OI có concentrate trên 1 surface cụ thể không?",
            "finding": f"Top-1 surface chiếm {surf_share:.1f}%. Top-3: {findings.get('A1_oi_top3_surfaces', 'N/A')}",
            "verdict": verdict,
            "interpretation": (
                "OI gắn với 1 surface cụ thể → có thể là feature đặt tên không rõ (vd save_listing)"
                if verdict == "concentrated" else
                "OI rải nhiều surface → có thể là tracking event passive"
                if verdict == "spread" else
                "OI có pattern nhưng không tập trung tuyệt đối"
            ),
        })

    # B1 — Dwell pattern
    oi_dwell_zero = findings.get("B1_oi_pct_dwell_zero", np.nan)
    oi_dwell_under_5 = findings.get("B1_oi_pct_dwell_under_5s", np.nan)
    if not np.isnan(oi_dwell_zero):
        verdict = (
            "noise_signal" if oi_dwell_zero > 80 else
            "real_engagement" if oi_dwell_zero < 30 else
            "mixed"
        )
        summary_rows.append({
            "angle": "B1_dwell",
            "question": "OI có dwell_time_sec hợp lý không?",
            "finding": f"{oi_dwell_zero:.1f}% OI events có dwell=0, {oi_dwell_under_5:.1f}% under 5s. Median: {findings.get('B1_oi_median_dwell', 'N/A')}s",
            "verdict": verdict,
            "interpretation": (
                "OI mostly dwell=0 → passive tracking event"
                if verdict == "noise_signal" else
                "OI có dwell time hợp lý → engagement thật"
                if verdict == "real_engagement" else
                "OI mixed: vừa có engagement vừa có noise"
            ),
        })

    # B2 — Session frequency
    oi_per_sess = findings.get("B2_oi_median_per_session", np.nan)
    dc_per_sess = findings.get("B2_dc_median_per_session", np.nan)
    if not np.isnan(oi_per_sess):
        ratio = oi_per_sess / max(dc_per_sess, 0.5) if not np.isnan(dc_per_sess) else np.nan
        verdict = "noise_signal" if ratio > 5 else "normal"
        dc_str = f"{dc_per_sess:.1f}" if not np.isnan(dc_per_sess) else "N/A"
        summary_rows.append({
            "angle": "B2_session_freq",
            "question": "OI có cộng dồn quá nhiều trong 1 session không?",
            "finding": f"Median OI/session: {oi_per_sess:.1f}; DC/session: {dc_str}",
            "verdict": verdict,
            "interpretation": (
                "OI per session quá cao so với DC → có thể là event tự fire (toast, scroll, hover)"
                if verdict == "noise_signal" else
                "OI per session ở mức bình thường"
            ),
        })

    # C — Direct-contact-only CR
    drop_pct = findings.get("C_drop_pct_excluding_oi", np.nan)
    if not np.isnan(drop_pct):
        verdict = (
            "story_collapses" if drop_pct > 80 else
            "story_holds" if drop_pct < 30 else
            "story_partial"
        )
        cr_full = findings.get("C_cr_full_sample_pct", np.nan)
        cr_dc = findings.get("C_cr_dc_only_sample_pct", np.nan)
        summary_rows.append({
            "angle": "C_dc_only_cr",
            "question": "Nếu loại OI khỏi numerator, CR D30 còn lại bao nhiêu?",
            "finding": f"CR FULL: {cr_full:.2f}%. CR DC-only: {cr_dc:.2f}%. Mất {drop_pct:.1f}% signal.",
            "verdict": verdict,
            "interpretation": (
                "Nếu OI là noise, framing CR D30 37% collapse — phải đổi sang DC-only"
                if verdict == "story_collapses" else
                "OI quan trọng nhưng không phải toàn bộ — story có thể giữ với caveat"
                if verdict == "story_partial" else
                "DC-only CR đủ cao để giữ framing nếu OI bị nghi"
            ),
        })

    # D — is_contact flag
    oi_flag1 = findings.get("D_oi_pct_is_contact_1", np.nan)
    vp_flag1 = findings.get("D_view_phone_pct_is_contact_1", np.nan)
    if not np.isnan(oi_flag1):
        verdict = (
            "btc_treats_as_contact" if oi_flag1 > 90 else
            "btc_inconsistent" if 30 < oi_flag1 < 90 else
            "btc_not_contact"
        )
        summary_rows.append({
            "angle": "D_is_contact_flag",
            "question": "BTC's is_contact flag có align với event_type=other_interaction không?",
            "finding": f"OI is_contact=1: {oi_flag1:.1f}%. view_phone is_contact=1: {vp_flag1:.1f}%.",
            "verdict": verdict,
            "interpretation": (
                "BTC's own flag confirms OI = contact → defendable trước BGK"
                if verdict == "btc_treats_as_contact" else
                "BTC's flag inconsistent với label → flag để defend cẩn thận"
                if verdict == "btc_inconsistent" else
                "BTC's flag KHÔNG xem OI là contact — đây là discrepancy"
            ),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "audit_other_interaction_summary.csv", index=False)

    # Final verdict
    verdicts = [r["verdict"] for r in summary_rows]
    print("\n" + "=" * 72)
    print("AUDIT OTHER_INTERACTION — TÓM TẮT")
    print("=" * 72)
    for r in summary_rows:
        print(f"\n[{r['angle']}] {r['question']}")
        print(f"  Finding:  {r['finding']}")
        print(f"  Verdict:  {r['verdict']}")
        print(f"  Diễn giải: {r['interpretation']}")

    print("\n" + "=" * 72)
    print("FINAL CALL — defend framing CR D30 37% được không?")
    print("=" * 72)
    bad_signals = sum(1 for v in verdicts if v in ("noise_signal", "story_collapses", "btc_not_contact"))
    good_signals = sum(1 for v in verdicts if v in ("real_engagement", "story_holds", "btc_treats_as_contact"))

    if bad_signals == 0 and good_signals >= 2:
        print("✓ GIỮ NGUYÊN. OI có pattern engagement thật, framing 37% defendable.")
    elif bad_signals >= 2:
        print("✗ ĐỔI FRAMING. OI có dấu hiệu noise, cần dùng DC-only CR backup.")
    else:
        print("⚠ MIXED. Cần discuss team — story có thể giữ với caveat trong speaker note.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Audit other_interaction event semantics")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root chứa train/fact_user_events/")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit"),
                        help="Output dir cho CSV files")
    parser.add_argument("--n-files", type=int, default=30,
                        help="Số file Parquet sample (default 30/500 ~6%)")
    args = parser.parse_args()

    events_dir = args.data_root / "train" / "fact_user_events"
    if not events_dir.is_dir():
        print(f"✗ Không tìm thấy {events_dir}")
        sys.exit(2)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ Audit other_interaction trên sample {args.n_files} files từ {events_dir}")
    files = select_sample_files(events_dir, args.n_files)
    print(f"  Sampled {len(files)} files (uniform across {len(list(events_dir.glob('*.parquet')))} total)")

    df = load_audit_sample(files)
    if df.empty:
        print("✗ Sample không có positive events nào")
        sys.exit(1)

    df["event_group"] = np.where(
        df["event_type"] == "other_interaction", "other_interaction", "direct_contact"
    )

    findings = {}
    findings.update(angle_a_attribute_distribution(df, args.output_dir))
    findings.update(angle_b_time_pattern(df, args.output_dir))
    findings.update(angle_d_is_contact_flag(df, args.output_dir))
    # Angle C re-scans the dataset; pass files separately
    findings.update(angle_c_dc_only_cr(events_dir, files, args.output_dir))

    build_summary(findings, args.output_dir)

    print(f"\n✓ Audit hoàn tất. Files tại: {args.output_dir}")
    print("  - audit_other_interaction_summary.csv  ← đọc trước hết")
    print("  - audit_oi_by_surface.csv              ← Angle A1")
    print("  - audit_oi_by_device.csv               ← Angle A2")
    print("  - audit_oi_by_position.csv             ← Angle A3")
    print("  - audit_oi_is_contact_crosstab.csv     ← Angle D")
    print("  - audit_oi_dwell_distribution.csv      ← Angle B1")
    print("  - audit_oi_session_pattern.csv         ← Angle B2")
    print("  - audit_oi_dc_vs_oi_cr_d30.csv         ← Angle C")


if __name__ == "__main__":
    main()
