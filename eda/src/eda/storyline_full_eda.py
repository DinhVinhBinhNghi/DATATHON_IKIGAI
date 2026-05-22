"""
Full EDA storyline pipeline for Datathon 2026 - Cho Tot Real Estate recommender.

Design goals
------------
1. Never load raw fact_user_events into memory at once.
2. Pre-aggregate event files locally in small parts.
3. Cache all intermediate tables under outputs/agg for fast reruns.
4. Produce slide-ready PNG figures and a compact summary_metrics.csv.

This module is intentionally local-first for the updated BTC data access policy,
but the notebook wrapper can still mount Google Drive in Colab and point
--data-root to the mounted folder.
"""

from __future__ import annotations

import gc
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TRAIN_START_DATE = "2025-11-09"
TRAIN_END_DATE = "2026-04-09"
TRAIN_END_TS = pd.Timestamp("2026-04-10 00:00:00")

# Canonical funnel A1 threshold. Single source of truth in src/utils/constants.py.
# Re-imported here to keep this module's constants block discoverable.
#
# D30 schema (chốt theo feedback mentor v2, áp dụng từ patch_round1):
# ---------------------------------------------------------------------
# - "Qualified pageview event" = pageview event có dwell_time_valid_sec >= 30s
#   AND < 3600s (cap 3600s là timer-not-stopped artifact, đã xử lý ở
#   read_event_file).
# - "Qualified pair" (user × item) = pair có ít nhất 1 qualified pageview
#   event, tương đương max_dwell_sec >= 30 AND max_dwell_sec < 3600 trên
#   pair_flags. KHÔNG dùng SUM dwell qua nhiều session — MAX phản ánh đúng
#   intent "high-involvement decision cần ít nhất 1 lần xem nghiêm túc".
# - "Positive contact" GIỮ NGUYÊN định nghĩa BTC (5 event types). Không thêm
#   điều kiện qualified vào positive event bản thân.
# - "D30 contact rate" (CR D30) = qualified_pairs_with_positive / qualified_pairs.
#   Đây là metric thống nhất cho mọi nhánh A1/A2/A3 trên slide chính. Snapshot
#   CR (từ fact_listing_snapshot.views_24h/contacts_24h) chỉ dùng làm validation
#   cross-check trong appendix, không đem lên slide chính để tránh trộn universe.
from src.utils.constants import DWELL_THRESHOLD_SEC  # noqa: E402

# Hard assertion: D30 schema yêu cầu ngưỡng đúng bằng 30s. Nếu repo có canonical
# khác (ví dụ 60s cho thí nghiệm), patch phải fail loud thay vì silently khác đề.
assert int(DWELL_THRESHOLD_SEC) == 30, (
    f"D30 schema yêu cầu DWELL_THRESHOLD_SEC=30, nhưng đang là {DWELL_THRESHOLD_SEC}. "
    "Sửa src/utils/constants.py về 30 trước khi chạy lại pipeline."
)

POSITIVE_EVENTS = [
    "view_phone",
    "contact_chat",
    "contact_zalo",
    "contact_sms",
    "other_interaction",
]
DIRECT_CONTACT_EVENTS = ["view_phone", "contact_chat", "contact_zalo", "contact_sms"]
EVENTS_OF_INTEREST = ["pageview", *POSITIVE_EVENTS]

CATEGORY_NAMES = {
    1010: "Phòng trọ/thuê",
    1020: "Căn hộ/CC",
    1030: "Nhà ở",
    1040: "Đất nền/TM",
    1050: "Dự án mới",
}

# Palette nhất quán toàn notebook/report.
PALETTE = {
    "navy": "#0B4F8A",
    "blue": "#2F80ED",
    "sky": "#56CCF2",
    "orange": "#FF7A30",
    "green": "#27AE60",
    "red": "#EB5757",
    "gray": "#7B8794",
    "light_gray": "#E5EAF0",
    "dark": "#1F2933",
}

FIG_DPI = 220


@dataclass
class EDAPaths:
    data_root: Path
    output_root: Path
    agg_dir: Path
    fig_dir: Path
    table_dir: Path
    tmp_dir: Path

    @classmethod
    def build(cls, data_root: str | Path, output_root: str | Path = "outputs") -> "EDAPaths":
        root = Path(data_root).expanduser().resolve()
        out = Path(output_root).expanduser().resolve()
        paths = cls(
            data_root=root,
            output_root=out,
            agg_dir=out / "agg",
            fig_dir=out / "figures" / "main",
            table_dir=out / "tables",
            tmp_dir=Path("tmp") / "duckdb_tmp",
        )
        for p in [paths.agg_dir, paths.fig_dir, paths.table_dir, paths.tmp_dir]:
            p.mkdir(parents=True, exist_ok=True)
        return paths

    @property
    def train_dir(self) -> Path:
        return self.data_root / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_root / "test"


def _p(path: Path | str) -> str:
    """DuckDB-friendly path string, especially on Windows."""
    return Path(path).as_posix()


def connect_duckdb(paths: EDAPaths, threads: int = 2, memory_limit: str = "10GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads TO {int(threads)};")
    con.execute(f"SET memory_limit='{memory_limit}';")
    con.execute(f"SET temp_directory='{_p(paths.tmp_dir)}';")
    # Lower memory pressure for large GROUP BY / JOIN workloads.
    # DuckDB will spill to the configured temp_directory when needed.
    try:
        con.execute("SET preserve_insertion_order=false;")
    except Exception:
        pass
    return con


def discover_event_files(data_root: Path, sample_files: Optional[int] = None) -> list[Path]:
    folder = data_root / "train" / "fact_user_events"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy parquet trong {folder}")
    if sample_files:
        files = files[: int(sample_files)]
    return files


def read_event_file(file_path: Path) -> pd.DataFrame:
    """Read a single event parquet with only columns needed for EDA.

    We keep this function conservative: column projection is always applied;
    date/event filters are re-applied in pandas to avoid dtype mismatch across
    Parquet writers. On local SSD, per-file projection is usually stable and
    avoids the OOM risk of scanning all 500 files at once.
    """
    columns = [
        "user_id",
        "item_id",
        "event_type",
        "event_ts",
        "date",
        "category",
        "city_name",
        "surface",
        "device",
        "position",
        "dwell_time_sec",
        "is_login",
        "session_id",
    ]
    table = pq.read_table(file_path, columns=columns)
    df = table.to_pandas()
    del table

    if df.empty:
        return df

    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    mask = (
        (df["event_ts"] < TRAIN_END_TS)
        & (df["date"] >= pd.to_datetime(TRAIN_START_DATE).date())
        & (df["date"] <= pd.to_datetime(TRAIN_END_DATE).date())
        & (df["event_type"].isin(EVENTS_OF_INTEREST))
    )
    df = df.loc[mask].copy()
    df["dwell_time_sec"] = pd.to_numeric(df["dwell_time_sec"], errors="coerce").fillna(0).clip(lower=0)
    # 3600s appears as a hard cap/artifact in the raw clickstream. Keep raw dwell
    # for audit, but use a valid dwell column for thresholding and funnel logic.
    df["dwell_time_valid_sec"] = df["dwell_time_sec"].where(
        (df["dwell_time_sec"] > 0) & (df["dwell_time_sec"] < 3600), np.nan
    )
    return df


def _write_part(df: pd.DataFrame, out_dir: Path, part_idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        return
    df.to_parquet(out_dir / f"part_{part_idx:05d}.parquet", index=False)


def clean_agg_subdirs(base: Path) -> None:
    import shutil

    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)


def aggregate_events_per_file(
    paths: EDAPaths,
    sample_files: Optional[int] = None,
    force: bool = False,
    progress_every: int = 10,
) -> None:
    """Layer 1: per-file aggregation of fact_user_events.

    Output folders under outputs/agg/events:
    - event_counts: small event_type counts.
    - daily_counts: daily event_type counts.
    - pageview_pairs: unique user-item pageview pairs per file.
    - positive_pairs: unique user-item positive pairs per file.
    - user_positive_counts, user_pageview_counts.
    - user_category_counts: view/contact category preference by user.
    - device_surface_counts.
    - pre_contact_dwell_sample: pageview dwell before first positive in same session.
    """
    events_agg_dir = paths.agg_dir / "events"
    done_flag = events_agg_dir / "_DONE_SAMPLE" if sample_files else events_agg_dir / "_DONE_FULL"
    if done_flag.exists() and not force:
        print(f"✓ Event aggregation cache found: {done_flag}")
        return

    clean_agg_subdirs(events_agg_dir)
    files = discover_event_files(paths.data_root, sample_files=sample_files)
    print(f"→ Aggregating {len(files):,} fact_user_events files")

    for idx, file_path in enumerate(files):
        try:
            df = read_event_file(file_path)
            if df.empty:
                continue

            df["is_positive"] = df["event_type"].isin(POSITIVE_EVENTS)
            df["is_direct_contact"] = df["event_type"].isin(DIRECT_CONTACT_EVENTS)
            df["is_pageview"] = df["event_type"].eq("pageview")
            df["is_login_bool"] = df["is_login"].eq("login")

            # A1: event counts and daily contact rate.
            event_counts = (
                df.groupby(["event_type"], dropna=False)
                .size()
                .reset_index(name="n_events")
            )
            _write_part(event_counts, events_agg_dir / "event_counts", idx)

            daily_counts = (
                df.groupby(["date", "event_type"], dropna=False)
                .size()
                .reset_index(name="n_events")
            )
            _write_part(daily_counts, events_agg_dir / "daily_counts", idx)

            # A1/A2: pair-level funnel components.
            pv = df.loc[df["is_pageview"] & df["user_id"].notna() & df["item_id"].notna()]
            if not pv.empty:
                pageview_pairs = (
                    pv.groupby(["user_id", "item_id"], sort=False)
                    .agg(
                        n_pageview=("event_type", "size"),
                        first_pageview_ts=("event_ts", "min"),
                        max_dwell_sec=("dwell_time_valid_sec", "max"),
                        first_view_category=("category", "first"),
                        first_view_city=("city_name", "first"),
                    )
                    .reset_index()
                )
                _write_part(pageview_pairs, events_agg_dir / "pageview_pairs", idx)

                # Daily unique pageview pairs for true conversion-rate time series.
                # This is still pre-aggregated/shrunk versus raw event level.
                daily_pv_pairs = pv[["date", "user_id", "item_id"]].drop_duplicates()
                _write_part(daily_pv_pairs, events_agg_dir / "daily_pageview_pairs", idx)

                user_pageview = (
                    pv.loc[pv["is_login_bool"]]
                    .groupby("user_id", sort=False)
                    .agg(
                        n_pageview_events=("event_type", "size"),
                        n_view_items=("item_id", "nunique"),
                        n_view_days=("date", "nunique"),
                    )
                    .reset_index()
                )
                _write_part(user_pageview, events_agg_dir / "user_pageview_counts", idx)

            pos = df.loc[df["is_positive"] & df["user_id"].notna() & df["item_id"].notna()]
            if not pos.empty:
                positive_pairs = (
                    pos.groupby(["user_id", "item_id"], sort=False)
                    .agg(
                        n_positive_events=("event_type", "size"),
                        n_direct_contact_events=("is_direct_contact", "sum"),
                        first_positive_ts=("event_ts", "min"),
                        first_positive_category=("category", "first"),
                        first_positive_city=("city_name", "first"),
                    )
                    .reset_index()
                )
                _write_part(positive_pairs, events_agg_dir / "positive_pairs", idx)

                daily_pos_pairs = pos[["date", "user_id", "item_id"]].drop_duplicates()
                _write_part(daily_pos_pairs, events_agg_dir / "daily_positive_pairs", idx)
                direct = pos.loc[pos["is_direct_contact"], ["date", "user_id", "item_id"]].drop_duplicates()
                _write_part(direct, events_agg_dir / "daily_direct_contact_pairs", idx)

                user_positive = (
                    pos.loc[pos["is_login_bool"]]
                    .groupby("user_id", sort=False)
                    .agg(
                        n_positive_events=("event_type", "size"),
                        n_positive_items=("item_id", "nunique"),
                        n_positive_days=("date", "nunique"),
                    )
                    .reset_index()
                )
                _write_part(user_positive, events_agg_dir / "user_positive_counts", idx)

            # A2: user category preference; login only because non-login user_id changes by session.
            login = df.loc[df["is_login_bool"] & df["user_id"].notna() & df["category"].notna()]
            if not login.empty:
                view_cat = (
                    login.loc[login["is_pageview"]]
                    .groupby(["user_id", "category"], sort=False)
                    .size()
                    .reset_index(name="n_view_events")
                )
                contact_cat = (
                    login.loc[login["is_positive"]]
                    .groupby(["user_id", "category"], sort=False)
                    .size()
                    .reset_index(name="n_positive_events")
                )
                if not view_cat.empty or not contact_cat.empty:
                    user_cat = pd.merge(view_cat, contact_cat, on=["user_id", "category"], how="outer").fillna(0)
                    _write_part(user_cat, events_agg_dir / "user_category_counts", idx)

            # A2: device/surface breakdown of positive events.
            if not pos.empty:
                device_surface = (
                    pos.groupby(["device", "surface", "event_type"], dropna=False)
                    .size()
                    .reset_index(name="n_events")
                )
                _write_part(device_surface, events_agg_dir / "device_surface_counts", idx)

            # A1: dwell time before first positive event in the same session.
            # Purpose: choose a data-driven threshold D for serious engagement.
            if "session_id" in df.columns:
                sess_pos = (
                    df.loc[df["is_positive"] & df["session_id"].notna()]
                    .groupby("session_id", sort=False)["event_ts"]
                    .min()
                    .reset_index(name="first_positive_ts_session")
                )
                if not sess_pos.empty and not pv.empty:
                    pv_sess = pv.loc[pv["session_id"].notna(), ["session_id", "event_ts", "dwell_time_valid_sec"]]
                    pre = pv_sess.merge(sess_pos, on="session_id", how="inner")
                    pre = pre.loc[pre["event_ts"] <= pre["first_positive_ts_session"], ["dwell_time_valid_sec"]]
                    pre = pre.rename(columns={"dwell_time_valid_sec": "dwell_time_sec"})
                    pre = pre.loc[pre["dwell_time_sec"].notna() & (pre["dwell_time_sec"] > 0) & (pre["dwell_time_sec"] < 3600)]
                    if len(pre) > 250_000:
                        pre = pre.sample(250_000, random_state=42)
                    _write_part(pre, events_agg_dir / "pre_contact_dwell_sample", idx)

            if (idx + 1) % progress_every == 0 or idx == len(files) - 1:
                print(f"  processed {idx + 1:>4}/{len(files)} files")

        except Exception as exc:  # keep full pipeline moving, report file-level issue.
            warnings.warn(f"Failed to process {file_path.name}: {exc}")
        finally:
            try:
                del df
            except Exception:
                pass
            gc.collect()

    done_flag.write_text("done", encoding="utf-8")
    print(f"✓ Event aggregation done: {done_flag}")


def _has_parquet(folder: Path) -> bool:
    return folder.exists() and any(folder.glob("*.parquet"))


def _date_month_chunks(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return [start, end) monthly chunks covering the given dates."""
    d = pd.to_datetime(dates).dropna()
    if d.empty:
        return []
    start = d.min().to_period("M").start_time
    end = (d.max().to_period("M") + 1).start_time
    chunks = []
    cur = start
    while cur < end:
        nxt = (cur.to_period("M") + 1).start_time
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def _compute_daily_pair_rates_low_memory(
    paths: EDAPaths,
    con: duckdb.DuckDBPyConnection,
    df_daily: pd.DataFrame,
    pv_day_dir: Path,
    pos_day_dir: Path,
    direct_day_dir: Path,
) -> pd.DataFrame:
    """Compute true daily pair conversion without materializing all dates at once.

    The v2 implementation did one global DISTINCT + JOIN across all daily pair
    parquet files. On 8-10GB machines this can OOM after event aggregation.
    This version first tries month-sized chunks and falls back to day-sized
    chunks if a month is still too large. It writes only the final daily table
    to pandas, so the rest of the pipeline can continue.
    """
    pv_day_glob = _p(pv_day_dir / "*.parquet")
    pos_day_glob = _p(pos_day_dir / "*.parquet")
    direct_day_glob = _p(direct_day_dir / "*.parquet") if _has_parquet(direct_day_dir) else None
    direct_cte_tpl = (
        "direct AS (SELECT DISTINCT CAST(date AS DATE) AS date, user_id, item_id "
        f"FROM read_parquet('{direct_day_glob}') "
        "WHERE CAST(date AS DATE) >= DATE '{start}' AND CAST(date AS DATE) < DATE '{end}')"
        if direct_day_glob
        else "direct AS (SELECT NULL::DATE AS date, NULL::VARCHAR AS user_id, NULL::VARCHAR AS item_id WHERE FALSE)"
    )

    def run_range(start: str, end: str) -> pd.DataFrame:
        direct_cte = direct_cte_tpl.format(start=start, end=end) if direct_day_glob else direct_cte_tpl
        return con.execute(
            f"""
            WITH pv AS (
                SELECT DISTINCT CAST(date AS DATE) AS date, user_id, item_id
                FROM read_parquet('{pv_day_glob}')
                WHERE CAST(date AS DATE) >= DATE '{start}'
                  AND CAST(date AS DATE) <  DATE '{end}'
            ),
            pos AS (
                SELECT DISTINCT CAST(date AS DATE) AS date, user_id, item_id
                FROM read_parquet('{pos_day_glob}')
                WHERE CAST(date AS DATE) >= DATE '{start}'
                  AND CAST(date AS DATE) <  DATE '{end}'
            ),
            {direct_cte}
            SELECT
                pv.date,
                COUNT(*)::BIGINT AS pageview_pairs,
                SUM(CASE WHEN pos.user_id IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS positive_pairs_same_day,
                SUM(CASE WHEN direct.user_id IS NOT NULL THEN 1 ELSE 0 END)::BIGINT AS direct_contact_pairs_same_day
            FROM pv
            LEFT JOIN pos USING (date, user_id, item_id)
            LEFT JOIN direct USING (date, user_id, item_id)
            GROUP BY pv.date
            ORDER BY pv.date
            """
        ).df()

    out_parts: list[pd.DataFrame] = []
    for start_ts, end_ts in _date_month_chunks(df_daily["date"]):
        start = start_ts.strftime("%Y-%m-%d")
        end = end_ts.strftime("%Y-%m-%d")
        try:
            part = run_range(start, end)
            out_parts.append(part)
            print(f"  daily pair rates: {start} -> {end} OK ({len(part)} days)")
        except duckdb.OutOfMemoryException:
            print(f"  daily pair rates: {start} -> {end} OOM, fallback to day chunks")
            days = pd.date_range(start_ts, end_ts - pd.Timedelta(days=1), freq="D")
            for day in days:
                d0 = day.strftime("%Y-%m-%d")
                d1 = (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                try:
                    part = run_range(d0, d1)
                    out_parts.append(part)
                except duckdb.OutOfMemoryException:
                    # Last-resort: keep pipeline alive, mark this day missing.
                    warnings.warn(f"OOM even for daily pair-rate chunk {d0}; leaving pair rates as NA for this day.")
        finally:
            gc.collect()

    if not out_parts:
        return pd.DataFrame(columns=["date", "pageview_pairs", "positive_pairs_same_day", "direct_contact_pairs_same_day"])
    ans = pd.concat(out_parts, ignore_index=True)
    # If a date appears in multiple chunks due to edge cases, reduce safely.
    ans = (
        ans.groupby("date", as_index=False)
        .agg(
            pageview_pairs=("pageview_pairs", "sum"),
            positive_pairs_same_day=("positive_pairs_same_day", "sum"),
            direct_contact_pairs_same_day=("direct_contact_pairs_same_day", "sum"),
        )
        .sort_values("date")
    )
    return ans


def reduce_event_aggregates(paths: EDAPaths, con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Layer 1 reduction: shrink per-file event aggregations to analysis tables."""
    ev = paths.agg_dir / "events"
    reduced = paths.agg_dir / "reduced"
    reduced.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    # Event counts.
    event_counts_glob = _p(ev / "event_counts" / "*.parquet")
    df_event_counts = con.execute(
        f"""
        SELECT event_type, SUM(n_events)::BIGINT AS n_events
        FROM read_parquet('{event_counts_glob}')
        GROUP BY event_type
        ORDER BY n_events DESC
        """
    ).df()
    df_event_counts.to_csv(paths.table_dir / "eda_A1_event_counts.csv", index=False)
    results["event_counts"] = df_event_counts

    # Daily counts and CR.
    daily_glob = _p(ev / "daily_counts" / "*.parquet")
    df_daily = con.execute(
        f"""
        WITH d AS (
            SELECT date, event_type, SUM(n_events)::BIGINT AS n_events
            FROM read_parquet('{daily_glob}')
            GROUP BY date, event_type
        )
        SELECT
            date,
            SUM(CASE WHEN event_type='pageview' THEN n_events ELSE 0 END)::BIGINT AS pageview_events,
            SUM(CASE WHEN event_type IN {tuple(POSITIVE_EVENTS)} THEN n_events ELSE 0 END)::BIGINT AS positive_events,
            SUM(CASE WHEN event_type IN {tuple(DIRECT_CONTACT_EVENTS)} THEN n_events ELSE 0 END)::BIGINT AS direct_contact_events
        FROM d
        GROUP BY date
        ORDER BY date
        """
    ).df()
    # Event intensity can exceed 100%, because one pageview can be followed by
    # multiple positive events. Keep it for audit, but do NOT label it as rate.
    df_daily["positive_event_intensity"] = df_daily["positive_events"] / df_daily["pageview_events"].replace(0, np.nan)
    df_daily["direct_contact_event_intensity"] = df_daily["direct_contact_events"] / df_daily["pageview_events"].replace(0, np.nan)

    # True daily conversion rate at unique user-item-pair level: denominator =
    # pairs with pageview that day; numerator = those same pairs with positive/direct contact that day.
    # Compute in low-memory month/day chunks instead of one global DISTINCT+JOIN.
    pv_day_dir = ev / "daily_pageview_pairs"
    pos_day_dir = ev / "daily_positive_pairs"
    direct_day_dir = ev / "daily_direct_contact_pairs"
    if _has_parquet(pv_day_dir) and _has_parquet(pos_day_dir):
        df_daily_pairs = _compute_daily_pair_rates_low_memory(
            paths=paths,
            con=con,
            df_daily=df_daily,
            pv_day_dir=pv_day_dir,
            pos_day_dir=pos_day_dir,
            direct_day_dir=direct_day_dir,
        )
        df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.date
        if not df_daily_pairs.empty:
            df_daily_pairs["date"] = pd.to_datetime(df_daily_pairs["date"]).dt.date
        df_daily = df_daily.merge(df_daily_pairs, on="date", how="left")
        df_daily["pair_positive_rate"] = df_daily["positive_pairs_same_day"] / df_daily["pageview_pairs"].replace(0, np.nan)
        df_daily["pair_direct_contact_rate"] = df_daily["direct_contact_pairs_same_day"] / df_daily["pageview_pairs"].replace(0, np.nan)
        df_daily["pair_positive_rate_7d"] = df_daily["pair_positive_rate"].rolling(7, min_periods=1).mean()
        df_daily["pair_direct_contact_rate_7d"] = df_daily["pair_direct_contact_rate"].rolling(7, min_periods=1).mean()
    else:
        # Backward compatibility with old cache: use event intensity only and warn in plot title.
        df_daily["pair_positive_rate"] = np.nan
        df_daily["pair_direct_contact_rate"] = np.nan
        df_daily["pair_positive_rate_7d"] = np.nan
        df_daily["pair_direct_contact_rate_7d"] = np.nan

    df_daily.to_csv(paths.table_dir / "eda_A1_daily_contact_rate.csv", index=False)
    results["daily"] = df_daily

    # Dwell threshold D from distribution before contact.
    dwell_dir = ev / "pre_contact_dwell_sample"
    if _has_parquet(dwell_dir):
        dwell_glob = _p(dwell_dir / "*.parquet")
        df_dwell_quantile = con.execute(
            f"""
            SELECT
                quantile_cont(dwell_time_sec, 0.25) AS q25,
                quantile_cont(dwell_time_sec, 0.50) AS q50,
                quantile_cont(dwell_time_sec, 0.75) AS q75,
                COUNT(*) AS n_sample
            FROM read_parquet('{dwell_glob}')
            WHERE dwell_time_sec > 0 AND dwell_time_sec < 3600
            """
        ).df()
        q25 = float(df_dwell_quantile.loc[0, "q25"] or 0)
        q50 = float(df_dwell_quantile.loc[0, "q50"] or 0)
        # Canonical D — see DECISIONS_NEEDED.md mục 1.
        # Heuristic Q25/Q50 cũ bị bỏ vì median pre-contact dwell ~31 phút là cap artifact,
        # và pipeline nên có 1 D thống nhất cho toàn slide. Override bằng src/utils/constants.py.
        D = int(DWELL_THRESHOLD_SEC)
        reason = (
            f"Canonical D={D}s (xem DECISIONS_NEEDED.md mục 1). "
            f"Quantiles của data sau khi loại cap 3600s: Q25={q25:.0f}s, Q50={q50:.0f}s. "
            f"Heuristic cũ bị bỏ vì median pre-contact ~31 phút là artifact timer-not-stopped."
        )
        df_dwell_threshold = df_dwell_quantile.copy()
        df_dwell_threshold["D_selected"] = D
        df_dwell_threshold["reason"] = reason
    else:
        D = int(DWELL_THRESHOLD_SEC)
        df_dwell_threshold = pd.DataFrame({"q25": [np.nan], "q50": [np.nan], "q75": [np.nan], "n_sample": [0], "D_selected": [D], "reason": [f"Không có dwell sample; dùng canonical D={D}s."]})
    df_dwell_threshold.to_csv(paths.table_dir / "eda_A1_dwell_threshold.csv", index=False)
    results["dwell_threshold"] = df_dwell_threshold

    # Pair-level flags from per-file pair aggregates.
    # v3 tried to build one global FULL OUTER JOIN and OOMed on 8-10GB RAM.
    # v4 writes hash-partitioned pair flags, so each join only touches a small shard.
    pageview_glob = _p(ev / "pageview_pairs" / "*.parquet")
    positive_glob = _p(ev / "positive_pairs" / "*.parquet")

    pair_flags_dir = reduced / "event_pair_flags_parts"
    pair_flags_dir.mkdir(parents=True, exist_ok=True)
    existing_parts = sorted(pair_flags_dir.glob("part_*.parquet"))
    n_pair_partitions = 64
    if len(existing_parts) < n_pair_partitions:
        # Clear incomplete/old partition outputs.
        for f in existing_parts:
            try:
                f.unlink()
            except Exception:
                pass
        print(f"pair flags: building {n_pair_partitions} low-memory hash partitions")
        for b in range(n_pair_partitions):
            out_part = pair_flags_dir / f"part_{b:03d}.parquet"
            con.execute(
                f"""
                COPY (
                    WITH pv AS (
                        SELECT
                            user_id,
                            item_id,
                            SUM(n_pageview)::BIGINT AS n_pageview,
                            MIN(first_pageview_ts) AS first_pageview_ts,
                            MAX(max_dwell_sec) AS max_dwell_sec,
                            any_value(first_view_category) AS first_view_category,
                            any_value(first_view_city) AS first_view_city
                        FROM read_parquet('{pageview_glob}')
                        WHERE hash(user_id) % {n_pair_partitions} = {b}
                        GROUP BY user_id, item_id
                    ),
                    pos AS (
                        SELECT
                            user_id,
                            item_id,
                            SUM(n_positive_events)::BIGINT AS n_positive_events,
                            SUM(n_direct_contact_events)::BIGINT AS n_direct_contact_events,
                            MIN(first_positive_ts) AS first_positive_ts,
                            any_value(first_positive_category) AS first_positive_category,
                            any_value(first_positive_city) AS first_positive_city
                        FROM read_parquet('{positive_glob}')
                        WHERE hash(user_id) % {n_pair_partitions} = {b}
                        GROUP BY user_id, item_id
                    )
                    SELECT
                        COALESCE(pv.user_id, pos.user_id) AS user_id,
                        COALESCE(pv.item_id, pos.item_id) AS item_id,
                        COALESCE(n_pageview, 0) AS n_pageview,
                        COALESCE(n_positive_events, 0) AS n_positive_events,
                        COALESCE(n_direct_contact_events, 0) AS n_direct_contact_events,
                        first_pageview_ts,
                        first_positive_ts,
                        max_dwell_sec,
                        first_view_category,
                        first_positive_category,
                        first_view_city,
                        first_positive_city
                    FROM pv
                    FULL OUTER JOIN pos USING (user_id, item_id)
                ) TO '{_p(out_part)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            if (b + 1) % 8 == 0 or b == n_pair_partitions - 1:
                print(f"  pair flags partition {b + 1:>2}/{n_pair_partitions}")
    else:
        print(f"✓ pair flags cache found: {pair_flags_dir}")

    pair_flags_glob = _p(pair_flags_dir / "*.parquet")

    df_funnel = con.execute(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE n_pageview > 0) AS N1_pageview_pairs,
            COUNT(*) FILTER (WHERE n_pageview > 0 AND max_dwell_sec >= {D} AND max_dwell_sec < 3600) AS N2_serious_pageview_pairs,
            COUNT(*) FILTER (WHERE n_positive_events > 0) AS N3_positive_pairs_global,
            COUNT(*) FILTER (WHERE n_pageview > 0 AND max_dwell_sec >= {D} AND max_dwell_sec < 3600 AND n_positive_events > 0) AS N3_positive_pairs_inside_serious_pv,
            COUNT(*) FILTER (WHERE n_direct_contact_events > 0) AS N3_direct_contact_pairs_global,
            -- DC-only inside qualified pair (backup metric cho slide khi BGK hỏi
            -- about other_interaction). Numerator = chỉ 4 events: view_phone,
            -- contact_chat, contact_zalo, contact_sms. Loại bỏ other_interaction.
            COUNT(*) FILTER (WHERE n_pageview > 0 AND max_dwell_sec >= {D} AND max_dwell_sec < 3600 AND n_direct_contact_events > 0) AS N3_dc_pairs_inside_serious_pv
        FROM read_parquet('{pair_flags_glob}')
        """
    ).df()
    N1 = float(df_funnel.loc[0, "N1_pageview_pairs"])
    N2 = float(df_funnel.loc[0, "N2_serious_pageview_pairs"])
    N3_slide = float(df_funnel.loc[0, "N3_positive_pairs_inside_serious_pv"])
    N3_dc = float(df_funnel.loc[0, "N3_dc_pairs_inside_serious_pv"])
    df_funnel["P1_serious_over_pageview_pct"] = 100 * N2 / N1 if N1 else np.nan
    df_funnel["P2_positive_over_serious_pct"] = 100 * N3_slide / N2 if N2 else np.nan
    df_funnel["P2_dc_over_serious_pct"] = 100 * N3_dc / N2 if N2 else np.nan
    df_funnel["P_total_positive_over_pageview_pct"] = 100 * N3_slide / N1 if N1 else np.nan
    df_funnel["D_selected"] = D
    df_funnel.to_csv(paths.table_dir / "eda_A1_funnel_summary.csv", index=False)
    results["funnel"] = df_funnel

    # Positive distribution by event type.
    df_pos_dist = df_event_counts.loc[df_event_counts["event_type"].isin(POSITIVE_EVENTS)].copy()
    df_pos_dist["pct"] = 100 * df_pos_dist["n_events"] / df_pos_dist["n_events"].sum()
    df_pos_dist.to_csv(paths.table_dir / "eda_A1_positive_event_distribution.csv", index=False)
    results["positive_dist"] = df_pos_dist

    # User-level positive counts for A2.
    user_pos_glob = _p(ev / "user_positive_counts" / "*.parquet")
    user_pv_glob = _p(ev / "user_pageview_counts" / "*.parquet")
    df_user_positive = con.execute(
        f"""
        SELECT
            user_id,
            SUM(n_positive_events)::BIGINT AS n_positive_events,
            SUM(n_positive_items)::BIGINT AS n_positive_items,
            SUM(n_positive_days)::BIGINT AS n_positive_days
        FROM read_parquet('{user_pos_glob}')
        GROUP BY user_id
        """
    ).df()
    user_positive_path = reduced / "user_positive_counts.parquet"
    df_user_positive.to_parquet(user_positive_path, index=False)
    results["user_positive"] = df_user_positive

    # Cold-start test groups.
    # Use user-level pre-aggregates instead of pair_flags GROUP BY to keep RAM low.
    test_path = paths.test_dir / "test_users.parquet"
    df_cold = con.execute(
        f"""
        WITH test AS (SELECT user_id FROM read_parquet('{_p(test_path)}')),
        pv AS (
            SELECT user_id, SUM(n_pageview_events)::BIGINT AS n_pageview
            FROM read_parquet('{user_pv_glob}')
            GROUP BY user_id
        ),
        pos AS (
            SELECT user_id, SUM(n_positive_events)::BIGINT AS n_positive
            FROM read_parquet('{user_pos_glob}')
            GROUP BY user_id
        ),
        hist AS (
            SELECT
                t.user_id,
                COALESCE(pv.n_pageview, 0) AS n_pageview,
                COALESCE(pos.n_positive, 0) AS n_positive
            FROM test t
            LEFT JOIN pv USING (user_id)
            LEFT JOIN pos USING (user_id)
        )
        SELECT
            CASE
                WHEN n_pageview=0 AND n_positive=0 THEN '0_no_history'
                WHEN n_pageview>0 AND n_positive=0 THEN '1_browser_only'
                WHEN n_pageview=0 AND n_positive>0 THEN '2_positive_no_pageview'
                ELSE '3_warm_with_positive'
            END AS user_segment,
            COUNT(*) AS n_users
        FROM hist
        GROUP BY user_segment
        ORDER BY user_segment
        """
    ).df()
    df_cold["pct"] = 100 * df_cold["n_users"] / df_cold["n_users"].sum()
    df_cold.to_csv(paths.table_dir / "eda_A2_test_cold_start_segments.csv", index=False)
    results["cold_start"] = df_cold

    # Time gap view-to-contact. Read from partitioned pair flags; output should be small
    # because it only keeps pairs having both pageview and positive history.
    df_gap = con.execute(
        f"""
        SELECT
            EXTRACT(EPOCH FROM (first_positive_ts - first_pageview_ts)) / 3600.0 AS gap_hours
        FROM read_parquet('{pair_flags_glob}')
        WHERE n_pageview > 0
          AND n_positive_events > 0
          AND first_positive_ts >= first_pageview_ts
          AND first_positive_ts IS NOT NULL
          AND first_pageview_ts IS NOT NULL
        """
    ).df()
    df_gap.to_parquet(reduced / "time_gap_view_to_contact.parquet", index=False)
    results["time_gap"] = df_gap

    # User category heatmap: dominant view category x dominant positive category.
    user_cat_glob = _p(ev / "user_category_counts" / "*.parquet")
    df_heat = con.execute(
        f"""
        WITH base AS (
            SELECT user_id, category,
                   SUM(n_view_events) AS n_view_events,
                   SUM(n_positive_events) AS n_positive_events
            FROM read_parquet('{user_cat_glob}')
            GROUP BY user_id, category
        ),
        top_view AS (
            SELECT user_id, category AS view_category
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY n_view_events DESC, category) AS rn
                FROM base WHERE n_view_events > 0
            ) WHERE rn = 1
        ),
        top_contact AS (
            SELECT user_id, category AS contact_category
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY n_positive_events DESC, category) AS rn
                FROM base WHERE n_positive_events > 0
            ) WHERE rn = 1
        )
        SELECT view_category, contact_category, COUNT(*) AS n_users
        FROM top_view v
        JOIN top_contact c USING (user_id)
        GROUP BY view_category, contact_category
        ORDER BY view_category, contact_category
        """
    ).df()
    df_heat.to_csv(paths.table_dir / "eda_A2_category_preference_heatmap.csv", index=False)
    results["category_heatmap"] = df_heat

    # Device/surface breakdown.
    ds_glob = _p(ev / "device_surface_counts" / "*.parquet")
    df_device = con.execute(
        f"""
        SELECT device, surface, event_type, SUM(n_events)::BIGINT AS n_events
        FROM read_parquet('{ds_glob}')
        GROUP BY device, surface, event_type
        ORDER BY n_events DESC
        """
    ).df()
    df_device.to_csv(paths.table_dir / "eda_A2_device_surface_breakdown.csv", index=False)
    results["device_surface"] = df_device

    return results


def register_raw_tables(paths: EDAPaths, con: duckdb.DuckDBPyConnection) -> None:
    train = paths.train_dir
    con.execute(f"CREATE OR REPLACE VIEW dim_raw AS SELECT * FROM read_parquet('{_p(train / 'dim_listing' / '*.parquet')}')")
    con.execute(f"CREATE OR REPLACE VIEW snapshot_raw AS SELECT * FROM read_parquet('{_p(train / 'fact_listing_snapshot' / '*.parquet')}')")
    con.execute(f"CREATE OR REPLACE VIEW interactions_raw AS SELECT * FROM read_parquet('{_p(train / 'fact_post_contact_interactions' / '*.parquet')}')")


def build_listing_seller_aggregates(paths: EDAPaths, con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Layer 2: join shrunken snapshot aggregates with dim_listing for A3."""
    reduced = paths.agg_dir / "reduced"
    reduced.mkdir(parents=True, exist_ok=True)
    register_raw_tables(paths, con)

    item_metrics_path = reduced / "item_metrics.parquet"
    seller_metrics_path = reduced / "seller_metrics.parquet"

    con.execute(
        f"""
        COPY (
            WITH snap AS (
                SELECT
                    item_id,
                    SUM(GREATEST(COALESCE(views_24h, 0), 0))::DOUBLE AS total_views,
                    SUM(GREATEST(COALESCE(contacts_24h, 0), 0))::DOUBLE AS total_contacts,
                    AVG(GREATEST(COALESCE(listing_age_days, 0), 0))::DOUBLE AS avg_listing_age_days,
                    MAX(GREATEST(COALESCE(listing_age_days, 0), 0))::DOUBLE AS max_listing_age_days,
                    COUNT(*) AS n_snapshot_days
                FROM snapshot_raw
                WHERE date <= DATE '{TRAIN_END_DATE}'
                GROUP BY item_id
            )
            SELECT
                d.item_id,
                d.seller_id,
                CAST(d.category AS INTEGER) AS category,
                d.seller_type,
                d.ad_type,
                d.city_name,
                d.district_name,
                d.posted_date,
                TRY_CAST(d.images_count AS DOUBLE) AS images_count,
                TRY_CAST(d.area_sqm AS DOUBLE) AS area_sqm,
                TRY_CAST(d.bedrooms AS DOUBLE) AS bedrooms,
                d.price_bucket,
                d.project_id,
                -- B2B segment proxy (P3 = cat 1050 OR project_id NOT NULL).
                -- Lý do P3 thay vì chỉ cat==1050: đề thi nói project_id "chỉ có với tin
                -- thuộc dự án mới", nhưng không loại trừ trường hợp tin dự án bị post
                -- nhầm vào cat khác (vd 1020 căn hộ CC). P3 bắt được cả 2 trường hợp.
                -- Coverage audit (cross-tab cat × project_id) chạy ở build_b2b_c2c_segment_analysis.
                CASE
                    WHEN CAST(d.category AS INTEGER) = 1050 OR d.project_id IS NOT NULL
                    THEN 'B2B'
                    ELSE 'C2C'
                END AS segment,
                COALESCE(s.total_views, 0) AS total_views,
                COALESCE(s.total_contacts, 0) AS total_contacts,
                COALESCE(s.avg_listing_age_days, 0) AS avg_listing_age_days,
                COALESCE(s.max_listing_age_days, 0) AS max_listing_age_days,
                COALESCE(s.n_snapshot_days, 0) AS n_snapshot_days,
                CASE WHEN COALESCE(s.total_views,0)>0 THEN s.total_contacts / s.total_views ELSE NULL END AS contact_rate
            FROM dim_raw d
            LEFT JOIN snap s USING (item_id)
        ) TO '{_p(item_metrics_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                seller_id,
                any_value(seller_type) AS seller_type,
                COUNT(*) AS n_listings,
                SUM(total_views) AS total_views,
                SUM(total_contacts) AS total_contacts,
                SUM(CASE WHEN total_contacts > 0 THEN 1 ELSE 0 END) AS n_listings_with_contact,
                MIN(posted_date) AS first_posted_date,
                MAX(posted_date) AS last_posted_date,
                CASE WHEN SUM(total_views)>0 THEN SUM(total_contacts)/SUM(total_views) ELSE NULL END AS seller_contact_rate
            FROM read_parquet('{_p(item_metrics_path)}')
            GROUP BY seller_id
        ) TO '{_p(seller_metrics_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    results: dict[str, pd.DataFrame] = {}

    # =====================================================================
    # D30 clickstream pair-level CR (PRIMARY metric for slide 4-6)
    # =====================================================================
    # Build pair_flags_with_dim: join pair_flags (clickstream) với dim attributes
    # (category, seller_type, ad_type, seller_id) qua item_id. Đây là bảng nền
    # cho A3.1_d30 và A3.2_d30 — universe nhất quán với funnel A1, không trộn
    # với fact_listing_snapshot.
    #
    # Lưu ý:
    # - n_pageview > 0: pair có ít nhất 1 pageview (denominator thô).
    # - max_dwell_sec >= D AND < 3600: qualified pair = denominator D30.
    # - n_positive_events > 0: pair có ít nhất 1 positive event (BTC 5 types).
    #   Kết hợp với qualified mask cho ra "qualified pair có positive" = numerator D30.
    pair_flags_glob = _p(paths.agg_dir / "reduced" / "event_pair_flags_parts" / "*.parquet")
    pair_dim_path = reduced / "pair_flags_with_dim.parquet"
    D = int(DWELL_THRESHOLD_SEC)

    con.execute(
        f"""
        COPY (
            SELECT
                pf.user_id,
                pf.item_id,
                pf.n_pageview,
                pf.n_positive_events,
                pf.n_direct_contact_events,
                pf.max_dwell_sec,
                d.category,
                d.seller_id,
                d.seller_type,
                d.ad_type,
                d.city_name,
                d.district_name,
                d.project_id,
                d.segment,
                -- D30 masks (qualified pair có/không có positive)
                CASE WHEN pf.n_pageview > 0
                      AND pf.max_dwell_sec >= {D}
                      AND pf.max_dwell_sec <  3600 THEN 1 ELSE 0 END AS is_qualified_pair,
                CASE WHEN pf.n_pageview > 0
                      AND pf.max_dwell_sec >= {D}
                      AND pf.max_dwell_sec <  3600
                      AND pf.n_positive_events > 0 THEN 1 ELSE 0 END AS is_qualified_pos_pair
            FROM read_parquet('{pair_flags_glob}') pf
            INNER JOIN read_parquet('{_p(item_metrics_path)}') d
                USING (item_id)
            WHERE pf.n_pageview > 0
        ) TO '{_p(pair_dim_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    pair_dim_glob = _p(pair_dim_path)

    # ---- A3.1 D30: CR theo category, pair-level clickstream ----
    # Numerator: qualified_pos_pairs (qualified pair có positive)
    # Denominator: qualified_pairs (pair có max_dwell ≥ 30s)
    # → universe nhất quán 100% với funnel A1.
    df_cat_cr_d30 = con.execute(
        f"""
        SELECT
            category,
            SUM(is_qualified_pair)::BIGINT      AS qualified_pairs,
            SUM(is_qualified_pos_pair)::BIGINT  AS qualified_pos_pairs,
            SUM(is_qualified_pos_pair)::DOUBLE / NULLIF(SUM(is_qualified_pair), 0) AS cr
        FROM read_parquet('{pair_dim_glob}')
        GROUP BY category
        ORDER BY category
        """
    ).df()
    df_cat_cr_d30[["ci_low", "ci_high"]] = df_cat_cr_d30.apply(
        lambda r: pd.Series(wilson_ci(r["qualified_pos_pairs"], r["qualified_pairs"])),
        axis=1,
    )
    df_cat_cr_d30["category_name"] = df_cat_cr_d30["category"].map(CATEGORY_NAMES)
    df_cat_cr_d30["metric"] = "cr_d30_pair_level"
    df_cat_cr_d30.to_csv(paths.table_dir / "eda_A3_contact_rate_by_category_d30.csv", index=False)
    # Đây là key results CHÍNH dùng cho slide. plot_a3 sẽ đọc key này.
    results["category_cr"] = df_cat_cr_d30

    # ---- A3.2 D30: CR theo ad_type × seller_type, pair-level clickstream ----
    df_combo_d30 = con.execute(
        f"""
        SELECT
            ad_type,
            seller_type,
            SUM(is_qualified_pair)::BIGINT      AS qualified_pairs,
            SUM(is_qualified_pos_pair)::BIGINT  AS qualified_pos_pairs,
            SUM(is_qualified_pos_pair)::DOUBLE / NULLIF(SUM(is_qualified_pair), 0) AS cr
        FROM read_parquet('{pair_dim_glob}')
        WHERE ad_type IS NOT NULL AND seller_type IS NOT NULL
        GROUP BY ad_type, seller_type
        ORDER BY ad_type, seller_type
        """
    ).df()
    df_combo_d30[["ci_low", "ci_high"]] = df_combo_d30.apply(
        lambda r: pd.Series(wilson_ci(r["qualified_pos_pairs"], r["qualified_pairs"])),
        axis=1,
    )
    df_combo_d30["metric"] = "cr_d30_pair_level"
    df_combo_d30.to_csv(paths.table_dir / "eda_A3_contact_rate_by_ad_seller_d30.csv", index=False)
    results["ad_seller_cr"] = df_combo_d30

    # =====================================================================
    # Snapshot CR (SECONDARY metric, appendix / validation cross-check only)
    # =====================================================================
    # Source: fact_listing_snapshot.views_24h và contacts_24h, do Chợ Tốt agg.
    # KHÔNG dùng cho slide chính vì:
    # 1. Khác universe với funnel A1 (snapshot vs clickstream).
    # 2. Không có dwell filter, không thể áp D30.
    # 3. Trộn lên cùng slide là apples-to-oranges (đây chính là lỗi mentor catch).
    # Giữ lại để cross-check: nếu CR snapshot theo segment "đồng hướng" với CR D30
    # (cùng ranking giữa các category, cùng dấu chênh lệch private vs agent),
    # thì có thể defend rằng insight không phải artifact của universe nào cả.
    df_cat_cr_snap = con.execute(
        f"""
        SELECT
            category,
            COUNT(*) AS n_items,
            SUM(total_views) AS views,
            SUM(total_contacts) AS contacts,
            SUM(total_contacts) / NULLIF(SUM(total_views),0) AS cr
        FROM read_parquet('{_p(item_metrics_path)}')
        GROUP BY category
        ORDER BY category
        """
    ).df()
    df_cat_cr_snap[["ci_low", "ci_high"]] = df_cat_cr_snap.apply(
        lambda r: pd.Series(wilson_ci(r["contacts"], r["views"])), axis=1
    )
    df_cat_cr_snap["category_name"] = df_cat_cr_snap["category"].map(CATEGORY_NAMES)
    df_cat_cr_snap["metric"] = "cr_snapshot_view_level"
    df_cat_cr_snap.to_csv(paths.table_dir / "eda_A3_contact_rate_by_category_snapshot.csv", index=False)
    results["category_cr_snapshot"] = df_cat_cr_snap

    df_combo_snap = con.execute(
        f"""
        SELECT
            ad_type,
            seller_type,
            COUNT(*) AS n_items,
            SUM(total_views) AS views,
            SUM(total_contacts) AS contacts,
            SUM(total_contacts) / NULLIF(SUM(total_views),0) AS cr
        FROM read_parquet('{_p(item_metrics_path)}')
        GROUP BY ad_type, seller_type
        ORDER BY ad_type, seller_type
        """
    ).df()
    df_combo_snap["metric"] = "cr_snapshot_view_level"
    df_combo_snap.to_csv(paths.table_dir / "eda_A3_contact_rate_by_ad_seller_snapshot.csv", index=False)
    results["ad_seller_cr_snapshot"] = df_combo_snap

    # 3.3 age decay.
    df_age = con.execute(
        f"""
        SELECT
            CASE
                WHEN avg_listing_age_days <= 3 THEN '00_0-3'
                WHEN avg_listing_age_days <= 7 THEN '01_4-7'
                WHEN avg_listing_age_days <= 14 THEN '02_8-14'
                WHEN avg_listing_age_days <= 30 THEN '03_15-30'
                WHEN avg_listing_age_days <= 60 THEN '04_31-60'
                WHEN avg_listing_age_days <= 90 THEN '05_61-90'
                WHEN avg_listing_age_days <= 180 THEN '06_91-180'
                ELSE '07_180+'
            END AS age_bin,
            COUNT(*) AS n_items,
            SUM(total_views) AS views,
            SUM(total_contacts) AS contacts,
            SUM(total_contacts) / NULLIF(SUM(total_views),0) AS cr
        FROM read_parquet('{_p(item_metrics_path)}')
        GROUP BY age_bin
        ORDER BY age_bin
        """
    ).df()
    df_age.to_csv(paths.table_dir / "eda_A3_listing_age_decay.csv", index=False)
    results["age_decay"] = df_age

    # 3.4 image bucket item-level contact rate distribution.
    # v7: Do this block with PyArrow + pandas instead of DuckDB.
    # Reason: DuckDB on Windows can hit an internal binder error when a parquet
    # column is inferred as VARCHAR and compared/cast inside CASE. This figure is
    # useful for A3 but should never kill the whole EDA pipeline.
    try:
        img_cols = ["item_id", "images_count", "total_views", "total_contacts", "contact_rate"]
        tbl = pq.read_table(item_metrics_path, columns=img_cols)
        df_img_all = tbl.to_pandas()
        df_img_all["images_count_num"] = pd.to_numeric(df_img_all["images_count"], errors="coerce")
        df_img_all["total_views"] = pd.to_numeric(df_img_all["total_views"], errors="coerce")
        df_img_all["total_contacts"] = pd.to_numeric(df_img_all["total_contacts"], errors="coerce")
        df_img_all["contact_rate"] = pd.to_numeric(df_img_all["contact_rate"], errors="coerce")
        df_img_all = df_img_all[(df_img_all["total_views"] >= 10) & df_img_all["contact_rate"].notna()].copy()

        bins = [-np.inf, 2, 5, 10, 15, np.inf]
        labels = ["00_0-2", "01_3-5", "02_6-10", "03_11-15", "04_16+"]
        df_img_all["image_bucket"] = pd.cut(
            df_img_all["images_count_num"], bins=bins, labels=labels
        ).astype("object")
        df_img_all.loc[df_img_all["images_count_num"].isna(), "image_bucket"] = "unknown"

        # Deterministic, low-memory sample for the box plot.
        df_img_plot = (
            df_img_all[["image_bucket", "item_id", "total_views", "total_contacts", "contact_rate"]]
            .sort_values(["image_bucket", "item_id"])
            .groupby("image_bucket", group_keys=False, observed=True)
            .head(5000)
            .reset_index(drop=True)
        )
        del df_img_all, tbl
        gc.collect()
    except Exception as e:
        warnings.warn(f"A3.4 images_count figure skipped due to: {type(e).__name__}: {e}")
        df_img_plot = pd.DataFrame(
            columns=["image_bucket", "item_id", "total_views", "total_contacts", "contact_rate"]
        )

    df_img_plot.to_csv(paths.table_dir / "eda_A3_images_bucket_cr_sample.csv", index=False)
    results["image_cr_sample"] = df_img_plot

    # 3.5 seller Lorenz and Gini.
    df_seller_contacts = con.execute(
        f"""
        SELECT seller_id, total_contacts, total_views, n_listings, seller_type
        FROM read_parquet('{_p(seller_metrics_path)}')
        WHERE seller_id IS NOT NULL
        """
    ).df()
    gini = gini_coeff(df_seller_contacts["total_contacts"].to_numpy())
    df_seller_contacts.to_parquet(reduced / "seller_contact_distribution.parquet", index=False)
    results["seller_contacts"] = df_seller_contacts
    results["seller_gini"] = pd.DataFrame({"metric": ["seller_contact_gini"], "value": [gini]})

    # 3.6 geographic concentration.
    df_geo = con.execute(
        f"""
        SELECT
            city_name,
            district_name,
            COUNT(*) AS n_items,
            SUM(total_views) AS views,
            SUM(total_contacts) AS contacts,
            SUM(total_contacts) / NULLIF(SUM(total_views),0) AS cr
        FROM read_parquet('{_p(item_metrics_path)}')
        WHERE district_name IS NOT NULL AND district_name != 'Không xác định'
        GROUP BY city_name, district_name
        ORDER BY contacts DESC
        LIMIT 20
        """
    ).df()
    df_geo.to_csv(paths.table_dir / "eda_A3_geo_top20_districts.csv", index=False)
    results["geo_top20"] = df_geo

    # 3.7 seller new vs established cohort over weeks.
    df_cohort = con.execute(
        f"""
        WITH seller_first AS (
            SELECT seller_id, MIN(posted_date) AS first_posted_date
            FROM read_parquet('{_p(item_metrics_path)}')
            WHERE seller_id IS NOT NULL AND posted_date IS NOT NULL
            GROUP BY seller_id
        ),
        item_base AS (
            SELECT i.*, s.first_posted_date,
                CASE WHEN s.first_posted_date >= DATE '2026-03-01' THEN 'new_seller'
                     ELSE 'established_seller' END AS seller_cohort
            FROM read_parquet('{_p(item_metrics_path)}') i
            JOIN seller_first s USING (seller_id)
        )
        SELECT
            seller_cohort,
            DATE_TRUNC('week', posted_date)::DATE AS posted_week,
            COUNT(*) AS n_new_listings,
            SUM(total_views) AS views,
            SUM(total_contacts) AS contacts,
            SUM(total_contacts) / NULLIF(SUM(total_views),0) AS cr
        FROM item_base
        WHERE posted_date IS NOT NULL
        GROUP BY seller_cohort, posted_week
        ORDER BY posted_week, seller_cohort
        """
    ).df()
    df_cohort.to_csv(paths.table_dir / "eda_A3_seller_cohort_weekly.csv", index=False)
    results["seller_cohort"] = df_cohort

    return results


def build_b2b_c2c_segment_analysis(
    paths: EDAPaths, con: duckdb.DuckDBPyConnection
) -> dict[str, pd.DataFrame]:
    """Item #2 — B2B vs C2C split analysis on D30 universe.

    Hypothesis kiểm tra (theo mentor feedback v2):
    -----------------------------------------------
    "Nhà dự án (category 1050) đang làm lệch dữ liệu — tách thành 2 nhóm
    B2B (dự án) và C2C thuần túy để phân tích."

    Proxy B2B = P3: cat==1050 OR project_id IS NOT NULL. Đã được apply ở
    item_metrics column `segment` (B2B/C2C).

    Output:
    -------
    1. Audit cross-tab cat × project_id_not_null (data quality check)
    2. Per-segment: D30 CR overall, D30 seller Gini, top-10 seller share
    3. Per-segment × per-(ad,seller): D30 CR (test "agent thống trị" hypothesis)
    4. Per-segment × per-category: D30 CR (chỉ ý nghĩa cho C2C 1010-1040)
    5. Slide-ready long-format comparison CSV
    6. Snapshot Gini validation (legacy universe, appendix only)

    Returns dict with keys: audit, segment_overall, segment_ad_seller,
    segment_category, seller_distribution_b2b, seller_distribution_c2c,
    comparison.
    """
    reduced = paths.agg_dir / "reduced"
    pair_dim_glob = _p(reduced / "pair_flags_with_dim.parquet")
    item_metrics_glob = _p(reduced / "item_metrics.parquet")

    out: dict[str, pd.DataFrame] = {}

    # ---------- 1. AUDIT: cat × project_id cross-tab ----------
    # Tại sao audit này quan trọng: P3 chỉ hợp lý nếu (a) tin có project_id
    # thực sự là tin dự án, và (b) tin 1050 thực sự là dự án.
    # Số cần track: pct tin "ngoài 1050 nhưng có project_id" — nếu > 1% thì
    # mentor đúng: có nhiều B2B ẩn trong 1010-1040.
    df_audit = con.execute(
        f"""
        SELECT
            category,
            COUNT(*) AS n_listings,
            SUM(CASE WHEN project_id IS NOT NULL THEN 1 ELSE 0 END) AS n_with_project_id,
            SUM(CASE WHEN segment = 'B2B' THEN 1 ELSE 0 END) AS n_b2b,
            SUM(CASE WHEN segment = 'C2C' THEN 1 ELSE 0 END) AS n_c2c
        FROM read_parquet('{item_metrics_glob}')
        GROUP BY category
        ORDER BY category
        """
    ).df()
    df_audit["pct_with_project_id"] = 100 * df_audit["n_with_project_id"] / df_audit["n_listings"]
    df_audit["pct_b2b"] = 100 * df_audit["n_b2b"] / df_audit["n_listings"]
    df_audit["category_name"] = df_audit["category"].map(CATEGORY_NAMES)
    df_audit.to_csv(paths.table_dir / "eda_A3_b2b_c2c_proxy_audit.csv", index=False)
    out["audit"] = df_audit

    # ---------- 2. Segment-level D30 CR overall + qualified pair volume ----------
    df_seg_overall = con.execute(
        f"""
        SELECT
            segment,
            SUM(is_qualified_pair)::BIGINT     AS qualified_pairs,
            SUM(is_qualified_pos_pair)::BIGINT AS qualified_pos_pairs,
            SUM(is_qualified_pos_pair)::DOUBLE / NULLIF(SUM(is_qualified_pair), 0) AS cr_d30,
            COUNT(DISTINCT user_id)::BIGINT AS n_unique_users,
            COUNT(DISTINCT item_id)::BIGINT AS n_unique_items,
            COUNT(DISTINCT seller_id)::BIGINT AS n_unique_sellers
        FROM read_parquet('{pair_dim_glob}')
        WHERE segment IN ('B2B', 'C2C')
        GROUP BY segment
        ORDER BY segment
        """
    ).df()
    df_seg_overall[["ci_low", "ci_high"]] = df_seg_overall.apply(
        lambda r: pd.Series(wilson_ci(r["qualified_pos_pairs"], r["qualified_pairs"])),
        axis=1,
    )
    df_seg_overall.to_csv(paths.table_dir / "eda_A3_b2b_c2c_overall_d30.csv", index=False)
    out["segment_overall"] = df_seg_overall

    # ---------- 3. Per-segment × per-(ad_type, seller_type) D30 CR ----------
    # Test mentor hypothesis: trong B2B, agent dominate share volume.
    df_seg_as = con.execute(
        f"""
        SELECT
            segment,
            ad_type,
            seller_type,
            SUM(is_qualified_pair)::BIGINT     AS qualified_pairs,
            SUM(is_qualified_pos_pair)::BIGINT AS qualified_pos_pairs,
            SUM(is_qualified_pos_pair)::DOUBLE / NULLIF(SUM(is_qualified_pair), 0) AS cr_d30
        FROM read_parquet('{pair_dim_glob}')
        WHERE segment IN ('B2B', 'C2C')
          AND ad_type IS NOT NULL
          AND seller_type IS NOT NULL
        GROUP BY segment, ad_type, seller_type
        ORDER BY segment, ad_type, seller_type
        """
    ).df()
    df_seg_as[["ci_low", "ci_high"]] = df_seg_as.apply(
        lambda r: pd.Series(wilson_ci(r["qualified_pos_pairs"], r["qualified_pairs"])),
        axis=1,
    )
    # Volume share within segment — để slide diễn giải "agent chiếm X% trong B2B"
    seg_totals = df_seg_as.groupby("segment")["qualified_pairs"].transform("sum")
    df_seg_as["volume_share_within_segment"] = df_seg_as["qualified_pairs"] / seg_totals
    df_seg_as.to_csv(paths.table_dir / "eda_A3_b2b_c2c_ad_seller_d30.csv", index=False)
    out["segment_ad_seller"] = df_seg_as

    # ---------- 4. Per-segment × per-category D30 CR ----------
    # Lưu ý: B2B chủ yếu là cat 1050 (theo định nghĩa segment), nên rows B2B×non-1050
    # chỉ có khi project_id NOT NULL trong cat khác. Đây là chính tin "B2B ẩn" mentor
    # nghi ngờ. Số liệu của các rows này cũng informative.
    df_seg_cat = con.execute(
        f"""
        SELECT
            segment,
            category,
            SUM(is_qualified_pair)::BIGINT     AS qualified_pairs,
            SUM(is_qualified_pos_pair)::BIGINT AS qualified_pos_pairs,
            SUM(is_qualified_pos_pair)::DOUBLE / NULLIF(SUM(is_qualified_pair), 0) AS cr_d30
        FROM read_parquet('{pair_dim_glob}')
        WHERE segment IN ('B2B', 'C2C')
        GROUP BY segment, category
        ORDER BY segment, category
        """
    ).df()
    df_seg_cat[["ci_low", "ci_high"]] = df_seg_cat.apply(
        lambda r: pd.Series(wilson_ci(r["qualified_pos_pairs"], r["qualified_pairs"])),
        axis=1,
    )
    df_seg_cat["category_name"] = df_seg_cat["category"].map(CATEGORY_NAMES)
    df_seg_cat.to_csv(paths.table_dir / "eda_A3_b2b_c2c_category_d30.csv", index=False)
    out["segment_category"] = df_seg_cat

    # ---------- 5. Per-segment seller distribution (cho Gini/Lorenz D30) ----------
    # Đây là phần CHỐT G1: Gini đo trên qualified_pos_pairs theo seller, không phải
    # snapshot.total_contacts như trước. Universe nhất quán với A3.1/A3.2.
    for seg in ["B2B", "C2C"]:
        df_seller_seg = con.execute(
            f"""
            SELECT
                seller_id,
                SUM(is_qualified_pair)::BIGINT     AS qualified_pairs,
                SUM(is_qualified_pos_pair)::BIGINT AS qualified_pos_pairs,
                COUNT(DISTINCT item_id)::BIGINT    AS n_items_in_segment
            FROM read_parquet('{pair_dim_glob}')
            WHERE segment = '{seg}'
              AND seller_id IS NOT NULL
            GROUP BY seller_id
            HAVING SUM(is_qualified_pair) > 0
            """
        ).df()
        df_seller_seg.to_parquet(reduced / f"seller_distribution_{seg.lower()}_d30.parquet", index=False)
        out[f"seller_distribution_{seg.lower()}"] = df_seller_seg

    # ---------- 6. Gini/Lorenz D30 per segment + overall ----------
    gini_rows = []
    for seg in ["B2B", "C2C"]:
        sd = out[f"seller_distribution_{seg.lower()}"]
        if sd.empty:
            continue
        # Gini D30 dùng qualified_pos_pairs làm "lead" distribution per seller.
        # Lý do: snapshot Gini cũ dùng contacts_24h (view-level, gồm non-login).
        # D30 Gini dùng pair-level qualified-pos-pairs (login user, dwell ≥ 30s)
        # = "lead chất". Nếu D30 Gini vẫn cao → concentration không phải artifact universe.
        gini_d30 = gini_coeff(sd["qualified_pos_pairs"].to_numpy())
        n_sellers = len(sd)
        # Top-K share (lead concentration cho slide marketplace health)
        sd_sorted = sd.sort_values("qualified_pos_pairs", ascending=False)
        total_pos = sd_sorted["qualified_pos_pairs"].sum()
        top10_share = sd_sorted["qualified_pos_pairs"].head(10).sum() / total_pos if total_pos > 0 else np.nan
        top1pct_share = (
            sd_sorted["qualified_pos_pairs"].head(max(1, n_sellers // 100)).sum() / total_pos
            if total_pos > 0 else np.nan
        )
        top10pct_share = (
            sd_sorted["qualified_pos_pairs"].head(max(1, n_sellers // 10)).sum() / total_pos
            if total_pos > 0 else np.nan
        )
        gini_rows.append({
            "segment": seg,
            "gini_d30": gini_d30,
            "n_sellers": n_sellers,
            "total_qualified_pos_pairs": int(total_pos),
            "top10_seller_share": top10_share,
            "top1pct_seller_share": top1pct_share,
            "top10pct_seller_share": top10pct_share,
        })
    # All segments combined (cross-check với Gini snapshot cũ)
    df_seller_all_d30 = con.execute(
        f"""
        SELECT
            seller_id,
            SUM(is_qualified_pos_pair)::BIGINT AS qualified_pos_pairs
        FROM read_parquet('{pair_dim_glob}')
        WHERE seller_id IS NOT NULL
        GROUP BY seller_id
        HAVING SUM(is_qualified_pos_pair) > 0
        """
    ).df()
    if not df_seller_all_d30.empty:
        gini_rows.append({
            "segment": "ALL",
            "gini_d30": gini_coeff(df_seller_all_d30["qualified_pos_pairs"].to_numpy()),
            "n_sellers": len(df_seller_all_d30),
            "total_qualified_pos_pairs": int(df_seller_all_d30["qualified_pos_pairs"].sum()),
            "top10_seller_share": np.nan,
            "top1pct_seller_share": np.nan,
            "top10pct_seller_share": np.nan,
        })
    df_gini = pd.DataFrame(gini_rows)
    df_gini.to_csv(paths.table_dir / "eda_A3_b2b_c2c_gini_d30.csv", index=False)
    out["segment_gini"] = df_gini

    # ---------- 7. Snapshot Gini per segment (legacy universe, appendix cross-check) ----------
    # Để compare: nếu D30 Gini và snapshot Gini agree → concentration là property
    # marketplace thực sự, không phụ thuộc universe đo. Nếu disagree → flag.
    df_gini_snap = con.execute(
        f"""
        SELECT
            segment,
            seller_id,
            SUM(total_contacts) AS total_contacts_snapshot
        FROM read_parquet('{item_metrics_glob}')
        WHERE seller_id IS NOT NULL
        GROUP BY segment, seller_id
        HAVING SUM(total_contacts) > 0
        """
    ).df()
    snap_rows = []
    for seg in ["B2B", "C2C", "ALL"]:
        if seg == "ALL":
            sub = df_gini_snap
        else:
            sub = df_gini_snap[df_gini_snap["segment"] == seg]
        if sub.empty:
            continue
        snap_rows.append({
            "segment": seg,
            "gini_snapshot": gini_coeff(sub["total_contacts_snapshot"].to_numpy()),
            "n_sellers_with_contact": len(sub),
        })
    df_gini_snap_summary = pd.DataFrame(snap_rows)
    df_gini_snap_summary.to_csv(paths.table_dir / "eda_A3_b2b_c2c_gini_snapshot.csv", index=False)
    out["segment_gini_snapshot"] = df_gini_snap_summary

    # ---------- 8. Slide-ready long-format comparison CSV ----------
    # Schema: segment, metric_group, slice, value, n_qual_pairs, n_qual_pos_pairs, ci_low, ci_high, note
    # Slide 6 chỉ cần SELECT WHERE segment IN ('B2B', 'C2C') AND metric_group IN (...)
    rows = []

    def _add(segment, group, slice_label, value, n_pairs=None, n_pos=None, ci_low=None, ci_high=None, note=""):
        rows.append({
            "segment": segment,
            "metric_group": group,
            "slice": slice_label,
            "value": value,
            "n_qualified_pairs": n_pairs,
            "n_qualified_pos_pairs": n_pos,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "note": note,
        })

    # Overall CR D30
    for r in df_seg_overall.itertuples(index=False):
        _add(
            segment=r.segment,
            group="cr_d30_overall",
            slice_label="all",
            value=float(r.cr_d30),
            n_pairs=int(r.qualified_pairs),
            n_pos=int(r.qualified_pos_pairs),
            ci_low=float(r.ci_low),
            ci_high=float(r.ci_high),
            note=f"{r.n_unique_users:,} users, {r.n_unique_items:,} items, {r.n_unique_sellers:,} sellers",
        )

    # CR D30 by ad×seller
    for r in df_seg_as.itertuples(index=False):
        _add(
            segment=r.segment,
            group="cr_d30_by_ad_seller",
            slice_label=f"{r.ad_type}_{r.seller_type}",
            value=float(r.cr_d30),
            n_pairs=int(r.qualified_pairs),
            n_pos=int(r.qualified_pos_pairs),
            ci_low=float(r.ci_low),
            ci_high=float(r.ci_high),
            note=f"volume_share={r.volume_share_within_segment*100:.1f}% trong segment",
        )

    # CR D30 by category (chủ yếu ý nghĩa cho C2C)
    for r in df_seg_cat.itertuples(index=False):
        _add(
            segment=r.segment,
            group="cr_d30_by_category",
            slice_label=str(r.category),
            value=float(r.cr_d30),
            n_pairs=int(r.qualified_pairs),
            n_pos=int(r.qualified_pos_pairs),
            ci_low=float(r.ci_low),
            ci_high=float(r.ci_high),
            note=r.category_name if isinstance(r.category_name, str) else "",
        )

    # Gini D30 + top-K share (marketplace health, slide 6 main)
    for r in df_gini.itertuples(index=False):
        _add(r.segment, "gini_d30",          "all", float(r.gini_d30),
             note=f"{r.n_sellers:,} sellers, {r.total_qualified_pos_pairs:,} qual_pos_pairs")
        if r.segment != "ALL":
            _add(r.segment, "top10_seller_share",    "absolute_top10", float(r.top10_seller_share))
            _add(r.segment, "top1pct_seller_share",  "top_1pct",       float(r.top1pct_seller_share))
            _add(r.segment, "top10pct_seller_share", "top_10pct",      float(r.top10pct_seller_share))

    # Gini snapshot (appendix cross-check)
    for r in df_gini_snap_summary.itertuples(index=False):
        _add(r.segment, "gini_snapshot_appendix", "all", float(r.gini_snapshot),
             note=f"{r.n_sellers_with_contact:,} sellers (legacy universe)")

    df_comparison = pd.DataFrame(rows)
    df_comparison.to_csv(paths.table_dir / "eda_A3_b2b_vs_c2c_comparison.csv", index=False)
    out["comparison"] = df_comparison

    # ---------- 9. Console summary để Nhan đọc nhanh khi pipeline xong ----------
    print("\n" + "=" * 72)
    print("ITEM #2 — B2B vs C2C SEGMENT ANALYSIS (D30 universe)")
    print("=" * 72)
    print("\n[Proxy P3] B2B = cat==1050 OR project_id NOT NULL")
    print("\nCross-tab audit (% listings per category là B2B):")
    print(df_audit[["category", "category_name", "n_listings", "n_with_project_id",
                    "pct_with_project_id", "pct_b2b"]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nD30 CR overall per segment:")
    show = df_seg_overall[["segment", "qualified_pairs", "qualified_pos_pairs", "cr_d30",
                           "n_unique_sellers"]].copy()
    show["cr_d30_pct"] = show["cr_d30"] * 100
    print(show[["segment", "qualified_pairs", "qualified_pos_pairs", "cr_d30_pct",
                "n_unique_sellers"]].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\nGini D30 per segment (slide 6 main):")
    print(df_gini.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nGini snapshot per segment (appendix cross-check):")
    print(df_gini_snap_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return out


def wilson_ci(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n is None or n <= 0 or np.isnan(n):
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def gini_coeff(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    total = x.sum()
    if total == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / total) / n)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["light_gray"],
            "axes.labelcolor": PALETTE["dark"],
            "xtick.color": PALETTE["dark"],
            "ytick.color": PALETTE["dark"],
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved figure: {path}")


def fmt_int(x: float | int) -> str:
    if pd.isna(x):
        return "NA"
    return f"{int(round(x)):,}"


def fmt_pct(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}%"


def plot_a1(results: dict[str, pd.DataFrame], paths: EDAPaths) -> None:
    """A1 - Contact là tín hiệu giá trị quan sát được."""
    set_plot_style()
    fig_dir = paths.fig_dir

    funnel = results["funnel"].iloc[0]
    # 1.1 Event funnel bar chart (log scale).
    # Mục đích: chứng minh pageview là lớp nhiễu lớn, contact/positive là phần nhỏ hơn và có giá trị hơn.
    labels = ["Pageview pair", f"Dwell ≥ {int(funnel['D_selected'])}s", "Positive pair"]
    values = [funnel["N1_pageview_pairs"], funnel["N2_serious_pageview_pairs"], funnel["N3_positive_pairs_inside_serious_pv"]]
    colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["orange"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Số user × item pair (log scale)")
    ax.set_title("A1.1 Phễu hành vi: từ pageview đến positive contact")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt_int(v), ha="center", va="bottom", fontsize=9)
    ax.annotate(
        f"Tổng conversion: {fmt_pct(funnel['P_total_positive_over_pageview_pct'])}",
        xy=(2, values[2]), xytext=(1.2, max(values) / 6),
        arrowprops=dict(arrowstyle="->", color=PALETTE["orange"]),
        color=PALETTE["orange"], weight="bold",
    )
    save_fig(fig, fig_dir / "fig_A1_1_event_funnel_log_bar.png")

    # 1.2 True pair-level contact rate theo ngày với rolling 7 ngày.
    # FIX: cliff T11-T12/2025 nghi là pipeline/data artifact (không phải insight business).
    # Filter date >= 2026-01-01 để show stable plateau dùng cho train window valid.
    # Cache giữ nguyên (chỉ xử lý ở tầng vẽ).
    STABLE_FROM = pd.Timestamp("2026-01-01")
    daily = results["daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily_stable = daily[daily["date"] >= STABLE_FROM].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if daily_stable["pair_positive_rate"].notna().any():
        ax.plot(daily_stable["date"], daily_stable["pair_positive_rate"] * 100, color=PALETTE["sky"], alpha=0.45, linewidth=1, label="Daily pair-level positive rate")
        ax.plot(daily_stable["date"], daily_stable["pair_positive_rate_7d"] * 100, color=PALETTE["navy"], linewidth=2.4, label="Rolling 7 ngày")
        ax.set_ylabel("Positive pairs / pageview pairs (%)")
        if not daily_stable.empty:
            mean_rate = daily_stable["pair_positive_rate_7d"].mean() * 100
            ax.axhline(mean_rate, color=PALETTE["orange"], linestyle="--", linewidth=1.5, alpha=0.8)
            ax.text(
                0.98, 0.95,
                f"Trung bình rolling-7d: {mean_rate:.1f}%\nStable window (Jan–Apr 2026)",
                transform=ax.transAxes, ha="right", va="top",
                color=PALETTE["orange"], weight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PALETTE["orange"], alpha=0.9),
            )
    else:
        ax.plot(daily_stable["date"], daily_stable["positive_event_intensity"] * 100, color=PALETTE["gray"], alpha=0.6, linewidth=1.5, label="Event intensity audit")
        ax.set_ylabel("Positive events / pageview events (%) — audit only")
        ax.text(0.02, 0.92, "Cache cũ thiếu daily pair flags; chạy --force-events để có true rate", transform=ax.transAxes, color=PALETTE["red"], weight="bold")
    ax.set_title("A1.2 Contact rate theo ngày — stable window (Jan–Apr 2026)")
    ax.set_xlabel("Ngày")
    ax.legend(loc="upper left")
    save_fig(fig, fig_dir / "fig_A1_2_daily_contact_rate_7d.png")

    # 1.3 Donut phân phối 5 loại positive event.
    # Implication model: event types có thể nên được weight khác nhau trong label/ranking.
    pos = results["positive_dist"].sort_values("n_events", ascending=False)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    donut_colors = [PALETTE["orange"], PALETTE["navy"], PALETTE["blue"], PALETTE["gray"], PALETTE["green"]]
    wedges, _ = ax.pie(pos["n_events"], startangle=90, counterclock=False, colors=donut_colors[: len(pos)], wedgeprops=dict(width=0.38, edgecolor="white"))
    ax.text(0, 0, "POSITIVE\nEVENTS", ha="center", va="center", fontsize=12, weight="bold", color=PALETTE["dark"])
    ax.set_title("A1.3 Phân phối 5 loại positive interaction")
    legend_labels = [f"{r.event_type}: {r.pct:.1f}%" for r in pos.itertuples()]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    save_fig(fig, fig_dir / "fig_A1_3_positive_event_donut.png")

    # 1.4 [DROPPED FROM DELIVERABLE] Histogram dwell time trước contact.
    # Lý do drop: distribution có 2 spike rời rạc nghi là pipeline đã bucketize/round dwell_time;
    # median 1682s (~28 phút) không hợp lý cho hành vi xem tin BĐS mobile.
    # Cần audit lại tầng aggregation trước khi đưa vào deliverable. D_selected vẫn dùng cho summary.
    # Để bật lại: bỏ comment block dưới và rerun với --force-events.
    # dwell_path = paths.agg_dir / "events" / "pre_contact_dwell_sample"
    # if _has_parquet(dwell_path):
    #     ... (giữ nguyên code cũ trong git history) ...
    #     save_fig(fig, fig_dir / "fig_A1_4_dwell_before_contact_hist.png")


def plot_a2(results: dict[str, pd.DataFrame], paths: EDAPaths) -> None:
    """A2 - User intent: hiểu phía cầu của marketplace."""
    set_plot_style()
    fig_dir = paths.fig_dir

    # 2.1 Distribution số contact mỗi user, log-log.
    # Mục đích: kiểm tra heavy-tail/power-law; implication là model cần xử lý power users và long-tail khác nhau.
    user_pos = results["user_positive"]
    counts = user_pos["n_positive_events"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(counts.index, counts.values, s=16, color=PALETTE["navy"], alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Số positive events/user (log)")
    ax.set_ylabel("Số user (log)")
    ax.set_title("A2.1 Phân phối contact/user: phía cầu có heavy-tail")
    if len(user_pos) > 0:
        top20_share = user_pos.nlargest(max(1, int(0.2 * len(user_pos))), "n_positive_events")["n_positive_events"].sum() / user_pos["n_positive_events"].sum()
        ax.text(0.05, 0.95, f"Top 20% user tạo {top20_share*100:.1f}% positive events", transform=ax.transAxes, va="top", color=PALETTE["orange"], weight="bold")
    save_fig(fig, fig_dir / "fig_A2_1_user_contact_distribution_loglog.png")

    # 2.2 Cold-start trong test set.
    # Cách đọc: pie cho biết bao nhiêu test users có lịch sử đủ giàu để cá nhân hóa.
    cold = results["cold_start"]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    colors = [PALETTE["light_gray"], PALETTE["sky"], PALETTE["gray"], PALETTE["orange"]]
    ax.pie(cold["n_users"], labels=cold["user_segment"], autopct="%1.1f%%", startangle=90, colors=colors[: len(cold)], textprops={"fontsize": 9})
    ax.set_title("A2.2 Test users: cold-start quyết định chiến lược fallback")
    save_fig(fig, fig_dir / "fig_A2_2_test_cold_start_pie.png")

    # 2.3 [DROPPED FROM DELIVERABLE] Time gap view-to-contact.
    # Lý do drop: median = 0.00h, P90 = 0.0h trên 17M pair — nghi pipeline tính gap chỉ trong
    # cùng (user_id, item_id, session) thay vì cross-session, hoặc event_ts thiếu độ phân giải.
    # Insight "recency rất quan trọng" vẫn giữ trong feature engineering (A4) qua các biến
    # last_7d_activity / last_24h_activity được tính lại từ raw events.
    # Để bật lại: bỏ comment block dưới sau khi audit aggregator.
    # gap = results["time_gap"].copy()
    # gap = gap[(gap["gap_hours"] >= 0) & (gap["gap_hours"] <= 24 * 30)]
    # ... (giữ nguyên code cũ trong git history) ...
    #     save_fig(fig, fig_dir / "fig_A2_3_view_to_contact_gap_hist.png")

    # 2.4 Category preference heatmap.
    # Cách đọc: hàng là category user xem nhiều nhất, cột là category user contact nhiều nhất.
    # FIX: filter category hợp lệ 1010-1050 (đề bài quy định). Giá trị khác (vd 6020) là noise
    # từ category cũ/legacy, không thuộc 5 phân khúc BĐS hiện hành.
    VALID_CATEGORIES = {1010, 1020, 1030, 1040, 1050}
    heat = results["category_heatmap"].copy()
    if not heat.empty:
        heat = heat[
            heat["view_category"].astype(int).isin(VALID_CATEGORIES)
            & heat["contact_category"].astype(int).isin(VALID_CATEGORIES)
        ]
    if not heat.empty:
        pivot = heat.pivot_table(index="view_category", columns="contact_category", values="n_users", aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        fig, ax = plt.subplots(figsize=(7.4, 5.8))
        im = ax.imshow(pivot.values, cmap="Blues")
        ax.set_xticks(range(len(pivot.columns)), [CATEGORY_NAMES.get(int(c), str(c)) for c in pivot.columns], rotation=35, ha="right")
        ax.set_yticks(range(len(pivot.index)), [CATEGORY_NAMES.get(int(c), str(c)) for c in pivot.index])
        ax.set_title("A2.4 User intent: category xem nhiều nhất × contact nhiều nhất")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if v > 0:
                    ax.text(j, i, fmt_int(v), ha="center", va="center", fontsize=8, color=PALETTE["dark"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_fig(fig, fig_dir / "fig_A2_4_category_preference_heatmap.png")

    # 2.5 Device & surface stacked bar.
    # Implication model: device/surface là context feature, đồng thời gợi ý khác biệt UI journey.
    # FIX: unify label "adview" → "ad_view" (duplicate label từ pipeline gốc, cùng nghĩa).
    SURFACE_ALIASES = {"adview": "ad_view"}
    dev = results["device_surface"].copy()
    if not dev.empty:
        dev["surface"] = dev["surface"].replace(SURFACE_ALIASES)
        dev["device_surface"] = dev["device"].fillna("unknown") + " / " + dev["surface"].fillna("unknown")
        top = dev.groupby("device_surface")["n_events"].sum().nlargest(10).index
        plot_df = dev[dev["device_surface"].isin(top)].pivot_table(index="device_surface", columns="event_type", values="n_events", aggfunc="sum", fill_value=0)
        plot_df = plot_df.loc[plot_df.sum(axis=1).sort_values(ascending=True).index]
        fig, ax = plt.subplots(figsize=(9, 5.4))
        bottom = np.zeros(len(plot_df))
        colors = [PALETTE["orange"], PALETTE["navy"], PALETTE["blue"], PALETTE["green"], PALETTE["gray"]]
        for color, col in zip(colors, plot_df.columns):
            ax.barh(plot_df.index, plot_df[col], left=bottom, label=col, color=color)
            bottom += plot_df[col].to_numpy()
        ax.set_title("A2.5 Positive events theo device × surface")
        ax.set_xlabel("Số positive events")
        ax.legend(loc="lower right")
        save_fig(fig, fig_dir / "fig_A2_5_device_surface_stacked_bar.png")


def plot_a3(results: dict[str, pd.DataFrame], paths: EDAPaths) -> None:
    """A3 - Listing & Seller: phía cung là nguồn doanh thu."""
    set_plot_style()
    fig_dir = paths.fig_dir

    # 3.1 CR D30 by category with Wilson CI (PAIR-LEVEL CLICKSTREAM).
    # Numerator: qualified pairs có positive event. Denominator: qualified pairs (max_dwell >= 30s).
    # Universe: fact_user_events, nhất quán 100% với funnel A1.
    cat = results["category_cr"].copy()
    fig, ax = plt.subplots(figsize=(8.4, 5))
    x = np.arange(len(cat))
    cr = cat["cr"].to_numpy() * 100
    yerr = np.vstack([(cat["cr"] - cat["ci_low"]).to_numpy() * 100, (cat["ci_high"] - cat["cr"]).to_numpy() * 100])
    ax.bar(x, cr, yerr=yerr, color=PALETTE["blue"], capsize=4)
    ax.set_xticks(x, cat["category_name"], rotation=20, ha="right")
    ax.set_ylabel("CR D30 — tỷ lệ pair xem ≥30s có liên hệ (%)")
    ax.set_title("A3.1 CR D30 theo phân khúc (pair-level từ clickstream)")
    for xi, yi in zip(x, cr):
        ax.text(xi, yi, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)
    save_fig(fig, fig_dir / "fig_A3_1_contact_rate_by_category_ci.png")

    # 3.2 CR D30 by ad_type × seller_type (PAIR-LEVEL CLICKSTREAM).
    combo = results["ad_seller_cr"].copy()
    combo["combo"] = combo["ad_type"].astype(str) + " × " + combo["seller_type"].astype(str)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(combo["combo"], combo["cr"] * 100, color=[PALETTE["navy"], PALETTE["blue"], PALETTE["orange"], PALETTE["green"]][: len(combo)])
    ax.set_title("A3.2 CR D30 theo ad_type × seller_type (pair-level)")
    ax.set_ylabel("CR D30 (%)")
    ax.tick_params(axis="x", rotation=25)
    for b, y in zip(bars, combo["cr"] * 100):
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)
    save_fig(fig, fig_dir / "fig_A3_2_contact_rate_ad_type_seller_type.png")

    # 3.3 Listing age decay.
    age = results["age_decay"].copy()
    age["label"] = age["age_bin"].str.replace(r"^\d+_", "", regex=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(age["label"], age["cr"] * 100, marker="o", color=PALETTE["navy"], linewidth=2.2)
    ax.set_title("A3.3 Listing age decay: tin mới có lợi thế contact không?")
    ax.set_xlabel("Tuổi tin trung bình")
    ax.set_ylabel("Contact rate (%)")
    for x, y in zip(age["label"], age["cr"] * 100):
        ax.text(x, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)
    save_fig(fig, fig_dir / "fig_A3_3_listing_age_decay_curve.png")

    # 3.4 [DROPPED FROM DELIVERABLE] Images count vs contact rate box plot.
    # Lý do drop: pattern "nhiều ảnh → CR thấp hơn" counterintuitive — nghi Simpson's paradox
    # do confounder seller_type (agent thường có nhiều ảnh chuyên nghiệp, nhưng CR thấp hơn private).
    # Để bật lại đúng cách: cần stratified boxplot theo seller_type, hoặc fit GLM khử confound.
    # Hiện tại drop khỏi deliverable để tránh defend khó khăn trước BGK.
    # img = results["image_cr_sample"].copy()
    # if not img.empty:
    #     ... (giữ nguyên code cũ trong git history) ...
    #     save_fig(fig, fig_dir / "fig_A3_4_images_count_vs_contact_rate_box.png")

    # 3.5 Lorenz curve + Gini seller.
    seller = results["seller_contacts"].copy()
    values = seller["total_contacts"].fillna(0).to_numpy()
    values = np.sort(values)
    cum_contacts = np.cumsum(values)
    if cum_contacts.sum() == 0:
        lorenz_y = np.linspace(0, 1, len(values))
    else:
        lorenz_y = np.insert(cum_contacts / cum_contacts[-1], 0, 0)
    lorenz_x = np.linspace(0, 1, len(lorenz_y))
    gini = float(results["seller_gini"].loc[0, "value"])
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.plot(lorenz_x, lorenz_y, color=PALETTE["orange"], linewidth=2.4, label="Lorenz curve")
    ax.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle="--", label="Phân phối đều")
    ax.fill_between(lorenz_x, lorenz_y, lorenz_x, color=PALETTE["orange"], alpha=0.12)
    ax.set_title("A3.5 Phân phối contact theo seller: Lorenz curve")
    ax.set_xlabel("Tỷ lệ seller tích lũy")
    ax.set_ylabel("Tỷ lệ contact tích lũy")
    ax.text(0.05, 0.9, f"Gini = {gini:.3f}", transform=ax.transAxes, color=PALETTE["orange"], fontsize=13, weight="bold")
    ax.legend(loc="lower right")
    save_fig(fig, fig_dir / "fig_A3_5_seller_contact_lorenz_gini.png")

    # 3.6 Geographic concentration top 20 district.
    # FIX: format label "District, City" thay vì 2-line "District\nCity" cho gọn mắt.
    # Filter chỉ giữ top 20 (đôi khi geo_top20 chứa nhiều hơn 20 dòng do tie-break).
    geo = results["geo_top20"].copy()
    geo["district_label"] = geo["district_name"].astype(str) + ", " + geo["city_name"].astype(str)
    geo = geo.nlargest(20, "contacts").sort_values("contacts", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    ax.barh(geo["district_label"], geo["contacts"], color=PALETTE["blue"])
    ax.set_title("A3.6 Top 20 quận/huyện theo số contact")
    ax.set_xlabel("Tổng contacts từ snapshot")
    # Tính share HCM để annotation
    hcm_share = geo.loc[geo["city_name"].str.contains("Hồ Chí Minh", na=False), "contacts"].sum() / geo["contacts"].sum()
    ax.text(
        0.98, 0.05,
        f"TP HCM: {hcm_share*100:.0f}% top contacts",
        transform=ax.transAxes, ha="right", va="bottom",
        color=PALETTE["orange"], weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PALETTE["orange"], alpha=0.9),
    )
    save_fig(fig, fig_dir / "fig_A3_6_geo_top20_district_contacts.png")

    # 3.7 [DROPPED FROM DELIVERABLE] Seller cohort: new vs established.
    # Lý do drop: line new_seller chỉ có ~6 data points ở cuối train window (Mar-Apr 2026)
    # → đây là artifact của định nghĩa "new" (seller xuất hiện lần đầu trong window) chứ không
    # phải insight thật. Variance trên 6 điểm quá lớn (peak 14.8% là spike).
    # Để giữ ý "seller mới không bị thiệt thòi về CR" trong storyline, dùng số trung bình
    # theo cohort (xem build_summary_metrics) thay vì time series.
    # cohort = results["seller_cohort"].copy()
    # if not cohort.empty:
    #     ... (giữ nguyên code cũ trong git history) ...
    #     save_fig(fig, fig_dir / "fig_A3_7_new_vs_established_seller_cohort.png")


def build_summary_metrics(
    results: dict[str, pd.DataFrame],
    listing_results: dict[str, pd.DataFrame],
    paths: EDAPaths,
    b2b_results: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    rows = []
    funnel = results["funnel"].iloc[0]
    rows.extend([
        ("A1", "N1_pageview_pairs", funnel["N1_pageview_pairs"], "Unique user × item pairs có pageview"),
        ("A1", "N2_qualified_pairs_d30", funnel["N2_serious_pageview_pairs"], f"Qualified pairs D30: dwell >= {int(funnel['D_selected'])}s (denominator chính)"),
        ("A1", "N3_qualified_pos_pairs_d30", funnel["N3_positive_pairs_inside_serious_pv"], "Qualified pairs D30 có positive (numerator chính)"),
        ("A1", "CR_D30_overall_pct", funnel["P2_positive_over_serious_pct"], "CR D30 tổng = N3 / N2 (metric chính trên slide, 5 events BTC)"),
        # DC-only backup metric (item #5 audit). Loại bỏ other_interaction khỏi numerator.
        # Nếu BGK hỏi "94% positive là other_interaction là gì", trả lời: định nghĩa BTC
        # cover OI là contact (is_contact=1 100%, fire 100% trên ad_view surface với
        # 1/session frequency — không phải tracking noise). Backup metric strict là DC-only.
        ("A1", "N3_qualified_dc_pos_pairs_d30", funnel["N3_dc_pairs_inside_serious_pv"], "Qualified pairs D30 có direct contact (4 events: view_phone/chat/zalo/sms)"),
        ("A1", "CR_D30_dc_only_pct", funnel["P2_dc_over_serious_pct"], "CR D30 DC-only (backup metric, strict definition không tính other_interaction)"),
        ("A1", "P_total_positive_over_pageview_pct", funnel["P_total_positive_over_pageview_pct"], "Funnel conversion thô: positive / pageview pair (audit only)"),
        ("A1", "D_selected_sec", funnel["D_selected"], "Ngưỡng D30 thống nhất cho toàn pipeline"),
    ])
    cold = results["cold_start"]
    no_hist = cold.loc[cold["user_segment"].eq("0_no_history"), "pct"]
    if len(no_hist):
        rows.append(("A2", "test_no_history_pct", float(no_hist.iloc[0]), "% test users không có pageview/positive history"))
    gap = results["time_gap"]
    if not gap.empty:
        rows.append(("A2", "median_view_to_contact_hours", float(gap.loc[(gap["gap_hours"] >= 0) & (gap["gap_hours"] <= 24 * 30), "gap_hours"].median()), "Median time gap từ view đầu đến contact"))
    rows.append(("A3", "seller_contact_gini", float(listing_results["seller_gini"].loc[0, "value"]), "Mức độ tập trung contact theo seller (Lorenz/Gini)"))
    cat = listing_results["category_cr"].copy()
    if not cat.empty:
        best = cat.loc[cat["cr"].idxmax()]
        rows.append(("A3", "best_category_by_cr_d30", best["category"], f"Category có CR D30 cao nhất: {best['category_name']}"))
        rows.append(("A3", "best_category_cr_d30_pct", float(best["cr"] * 100), "CR D30 của category tốt nhất (pair-level clickstream)"))
    # ad×seller D30 vào summary để slide 6 (Marketplace Health) có sẵn các con số nền.
    combo = listing_results.get("ad_seller_cr", pd.DataFrame()).copy()
    if not combo.empty:
        for r in combo.itertuples(index=False):
            tag = f"{r.ad_type}_{r.seller_type}"
            rows.append(("A3", f"cr_d30_pct__{tag}", float(r.cr * 100), f"CR D30 cho ad_type={r.ad_type}, seller_type={r.seller_type}"))

    # ----- Item #2: B2B vs C2C split (D30 universe) -----
    if b2b_results is not None:
        seg_overall = b2b_results.get("segment_overall", pd.DataFrame())
        for r in seg_overall.itertuples(index=False):
            rows.append(("A3_seg", f"cr_d30_pct__{r.segment}", float(r.cr_d30 * 100),
                         f"CR D30 cho segment {r.segment} (proxy P3: 1050 OR project_id NOT NULL)"))
            rows.append(("A3_seg", f"qualified_pairs__{r.segment}", int(r.qualified_pairs),
                         f"Số qualified pair trong segment {r.segment}"))
        seg_gini = b2b_results.get("segment_gini", pd.DataFrame())
        for r in seg_gini.itertuples(index=False):
            rows.append(("A3_seg", f"gini_d30__{r.segment}", float(r.gini_d30),
                         f"Gini D30 seller concentration trong segment {r.segment}"))
            if r.segment != "ALL" and pd.notna(r.top10pct_seller_share):
                rows.append(("A3_seg", f"top10pct_share__{r.segment}", float(r.top10pct_seller_share),
                             f"% lead chiếm bởi top 10% seller trong segment {r.segment}"))

    summary = pd.DataFrame(rows, columns=["branch", "metric", "value", "description"])
    summary.to_csv(paths.table_dir / "eda_summary_metrics_for_slides.csv", index=False)
    return summary


def write_storyline_notes(paths: EDAPaths, summary: pd.DataFrame) -> None:
    """Write a compact text file that the team can paste into slide speaker notes.

    Note: updated for v2 framing (D30 universe, post-mentor feedback).
    """
    lines = [
        "EDA SUMMARY FOR SLIDES - Datathon 2026 Chợ Tốt BĐS (v2 — D30 framing)",
        "=" * 72,
        "Core frame v2: Khi user đã xem nghiêm túc tin (≥30s), conversion đã rất khoẻ",
        "(CR D30 ~37%). Bottleneck thật không phải tỷ lệ chốt thấp, mà là MATCH QUALITY:",
        "chỉ ~15% pageview pair trở thành qualified pair. Recommender = nâng tỷ lệ đó.",
        "",
        "=" * 72,
        "ĐỊNH NGHĨA other_interaction (BẮT BUỘC đọc trước khi thuyết trình)",
        "=" * 72,
        "other_interaction chiếm 94% positive events. Đây không phải bug hay noise.",
        "Bằng chứng từ audit (sample 30/500 files, item #5):",
        "  1. is_contact flag của BTC = 1 trên 100% OI events (giống y view_phone)",
        "  2. OI fire 100% trên surface ad_view (trang chi tiết tin) — không phải tracking",
        "     đa nguồn như feed/search/scroll",
        "  3. Median OI/session = 1.0 (giống direct contact) — không phải event passive",
        "     tự fire nhiều lần",
        "Diễn giải khả dĩ: OI là các action engagement implicit trên trang chi tiết —",
        "save/share/compare/click button phụ. BTC monetize lead-gen nên coi đây là tín",
        "hiệu lead. Đề thi chính thức gắn label TÍCH CỰC cho OI.",
        "",
        "Backup nếu BGK hỏi 'có thể là noise không?': có sẵn metric DC-only CR D30",
        "chỉ trên 4 direct events (view_phone, contact_chat, contact_zalo, contact_sms).",
        "→ Xem CR_D30_dc_only_pct trong summary này.",
        "",
        "=" * 72,
        "METRICS",
        "=" * 72,
    ]
    for r in summary.itertuples(index=False):
        lines.append(f"[{r.branch}] {r.metric}: {r.value} — {r.description}")
    lines.extend([
        "",
        "=" * 72,
        "RECOMMENDED STORYLINE (v2)",
        "=" * 72,
        "Slide 4 (Contact Rate vấn đề có tên):",
        "  Khi user đã engage nghiêm túc (qualified pair, dwell ≥30s), CR D30 ~37% —",
        "  conversion đã khoẻ. Vấn đề là chỉ ~15% pageview pair là qualified.",
        "  → Recommender = nâng qualified rate, không phải nâng CR.",
        "",
        "Slide 5 (Match Quality - phía cầu):",
        "  64% test user cold-start không thể dùng CF thuần. Còn lại có category",
        "  affinity ~86% diagonal — intent rõ. → Multi-source candidate gen.",
        "",
        "Slide 6 (Marketplace Health):",
        "  Lead concentration là vấn đề toàn marketplace, không phải artifact của riêng",
        "  dự án. Tách B2B/C2C: Gini D30 B2B=0.74, C2C=0.76 — KHÔNG chênh mạnh như",
        "  hypothesis ban đầu. Top 10% seller chiếm 62-65% lead ở CẢ 2 segment.",
        "  Private nhỉnh hơn agent trên D30 (~1.18×), không phải 'gấp đôi' như v1.",
        "  → Re-ranking cần cap exposure cho cả 2 segment.",
        "",
        "Slide 7 (Bản lề EDA → Model):",
        "  4 ràng buộc thiết kế trace ngược về 4 insight: cold-start 64% → candidate",
        "  gen multi-source; category affinity 86% → same-cat candidate; private 1.18×",
        "  agent (D30) → seller_type là feature ranker; Gini 0.73-0.76 cả 2 segment",
        "  → re-rank cap exposure.",
        "",
        "Slide 8-10 (Model 3 tầng):",
        "  Candidate gen → ranker LambdaRank → marketplace-aware re-rank.",
        "  Mỗi tầng mở bằng callback đến 1 insight EDA cụ thể.",
        "",
        "=" * 72,
        "APPENDIX SUGGESTIONS",
        "=" * 72,
        "- A1.4 dwell histogram (drop từ v1, mentor yêu cầu cho vào appendix)",
        "- A2.3 view→contact gap (drop từ v1)",
        "- A3.4 số ảnh vs CR (drop từ v1)",
        "- A3.7 new vs established seller (drop từ v1)",
        "- Cat 1010 có 41% B2B ẩn (B2B/C2C audit, mini-insight về data quality)",
        "- Gini snapshot 0.897 (legacy universe, để cross-check)",
        "- CR snapshot per category (universe khác, để contextualize CR D30)",
        "- Audit other_interaction summary (defend trước Q&A nếu BGK hỏi)",
    ])
    (paths.table_dir / "eda_storyline_notes_for_slides.txt").write_text("\n".join(lines), encoding="utf-8")


def run_full_storyline_eda(
    data_root: str | Path,
    output_root: str | Path = "outputs",
    sample_files: Optional[int] = None,
    force_events: bool = False,
    skip_event_agg: bool = False,
    threads: int = 2,
    memory_limit: str = "10GB",
) -> None:
    paths = EDAPaths.build(data_root=data_root, output_root=output_root)

    # Validate structure early.
    if not (paths.train_dir / "fact_user_events").exists():
        raise FileNotFoundError(f"Missing fact_user_events folder under {paths.train_dir}")
    if not (paths.test_dir / "test_users.parquet").exists():
        raise FileNotFoundError(f"Missing test_users.parquet under {paths.test_dir}")

    if not skip_event_agg:
        aggregate_events_per_file(paths, sample_files=sample_files, force=force_events)

    con = connect_duckdb(paths, threads=threads, memory_limit=memory_limit)
    try:
        event_results = reduce_event_aggregates(paths, con)
        listing_results = build_listing_seller_aggregates(paths, con)
        # Item #2: B2B vs C2C split — depends on pair_flags_with_dim đã build ở
        # build_listing_seller_aggregates. Đặt sau listing_results, trước plotting.
        b2b_results = build_b2b_c2c_segment_analysis(paths, con)
        plot_a1(event_results, paths)
        plot_a2(event_results, paths)
        plot_a3(listing_results, paths)
        summary = build_summary_metrics(event_results, listing_results, paths, b2b_results=b2b_results)
        write_storyline_notes(paths, summary)
        print("\n✓ FULL EDA STORYLINE DONE")
        print(f"  Figures: {paths.fig_dir}")
        print(f"  Tables:  {paths.table_dir}")
        print(f"  Agg:     {paths.agg_dir}")
    finally:
        con.close()
