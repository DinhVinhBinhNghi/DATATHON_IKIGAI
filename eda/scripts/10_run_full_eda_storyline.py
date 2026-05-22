from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eda.storyline_full_eda import run_full_storyline_eda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full 5-branch EDA storyline for Datathon 2026 Cho Tot BDS recommender."
    )
    parser.add_argument("--data-root", required=True, help="Folder containing train/ and test/, e.g. C:/Datathon_Data")
    parser.add_argument("--output-root", default="outputs", help="Output folder for agg/tables/figures")
    parser.add_argument("--sample-files", type=int, default=None, help="Exploratory mode: only scan first N fact_user_events files")
    parser.add_argument("--force-events", action="store_true", help="Rebuild event pre-aggregations even if cache exists")
    parser.add_argument("--skip-event-agg", action="store_true", help="Skip Layer-1 event aggregation and reuse outputs/agg/events")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="10GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full_storyline_eda(
        data_root=args.data_root,
        output_root=args.output_root,
        sample_files=args.sample_files,
        force_events=args.force_events,
        skip_event_agg=args.skip_event_agg,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )


if __name__ == "__main__":
    main()
