"""Config loader: đọc config/local.yaml, resolve paths, cache singleton.

Pattern:
    from src.common import get_config
    cfg = get_config()
    print(cfg.paths.raw_root)
    print(cfg.weights.hard_contact)
    print(cfg.popularity.window_days)

Config được load 1 lần (cached). Tất cả nested dicts được wrap thành SimpleNamespace
để truy cập bằng dot notation. Paths được resolve thành pathlib.Path absolute.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Cache singleton (load 1 lần per process)
_CONFIG: SimpleNamespace | None = None


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dict → SimpleNamespace để dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj


def _resolve_paths(cfg: SimpleNamespace, project_root: Path) -> None:
    """Resolve các path trong cfg.paths thành Path absolute.

    Path tương đối (vd "./cache") resolve relative tới project_root.
    Path tuyệt đối (vd "C:/Datathon_Data") giữ nguyên.
    """
    if not hasattr(cfg, "paths"):
        return
    p = cfg.paths
    for attr in ["raw_root", "cache_root", "submissions_root",
                 "outputs_root", "logs_root"]:
        if hasattr(p, attr):
            val = getattr(p, attr)
            path = Path(val)
            if not path.is_absolute():
                path = (project_root / path).resolve()
            setattr(p, attr, path)

    # Convenient derived paths
    if hasattr(p, "raw_root"):
        p.dim_listing_dir = p.raw_root / "train" / "dim_listing"
        p.fact_events_dir = p.raw_root / "train" / "fact_user_events"
        p.fact_snapshot_dir = p.raw_root / "train" / "fact_listing_snapshot"
        p.fact_interactions_dir = p.raw_root / "train" / "fact_post_contact_interactions"
        p.test_users_file = p.raw_root / "test" / "test_users.parquet"

    if hasattr(p, "cache_root"):
        p.agg_dir = p.cache_root / "agg"
        p.candidates_dir = p.cache_root / "candidates"
        p.features_dir = p.cache_root / "features"
        p.models_dir = p.cache_root / "models"
        p.gt_dir = p.cache_root / "ground_truth"

    if hasattr(p, "outputs_root"):
        p.tables_dir = p.outputs_root / "tables"

    # Create dirs if not exist (only output dirs, not raw_root)
    for attr in ["cache_root", "submissions_root", "outputs_root", "logs_root",
                 "agg_dir", "candidates_dir", "features_dir", "models_dir",
                 "gt_dir", "tables_dir"]:
        if hasattr(p, attr):
            getattr(p, attr).mkdir(parents=True, exist_ok=True)


def _find_project_root() -> Path:
    """Tìm project root bằng cách walk up từ file này tới khi gặp config/."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent
    raise RuntimeError(
        f"Không tìm được project root (config/ + src/) từ {here}"
    )


def get_config(force_reload: bool = False) -> SimpleNamespace:
    """Load config singleton.

    Args:
        force_reload: nếu True, reload từ disk (dùng cho testing).

    Returns:
        SimpleNamespace với dot-access, paths đã resolve thành Path absolute.
    """
    global _CONFIG
    if _CONFIG is not None and not force_reload:
        return _CONFIG

    root = _find_project_root()
    cfg_path = root / "config" / "local.yaml"
    if not cfg_path.exists():
        # Fallback: dùng local.example.yaml để test/CI work
        example = root / "config" / "local.example.yaml"
        if example.exists():
            cfg_path = example
        else:
            raise FileNotFoundError(
                f"Không tìm thấy config: {cfg_path} hoặc {example}. "
                "Hãy copy config/local.example.yaml -> config/local.yaml"
            )

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = _to_namespace(raw)
    cfg.project_root = root
    cfg.config_file = cfg_path
    _resolve_paths(cfg, root)

    _CONFIG = cfg
    return cfg
