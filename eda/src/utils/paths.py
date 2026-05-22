from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    train_path: Path
    test_path: Path
    clean_dir: Path
    cache_dir: Path
    table_dir: Path
    figure_dir: Path
    model_dir: Path
    submission_dir: Path
    duckdb_tmp: Path


def resolve_paths(data_root: str | Path, project_root: str | Path | None = None) -> ProjectPaths:
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    data_root = Path(data_root).expanduser().resolve()
    paths = ProjectPaths(
        project_root=root,
        data_root=data_root,
        train_path=data_root / "train",
        test_path=data_root / "test",
        clean_dir=root / "data" / "clean",
        cache_dir=root / "outputs" / "cache",
        table_dir=root / "outputs" / "tables",
        figure_dir=root / "outputs" / "figures",
        model_dir=root / "outputs" / "models",
        submission_dir=root / "submissions",
        duckdb_tmp=root / "tmp" / "duckdb_tmp",
    )
    for p in [
        paths.clean_dir,
        paths.cache_dir,
        paths.table_dir,
        paths.figure_dir,
        paths.model_dir,
        paths.submission_dir,
        paths.duckdb_tmp,
    ]:
        p.mkdir(parents=True, exist_ok=True)
    return paths
