from pathlib import Path


def ensure_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)