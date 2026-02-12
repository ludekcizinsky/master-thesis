from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import re


def is_epoch_dir(path: Path) -> bool:
    return path.is_dir() and re.fullmatch(r"epoch_\d+", path.name) is not None


def epoch_key(epoch_name: str) -> int:
    return int(epoch_name.split("_", 1)[1])


def resolve_epochs(eval_root: Path, epoch: str) -> List[str]:
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Missing evaluation directory: {eval_root}")

    epoch_dirs = [path for path in eval_root.iterdir() if is_epoch_dir(path)]
    epoch_names = sorted((path.name for path in epoch_dirs), key=epoch_key)
    if not epoch_names:
        raise RuntimeError(f"No epoch directories found in {eval_root}")

    if epoch == "all":
        return epoch_names
    if epoch == "latest":
        return [epoch_names[-1]]
    if epoch.startswith("epoch_"):
        if epoch not in epoch_names:
            raise ValueError(f"Epoch '{epoch}' not found in {eval_root}")
        return [epoch]
    if epoch.isdigit():
        canonical = f"epoch_{int(epoch):04d}"
        if canonical not in epoch_names:
            raise ValueError(f"Epoch '{canonical}' not found in {eval_root}")
        return [canonical]
    raise ValueError(
        f"Unsupported epoch value '{epoch}'. Use one of: all, latest, epoch_0030, 30."
    )


def sorted_numeric_stems(paths: Iterable[Path]) -> List[Path]:
    numeric = [path for path in paths if path.stem.isdigit()]
    return sorted(numeric, key=lambda path: int(path.stem))
