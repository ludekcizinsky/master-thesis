from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re


@dataclass(frozen=True)
class RuntimePaths:
    misc_root_dir: Path
    slurm_dir: Path
    hf_cache_dir: Path
    wandb_root_dir: Path
    hydra_root_dir: Path
    results_root_dir: Path
    preprocessing_root_dir: Path
    canonical_gt_root_dir: Path


def _default_paths_config() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"


@lru_cache(maxsize=8)
def load_runtime_paths(config_path: str | Path | None = None) -> RuntimePaths:
    path = Path(config_path) if config_path is not None else _default_paths_config()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Runtime paths config not found: {path}")

    resolved = _load_simple_yaml_with_interpolation(path)
    required = {
        "misc_root_dir",
        "slurm_dir",
        "hf_cache_dir",
        "wandb_root_dir",
        "hydra_root_dir",
        "results_root_dir",
        "preprocessing_root_dir",
        "canonical_gt_root_dir",
    }
    missing = required - set(resolved.keys())
    if missing:
        raise ValueError(f"Missing required path keys in {path}: {sorted(missing)}")

    return RuntimePaths(
        misc_root_dir=Path(str(resolved["misc_root_dir"])),
        slurm_dir=Path(str(resolved["slurm_dir"])),
        hf_cache_dir=Path(str(resolved["hf_cache_dir"])),
        wandb_root_dir=Path(str(resolved["wandb_root_dir"])),
        hydra_root_dir=Path(str(resolved["hydra_root_dir"])),
        results_root_dir=Path(str(resolved["results_root_dir"])),
        preprocessing_root_dir=Path(str(resolved["preprocessing_root_dir"])),
        canonical_gt_root_dir=Path(str(resolved["canonical_gt_root_dir"])),
    )


def ensure_runtime_dirs(paths: RuntimePaths) -> None:
    for path in (
        paths.misc_root_dir,
        paths.slurm_dir,
        paths.hf_cache_dir,
        paths.wandb_root_dir,
        paths.hydra_root_dir,
        paths.results_root_dir,
        paths.preprocessing_root_dir,
        paths.canonical_gt_root_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _load_simple_yaml_with_interpolation(path: Path) -> dict[str, str]:
    raw: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid line in {path}: {line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        raw[key] = value

    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

    def resolve_value(value: str, depth: int = 0) -> str:
        if depth > 20:
            raise ValueError(f"Interpolation depth exceeded while parsing {path}: {value!r}")

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in raw:
                raise KeyError(f"Unknown interpolation key '{key}' in {path}")
            return resolve_value(raw[key], depth + 1)

        return pattern.sub(repl, value)

    return {key: resolve_value(value) for key, value in raw.items()}
