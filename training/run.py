from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import tyro


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.path_config import ensure_runtime_dirs, load_runtime_paths


@dataclass
class Scene:
    seq_name: str
    cam_id: int


@dataclass
class SlurmConfig:
    job_name: str = "train_scene"
    slurm_script: Path = Path("training/submit.slurm")
    time: str = "03:00:00"
    array_parallelism: Optional[int] = None


@dataclass
class Config:
    repo_dir: Path = REPO_ROOT
    train_script: Path = Path("training/simple_multi_human_trainer.py")
    paths_config: Path = Path("configs/paths.yaml")
    scenes_dir: Optional[Path] = Path("preprocess/scenes")
    scenes: List[Scene] = field(default_factory=list)
    exp_name: str = "debug_run"
    overrides: List[str] = field(default_factory=list)
    scene_name_includes: Optional[str] = None
    run_all: bool = False
    submit: bool = False
    dry_run: bool = False
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def _resolve_repo_path(repo_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_dir / path


def _parse_scene_data(data: dict, path: Path) -> Scene:
    if "seq_name" not in data:
        raise ValueError(f"Missing 'seq_name' in {path}.")
    if "cam_id" not in data:
        raise ValueError(f"Missing 'cam_id' in {path}.")
    return Scene(
        seq_name=str(data["seq_name"]),
        cam_id=int(data["cam_id"]),
    )


def _load_scenes_from_dir(path: Path) -> List[Scene]:
    if not path.exists():
        return []
    json_files = sorted(path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No scene JSON files found in {path}.")
    scenes: List[Scene] = []
    for json_path in json_files:
        with json_path.open() as f:
            data = json.load(f)
        scenes.append(_parse_scene_data(data, json_path))
    return scenes


def _load_scenes(cfg: Config) -> List[Scene]:
    scenes_dir = cfg.scenes_dir
    if scenes_dir is not None:
        scenes_dir = _resolve_repo_path(cfg.repo_dir, scenes_dir)
        scenes = _load_scenes_from_dir(scenes_dir)
        if scenes:
            return scenes
    if not cfg.scenes:
        raise ValueError("No scenes provided.")
    return cfg.scenes


def _filter_scenes(cfg: Config, scenes: Sequence[Scene]) -> List[Scene]:
    if not cfg.scene_name_includes:
        return list(scenes)
    needle = cfg.scene_name_includes
    return [scene for scene in scenes if needle in scene.seq_name]


def _resolve_scene_index() -> Optional[int]:
    env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
    if env_idx is None or env_idx == "":
        return None
    try:
        return int(env_idx)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID='{env_idx}'") from exc


def _read_slurm_directives(path: Path) -> List[str]:
    directives: List[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("#SBATCH"):
                directives.append(line)
    return directives


def _print_submission_summary(
    cfg: Config,
    scenes: Sequence[Scene],
    slurm_script: Path,
    array_spec: str,
) -> None:
    print("About to submit training array with:")
    print(f"  Slurm script: {slurm_script}")
    print(f"  Job name: {cfg.slurm.job_name}")
    print(f"  Time: {cfg.slurm.time}")
    print(f"  Array: {array_spec} ({len(scenes)} scenes)")
    print(f"  exp_name: {cfg.exp_name}")
    if cfg.scene_name_includes:
        print(f"  Filter: seq_name includes '{cfg.scene_name_includes}'")
    if cfg.overrides:
        print("  Hydra overrides:")
        for override in cfg.overrides:
            print(f"    {override}")
    if cfg.slurm.array_parallelism:
        print(f"  Max parallelism: {cfg.slurm.array_parallelism}")
    directives = _read_slurm_directives(slurm_script)
    if directives:
        print("  Slurm directives:")
        for directive in directives:
            print(f"    {directive}")
    print("  Scenes:")
    for scene in scenes:
        print(f"    {scene.seq_name} | cam_id={scene.cam_id}")


def _confirm_submit() -> bool:
    try:
        response = input("Press Enter to submit, or type anything to cancel: ")
    except EOFError:
        return False
    return response.strip() == ""


def _forwarded_args(argv: Sequence[str]) -> List[str]:
    forwarded: List[str] = []
    skip_next = False
    blocked_exact = {"--submit", "--no-submit", "--run-all", "--no-run-all", "--dry-run", "--no-dry-run"}
    blocked_prefix = ("--submit=", "--run-all=", "--dry-run=")
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in blocked_exact:
            continue
        if arg.startswith(blocked_prefix):
            continue
        forwarded.append(arg)
    return forwarded


def _run_scene(cfg: Config, scene: Scene) -> None:
    script_path = _resolve_repo_path(cfg.repo_dir, cfg.train_script)
    runtime_paths = load_runtime_paths(_resolve_repo_path(cfg.repo_dir, cfg.paths_config))
    ensure_runtime_dirs(runtime_paths)

    cmd = [
        "python",
        str(script_path),
        f"shared.scene_name={scene.seq_name}",
        f"shared.exp_name={cfg.exp_name}",
    ]
    cmd.extend(cfg.overrides)
    if cfg.dry_run:
        print(" ".join(cmd))
        return
    env = os.environ.copy()
    env.setdefault("THESIS_WANDB_ROOT_DIR", str(runtime_paths.wandb_root_dir))
    env.setdefault("THESIS_HYDRA_ROOT_DIR", str(runtime_paths.hydra_root_dir))
    env.setdefault("THESIS_RESULTS_ROOT_DIR", str(runtime_paths.results_root_dir))
    env.setdefault("THESIS_PREPROCESSING_ROOT_DIR", str(runtime_paths.preprocessing_root_dir))
    env.setdefault("THESIS_CANONICAL_GT_ROOT_DIR", str(runtime_paths.canonical_gt_root_dir))
    subprocess.run(cmd, check=True, cwd=str(cfg.repo_dir), env=env)


def _submit_array(cfg: Config, scenes: Sequence[Scene]) -> None:
    if not scenes:
        raise ValueError("No scenes to submit.")
    array_spec = f"0-{len(scenes) - 1}"
    if cfg.slurm.array_parallelism:
        array_spec = f"{array_spec}%{cfg.slurm.array_parallelism}"

    slurm_script = _resolve_repo_path(cfg.repo_dir, cfg.slurm.slurm_script)
    if not slurm_script.exists():
        raise FileNotFoundError(f"Slurm script not found: {slurm_script}")
    runtime_paths = load_runtime_paths(_resolve_repo_path(cfg.repo_dir, cfg.paths_config))
    ensure_runtime_dirs(runtime_paths)

    _print_submission_summary(cfg, scenes, slurm_script, array_spec)
    if not _confirm_submit():
        print("Submission cancelled.")
        return

    cmd: List[str] = [
        "sbatch",
        "--job-name",
        cfg.slurm.job_name,
        "--time",
        cfg.slurm.time,
        "--output",
        str(runtime_paths.slurm_dir / "%x.%A_%a.out"),
        "--error",
        str(runtime_paths.slurm_dir / "%x.%A_%a.err"),
        "--array",
        array_spec,
        "--export",
        ",".join(
            [
                "ALL",
                f"THESIS_WANDB_ROOT_DIR={runtime_paths.wandb_root_dir}",
                f"THESIS_HYDRA_ROOT_DIR={runtime_paths.hydra_root_dir}",
                f"THESIS_RESULTS_ROOT_DIR={runtime_paths.results_root_dir}",
                f"THESIS_PREPROCESSING_ROOT_DIR={runtime_paths.preprocessing_root_dir}",
                f"THESIS_CANONICAL_GT_ROOT_DIR={runtime_paths.canonical_gt_root_dir}",
            ]
        ),
        str(slurm_script),
    ]
    cmd.extend(_forwarded_args(sys.argv[1:]))

    if cfg.dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    cfg = tyro.cli(Config)
    scenes = _filter_scenes(cfg, _load_scenes(cfg))

    if cfg.submit:
        _submit_array(cfg, scenes)
        return

    scene_index = _resolve_scene_index()
    if cfg.run_all:
        for scene in scenes:
            _run_scene(cfg, scene)
        return

    if scene_index is None:
        if len(scenes) == 1:
            _run_scene(cfg, scenes[0])
            return
        if not scenes:
            print("No scenes matched the filter.")
            sys.exit(1)
        print("Multiple scenes matched. Use --run-all or --submit.")
        sys.exit(1)

    if scene_index < 0 or scene_index >= len(scenes):
        raise IndexError(f"Scene index {scene_index} out of range (0..{len(scenes)-1}).")
    _run_scene(cfg, scenes[scene_index])


if __name__ == "__main__":
    main()
