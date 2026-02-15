from __future__ import annotations

import json
import os
import re
import shlex
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
    auto_schedule_aggregate: bool = True
    aggregate_job_name: str = "difix_data_aggregate"
    aggregate_time: str = "01:00:00"


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


def _resource_sbatch_args_from_script(path: Path) -> List[str]:
    """Extract resource-related sbatch args from a slurm script.

    We intentionally ignore job-name/output/error/time because those are set
    explicitly by the caller for aggregate jobs.
    """
    passthrough_keys = {
        "account",
        "partition",
        "qos",
        "constraint",
        "nodes",
        "ntasks",
        "ntasks-per-node",
        "cpus-per-task",
        "cpus-per-gpu",
        "mem",
        "mem-per-cpu",
        "gres",
        "gpus",
        "gpus-per-node",
        "gpus-per-task",
    }
    args: List[str] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith("#SBATCH"):
                continue
            payload = line[len("#SBATCH") :].strip()
            if not payload.startswith("--"):
                continue
            if "=" in payload:
                key, value = payload[2:].split("=", 1)
                key = key.strip()
                value = value.strip()
            else:
                parts = payload[2:].split(None, 1)
                key = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ""
            if key not in passthrough_keys or value == "":
                continue
            args.extend([f"--{key}", value])
    return args


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


def _get_override_value(overrides: Sequence[str], key: str) -> Optional[str]:
    value: Optional[str] = None
    for override in overrides:
        if "=" not in override:
            continue
        lhs, rhs = override.split("=", 1)
        if lhs == key:
            value = rhs
    return value


def _set_override_value(overrides: Sequence[str], key: str, value: str) -> List[str]:
    out: List[str] = []
    replaced = False
    for override in overrides:
        if "=" not in override:
            out.append(override)
            continue
        lhs, _ = override.split("=", 1)
        if lhs == key:
            if not replaced:
                out.append(f"{key}={value}")
                replaced = True
        else:
            out.append(override)
    if not replaced:
        out.append(f"{key}={value}")
    return out


def _extract_submitted_job_id(stdout_text: str) -> Optional[str]:
    m = re.search(r"Submitted batch job (\d+)", stdout_text)
    if m is None:
        return None
    return m.group(1)


def _build_export_env(runtime_paths) -> str:
    return ",".join(
        [
            "ALL",
            f"THESIS_WANDB_ROOT_DIR={runtime_paths.wandb_root_dir}",
            f"THESIS_HYDRA_ROOT_DIR={runtime_paths.hydra_root_dir}",
            f"THESIS_RESULTS_ROOT_DIR={runtime_paths.results_root_dir}",
            f"THESIS_PREPROCESSING_ROOT_DIR={runtime_paths.preprocessing_root_dir}",
            f"THESIS_CANONICAL_GT_ROOT_DIR={runtime_paths.canonical_gt_root_dir}",
        ]
    )


def _is_difix_generate_submit(cfg: Config) -> bool:
    run_mode = _get_override_value(cfg.overrides, "run_mode")
    return run_mode in {"difix_data_generation_generate", "difix_data_generation"}


def _submit_dependent_aggregate_job(
    cfg: Config,
    scenes: Sequence[Scene],
    dependency_job_id: str,
    runtime_paths,
) -> None:
    if len(scenes) == 0:
        raise ValueError("No scenes available to bootstrap aggregate job.")
    script_path = _resolve_repo_path(cfg.repo_dir, cfg.train_script)
    slurm_script = _resolve_repo_path(cfg.repo_dir, cfg.slurm.slurm_script)
    aggregate_overrides = _set_override_value(
        cfg.overrides,
        "run_mode",
        "difix_data_generation_aggregate",
    )
    aggregate_cmd = [
        "python",
        str(script_path),
        f"shared.scene_name={scenes[0].seq_name}",
        f"shared.exp_name={cfg.exp_name}",
    ] + aggregate_overrides
    wrapped = (
        f"cd {shlex.quote(str(cfg.repo_dir))} && "
        "source /home/cizinsky/miniconda3/etc/profile.d/conda.sh && "
        "conda activate thesis && "
        "module load gcc ffmpeg && "
        + " ".join(shlex.quote(arg) for arg in aggregate_cmd)
    )

    cmd: List[str] = [
        "sbatch",
        "--job-name",
        cfg.aggregate_job_name,
        "--time",
        cfg.aggregate_time,
        "--output",
        str(runtime_paths.slurm_dir / "%x.%j.out"),
        "--error",
        str(runtime_paths.slurm_dir / "%x.%j.err"),
        "--dependency",
        f"afterok:{dependency_job_id}",
        "--export",
        _build_export_env(runtime_paths),
    ] + _resource_sbatch_args_from_script(slurm_script) + [
        "--wrap",
        wrapped,
    ]

    if cfg.dry_run:
        print(" ".join(cmd))
        return

    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if res.stdout.strip():
        print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip())
    agg_job_id = _extract_submitted_job_id(res.stdout)
    if agg_job_id is not None:
        print(
            "Scheduled dependent DiFix aggregate job "
            f"{agg_job_id} (afterok:{dependency_job_id})."
        )
    else:
        print(
            "Scheduled dependent DiFix aggregate job "
            f"(afterok:{dependency_job_id})."
        )


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
    if cfg.auto_schedule_aggregate and _is_difix_generate_submit(cfg):
        print(
            "  Post-step: dependent aggregate job will be auto-submitted "
            f"(job_name={cfg.aggregate_job_name}, time={cfg.aggregate_time})."
        )
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
        _build_export_env(runtime_paths),
        str(slurm_script),
    ]
    cmd.extend(_forwarded_args(sys.argv[1:]))

    if cfg.dry_run:
        print(" ".join(cmd))
        if cfg.auto_schedule_aggregate and _is_difix_generate_submit(cfg):
            _submit_dependent_aggregate_job(
                cfg=cfg,
                scenes=scenes,
                dependency_job_id="<array_job_id>",
                runtime_paths=runtime_paths,
            )
        return
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if res.stdout.strip():
        print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip())

    if cfg.auto_schedule_aggregate and _is_difix_generate_submit(cfg):
        array_job_id = _extract_submitted_job_id(res.stdout)
        if array_job_id is None:
            raise RuntimeError(
                "Could not parse submitted array job id from sbatch output. "
                f"stdout={res.stdout!r}"
            )
        _submit_dependent_aggregate_job(
            cfg=cfg,
            scenes=scenes,
            dependency_job_id=array_job_id,
            runtime_paths=runtime_paths,
        )


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
