from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import tyro


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Scene:
    video_path: str
    seq_name: str
    cam_id: int
    ref_frame_idx: int = 0


@dataclass
class SlurmConfig:
    job_name: str = "preprocess_scene"
    slurm_script: Path = Path("preprocess/train/submit.slurm")
    array_parallelism: Optional[int] = None


@dataclass
class Config:
    repo_dir: Path = REPO_ROOT
    preprocess_dir: Path = Path("/scratch/izar/cizinsky/thesis/v2_preprocessing")
    infer_scene_script: Path = Path("preprocess/train/helpers/infer_initial_scene.py")
    scenes_dir: Optional[Path] = Path("preprocess/scenes")
    scenes: List[Scene] = field(default_factory=list)
    seq_name_includes: Optional[str] = None
    run_all: bool = False
    submit: bool = False
    dry_run: bool = False
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def _resolve_repo_path(repo_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_dir / path


def _parse_scene_data(data: dict, path: Path) -> Scene:
    required = ["video_path", "seq_name", "cam_id"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing '{key}' in {path}.")
    ref_frame = data.get("ref_frame_idx", 0)
    return Scene(
        video_path=str(data["video_path"]),
        seq_name=str(data["seq_name"]),
        cam_id=int(data["cam_id"]),
        ref_frame_idx=int(ref_frame),
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


def _forwarded_args(argv: Sequence[str]) -> List[str]:
    forwarded: List[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--submit", "--no-submit", "--run-all", "--no-run-all", "--dry-run", "--no-dry-run"}:
            continue
        if arg.startswith("--submit=") or arg.startswith("--run-all=") or arg.startswith("--dry-run="):
            continue
        forwarded.append(arg)
    return forwarded


def _submit_array(cfg: Config, scenes: Sequence[Scene]) -> None:
    if not scenes:
        raise ValueError("No scenes to submit.")
    array_spec = f"0-{len(scenes) - 1}"
    if cfg.slurm.array_parallelism:
        array_spec = f"{array_spec}%{cfg.slurm.array_parallelism}"

    slurm_script = _resolve_repo_path(cfg.repo_dir, cfg.slurm.slurm_script)
    if not slurm_script.exists():
        raise FileNotFoundError(f"Slurm script not found: {slurm_script}")

    _print_submission_summary(cfg, scenes, slurm_script, array_spec)
    if not _confirm_submit():
        print("Submission cancelled.")
        return

    cmd: List[str] = [
        "sbatch",
        "--job-name",
        cfg.slurm.job_name,
        "--array",
        array_spec,
        "--export",
        "ALL",
        str(slurm_script),
    ]
    cmd.extend(_forwarded_args(sys.argv[1:]))

    if cfg.dry_run:
        print(" ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def _resolve_scene_index() -> Optional[int]:
    env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
    if env_idx is None or env_idx == "":
        return None
    try:
        return int(env_idx)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID='{env_idx}'") from exc


def _run_scene(cfg: Config, scene: Scene) -> None:
    script_path = _resolve_repo_path(cfg.repo_dir, cfg.infer_scene_script)
    cmd = [
        "python",
        str(script_path),
        "--video-path",
        scene.video_path,
        "--seq-name",
        scene.seq_name,
        "--cam-id",
        str(scene.cam_id),
        "--output-dir",
        str(cfg.preprocess_dir),
        "--ref-frame-idx",
        str(scene.ref_frame_idx),
    ]
    subprocess.run(cmd, check=True, cwd=str(cfg.repo_dir))


def _filter_scenes(cfg: Config, scenes: Sequence[Scene]) -> List[Scene]:
    if not cfg.seq_name_includes:
        return list(scenes)
    needle = cfg.seq_name_includes
    return [scene for scene in scenes if needle in scene.seq_name]


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
    print("About to submit preprocessing array with:")
    print(f"  Slurm script: {slurm_script}")
    print(f"  Job name: {cfg.slurm.job_name}")
    print(f"  Array: {array_spec} ({len(scenes)} scenes)")
    if cfg.seq_name_includes:
        print(f"  Filter: seq_name includes '{cfg.seq_name_includes}'")
    if cfg.slurm.array_parallelism:
        print(f"  Max parallelism: {cfg.slurm.array_parallelism}")
    directives = _read_slurm_directives(slurm_script)
    if directives:
        print("  Slurm directives:")
        for directive in directives:
            print(f"    {directive}")
    print("  Scenes:")
    for scene in scenes:
        print(
            f"    {scene.seq_name} | cam_id={scene.cam_id} | ref_frame_idx={scene.ref_frame_idx} | video_path={scene.video_path}"
        )


def _confirm_submit() -> bool:
    try:
        response = input("Press Enter to submit, or type anything to cancel: ")
    except EOFError:
        return False
    return response.strip() == ""


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
