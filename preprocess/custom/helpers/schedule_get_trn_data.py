"""Submit preprocessing scenes via sbatch using a tyro config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence
import subprocess
import sys

import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]

@dataclass
class Scene:
    seq_name: str
    source_cam_id: int
    root_gt_dir_path: str


@dataclass
class ScheduleConfig:
    part: str = "part1"
    job_name_prefix: str = "preprocess"
    time: str = "01:00:00"
    slurm_script_part1: str = "preprocess/custom/helpers/get_trn_data_part1.slurm"
    slurm_script_part2: str = "preprocess/custom/helpers/get_trn_data_part2.slurm"
    dry_run: bool = False
    scenes: List[Scene] = field(
        default_factory=lambda: [
            Scene(
                "hi4d_pair15_fight",
                4,
                "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15",
            ),
            Scene(
                "hi4d_pair16_jump",
                4,
                "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16",
            ),
            Scene(
                "hi4d_pair17_dance",
                28,
                "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17",
            ),
            Scene(
                "hi4d_pair19_piggyback",
                4,
                "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19",
            ),
#            Scene(
                #"mmm_dance",
                #0,
                #"/scratch/izar/cizinsky/ait_datasets/full/mmm/dance",
            #),
            #Scene(
                #"mmm_lift",
                #0,
                #"/scratch/izar/cizinsky/ait_datasets/full/mmm/lift",
            #),
            #Scene(
                #"mmm_walkdance",
                #0,
                #"/scratch/izar/cizinsky/ait_datasets/full/mmm/walkdance",
            #),
            #Scene(
                #"taichi",
                #0,
                #"/scratch/izar/cizinsky/thesis/in_the_wild/scenes/taichi",
            #),
        ]
    )


def _parse_part_flags(argv: Sequence[str]) -> tuple[Optional[str], list[str]]:
    part = None
    cleaned: list[str] = []
    for arg in argv:
        if arg == "--part1":
            part = "part1" if part is None else part
            if part != "part1":
                raise ValueError("Use only one of --part1 or --part2.")
            continue
        if arg == "--part2":
            part = "part2" if part is None else part
            if part != "part2":
                raise ValueError("Use only one of --part1 or --part2.")
            continue
        cleaned.append(arg)
    return part, cleaned


def _normalize_part(part: str) -> str:
    part = part.strip().lower()
    if part not in {"part1", "part2"}:
        raise ValueError("part must be 'part1' or 'part2'.")
    return part


def _resolve_slurm_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def submit_scene(cfg: ScheduleConfig, scene: Scene) -> None:
    job_name = f"{cfg.job_name_prefix}_{cfg.part}_{scene.seq_name}"
    export_env = (
        "ALL,"
        f"SEQ_NAME={scene.seq_name},"
        f"SRC_CAMERA_ID={scene.source_cam_id},"
        f"ROOT_GT_DIR_PATH={scene.root_gt_dir_path},"
        f"PREPROCESS_PART={cfg.part}"
    )
    slurm_script = (
        cfg.slurm_script_part1 if cfg.part == "part1" else cfg.slurm_script_part2
    )
    slurm_script = _resolve_slurm_path(slurm_script)
    cmd = [
        "sbatch",
        "--job-name",
        job_name,
        "--time",
        cfg.time,
        "--export",
        export_env,
        slurm_script,
    ]

    if cfg.dry_run:
        print(" ".join(cmd))
        return

    subprocess.run(cmd, check=True)


def main() -> None:
    part_override, argv = _parse_part_flags(sys.argv[1:])
    cfg = tyro.cli(ScheduleConfig, args=argv)
    if part_override is not None:
        cfg.part = part_override
    cfg.part = _normalize_part(cfg.part)
    if not cfg.scenes:
        print("No scenes provided.")
        sys.exit(1)
    for scene in cfg.scenes:
        submit_scene(cfg, scene)


if __name__ == "__main__":
    main()
