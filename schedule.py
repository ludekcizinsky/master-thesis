"""Submit a batch of scenes via sbatch using a tyro config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import subprocess
import sys

import tyro



@dataclass
class Scene:
    seq_name: str
    source_cam_id: int
    root_gt_dir_path: str


@dataclass
class ScheduleConfig:
    job_name_prefix: str = "difix_v9_baseline"
    exp_name: str = "difix_v8_baseline"
    num_persons: int = 2
    time: str = "05:00:00"
    slurm_script: str = "train.slurm"
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
        ]
    )


def submit_scene(cfg: ScheduleConfig, scene: Scene) -> None:
    job_name = f"{cfg.job_name_prefix}_{scene.seq_name}"
    export_env = (
        "ALL,"
        f"EXP_NAME={cfg.exp_name},"
        f"NUM_PERSONS={cfg.num_persons},"
        f"SEQ_NAME={scene.seq_name},"
        f"SOURCE_CAM_ID={scene.source_cam_id},"
        f"ROOT_GT_DIR_PATH={scene.root_gt_dir_path}"
    )
    cmd = [
        "sbatch",
        "--job-name",
        job_name,
        "--time",
        cfg.time,
        "--export",
        export_env,
        cfg.slurm_script,
    ]

    if cfg.dry_run:
        print(" ".join(cmd))
        return

    subprocess.run(cmd, check=True)


def main() -> None:
    cfg = tyro.cli(ScheduleConfig)
    if not cfg.scenes:
        print("No scenes provided.")
        sys.exit(1)
    for scene in cfg.scenes:
        submit_scene(cfg, scene)


if __name__ == "__main__":
    main()
