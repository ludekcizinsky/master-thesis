"""Submit a batch of scenes via sbatch using a tyro config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import base64
import os
import subprocess
import sys
import time

import tyro

# - Target camera IDs for novel view eval
HI4D_TARGET_CAM_IDS = [4, 16, 28, 40, 52, 64, 76, 88]
# - Training and novel view generation camera IDs
HI4D_TRN_NV_GEN_CAM_IDS = [4, 16, 28, 40, 52, 64, 76, 88]
HI4D_PAIR17_TRN_NV_GEN_CAM_IDS = [4, 16, 28, 40, 52, 64, 76, 88]
MMM_TRN_NV_GEN_CAM_IDS = [0, 100, 101, 102, 103, 104, 105, 107]
WILD_TRN_NV_GEN_CAM_IDS = [0, 100, 101, 102, 103, 104, 105, 107]
# - Preprocessing directory path
PREPROCESSING_DIR = "/scratch/izar/cizinsky/thesis/preprocessing"


@dataclass
class Scene:
    seq_name: str
    source_cam_id: int
    root_gt_dir_path: str
    num_persons: int
    target_camera_ids: List[int]
    trn_nv_gen_cam_ids: List[int]
    preprocessing_dir_path: str | None = None
    test_masks_scene_dir: str | None = None
    test_frames_scene_dir: str | None = None
    test_smpl_params_scene_dir: str | None = None
    smpl_params_scene_dir: str | None = None
    test_smplx_params_scene_dir: str | None = None
    test_meshes_scene_dir: str | None = None
    cameras_scene_dir: str | None = None

    def __post_init__(self) -> None:
        if self.preprocessing_dir_path is None:
            self.preprocessing_dir_path = os.path.join(PREPROCESSING_DIR, self.seq_name)
        if self.test_masks_scene_dir is None:
            self.test_masks_scene_dir = self.root_gt_dir_path
        if self.test_frames_scene_dir is None:
            self.test_frames_scene_dir = self.root_gt_dir_path
        if self.test_smpl_params_scene_dir is None:
            self.test_smpl_params_scene_dir = self.root_gt_dir_path
        if self.smpl_params_scene_dir is None:
            self.smpl_params_scene_dir = self.preprocessing_dir_path
        if self.test_smplx_params_scene_dir is None:
            self.test_smplx_params_scene_dir = self.root_gt_dir_path
        if self.test_meshes_scene_dir is None:
            self.test_meshes_scene_dir = self.root_gt_dir_path
        if self.cameras_scene_dir is None:
            self.cameras_scene_dir = self.root_gt_dir_path


@dataclass
class ScheduleConfig:
    job_name_prefix: str = "difix_v9_baseline"
    exp_name: str = "difix_v8_baseline"
    time: str = "10:00:00"
    slurm_script: str = "train.slurm"
    scene_name_includes: str | None = None
    dry_run: bool = False
    submit_sleep_s: float = 2.0
    overrides: List[str] = field(default_factory=list)
    scenes: List[Scene] = field(
        default_factory=lambda: [
            Scene(
                "hi4d_pair15_fight",
                4,
                "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15",
                2,
                list(HI4D_TARGET_CAM_IDS),
                list(HI4D_TRN_NV_GEN_CAM_IDS),
            ),
            Scene(
                "hi4d_pair16_jump",
                4,
                "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16",
                2,
                list(HI4D_TARGET_CAM_IDS),
                list(HI4D_TRN_NV_GEN_CAM_IDS),
            ),
            Scene(
                "hi4d_pair17_dance",
                28,
                "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17",
                2,
                list(HI4D_TARGET_CAM_IDS),
                list(HI4D_PAIR17_TRN_NV_GEN_CAM_IDS),
            ),
            Scene(
                "hi4d_pair19_piggyback",
                4,
                "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19",
                2,
                list(HI4D_TARGET_CAM_IDS),
                list(HI4D_TRN_NV_GEN_CAM_IDS),
            ),
            Scene(
                "mmm_dance",
                0,
                "/scratch/izar/cizinsky/ait_datasets/full/mmm/dance",
                4,
                [],
                list(MMM_TRN_NV_GEN_CAM_IDS),
                test_masks_scene_dir="null",
                test_smpl_params_scene_dir="null",
                smpl_params_scene_dir="null",
                test_smplx_params_scene_dir="null",
                cameras_scene_dir=os.path.join(PREPROCESSING_DIR, "mmm_dance"),
            ),
            Scene(
                "mmm_lift",
                0,
                "/scratch/izar/cizinsky/ait_datasets/full/mmm/lift",
                3,
                [],
                list(MMM_TRN_NV_GEN_CAM_IDS),
                test_masks_scene_dir="null",
                test_smpl_params_scene_dir="null",
                smpl_params_scene_dir="null",
                test_smplx_params_scene_dir="null",
                cameras_scene_dir=os.path.join(PREPROCESSING_DIR, "mmm_lift"),
            ),
            Scene(
                "mmm_walkdance",
                0,
                "/scratch/izar/cizinsky/ait_datasets/full/mmm/walkdance",
                3,
                [],
                list(MMM_TRN_NV_GEN_CAM_IDS),
                test_masks_scene_dir="null",
                test_smpl_params_scene_dir="null",
                smpl_params_scene_dir="null",
                test_smplx_params_scene_dir="null",
                cameras_scene_dir=os.path.join(PREPROCESSING_DIR, "mmm_walkdance"),
            ),
            Scene(
                "taichi",
                0,
                "/scratch/izar/cizinsky/ait_datasets/full/mmm/taichi",
                2,
                [],
                list(WILD_TRN_NV_GEN_CAM_IDS),
                test_masks_scene_dir="null",
                test_frames_scene_dir="null",
                test_smpl_params_scene_dir="null",
                smpl_params_scene_dir="null",
                test_smplx_params_scene_dir="null",
                test_meshes_scene_dir="null",
                cameras_scene_dir=os.path.join(PREPROCESSING_DIR, "taichi"),
            ),

        ]
    )


def submit_scene(cfg: ScheduleConfig, scene: Scene) -> None:
    job_name = f"{cfg.job_name_prefix}_{scene.seq_name}"
    def encode_cam_ids(cam_ids: List[int]) -> str:
        if cam_ids:
            return ":".join(str(cam_id) for cam_id in cam_ids)
        return "[]"

    target_camera_ids = encode_cam_ids(scene.target_camera_ids)
    trn_nv_gen_cam_ids = encode_cam_ids(scene.trn_nv_gen_cam_ids)
    export_env = (
        "ALL,"
        f"EXP_NAME={cfg.exp_name},"
        f"NUM_PERSONS={scene.num_persons},"
        f"SEQ_NAME={scene.seq_name},"
        f"SOURCE_CAM_ID={scene.source_cam_id},"
        f"ROOT_GT_DIR_PATH={scene.root_gt_dir_path},"
        f"TARGET_CAMERA_IDS={target_camera_ids},"
        f"TRN_NV_GEN_CAMERA_IDS={trn_nv_gen_cam_ids},"
        f"PREPROCESSING_DIR_PATH={scene.preprocessing_dir_path},"
        f"TEST_MASKS_SCENE_DIR={scene.test_masks_scene_dir},"
        f"TEST_FRAMES_SCENE_DIR={scene.test_frames_scene_dir},"
        f"TEST_SMPL_PARAMS_SCENE_DIR={scene.test_smpl_params_scene_dir},"
        f"SMPL_PARAMS_SCENE_DIR={scene.smpl_params_scene_dir},"
        f"TEST_SMPLX_PARAMS_SCENE_DIR={scene.test_smplx_params_scene_dir},"
        f"TEST_MESHES_SCENE_DIR={scene.test_meshes_scene_dir},"
        f"CAMERAS_SCENE_DIR={scene.cameras_scene_dir}"
    )
    if cfg.overrides:
        overrides_blob = "\n".join(cfg.overrides)
        overrides_b64 = base64.b64encode(overrides_blob.encode("utf-8")).decode("ascii")
        export_env = f"{export_env},HYDRA_OVERRIDES_B64={overrides_b64}"
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
    scenes = cfg.scenes
    if cfg.scene_name_includes:
        scenes = [
            scene
            for scene in scenes
            if cfg.scene_name_includes in scene.seq_name
        ]

    # print the config for verification
    print("Scheduling with the following configuration:")
    for field_name, value in vars(cfg).items():
        if field_name != "scenes":
            print(f"  {field_name}: {value}")
        else:
            print(f"  {field_name}:")
            for scene in scenes:
                for scene_field, scene_value in vars(scene).items():
                    print(f"    {scene_field}: {scene_value}")
                print("\n---\n")

    if not scenes:
        print("No scenes provided.")
        sys.exit(1)
    for scene_idx, scene in enumerate(scenes):
        submit_scene(cfg, scene)
        if (
            not cfg.dry_run
            and cfg.submit_sleep_s > 0
            and scene_idx < (len(scenes) - 1)
        ):
            time.sleep(cfg.submit_sleep_s)


if __name__ == "__main__":
    main()
