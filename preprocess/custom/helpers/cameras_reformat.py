import numpy as np
from dataclasses import dataclass
from pathlib import Path

import tyro

def root_dir_to_target_format_cameras_file(root_dir: Path) -> Path:
    return root_dir / "cameras" / "rgb_cameras.npz"

def root_dir_to_source_format_cameras_file(root_dir: Path) -> Path:
    return root_dir / "motion_human3r" / "cameras.npz"

@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance"
    src_cam_id: int = 28
    fname_num_digits: int = 6


def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    scene_root_dir = Path(cfg.scene_root_dir)
    current_cameras_file = root_dir_to_source_format_cameras_file(scene_root_dir)
    tgt_cameras_file = root_dir_to_target_format_cameras_file(scene_root_dir)
    tgt_cameras_file.parent.mkdir(parents=True, exist_ok=True)

    # Load source cameras data
    # Note: We always use the first frame's camera parameters, as we assume cameras are static.
    src_cameras_data = np.load(current_cameras_file, allow_pickle=True)
    # - cam ids
    ids = np.array([cfg.src_cam_id]) # (1,)
    # - intrinsics
    intrinsics = src_cameras_data["K"][0][None, ...]  # (1, 3, 3)
    # - extrinsics
    R_w2c = src_cameras_data["R_world2cam"][0] # (3, 3)
    t_w2c = src_cameras_data["t_world2cam"][0] # (3,)
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R_w2c
    extrinsics[:3, 3] = t_w2c
    extrinsics = extrinsics[None, :3, :]  # (1, 3, 4)

    # Save with the keys expected by the target format
    dict_to_save = {
        "ids": ids,
        "intrinsics": intrinsics, 
        "extrinsics": extrinsics 

    }
    np.savez(tgt_cameras_file, **dict_to_save)


if __name__ == "__main__":
    main()
