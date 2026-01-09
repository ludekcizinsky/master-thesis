import numpy as np
from dataclasses import dataclass
from pathlib import Path

import tyro

def root_dir_to_target_format_cameras_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "all_cameras" / f"{cam_id}"

def root_dir_to_source_format_cameras_file(root_dir: Path) -> Path:
    return root_dir / "motion_human3r" / "cameras.npz"

@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance"
    src_cam_id: int = 28
    fname_num_digits: int = 6
    first_frame_number: int = 1


def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    scene_root_dir = Path(cfg.scene_root_dir)
    current_cameras_file = root_dir_to_source_format_cameras_file(scene_root_dir)
    tgt_cameras_dir = root_dir_to_target_format_cameras_dir(scene_root_dir, cfg.src_cam_id)
    tgt_cameras_dir.mkdir(parents=True, exist_ok=True)

    # Load source cameras data
    src_cameras_data = np.load(current_cameras_file, allow_pickle=True)
    n_frames = len(src_cameras_data["frame_idx"])

    # Save the camera data in the target format
    current_frame_number = cfg.first_frame_number
    for frame_idx in range(n_frames):

        # - intrinsics
        intrinsics = src_cameras_data["K"][frame_idx][None, ...]  # (1, 3, 3)
        # - extrinsics
        R_w2c = src_cameras_data["R_world2cam"][frame_idx] # (3, 3)
        t_w2c = src_cameras_data["t_world2cam"][frame_idx] # (3,)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R_w2c
        extrinsics[:3, 3] = t_w2c
        extrinsics = extrinsics[None, :3, :]  # (1, 3, 4)

        # Save with the keys expected by the target format
        dict_to_save = {
            "intrinsics": intrinsics, 
            "extrinsics": extrinsics 

        }
        tgt_cameras_file = tgt_cameras_dir / f"{current_frame_number:0{cfg.fname_num_digits}d}.npz"
        np.savez(tgt_cameras_file, **dict_to_save)
        current_frame_number += 1


if __name__ == "__main__":
    main()
