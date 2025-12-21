from collections import defaultdict
import json
from dataclasses import dataclass
from pathlib import Path

import tyro

def root_dir_to_target_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "smplx" 

def root_dir_to_src_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "motion_human3r" 

@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance"
    first_frame_number: int = 1
    fname_num_digits: int = 6


def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    scene_root_dir = Path(cfg.scene_root_dir)
    motion_dir = root_dir_to_src_format_smplx_dir(scene_root_dir)
    track_ids = sorted([d.name for d in motion_dir.iterdir() if d.is_dir()])

    filename_to_poses = defaultdict(list)
    for track_id in track_ids:
        track_smplx_dir = motion_dir / track_id / "smplx_params"
        all_poses_files = sorted(track_smplx_dir.iterdir())
        for pose_estimate_file in all_poses_files:
            file_name = pose_estimate_file.stem  # e.g., "000001"
            with open(pose_estimate_file, "r") as f:
                pose_data = json.load(f)
            filename_to_poses[file_name].append(pose_data)

    for filename, poses_list in filename_to_poses.items():
        assert len(poses_list) == len(track_ids), f"Frame {filename} is missing poses for some tracks."


if __name__ == "__main__":
    main()
