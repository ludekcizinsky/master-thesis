import numpy as np
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
    print(f"Reformatting SMPL-X data in scene root dir: {cfg.scene_root_dir}")
    scene_root_dir = Path(cfg.scene_root_dir)
    motion_dir = root_dir_to_src_format_smplx_dir(scene_root_dir)
    track_ids = sorted([d.name for d in motion_dir.iterdir() if d.is_dir()])

    # Load which frames to skip across both tracks
    path_to_skip_txt_file = motion_dir / "skip_frames.txt"
    frames_to_skip = set()
    if path_to_skip_txt_file.exists():
        with open(path_to_skip_txt_file, "r") as f:
            for line in f:
                frame_range = line.strip()
                start_frame, end_frame = map(int, frame_range.split("-"))
                frames_to_skip.update(range(start_frame, end_frame + 1))
    print(f"Frames to skip: {sorted(frames_to_skip)}")

    filename_to_poses = defaultdict(list)
    for track_id in track_ids:
        track_smplx_dir = motion_dir / track_id / "smplx_params"
        all_poses_files = sorted(track_smplx_dir.iterdir())
        for pose_estimate_file in all_poses_files:
            file_name = pose_estimate_file.stem  # e.g., "000001"
            with open(pose_estimate_file, "r") as f:
                pose_data = json.load(f)
            filename_to_poses[file_name].append(pose_data)
    
    current_frame_number = cfg.first_frame_number
    sorted_filenames = sorted(filename_to_poses.keys(), key=lambda x: int(x))
    tgt_folder_skip_frame_numbers = []
    for filename in sorted_filenames:
        poses_list = filename_to_poses[filename]
        keys = poses_list[0].keys()
        merged_pose_data = {}

        # Determine if we should skip this frame
        src_folder_frame_number = int(filename)
        skip_this_frame = src_folder_frame_number in frames_to_skip

        # If not skipping, perform quality checks and add the data to merged_pose_data
        if not skip_this_frame:
            # Quality checks
            # - Ensure we have poses for all tracks
            assert len(poses_list) == len(track_ids), f"Frame {filename} is missing poses for some tracks."
            # - Ensure all pose data have the same shape for each key
            for key in keys:
                shapes = [np.array(pose_data[key]).shape for pose_data in poses_list]
                assert all(shape == shapes[0] for shape in shapes), f"Shape mismatch for key {key} in frame {filename}."

            # If checks, passed merge the data
            for key in keys:
                merged_pose_data[key] = np.stack([np.array(pose_data[key]) for pose_data in poses_list], axis=0)
        # If skipping, create empty arrays for each key -> do not save any data for this frame
        else:
            tgt_folder_skip_frame_numbers.append(current_frame_number)

        # And finally save the data
        target_smplx_dir = root_dir_to_target_format_smplx_dir(scene_root_dir)
        target_smplx_dir.mkdir(parents=True, exist_ok=True)
        frame_number = current_frame_number
        target_filename = f"{frame_number:0{cfg.fname_num_digits}d}.npz"
        target_file_path = target_smplx_dir / target_filename
        np.savez(target_file_path, **merged_pose_data)   
        current_frame_number += 1

    # Save the skipped frame numbers
    skip_frames_file = scene_root_dir / "skip_frames.csv" 
    with open(skip_frames_file, "w") as f:
        f.write(",".join(map(str, tgt_folder_skip_frame_numbers)))
    print(f"Saved skipped frame numbers to {skip_frames_file}: {tgt_folder_skip_frame_numbers}")   

if __name__ == "__main__":
    main()
