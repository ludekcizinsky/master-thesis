import numpy as np
from collections import defaultdict
import json
from dataclasses import dataclass
from pathlib import Path
import cv2
from typing import List

import tyro

def root_dir_to_target_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "smplx" 

def root_dir_to_src_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "motion_human3r" 

def root_dir_to_target_format_cameras_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "all_cameras" / f"{cam_id}"

def cam_to_world_translation(transl_cam, R_c2w, t_c2w):
    """
    transl_cam: (N, 3) in camera coords
    R_c2w: (3, 3)
    t_c2w: (3,)
    returns: (N, 3) in world coords
    """
    transl_cam = np.asarray(transl_cam)
    return (R_c2w @ transl_cam.T).T + t_c2w.reshape(1, 3)

def cam_to_world_root_pose(root_pose_cam, R_c2w):
    """
    root_pose_cam: (N, 3) axis-angle (rotvec) in camera coords
    R_c2w: (3, 3)
    returns: (N, 3) axis-angle (rotvec) in world coords
    """
    root_pose_cam = np.asarray(root_pose_cam)
    root_pose_world = np.zeros_like(root_pose_cam)
    for i in range(root_pose_cam.shape[0]):
        R_cam, _ = cv2.Rodrigues(root_pose_cam[i])
        R_world = R_c2w @ R_cam
        rotvec_world, _ = cv2.Rodrigues(R_world)
        root_pose_world[i] = rotvec_world.reshape(3)
    return root_pose_world

def root_dir_to_skip_frames_path(root_dir: Path) -> Path:
    return root_dir / "skip_frames.csv"

def load_skip_frames(scene_dir: Path) -> List[int]:
    """
    Load skip frames from skip_frames.csv in the scene directory.

    Note: the frame indicies are actual frame indexes and the frames dir may
    not always start with frame 0. Therefore, the returnd indices correspond to
    the actual frame indices to skip. e.g. if we have frames 10, 11, 12, 13 and skip_frames.csv contains "11,13",
    we will skip frames 11 and 13.

    Args:
        scene_dir: Path to the scene directory.
    Returns:
        List of frame indices to skip.
    """

    skip_frames_file = root_dir_to_skip_frames_path(scene_dir)
    if not skip_frames_file.exists():
        return []
    with open(skip_frames_file, "r") as f:
        line = f.readline().strip()
        skip_frames = [int(idx_str) for idx_str in line.split(",") if idx_str.isdigit()]
    return skip_frames

def load_frame_extrinsics(
    scene_root_dir: Path,
    cam_id: int,
    frame_number: int,
    fname_num_digits: int,
) -> tuple[np.ndarray, np.ndarray]:
    cameras_dir = root_dir_to_target_format_cameras_dir(scene_root_dir, cam_id)
    cameras_file = cameras_dir / f"{frame_number:0{fname_num_digits}d}.npz"
    if not cameras_file.exists():
        raise FileNotFoundError(f"Missing camera file: {cameras_file}")

    with np.load(cameras_file, allow_pickle=True) as data:
        extrinsics = data["extrinsics"]

    if extrinsics.shape == (3, 4):
        extrinsics = extrinsics[None, ...]
    if extrinsics.shape != (1, 3, 4):
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    extrinsics = extrinsics[0]
    R_w2c = extrinsics[:3, :3]
    t_w2c = extrinsics[:3, 3]
    return R_w2c, t_w2c

@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance"
    first_frame_number: int = 1
    fname_num_digits: int = 6
    src_cam_id: int = 4

def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    print(f"Reformatting SMPL-X data in scene root dir: {cfg.scene_root_dir}")
    scene_root_dir = Path(cfg.scene_root_dir)
    motion_dir = root_dir_to_src_format_smplx_dir(scene_root_dir)
    track_ids = sorted([d.name for d in motion_dir.iterdir() if d.is_dir() and d.name.isdigit()])

    # Load which frames to skip across all tracks
    frames_to_skip = load_skip_frames(scene_root_dir)
    print(f"Frames to skip: {frames_to_skip}")

    # Gather SMPL-X pose data from all tracks, organized by filename
    filename_to_poses = defaultdict(list)
    for track_id in track_ids:
        track_smplx_dir = motion_dir / track_id / "smplx_params"
        all_poses_files = sorted(track_smplx_dir.iterdir())
        for pose_estimate_file in all_poses_files:
            file_name = pose_estimate_file.stem  # e.g., "000001"
            with open(pose_estimate_file, "r") as f:
                pose_data = json.load(f)
            filename_to_poses[file_name].append(pose_data)

    # Now, for each frame filename, merge the pose data from all tracks and save to target format 
    current_frame_number = cfg.first_frame_number
    sorted_filenames = sorted(filename_to_poses.keys(), key=lambda x: int(x))
    tgt_folder_skip_frame_numbers = []
    for filename in sorted_filenames:
        poses_list = filename_to_poses[filename]
        keys = poses_list[0].keys()
        merged_pose_data = {}

        # Determine if we should skip this frame
        skip_this_frame = current_frame_number in frames_to_skip

        # If not skipping, perform quality checks and add the data to merged_pose_data
        if not skip_this_frame:
            # Load per-frame camera extrinsics (world-to-camera)
            R_w2c, t_w2c = load_frame_extrinsics(
                scene_root_dir,
                cam_id=cfg.src_cam_id,
                frame_number=current_frame_number,
                fname_num_digits=cfg.fname_num_digits,
            )
            R_c2w = R_w2c.T
            t_c2w = -R_w2c.T @ t_w2c

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
                if key == "trans":
                    # Convert translations from camera to world coords
                    merged_pose_data[key] = cam_to_world_translation(merged_pose_data[key], R_c2w, t_c2w)
                if key == "root_pose":
                    # Convert root_pose from camera to world coords
                    merged_pose_data[key] = cam_to_world_root_pose(merged_pose_data[key], R_c2w)

            # Human3r specific fix: add expression key if missing
            if "expression" not in merged_pose_data:
                num_persons = merged_pose_data["trans"].shape[0]
                merged_pose_data["expression"] = np.zeros((num_persons, 100), dtype=np.float32)
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


if __name__ == "__main__":
    main()
