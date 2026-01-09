import numpy as np
from collections import defaultdict
import json
from dataclasses import dataclass
from pathlib import Path
import cv2

import tyro

def root_dir_to_target_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "smplx" 

def root_dir_to_src_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "motion_human3r" 

def root_dir_to_target_format_cameras_file(root_dir: Path) -> Path:
    return root_dir / "cameras" / "rgb_cameras.npz"

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

    # Gather SMPL-X pose data from all tracks, organized by filename
    filename_to_poses = defaultdict(list)
    for track_id in track_ids:
        # check if track id is numeric if not continue
        if not track_id.isdigit():
            continue
        track_smplx_dir = motion_dir / track_id / "smplx_params"
        all_poses_files = sorted(track_smplx_dir.iterdir())
        for pose_estimate_file in all_poses_files:
            file_name = pose_estimate_file.stem  # e.g., "000001"
            with open(pose_estimate_file, "r") as f:
                pose_data = json.load(f)
            filename_to_poses[file_name].append(pose_data)

    # Load the camera data to map smplx from camera space to world space
    # - Collect the extrinsics for the source camera id
    cameras_file = root_dir_to_target_format_cameras_file(scene_root_dir)
    cameras_data = np.load(cameras_file, allow_pickle=True)
    cam_ids = cameras_data["ids"]  # (N_cams,)
    extrinsics = cameras_data["extrinsics"]  # (N_cams, 3, 4)
    cam_id_to_extrinsics = {int(cam_id): extrinsics[i] for i, cam_id in enumerate(cam_ids)}
    src_cam_extrinsics = cam_id_to_extrinsics[cfg.src_cam_id]  # (3, 4)
    R_w2c = src_cam_extrinsics[:3, :3]
    t_w2c = src_cam_extrinsics[:3, 3]
    # - Compute camera to world transform
    R_c2w = R_w2c.T
    t_c2w = -R_w2c.T @ t_w2c

    # Now, for each frame filename, merge the pose data from all tracks and save to target format 
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

    # Save the skipped frame numbers
    skip_frames_file = scene_root_dir / "skip_frames.csv" 
    with open(skip_frames_file, "w") as f:
        f.write(",".join(map(str, tgt_folder_skip_frame_numbers)))
    print(f"Saved skipped frame numbers to {skip_frames_file}: {tgt_folder_skip_frame_numbers}")   

if __name__ == "__main__":
    main()
