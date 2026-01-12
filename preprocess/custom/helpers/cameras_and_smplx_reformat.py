import numpy as np
from collections import defaultdict
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import cv2
from typing import List, Optional, Tuple

import torch

import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from submodules.smplx import smplx

def root_dir_to_target_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "smplx" 

def root_dir_to_src_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "motion_human3r" 

def root_dir_to_target_format_cameras_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "all_cameras" / f"{cam_id}"

def root_dir_to_source_format_cameras_file(root_dir: Path) -> Path:
    return root_dir / "motion_human3r" / "cameras.npz"

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

def load_frame_extrinsics_from_new_format(
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


def load_frame_intrinsics_from_new_format(
    scene_root_dir: Path,
    cam_id: int,
    frame_number: int,
    fname_num_digits: int,
) -> np.ndarray:
    cameras_dir = root_dir_to_target_format_cameras_dir(scene_root_dir, cam_id)
    cameras_file = cameras_dir / f"{frame_number:0{fname_num_digits}d}.npz"
    if not cameras_file.exists():
        raise FileNotFoundError(f"Missing camera file: {cameras_file}")

    with np.load(cameras_file, allow_pickle=True) as data:
        intrinsics = data["intrinsics"]

    if intrinsics.shape == (3, 3):
        intrinsics = intrinsics[None, ...]
    if intrinsics.shape != (1, 3, 3):
        raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")

    return intrinsics[0]


def _camera_center_from_w2c(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    return -R_w2c.T @ t_w2c


def _estimate_min_fov_from_intrinsics(intrinsics: np.ndarray) -> float:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    width = max(1.0, 2.0 * cx)
    height = max(1.0, 2.0 * cy)
    fov_x = 2.0 * np.arctan2(width, 2.0 * fx)
    fov_y = 2.0 * np.arctan2(height, 2.0 * fy)
    return float(min(fov_x, fov_y))


def _look_at_w2c(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    forward = target - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        raise ValueError("Camera position and target are too close for look-at.")
    forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise ValueError("Up vector is parallel to view direction.")
    right = right / right_norm

    down = np.cross(forward, right)
    R_w2c = np.stack([right, down, forward], axis=0)
    t_w2c = -R_w2c @ camera_pos
    return R_w2c, t_w2c


def _pad_or_truncate(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(vec.shape[-1])
    if current_dim == target_dim:
        return vec
    if current_dim > target_dim:
        return vec[..., :target_dim]
    pad = torch.zeros((*vec.shape[:-1], target_dim - current_dim), device=vec.device, dtype=vec.dtype)
    return torch.cat([vec, pad], dim=-1)


def _build_smplx_layer(model_folder: Path, gender: str, ext: str, device: torch.device):
    layer = smplx.create(
        str(model_folder),
        model_type="smplx",
        gender=gender,
        ext=ext,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
    )
    return layer.to(device)


def _load_npz(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        if not data.files:
            return None
        return {k: np.asarray(data[k]) for k in data.files}


def _smplx_joints(layer, params: dict, device: torch.device) -> Optional[np.ndarray]:
    required = {
        "betas",
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
    }
    if not required.issubset(params.keys()):
        return None

    betas = torch.tensor(params["betas"], dtype=torch.float32, device=device)
    root_pose = torch.tensor(params["root_pose"], dtype=torch.float32, device=device)
    body_pose = torch.tensor(params["body_pose"], dtype=torch.float32, device=device)
    jaw_pose = torch.tensor(params["jaw_pose"], dtype=torch.float32, device=device)
    leye_pose = torch.tensor(params["leye_pose"], dtype=torch.float32, device=device)
    reye_pose = torch.tensor(params["reye_pose"], dtype=torch.float32, device=device)
    lhand_pose = torch.tensor(params["lhand_pose"], dtype=torch.float32, device=device)
    rhand_pose = torch.tensor(params["rhand_pose"], dtype=torch.float32, device=device)
    trans = torch.tensor(params["trans"], dtype=torch.float32, device=device)

    if betas.ndim == 1:
        betas = betas[None, :]
    if root_pose.ndim == 1:
        root_pose = root_pose[None, :]
    if body_pose.ndim == 2:
        body_pose = body_pose[None, :, :]
    if jaw_pose.ndim == 1:
        jaw_pose = jaw_pose[None, :]
    if leye_pose.ndim == 1:
        leye_pose = leye_pose[None, :]
    if reye_pose.ndim == 1:
        reye_pose = reye_pose[None, :]
    if lhand_pose.ndim == 2:
        lhand_pose = lhand_pose[None, :, :]
    if rhand_pose.ndim == 2:
        rhand_pose = rhand_pose[None, :, :]
    if trans.ndim == 1:
        trans = trans[None, :]

    expected_betas = int(getattr(layer, "num_betas", betas.shape[-1]))
    betas = _pad_or_truncate(betas, expected_betas)

    expr_dim = int(getattr(layer, "num_expression_coeffs", 0))
    if expr_dim > 0:
        expr = params.get("expression")
        if expr is None:
            expr = torch.zeros((betas.shape[0], expr_dim), device=device, dtype=betas.dtype)
        else:
            expr = torch.tensor(expr, dtype=torch.float32, device=device)
            if expr.ndim == 1:
                expr = expr[None, :]
            expr = _pad_or_truncate(expr, expr_dim)
    else:
        expr = None

    call_args = dict(
        global_orient=root_pose,
        body_pose=body_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=lhand_pose,
        right_hand_pose=rhand_pose,
        betas=betas,
        transl=trans,
    )
    if expr is not None:
        call_args["expression"] = expr

    with torch.no_grad():
        output = layer(**call_args)
    return output.joints.detach().cpu().numpy()


def _load_smplx_joints_sequence(
    scene_root_dir: Path,
    frame_numbers: List[int],
    fname_num_digits: int,
    model_folder: Path,
    gender: str,
    ext: str,
    device: torch.device,
) -> List[np.ndarray]:
    smplx_dir = root_dir_to_target_format_smplx_dir(scene_root_dir)
    layer = _build_smplx_layer(model_folder, gender, ext, device)
    joints_per_frame = []
    for frame_number in frame_numbers:
        frame_path = smplx_dir / f"{frame_number:0{fname_num_digits}d}.npz"
        params = _load_npz(frame_path)
        if params is None:
            continue
        joints = _smplx_joints(layer, params, device)
        if joints is None or joints.size == 0:
            continue
        joints_per_frame.append(joints.reshape(-1, 3))
    return joints_per_frame


def _points_visible(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    width: float,
    height: float,
) -> bool:
    cam = (R_w2c @ points_world.T).T + t_w2c.reshape(1, 3)
    z = cam[:, 2]
    valid_z = z > 1e-6
    if not np.all(valid_z):
        return False
    x = (intrinsics[0, 0] * cam[:, 0] / z) + intrinsics[0, 2]
    y = (intrinsics[1, 1] * cam[:, 1] / z) + intrinsics[1, 2]
    in_bounds = (x >= 0.0) & (x <= width) & (y >= 0.0) & (y <= height)
    return bool(np.all(in_bounds))


def _enforce_static_camera_coverage(
    radius: float,
    center: np.ndarray,
    intrinsics: np.ndarray,
    joints_per_frame: List[np.ndarray],
    num_cams: int,
    up: np.ndarray,
    radius_scale: float,
    max_iters: int,
) -> float:
    if not joints_per_frame:
        return radius
    width = max(1.0, 2.0 * float(intrinsics[0, 2]))
    height = max(1.0, 2.0 * float(intrinsics[1, 2]))

    current_radius = radius
    for _ in range(max_iters):
        ok = True
        for cam_idx in range(num_cams):
            theta = 2.0 * np.pi * cam_idx / num_cams
            cam_pos = center + np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float32) * current_radius
            R_w2c, t_w2c = _look_at_w2c(cam_pos, center, up)
            for points in joints_per_frame:
                if not _points_visible(points, intrinsics, R_w2c, t_w2c, width, height):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return current_radius
        current_radius *= radius_scale
    print("Warning: static camera coverage not fully satisfied after max iterations.")
    return current_radius


def _compute_static_camera_radius(
    centers_world: np.ndarray,
    centers_frame_numbers: List[int],
    body_radius_m: float,
    intrinsics: np.ndarray,
    scene_root_dir: Path,
    src_cam_id: int,
    fname_num_digits: int,
) -> float:
    center = np.median(centers_world, axis=0)
    offsets = np.linalg.norm(centers_world - center[None, :], axis=1)
    max_offset = float(np.max(offsets)) if offsets.size else 0.0

    min_fov = _estimate_min_fov_from_intrinsics(intrinsics)
    if min_fov <= 1e-6:
        r_min = body_radius_m + max_offset
    else:
        r_min = (body_radius_m + max_offset) / max(np.tan(min_fov / 2.0), 1e-6)

    src_dists = []
    for frame_number, center_t in zip(centers_frame_numbers, centers_world):
        try:
            R_w2c, t_w2c = load_frame_extrinsics_from_new_format(
                scene_root_dir,
                cam_id=src_cam_id,
                frame_number=frame_number,
                fname_num_digits=fname_num_digits,
            )
        except FileNotFoundError:
            continue
        cam_center = _camera_center_from_w2c(R_w2c, t_w2c)
        src_dists.append(float(np.linalg.norm(center_t - cam_center)))

    r_src = float(np.median(src_dists)) if src_dists else r_min
    return max(r_min, r_src)


def generate_static_cameras(
    scene_root_dir: Path,
    frame_numbers: List[int],
    centers_world: np.ndarray,
    centers_frame_numbers: List[int],
    body_radius_m: float,
    src_cam_id: int,
    start_cam_id: int,
    num_cams: int,
    fname_num_digits: int,
    model_folder: Optional[Path] = None,
    gender: str = "neutral",
    smplx_model_ext: str = "npz",
    device: Optional[torch.device] = None,
    enforce_coverage: bool = False,
    radius_scale: float = 1.05,
    max_iters: int = 50,
    up: Optional[np.ndarray] = None,
) -> None:
    if centers_world.size == 0:
        return

    if up is None:
        up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    intrinsics = load_frame_intrinsics_from_new_format(
        scene_root_dir,
        cam_id=src_cam_id,
        frame_number=frame_numbers[0],
        fname_num_digits=fname_num_digits,
    )

    center = np.median(centers_world, axis=0)
    radius = _compute_static_camera_radius(
        centers_world=centers_world,
        centers_frame_numbers=centers_frame_numbers,
        body_radius_m=body_radius_m,
        intrinsics=intrinsics,
        scene_root_dir=scene_root_dir,
        src_cam_id=src_cam_id,
        fname_num_digits=fname_num_digits,
    )

    if enforce_coverage:
        if model_folder is None or device is None:
            raise ValueError("SMPL-X model folder and device are required to enforce coverage.")
        joints_per_frame = _load_smplx_joints_sequence(
            scene_root_dir=scene_root_dir,
            frame_numbers=centers_frame_numbers,
            fname_num_digits=fname_num_digits,
            model_folder=model_folder,
            gender=gender,
            ext=smplx_model_ext,
            device=device,
        )
        if joints_per_frame:
            radius = _enforce_static_camera_coverage(
                radius=radius,
                center=center,
                intrinsics=intrinsics,
                joints_per_frame=joints_per_frame,
                num_cams=num_cams,
                up=up,
                radius_scale=radius_scale,
                max_iters=max_iters,
            )

    intrinsics = intrinsics[None, ...]

    for cam_offset in range(num_cams):
        cam_id = start_cam_id + cam_offset
        cam_dir = root_dir_to_target_format_cameras_dir(scene_root_dir, cam_id)
        cam_dir.mkdir(parents=True, exist_ok=True)
        theta = 2.0 * np.pi * cam_offset / num_cams
        cam_pos = center + np.array([np.cos(theta), 0.0, np.sin(theta)], dtype=np.float32) * radius

        R_w2c, t_w2c = _look_at_w2c(cam_pos, center, up)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R_w2c.astype(np.float32)
        extrinsics[:3, 3] = t_w2c.astype(np.float32)
        extrinsics = extrinsics[None, :3, :]

        for frame_number in frame_numbers:
            cam_file = cam_dir / f"{frame_number:0{fname_num_digits}d}.npz"
            np.savez(cam_file, intrinsics=intrinsics, extrinsics=extrinsics)


def reformat_cameras(scene_root_dir: Path, src_cam_id: int, first_frame_number: int, fname_num_digits: int) -> None:

    current_cameras_file = root_dir_to_source_format_cameras_file(scene_root_dir)
    tgt_cameras_dir = root_dir_to_target_format_cameras_dir(scene_root_dir, src_cam_id)
    tgt_cameras_dir.mkdir(parents=True, exist_ok=True)

    # Load source cameras data
    src_cameras_data = np.load(current_cameras_file, allow_pickle=True)
    n_frames = len(src_cameras_data["frame_idx"])

    # Save the camera data in the target format
    current_frame_number = first_frame_number
    for frame_idx in range(n_frames):

        # - intrinsics
        intrinsics = src_cameras_data["K"][frame_idx][None, ...]  # (1, 3, 3)
        # - extrinsics
        R_w2c = src_cameras_data["R_world2cam"][frame_idx] # (3, 3)
        t_w2c = src_cameras_data["t_world2cam"][frame_idx] # (3,)
        # R_up = np.diag([1.0, -1.0, -1.0])  # -y up -> +y up, keep right-handed
        # R_w2c = R_w2c @ R_up.T
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R_w2c
        extrinsics[:3, 3] = t_w2c
        extrinsics = extrinsics[None, :3, :]  # (1, 3, 4)

        # Save with the keys expected by the target format
        dict_to_save = {
            "intrinsics": intrinsics, 
            "extrinsics": extrinsics 

        }
        tgt_cameras_file = tgt_cameras_dir / f"{current_frame_number:0{fname_num_digits}d}.npz"
        np.savez(tgt_cameras_file, **dict_to_save)
        current_frame_number += 1


@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance"
    first_frame_number: int = 1
    fname_num_digits: int = 6
    src_cam_id: int = 4
    generate_static_cams: bool = True
    static_cam_start_id: int = 100
    static_cam_count: int = 8
    static_body_radius_m: float = 1.3
    enforce_static_cam_coverage: bool = True
    static_cam_radius_scale: float = 1.05
    static_cam_max_iters: int = 50
    smplx_model_folder: str = "/home/cizinsky/body_models"
    smplx_model_ext: str = "npz"
    gender: str = "neutral"
    device: str = "cuda"

def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    scene_root_dir = Path(cfg.scene_root_dir)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    motion_dir = root_dir_to_src_format_smplx_dir(scene_root_dir)
    track_ids = sorted([d.name for d in motion_dir.iterdir() if d.is_dir() and d.name.isdigit()])

    # First, reformat cameras and save them to disk with the expected structure]
    reformat_cameras(
        scene_root_dir,
        src_cam_id=cfg.src_cam_id,
        first_frame_number=cfg.first_frame_number,
        fname_num_digits=cfg.fname_num_digits,
    )

    # Load which frames to skip across all tracks
    frames_to_skip = load_skip_frames(scene_root_dir)

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
    centers_world = []
    centers_frame_numbers = []
    all_frame_numbers = []
    for filename in sorted_filenames:
        poses_list = filename_to_poses[filename]
        keys = poses_list[0].keys()
        merged_pose_data = {}

        # Determine if we should skip this frame
        skip_this_frame = current_frame_number in frames_to_skip
        all_frame_numbers.append(current_frame_number)

        # If not skipping, perform quality checks and add the data to merged_pose_data
        if not skip_this_frame:
            # Load per-frame camera extrinsics (world-to-camera)
            R_w2c, t_w2c = load_frame_extrinsics_from_new_format(
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

            centers_world.append(np.mean(merged_pose_data["trans"], axis=0))
            centers_frame_numbers.append(current_frame_number)
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

    if cfg.generate_static_cams and centers_world:
        generate_static_cameras(
            scene_root_dir=scene_root_dir,
            frame_numbers=all_frame_numbers,
            centers_world=np.stack(centers_world, axis=0),
            centers_frame_numbers=centers_frame_numbers,
            body_radius_m=cfg.static_body_radius_m,
            src_cam_id=cfg.src_cam_id,
            start_cam_id=cfg.static_cam_start_id,
            num_cams=cfg.static_cam_count,
            fname_num_digits=cfg.fname_num_digits,
            model_folder=Path(cfg.smplx_model_folder),
            gender=cfg.gender,
            smplx_model_ext=cfg.smplx_model_ext,
            device=device,
            enforce_coverage=cfg.enforce_static_cam_coverage,
            radius_scale=cfg.static_cam_radius_scale,
            max_iters=cfg.static_cam_max_iters,
        )


if __name__ == "__main__":
    main()
