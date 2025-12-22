import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import tyro
import cv2
import torch
from submodules.smplx import smplx
from collections import defaultdict


def root_dir_to_target_format_cameras_file(root_dir: Path) -> Path:
    return root_dir / "cameras" / "rgb_cameras.npz"

def root_dir_to_target_format_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "smplx" 

def load_cam_w2c_params(scene_root_dir: Path, src_cam_id: int):
    """
    Load the world-to-camera rotation and translation for the specified camera ID.

    Args:
        scene_root_dir: Path to the root directory of the scene.
        src_cam_id: Camera ID to load parameters for.
    
    Returns:
        R_w2c: (3, 3) rotation matrix from world to camera coordinates
        t_w2c: (3,) translation vector from world to camera coordinates
    """
    cameras_file = root_dir_to_target_format_cameras_file(scene_root_dir)
    cameras_data = np.load(cameras_file, allow_pickle=True)
    cam_ids = cameras_data["ids"]  # (N_cams,)
    extrinsics = cameras_data["extrinsics"]  # (N_cams, 3, 4)
    cam_id_to_extrinsics = {int(cam_id): extrinsics[i] for i, cam_id in enumerate(cam_ids)}
    src_cam_extrinsics = cam_id_to_extrinsics[src_cam_id]  # (3, 4)
    R_w2c = src_cam_extrinsics[:3, :3]
    t_w2c = src_cam_extrinsics[:3, 3]

    return R_w2c, t_w2c


def w2c_to_4x4(R_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = t_w2c
    return w2c


def align_extrinsics(extrinsics: np.ndarray, T_h3r_to_gt: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(extrinsics)
    T_inv = np.linalg.inv(T_h3r_to_gt)
    for i in range(extrinsics.shape[0]):
        w2c = np.eye(4, dtype=extrinsics.dtype)
        w2c[:3, :4] = extrinsics[i]
        w2c_aligned = w2c @ T_inv
        aligned[i] = w2c_aligned[:3, :4]
    return aligned


def align_root_pose_and_trans(
    root_pose: np.ndarray, trans: np.ndarray, T_h3r_to_gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    R = T_h3r_to_gt[:3, :3]
    t = T_h3r_to_gt[:3, 3]

    aligned_trans = (R @ trans.T).T + t.reshape(1, 3)

    aligned_root_pose = np.empty_like(root_pose)
    for i in range(root_pose.shape[0]):
        R_root, _ = cv2.Rodrigues(root_pose[i])
        R_root_aligned = R @ R_root
        rotvec_aligned, _ = cv2.Rodrigues(R_root_aligned)
        aligned_root_pose[i] = rotvec_aligned.reshape(3)

    return aligned_root_pose, aligned_trans


def build_smplx_layer(model_root: Path, num_betas: int, num_expr: int, device: torch.device):
    layer = smplx.create(
        str(model_root),
        "smplx",
        gender="neutral",
        num_betas=num_betas,
        num_expression_coeffs=num_expr,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
    )
    return layer.to(device)


def load_smplx_npz(path: Path, device: torch.device, expr_dim: int) -> dict:
    npz = np.load(path)

    def to_tensor(key, default=None):
        if key in npz:
            arr = npz[key]
        elif default is not None:
            arr = default
        else:
            raise KeyError(f"Missing key {key} in {path}")
        return torch.from_numpy(arr).float().to(device)

    params = {
        "betas": to_tensor("betas"),
        "root_pose": to_tensor("root_pose"),
        "body_pose": to_tensor("body_pose"),
        "jaw_pose": to_tensor("jaw_pose"),
        "leye_pose": to_tensor("leye_pose"),
        "reye_pose": to_tensor("reye_pose"),
        "lhand_pose": to_tensor("lhand_pose"),
        "rhand_pose": to_tensor("rhand_pose"),
        "trans": to_tensor("trans"),
    }

    if "expression" in npz:
        expr = to_tensor("expression")
        if expr.shape[-1] < expr_dim:
            pad = torch.zeros((*expr.shape[:-1], expr_dim - expr.shape[-1]), device=device, dtype=expr.dtype)
            expr = torch.cat([expr, pad], dim=-1)
        elif expr.shape[-1] > expr_dim:
            expr = expr[..., :expr_dim]
    else:
        expr = torch.zeros((params["betas"].shape[0], expr_dim), device=device, dtype=params["betas"].dtype)
    params["expr"] = expr
    return params


def pelvis_positions(layer, params: dict) -> torch.Tensor:
    smpl_in = {
        "global_orient": params["root_pose"],
        "body_pose": params["body_pose"],
        "jaw_pose": params["jaw_pose"],
        "leye_pose": params["leye_pose"],
        "reye_pose": params["reye_pose"],
        "left_hand_pose": params["lhand_pose"],
        "right_hand_pose": params["rhand_pose"],
        "betas": params["betas"],
        "transl": params["trans"],
        "expression": params["expr"],
    }
    with torch.no_grad():
        out = layer(**smpl_in)
    return out.joints[:, 0]  # [P,3]


@dataclass
class AlignmentConfig:
    scene_root_dir: Path = Path("/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance")
    gt_scene_root_dir: Path = Path("/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17")
    src_cam_id: int = 28
    smplx_model_path: Path = Path("/scratch/izar/cizinsky/pretrained/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz")
    device: str = "cpu"

def main() -> None:
    cfg = tyro.cli(AlignmentConfig)
    device = torch.device(cfg.device)

    # Compute the rigid transform from my estimated SMPL-X to the GT SMPL-X
    # - both in world coordinates of the same camera
    R_w2c, t_w2c = load_cam_w2c_params(cfg.scene_root_dir, cfg.src_cam_id)
    gt_R_w2c, gt_t_w2c = load_cam_w2c_params(cfg.gt_scene_root_dir, cfg.src_cam_id)
    # - compute the transform 
    w2c_h3r = w2c_to_4x4(R_w2c, t_w2c)
    w2c_gt = w2c_to_4x4(gt_R_w2c, gt_t_w2c)
    T_h3r_to_gt = np.linalg.inv(w2c_gt) @ w2c_h3r

    # Load camera extrinsics and align them
    cameras_file = root_dir_to_target_format_cameras_file(cfg.scene_root_dir)
    cameras_data = np.load(cameras_file, allow_pickle=True)
    ids = cameras_data["ids"]
    intrinsics = cameras_data["intrinsics"]
    extrinsics = cameras_data["extrinsics"]

    aligned_extrinsics = align_extrinsics(extrinsics, T_h3r_to_gt)
    np.savez(cameras_file, ids=ids, intrinsics=intrinsics, extrinsics=aligned_extrinsics)

    # Align SMPL-X parameters as well
    smplx_dir = root_dir_to_target_format_smplx_dir(cfg.scene_root_dir)
    smplx_files = sorted(smplx_dir.iterdir())
    gt_smplx_dir = root_dir_to_target_format_smplx_dir(cfg.gt_scene_root_dir)
    gt_files = sorted(gt_smplx_dir.iterdir())

    # Load frames to skip
    frames_to_skip_path = cfg.scene_root_dir / "skip_frames.csv"
    if os.path.exists(frames_to_skip_path):
        with open(frames_to_skip_path, "r") as f:
            lines = f.readlines()
        frames_to_skip = set()
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            fidxs = [int(x) for x in line.split(",")]
            frames_to_skip.update(fidxs)

        print(f"Skipping {len(frames_to_skip)} frames based on {frames_to_skip_path}")


    # Build SMPL-X layer for pelvis computations
    sample_h = np.load(smplx_files[0])
    sample_g = np.load(gt_files[0])
    num_betas = max(sample_h["betas"].shape[-1], sample_g["betas"].shape[-1])
    num_expr = max(
        sample_h["expression"].shape[-1] if "expression" in sample_h else 0,
        sample_g["expression"].shape[-1] if "expression" in sample_g else 0,
        100,
    )
    layer = build_smplx_layer(
        cfg.smplx_model_path.parent.parent if cfg.smplx_model_path.is_file() else cfg.smplx_model_path,
        num_betas,
        num_expr,
        device,
    )

    fname_set = sorted(set(f.name for f in smplx_files) & set(f.name for f in gt_files))
    for fname in fname_set:
        fidx = int(fname.split(".")[0])
        if fidx in frames_to_skip:
            continue
        h_path = smplx_dir / fname
        g_path = gt_smplx_dir / fname

        h_npz = np.load(h_path, allow_pickle=True)
        g_npz = np.load(g_path, allow_pickle=True)
        if h_npz["trans"].shape[0] != g_npz["trans"].shape[0]:
            continue

        # Align Human3R root pose + trans to GT world
        aligned_root_pose, aligned_trans = align_root_pose_and_trans(
            h_npz["root_pose"], h_npz["trans"], T_h3r_to_gt
        )

        # Load as tensors for pelvis computation
        h_params = load_smplx_npz(h_path, device, num_expr)
        g_params = load_smplx_npz(g_path, device, num_expr)
        h_params["root_pose"] = torch.from_numpy(aligned_root_pose).float().to(device)
        h_params["trans"] = torch.from_numpy(aligned_trans).float().to(device)

        pelvis_h = pelvis_positions(layer, h_params)
        pelvis_g = pelvis_positions(layer, g_params)
        delta = pelvis_g - pelvis_h  # [P,3]

        # Apply per-person, per-frame pelvis offset
        aligned_trans_with_delta = aligned_trans + delta.detach().cpu().numpy()

        updated = {k: h_npz[k] for k in h_npz.files}
        updated["root_pose"] = aligned_root_pose
        updated["trans"] = aligned_trans_with_delta
        np.savez(h_path, **updated)


if __name__ == "__main__":
    main()
