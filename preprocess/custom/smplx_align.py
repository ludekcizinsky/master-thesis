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
    print(f"Saved aligned cameras to {cameras_file}")

    # Align SMPL-X parameters as well
    smplx_dir = root_dir_to_target_format_smplx_dir(cfg.scene_root_dir)
    smplx_files = sorted(smplx_dir.iterdir())

    for smplx_npz_file in smplx_files:

        smplx_data = np.load(smplx_npz_file, allow_pickle=True)
        if "root_pose" in smplx_data.files:
            root_pose_world = smplx_data["root_pose"]
            trans_world = smplx_data["trans"]

            aligned_root_pose, aligned_trans = align_root_pose_and_trans(
                root_pose_world, trans_world, T_h3r_to_gt
            )

            updated = {k: smplx_data[k] for k in smplx_data.files}
            updated["root_pose"] = aligned_root_pose
            updated["trans"] = aligned_trans
            np.savez(smplx_npz_file, **updated)


if __name__ == "__main__":
    main()
