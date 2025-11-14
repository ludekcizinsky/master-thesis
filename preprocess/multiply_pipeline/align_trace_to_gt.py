#!/usr/bin/env python3
"""
Compute per-frame similarity transforms that align canonical Trace SMPL outputs to metric GT SMPL joints.

This script expects the Multiply preprocessing directory (with poses.npy etc.) and a directory containing
per-frame *.npz GT SMPL files (with joints_3d). It outputs an .npz file alongside the preprocessing
artifacts that stores scale/rotation/translation needed to map canonical poses into metric space.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import os
import numpy as np
import torch

import sys

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from PIL import Image
import cv2

from training.smpl_deformer.smpl_server import SMPLServer
from training.helpers.geom_utils import load_K_Rt_from_P


def _build_canonical_params(
    poses: np.ndarray,
    trans: np.ndarray,
    shape: np.ndarray,
    scale_value: float,
    frame_idx: int,
) -> np.ndarray:
    """
    Assemble canonical SMPL parameter tensor (P, 86) for a specific frame.
    """
    num_persons = poses.shape[1]
    param = np.zeros((num_persons, 86), dtype=np.float32)
    param[:, 0] = scale_value
    param[:, 1:4] = trans[frame_idx] * scale_value
    param[:, 4:76] = poses[frame_idx]
    param[:, 76:] = shape
    return param

def rigid_transform_3D(src, dst):
    """
    Compute the rotation R and translation t that aligns src to dst
    such that dst ≈ src @ R.T + t

    src, dst: (N, 3) arrays of corresponding 3D points
    returns: R (3x3), t (3,)
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    assert src.shape == dst.shape
    assert src.shape[1] == 3

    # 1. Compute centroids
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)

    # 2. Center the points
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst

    # 3. Compute covariance matrix
    H = src_centered.T @ dst_centered  # (3x3)

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Compute rotation
    R = Vt.T @ U.T

    # Reflection correction (ensure det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 6. Compute translation
    t = centroid_dst - centroid_src @ R.T

    # 7. compute reprojection error per point
    for i in range(src.shape[0]):
        src_pt = src[i]
        dst_pt = dst[i]
        src_transformed = R @ src_pt + t
        error = np.linalg.norm(dst_pt - src_transformed)
        print(f"Joint {i}: reproj error = {error:.4f} m")
    return R, t.squeeze()


def similarity_transform_3D(src, dst):
    """
    Compute the rotation R and translation t that aligns src to dst
    such that dst ≈ src @ R.T + t

    src, dst: (N, 3) arrays of corresponding 3D points
    returns: s, R (3x3), t (3,), avg_reproj_error
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    assert src.shape == dst.shape
    assert src.shape[1] == 3

    # 1. Compute centroids
    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)

    # 2. Center the points
    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst

    # 3. Compute covariance matrix
    H = src_centered.T @ dst_centered  # (3x3)

    # 4. SVD
    U, S, Vt = np.linalg.svd(H)

    # 5. Compute rotation
    R = Vt.T @ U.T

    # Reflection correction (ensure det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # 6. Compute scale
    var_src = np.sum(np.sum(src_centered**2, axis=1))
    s = np.sum(S) / var_src

    # 7. Compute translation
    t = centroid_dst - s*(centroid_src @ R.T)  

    # 8. compute reprojection error per point
    avg_reproj_error = 0.0
    for i in range(src.shape[0]):
        src_pt = src[i]
        dst_pt = dst[i]
        src_transformed = s * (R @ src_pt) + t
        error = np.linalg.norm(dst_pt - src_transformed)
        avg_reproj_error += error
    
    avg_reproj_error *= 1.0 / src.shape[0] 
    return s, R, t.squeeze(), avg_reproj_error


def _project_points(points: np.ndarray, K: np.ndarray, w2c_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam_pts_h = (w2c_cv @ pts_h.T).T
    cam_pts = cam_pts_h[:, :3]
    z = cam_pts[:, 2]
    valid = z > 1e-4
    cam_pts = cam_pts[valid]
    z = z[valid]
    if cam_pts.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0,))
    K3 = K[:3, :3]
    proj = (K3 @ cam_pts.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv, z


def _draw_alignment_overlay(
    image: np.ndarray,
    joints_list: List[np.ndarray],
    K: np.ndarray,
    w2c_cv: np.ndarray,
    out_path: Path,
) -> None:
    overlay = image.copy()
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]
    for idx, joints in enumerate(joints_list):
        uv, z = _project_points(joints, K, w2c_cv)
        if uv.shape[0] == 0:
            continue
        person_color = colors[idx % len(colors)]
        for joint_id, pt in enumerate(uv):
            center = (int(pt[0]), int(pt[1]))
            cv2.circle(overlay, center, 3, person_color, -1)
            label_pos = (center[0] + 4, center[1] - 4)
            cv2.putText(
                overlay,
                str(joint_id),
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                person_color,
                1,
                cv2.LINE_AA,
            )
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    Image.fromarray(blended).save(out_path)


def align_trace_to_metric(
    preprocess_dir: Path,
    gt_smpl_dir: Path,
    cameras: List[Tuple[np.ndarray, np.ndarray]],
    dataset: str = "hi4d",
    visualize: bool = False,
) -> None:
    
    # Load preprocessing artifacts
    poses = np.load(preprocess_dir / "poses.npy")
    trans = np.load(preprocess_dir / "normalize_trans.npy")
    shape = np.load(preprocess_dir / "mean_shape.npy")

    cam_norm = np.load(preprocess_dir / "cameras_normalize.npz")
    scale_mat = cam_norm["scale_mat_0"]
    scale_value = 1.0 / float(scale_mat[0, 0])

    # Load GT SMPL files
    frame_files = sorted(gt_smpl_dir.glob("*.npz"))
    if not frame_files:
        raise FileNotFoundError(f"No .npz files found in {gt_smpl_dir}")
    if len(frame_files) != poses.shape[0]:
        raise ValueError(
            f"Mismatch between GT frames ({len(frame_files)}) and canonical poses ({poses.shape[0]})"
        )

    # Prepare SMPL server
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_server = SMPLServer().to(device).eval()

    # Prepare storage for alignment transforms
    rotations = np.zeros((poses.shape[0], poses.shape[1], 3, 3), dtype=np.float32)
    translations = np.zeros((poses.shape[0], poses.shape[1], 3), dtype=np.float32)
    scales = np.ones((poses.shape[0], poses.shape[1]), dtype=np.float32)

    # Iterate over frames and compute alignment transforms
    with torch.no_grad():
        total_repr_error, total_alignments = 0.0, 0
        for frame_idx, npz_path in tqdm(enumerate(frame_files), total=len(frame_files), desc="Aligning frames"):
            # collect gt smpl joints
            gt_payload = np.load(npz_path)
            gt_joints = gt_payload["joints_3d"]
            if gt_joints.shape[0] != poses.shape[1]:
                raise ValueError(
                    f"GT persons ({gt_joints.shape[0]}) != canonical persons ({poses.shape[1]}) at frame {npz_path}"
                )

            # get trace canonical joints
            canon_params = _build_canonical_params(poses, trans, shape, scale_value, frame_idx)
            canon_tensor = torch.from_numpy(canon_params).to(device)
            smpl_output = smpl_server(canon_tensor, absolute=True)
            canon_joints = smpl_output["smpl_jnts"].cpu().numpy()

            # Compute alignment transforms per person
            for person_idx in range(gt_joints.shape[0]):
                # Get src and dst joints
                src = canon_joints[person_idx]
                dst = gt_joints[person_idx]

                # Compute similarity transform from canonical to GT joints
                scale, rot, t, reproj_error = similarity_transform_3D(src, dst)
                total_repr_error += reproj_error
                total_alignments += 1

                # Store the results
                scales[frame_idx, person_idx] = scale
                rotations[frame_idx, person_idx] = rot.astype(np.float32)
                translations[frame_idx, person_idx] = t.astype(np.float32)

            # Visualization (optional)
            if visualize:
                vis_dir = preprocess_dir / "smpl_joints_alignment_check" / dataset
                vis_dir.mkdir(parents=True, exist_ok=True)
                images_dir = preprocess_dir / "image"
                frame_image_path = images_dir / f"{frame_idx:04d}.png"
                out_path = vis_dir / f"{frame_idx:04d}.png"
                if frame_image_path.exists():
                    image_np = np.array(Image.open(frame_image_path).convert("RGB"))
                    joints_metric_list: List[np.ndarray] = []
                    for person_idx in range(gt_joints.shape[0]):

                        # parse the transforms
                        rot = rotations[frame_idx, person_idx]
                        trans_vec = translations[frame_idx, person_idx]
                        scale_val = scales[frame_idx, person_idx]
                        joints_canon = canon_joints[person_idx]

                        # apply the similarity transform to map 
                        joints_metric = (rot @ joints_canon.T).T * scale_val + trans_vec
                        joints_metric_list.append(joints_metric)

                    _draw_alignment_overlay(
                        image=image_np,
                        joints_list=joints_metric_list,
                        K=cameras[frame_idx][1],
                        w2c_cv=cameras[frame_idx][0],
                        out_path=out_path,
                    )
                else:
                    raise FileNotFoundError(f"Frame image not found for visualization: {frame_image_path}")

    # Save the alignment transforms
    output_path = preprocess_dir / "smpl_joints_alignment_transforms" / f"{dataset}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    avg_reproj_error = total_repr_error / total_alignments if total_alignments > 0 else float('nan')
    np.savez(
        output_path,
        rotations=rotations,
        translations=translations,
        scales=scales,
    )

    print(f"[align_trace_to_gt] Saved alignment transforms to {output_path}")
    print(f"[align_trace_to_gt] Average reprojection error over all alignments: {avg_reproj_error:.4f} m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align canonical Trace SMPL params to metric GT joints.")
    parser.add_argument("--preprocess_dir", type=Path, required=True, help="Path to Multiply preprocessing output.")
    parser.add_argument("--gt_smpl_dir", type=Path, required=True, help="Directory with per-frame GT SMPL .npz files.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, save overlay images of aligned SMPL meshes to <preprocess_dir>/smpl_alignment_check.",
    )
    parser.add_argument(
        "--gt_camera_file",
        type=Path,
        default=None,
        help="Path to the GT camera metadata file (e.g. rgb_cameras.npz).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="hi4d",
    )
    parser.add_argument(
        "--cam_id",
        type=int,
        required=True,
    )
    return parser.parse_args()


def load_camera_data(camera_file: Path, camera_id: int, dataset:str="hi4d", n_frames: Optional[int] = None) -> List[Tuple[np.array, np.array]]:

    if dataset=="hi4d":
        gt_data = np.load(camera_file)
        ids = gt_data["ids"]
        matches = np.where(ids == camera_id)[0]
        if matches.size == 0:
            raise ValueError(f"Camera id {camera_id} not found in GT camera file {camera_file}.")
        gt_idx = int(matches[0])

        # Extrinsics and intrinsics from GT camera file
        extr = gt_data["extrinsics"][gt_idx]  # [3,4], assumes world->camera
        pose_gt = np.eye(4, dtype=np.float32)
        pose_gt[:3, :3] = extr[:, :3]
        pose_gt[:3, 3] = extr[:, 3]

        K = gt_data["intrinsics"][gt_idx]
        gt_intrinsics = np.eye(4, dtype=np.float32)
        gt_intrinsics[:3, :3] = K.astype(np.float32)

        results = [(pose_gt, gt_intrinsics) for _ in range(n_frames)]

        return results

    elif dataset=="custom":
        camera_dict = np.load(camera_file)
        scale_mats = [camera_dict[f'scale_mat_{idx}'].astype(np.float32) for idx in range(n_frames)]
        world_mats = [camera_dict[f'world_mat_{idx}'].astype(np.float32) for idx in range(n_frames)]
        results = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]

            intrinsics, pose = load_K_Rt_from_P(P)
            results.append((pose, intrinsics))
        return results
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported yet.")

def main():
    args = parse_args()
    preprocess_dir: Path = args.preprocess_dir
    gt_dir: Path = args.gt_smpl_dir
    gt_camera_file: Path = args.gt_camera_file
    camera_id: int = int(args.cam_id)
    dataset: str = args.dataset

    camera_params = load_camera_data(
        gt_camera_file,
        camera_id,
        dataset=dataset,
        n_frames=len(sorted(gt_dir.glob("*.npz"))),
    )

    align_trace_to_metric(
        preprocess_dir,
        gt_dir,
        visualize=args.visualize,
        cameras=camera_params,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
