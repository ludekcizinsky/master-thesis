from typing import Optional, Dict, Tuple, List
from pathlib import Path
from tqdm import tqdm

from scipy.spatial import cKDTree
import trimesh
from trimesh import registration

import torch
import torch.nn.functional as F

import numpy as np

import kornia
import pyiqa


# ---------------------------------------------------------------------------
# Reconstruction evaluation metrics
# ---------------------------------------------------------------------------
def invert_se3(T: np.ndarray) -> np.ndarray:
    """
    Invert an SE(3) transform (4x4) assuming last row is [0,0,0,1].
    """
    assert T.shape == (4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=T.dtype)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def compose(T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Compose two SE(3) transforms: T_ab = T_a @ T_b
    """
    assert T_a.shape == (4, 4) and T_b.shape == (4, 4)
    return T_a @ T_b

def apply_se3_to_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Apply SE(3) transform T to 3D points P.

    P: (N, 3) row-vector points.
    Returns: (N, 3)
    """
    assert T.shape == (4, 4)
    assert P.ndim == 2 and P.shape[1] == 3
    R = T[:3, :3]
    t = T[:3, 3]
    return (P @ R.T) + t  # row vectors

def compute_world_align_from_gt_c2w_and_pred_c2w(
    T_gt_c2w: np.ndarray,
    T_pred_c2w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given:
      - GT camera-to-world (c2w):  T_gt_c2w  = T_{gt_world <- cam}
      - Pred camera-to-world (c2w): T_pred_c2w = T_{pred_world <- cam}

    Compute:
      - T_align = T_{gt_world <- pred_world}
      - R_align (3x3), t_align (3,)
    such that: X_gt_world = R_align * X_pred_world + t_align
    """
    assert T_gt_c2w.shape == (4, 4)
    assert T_pred_c2w.shape == (4, 4)

    # We want T_{gt_world <- pred_world}.
    # Using c2w poses:
    #   X_gt_world = T_gt_c2w * X_cam
    #   X_pred_world = T_pred_c2w * X_cam
    # So:
    #   X_gt_world = T_gt_c2w * (T_pred_c2w)^{-1} * X_pred_world
    T_align = compose(T_gt_c2w, invert_se3(T_pred_c2w))

    R_align = T_align[:3, :3]
    t_align = T_align[:3, 3]
    return T_align, R_align, t_align

def posed_gs_list_to_serializable_dict(posed_gs_list) -> Dict[str, torch.Tensor]:
    if len(posed_gs_list) == 0:
        raise ValueError("posed_gs_list is empty; expected at least one person.")

    merged_xyz = torch.cat([gs.xyz for gs in posed_gs_list], dim=0)
    merged_opacity = torch.cat([gs.opacity for gs in posed_gs_list], dim=0)
    merged_rotation = torch.cat([gs.rotation for gs in posed_gs_list], dim=0)
    merged_scaling = torch.cat([gs.scaling for gs in posed_gs_list], dim=0)
    merged_shs = torch.cat([gs.shs for gs in posed_gs_list], dim=0)

    person_ids = torch.cat(
        [
            torch.full((gs.xyz.shape[0],), i, dtype=torch.int32, device=gs.xyz.device)
            for i, gs in enumerate(posed_gs_list)
        ],
        dim=0,
    )
    person_gaussian_counts = torch.tensor(
        [gs.xyz.shape[0] for gs in posed_gs_list], dtype=torch.int32, device=merged_xyz.device
    )

    return {
        "xyz": merged_xyz.detach().cpu(),
        "opacity": merged_opacity.detach().cpu(),
        "rotation": merged_rotation.detach().cpu(),
        "scaling": merged_scaling.detach().cpu(),
        "shs": merged_shs.detach().cpu(),
        "person_ids": person_ids.detach().cpu(),
        "person_gaussian_counts": person_gaussian_counts.detach().cpu(),
        "use_rgb": torch.tensor(bool(posed_gs_list[0].use_rgb), dtype=torch.bool),
    }


def gt_mesh_from_sample(meshes: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    num_vertices = int(meshes["num_vertices"].detach().cpu().item())
    num_faces = int(meshes["num_faces"].detach().cpu().item())
    vertices = meshes["vertices"].detach().cpu().numpy()
    faces = meshes["faces"].detach().cpu().numpy()
    return vertices[:num_vertices].astype(np.float32), faces[:num_faces].astype(np.int32)


def merge_mesh_dict(meshes: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if not meshes:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    vertices_out = []
    faces_out = []
    vert_offset = 0
    for _pid, (verts, faces) in meshes.items():
        if verts.size == 0 or faces.size == 0:
            continue
        vertices_out.append(verts.astype(np.float32))
        faces_out.append((faces + vert_offset).astype(np.int32))
        vert_offset += verts.shape[0]
    if not vertices_out:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    return np.concatenate(vertices_out, axis=0), np.concatenate(faces_out, axis=0)


def icp_align_mesh_similarity(
    pred_mesh: Tuple[np.ndarray, np.ndarray],
    gt_mesh: Tuple[np.ndarray, np.ndarray],
    *,
    n_samples: int = 50000,
    max_iterations: int = 20,
    threshold: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Align pred mesh to GT mesh with rigid ICP (no scale).
    Returns transformed vertices, faces, transform matrix, and final cost.
    """

    pred_v, pred_f = pred_mesh
    gt_v, gt_f = gt_mesh

    if pred_v.size == 0 or gt_v.size == 0 or pred_f.size == 0 or gt_f.size == 0:
        return pred_v, pred_f, np.eye(4, dtype=np.float64), float("inf")

    pred_tm = trimesh.Trimesh(vertices=pred_v, faces=pred_f, process=False)
    gt_tm = trimesh.Trimesh(vertices=gt_v, faces=gt_f, process=False)

    pred_pts = pred_tm.sample(int(n_samples))
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = gt_tm.centroid - pred_tm.centroid

    matrix, _transformed, cost = registration.icp(
        pred_pts,
        gt_tm,
        initial=init,
        threshold=threshold,
        max_iterations=max_iterations,
        scale=False,
        reflection=False,
    )
    aligned_vertices = trimesh.transformations.transform_points(pred_v, matrix)
    return aligned_vertices.astype(np.float32), pred_f.astype(np.int32), matrix, float(cost)


def align_pred_meshes_icp(
    pred_meshes: List[Tuple[np.ndarray, np.ndarray]],
    gt_meshes: List[Tuple[np.ndarray, np.ndarray]],
    cameras: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None,
    n_samples: int = 50000,
    max_iterations: int = 20,
    threshold: float = 1e-5,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]:
    """
    Align each predicted mesh to GT mesh using rigid ICP.
    Returns the aligned meshes and per-frame transforms (pred -> aligned).
    """
    aligned = []
    transforms = []

    if cameras is not None:
        pred_c2ws, gt_c2ws = cameras

    i = 0
    for (pred_frame, gt_frame) in tqdm(zip(pred_meshes, gt_meshes), desc="Aligning Pred Meshes ICP", total=len(pred_meshes), leave=False):
        pred_v, pred_f = pred_frame
        gt_v, gt_f = gt_frame
        if pred_v.size == 0 or pred_f.size == 0 or gt_v.size == 0 or gt_f.size == 0:
            aligned.append(pred_frame)
            transforms.append(np.eye(4, dtype=np.float32))
            continue


        pred_v_init = pred_v.astype(np.float64)
        T_align = np.eye(4, dtype=np.float64)
        if cameras is not None:
            print(f"Aligning frame {i} with camera-based initialization.")
            T_align, R_align, t_align = compute_world_align_from_gt_c2w_and_pred_c2w(
                T_gt_c2w=gt_c2ws[i],
                T_pred_c2w=pred_c2ws[i],
            )
            pred_v_init = (pred_v_init @ R_align.T) + t_align

        aligned_v, aligned_f, _matrix, _cost = icp_align_mesh_similarity(
            (pred_v_init, pred_f),
            gt_frame,
            n_samples=n_samples,
            max_iterations=max_iterations,
            threshold=threshold,
        )
        aligned.append((aligned_v, aligned_f))
        transforms.append(compose(_matrix, T_align).astype(np.float32))
        i += 1
    return aligned, transforms


def _nearest_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    tree = cKDTree(points_b)
    distances, _ = tree.query(points_a, k=1)
    return distances


def _sample_mesh_points_and_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, "trimesh.Trimesh"]:

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    points, face_idx = mesh.sample(int(n_samples), return_index=True)
    normals = mesh.face_normals[face_idx]
    return points, normals, mesh


def _metric_units_scale(units: str) -> float:
    units_norm = units.strip().lower()
    if units_norm in ("m", "meter", "meters"):
        return 1.0
    if units_norm in ("cm", "centimeter", "centimeters"):
        return 100.0
    if units_norm in ("mm", "millimeter", "millimeters"):
        return 1000.0
    raise ValueError(f"Unsupported units: {units}")


def _format_mesh_frame_name(frame_name: str) -> str:
    if frame_name.isdigit():
        return f"{int(frame_name):06d}"
    return frame_name


def save_aligned_meshes(
    pred_meshes: List[Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    frame_names: List[str],
) -> None:
    for frame_idx, mesh in enumerate(pred_meshes):
        frame_name = _format_mesh_frame_name(str(frame_names[frame_idx]))
        vertices, faces = mesh
        if vertices.size == 0 or faces.size == 0:
            continue
        out_path = output_dir / f"{frame_name}.obj"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(out_path)


def compute_chamfer_distance(
    pred_meshes: List[Tuple[np.ndarray, np.ndarray]],
    gt_meshes: List[Tuple[np.ndarray, np.ndarray]],
    *,
    n_samples: int = 50000,
    units: str = "cm",
) -> torch.Tensor:

    scale = _metric_units_scale(units)
    per_frame = []
    for pred_frame, gt_frame in tqdm(zip(pred_meshes, gt_meshes), desc="Computing Chamfer Distance", total=len(pred_meshes), leave=False):
        pred_v, pred_f = pred_frame
        gt_v, gt_f = gt_frame
        if pred_v.size == 0 or pred_f.size == 0 or gt_v.size == 0 or gt_f.size == 0:
            per_frame.append(float("nan"))
            continue
        pred_pts, _pred_normals, _ = _sample_mesh_points_and_normals(pred_v, pred_f, n_samples)
        gt_pts, _gt_normals, _ = _sample_mesh_points_and_normals(gt_v, gt_f, n_samples)
        d_pred = _nearest_distances(pred_pts, gt_pts)
        d_gt = _nearest_distances(gt_pts, pred_pts)
        chamfer = 0.5 * (d_pred.mean() + d_gt.mean())
        per_frame.append(float(chamfer * scale))
    return torch.tensor(per_frame, dtype=torch.float32)


def compute_p2s_distance(
    pred_meshes: List[Tuple[np.ndarray, np.ndarray]],
    gt_meshes: List[Tuple[np.ndarray, np.ndarray]],
    *,
    n_samples: int = 50000,
    units: str = "cm",
) -> torch.Tensor:

    scale = _metric_units_scale(units)
    per_frame = []
    for pred_frame, gt_frame in tqdm(zip(pred_meshes, gt_meshes), desc="Computing P2S Distance", total=len(pred_meshes), leave=False):
        pred_v, pred_f = pred_frame
        gt_v, gt_f = gt_frame
        if pred_v.size == 0 or pred_f.size == 0 or gt_v.size == 0 or gt_f.size == 0:
            per_frame.append(float("nan"))
            continue
        pred_pts, _pred_normals, _ = _sample_mesh_points_and_normals(pred_v, pred_f, n_samples)
        _gt_pts, _gt_normals, gt_mesh = _sample_mesh_points_and_normals(gt_v, gt_f, 1)
        _closest, distances, _face_idx = gt_mesh.nearest.on_surface(pred_pts)
        per_frame.append(float(distances.mean() * scale))
    return torch.tensor(per_frame, dtype=torch.float32)


def compute_normal_consistency(
    pred_meshes: List[Tuple[np.ndarray, np.ndarray]],
    gt_meshes: List[Tuple[np.ndarray, np.ndarray]],
    *,
    n_samples: int = 50000,
) -> torch.Tensor:

    per_frame = []
    for pred_frame, gt_frame in tqdm(zip(pred_meshes, gt_meshes), desc="Computing Normal Consistency", total=len(pred_meshes), leave=False):
        pred_v, pred_f = pred_frame
        gt_v, gt_f = gt_frame
        if pred_v.size == 0 or pred_f.size == 0 or gt_v.size == 0 or gt_f.size == 0:
            per_frame.append(float("nan"))
            continue
        pred_pts, pred_normals, _ = _sample_mesh_points_and_normals(pred_v, pred_f, n_samples)
        _gt_pts, _gt_normals, gt_mesh = _sample_mesh_points_and_normals(gt_v, gt_f, 1)
        _closest, _distances, face_idx = gt_mesh.nearest.on_surface(pred_pts)
        gt_normals = gt_mesh.face_normals[face_idx]
        dot = np.abs(np.einsum("ij,ij->i", pred_normals, gt_normals))
        per_frame.append(float(dot.mean()))
    return torch.tensor(per_frame, dtype=torch.float32)


def compute_volumetric_iou(
    pred_meshes: List[Tuple[np.ndarray, np.ndarray]],
    gt_meshes: List[Tuple[np.ndarray, np.ndarray]],
    *,
    voxel_size: float = 0.02,
    padding: float = 0.05,
) -> torch.Tensor:
    """
    Compute volumetric IoU on a per-frame basis assuming watertight meshes.
    Uses a shared voxel grid defined by the union bounds of pred+gt.
    """

    def _mesh_occupancy(mesh: trimesh.Trimesh, origin: np.ndarray, dims: Tuple[int, int, int]) -> np.ndarray:
        vox = mesh.voxelized(pitch=voxel_size).fill()
        occ = np.zeros(dims, dtype=bool)
        if vox.points.size == 0:
            return occ
        idx = np.floor((vox.points - origin) / voxel_size).astype(np.int32)
        valid = np.all((idx >= 0) & (idx < np.asarray(dims)), axis=1)
        idx = idx[valid]
        if idx.size > 0:
            occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return occ

    per_frame = []
    for pred_frame, gt_frame in tqdm(
        zip(pred_meshes, gt_meshes),
        desc="Computing Volumetric IoU",
        total=len(pred_meshes),
        leave=False,
    ):
        pred_v, pred_f = pred_frame
        gt_v, gt_f = gt_frame
        if pred_v.size == 0 or pred_f.size == 0 or gt_v.size == 0 or gt_f.size == 0:
            per_frame.append(float("nan"))
            continue
        all_verts = np.concatenate([pred_v, gt_v], axis=0)
        min_corner = all_verts.min(axis=0) - padding
        max_corner = all_verts.max(axis=0) + padding
        dims = np.maximum(np.ceil((max_corner - min_corner) / voxel_size).astype(np.int32), 1)
        dims_tuple = (int(dims[0]), int(dims[1]), int(dims[2]))

        pred_tm = trimesh.Trimesh(vertices=pred_v, faces=pred_f, process=False)
        gt_tm = trimesh.Trimesh(vertices=gt_v, faces=gt_f, process=False)
        pred_occ = _mesh_occupancy(pred_tm, min_corner, dims_tuple)
        gt_occ = _mesh_occupancy(gt_tm, min_corner, dims_tuple)
        union = np.logical_or(pred_occ, gt_occ).sum()
        if union == 0:
            per_frame.append(float("nan"))
            continue
        inter = np.logical_and(pred_occ, gt_occ).sum()
        per_frame.append(float(inter) / float(union))
    return torch.tensor(per_frame, dtype=torch.float32)

# ---------------------------------------------------------------------------
# Pose evaluation metrics
# ---------------------------------------------------------------------------

# ----- MPJPE
def _flatten_smpl_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.dim() == 4:
        return pose.reshape(pose.shape[0], pose.shape[1], -1)
    if pose.dim() == 3:
        return pose
    raise ValueError(f"Unexpected pose shape: {pose.shape}")


def _expand_body_model_layers(layers, npeople: int, label: str) -> List:
    if isinstance(layers, (list, tuple)):
        if len(layers) != npeople:
            raise ValueError(
                f"Expected {npeople} {label} layers, got {len(layers)}"
            )
        return list(layers)
    return [layers for _ in range(npeople)]


def _prepare_param_for_people(
    param: Optional[torch.Tensor],
    bsize: int,
    npeople: int,
    is_pose: bool,
    name: str,
) -> Optional[torch.Tensor]:
    if param is None:
        return None
    if param.dim() == 2:
        if bsize != 1:
            raise ValueError(
                f"Param '{name}' missing batch dimension (got {param.shape})"
            )
        param = param.unsqueeze(0)
    if is_pose:
        param = _flatten_smpl_pose(param)
    if param.dim() < 3:
        raise ValueError(
            f"Expected '{name}' shape [B,P,*], got {param.shape}"
        )
    if param.shape[0] != bsize or param.shape[1] != npeople:
        raise ValueError(
            f"Param '{name}' shape {param.shape} does not match [B={bsize}, P={npeople}]"
        )
    return param.reshape(bsize, npeople, -1)


def _smpl_params_to_joints(
    smpl_params: Dict[str, torch.Tensor],
    smpl_layers,
) -> torch.Tensor:
    if "contact" in smpl_params:
        smpl_params = {k: v for k, v in smpl_params.items() if k != "contact"}

    betas = smpl_params["betas"]
    body_pose = smpl_params["body_pose"]
    root_pose = smpl_params["root_pose"]
    trans = smpl_params["trans"]

    if betas.dim() != 3:
        raise ValueError(f"Expected betas shape [B,P,?], got {betas.shape}")

    bsize, npeople = betas.shape[:2]
    betas = _prepare_param_for_people(betas, bsize, npeople, is_pose=False, name="betas")
    body_pose = _prepare_param_for_people(body_pose, bsize, npeople, is_pose=True, name="body_pose")
    root_pose = _prepare_param_for_people(root_pose, bsize, npeople, is_pose=True, name="root_pose")
    trans = _prepare_param_for_people(trans, bsize, npeople, is_pose=False, name="trans")

    smpl_layers = _expand_body_model_layers(smpl_layers, npeople, "SMPL")
    joints_per_person = []
    for p, smpl_layer in enumerate(smpl_layers):
        output = smpl_layer(
            global_orient=root_pose[:, p],
            body_pose=body_pose[:, p],
            betas=betas[:, p],
            transl=trans[:, p],
        )
        joints_per_person.append(output.joints)
    return torch.stack(joints_per_person, dim=1)


def _smplx_params_to_joints(
    smplx_params: Dict[str, torch.Tensor],
    smplx_layers,
) -> torch.Tensor:
    if "contact" in smplx_params:
        smplx_params = {k: v for k, v in smplx_params.items() if k != "contact"}

    betas = smplx_params["betas"]
    body_pose = smplx_params["body_pose"]
    root_pose = smplx_params["root_pose"]
    trans = smplx_params["trans"]
    jaw_pose = smplx_params.get("jaw_pose")
    leye_pose = smplx_params.get("leye_pose")
    reye_pose = smplx_params.get("reye_pose")
    lhand_pose = smplx_params.get("lhand_pose")
    rhand_pose = smplx_params.get("rhand_pose")
    expr = smplx_params.get("expr", smplx_params.get("expression"))

    if betas.dim() != 3:
        raise ValueError(f"Expected betas shape [B,P,?], got {betas.shape}")

    bsize, npeople = betas.shape[:2]

    betas = _prepare_param_for_people(betas, bsize, npeople, is_pose=False, name="betas")
    body_pose = _prepare_param_for_people(body_pose, bsize, npeople, is_pose=True, name="body_pose")
    root_pose = _prepare_param_for_people(root_pose, bsize, npeople, is_pose=True, name="root_pose")
    trans = _prepare_param_for_people(trans, bsize, npeople, is_pose=False, name="trans")
    jaw_pose = _prepare_param_for_people(jaw_pose, bsize, npeople, is_pose=True, name="jaw_pose")
    leye_pose = _prepare_param_for_people(leye_pose, bsize, npeople, is_pose=True, name="leye_pose")
    reye_pose = _prepare_param_for_people(reye_pose, bsize, npeople, is_pose=True, name="reye_pose")
    lhand_pose = _prepare_param_for_people(lhand_pose, bsize, npeople, is_pose=True, name="lhand_pose")
    rhand_pose = _prepare_param_for_people(rhand_pose, bsize, npeople, is_pose=True, name="rhand_pose")
    expr = _prepare_param_for_people(expr, bsize, npeople, is_pose=False, name="expression")

    smplx_layers = _expand_body_model_layers(smplx_layers, npeople, "SMPL-X")
    joints_per_person = []
    for p, smplx_layer in enumerate(smplx_layers):
        expr_dim = int(
            getattr(
                smplx_layer,
                "num_expression_coeffs",
                getattr(smplx_layer, "num_expression", getattr(smplx_layer, "num_expr", 0)),
            )
        )
        if expr_dim > 0:
            if expr is None:
                expr_p = torch.zeros((bsize, expr_dim), device=betas.device, dtype=betas.dtype)
            else:
                expr_p = expr[:, p]
                if expr_p.shape[1] >= expr_dim:
                    expr_p = expr_p[:, :expr_dim]
                else:
                    pad = torch.zeros(
                        (expr_p.shape[0], expr_dim - expr_p.shape[1]),
                        device=expr_p.device,
                        dtype=expr_p.dtype,
                    )
                    expr_p = torch.cat([expr_p, pad], dim=1)
        else:
            expr_p = torch.zeros((bsize, 0), device=betas.device, dtype=betas.dtype)

        smplx_kwargs = {
            "global_orient": root_pose[:, p],
            "body_pose": body_pose[:, p],
            "jaw_pose": jaw_pose[:, p] if jaw_pose is not None else None,
            "leye_pose": leye_pose[:, p] if leye_pose is not None else None,
            "reye_pose": reye_pose[:, p] if reye_pose is not None else None,
            "left_hand_pose": lhand_pose[:, p] if lhand_pose is not None else None,
            "right_hand_pose": rhand_pose[:, p] if rhand_pose is not None else None,
            "betas": betas[:, p],
            "transl": trans[:, p],
            "expression": expr_p,
        }
        output = smplx_layer(**smplx_kwargs)
        joints_per_person.append(output.joints)
    return torch.stack(joints_per_person, dim=1)


def _select_layers_for_params(
    body_model_layer,
    smpl_params: Dict[str, torch.Tensor],
    label: str,
    kind: str,
):
    if "betas" not in smpl_params:
        raise ValueError(f"Missing 'betas' in {label} params.")
    betas = smpl_params["betas"]
    if betas.dim() != 3:
        raise ValueError(f"Expected betas shape [B,P,?], got {betas.shape}")
    npeople = betas.shape[1]
    layers = body_model_layer
    if isinstance(body_model_layer, dict):
        if kind not in body_model_layer:
            raise ValueError(f"Missing '{kind}' layers in body_model_layer.")
        layers = body_model_layer[kind]
    return _expand_body_model_layers(layers, npeople, f"{label} {kind}")


def compute_smpl_mpjpe_per_frame(
    pred_smpl_params: Dict[str, torch.Tensor],
    gt_smpl_params: Dict[str, torch.Tensor],
    body_model_layer,
    unit: str = "mm",
) -> torch.Tensor:
    pred_layers = _select_layers_for_params(body_model_layer, pred_smpl_params, "SMPL", "pred")
    gt_layers = _select_layers_for_params(body_model_layer, gt_smpl_params, "SMPL", "gt")
    pred_joints = _smpl_params_to_joints(pred_smpl_params, pred_layers)
    gt_joints = _smpl_params_to_joints(gt_smpl_params, gt_layers)
    per_joint = torch.linalg.norm(pred_joints - gt_joints, dim=-1)
    per_frame = per_joint.mean(dim=(1, 2))

    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported MPJPE unit: {unit}")

    return per_frame * scale


def compute_smplx_mpjpe_per_frame(
    pred_smplx_params: Dict[str, torch.Tensor],
    gt_smplx_params: Dict[str, torch.Tensor],
    body_model_layer,
    unit: str = "mm",
) -> torch.Tensor:
    pred_layers = _select_layers_for_params(body_model_layer, pred_smplx_params, "SMPL-X", "pred")
    gt_layers = _select_layers_for_params(body_model_layer, gt_smplx_params, "SMPL-X", "gt")
    pred_joints = _smplx_params_to_joints(pred_smplx_params, pred_layers)
    gt_joints = _smplx_params_to_joints(gt_smplx_params, gt_layers)
    per_joint = torch.linalg.norm(pred_joints - gt_joints, dim=-1)
    per_frame = per_joint.mean(dim=(1, 2))

    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported MPJPE unit: {unit}")

    return per_frame * scale


def compute_pose_mpjpe_per_frame(
    pred_params: Dict[str, torch.Tensor],
    gt_params: Dict[str, torch.Tensor],
    body_model_layer,
    unit: str = "mm",
    pose_type: str = "smplx",
) -> torch.Tensor:
    if pose_type == "smplx":
        return compute_smplx_mpjpe_per_frame(pred_params, gt_params, body_model_layer, unit=unit)
    if pose_type == "smpl":
        return compute_smpl_mpjpe_per_frame(pred_params, gt_params, body_model_layer, unit=unit)
    raise ValueError(f"Unknown pose_type: {pose_type}")


# ------- Vertex Error
def _smpl_params_to_vertices(
    smpl_params: Dict[str, torch.Tensor],
    smpl_layers,
) -> torch.Tensor:
    if "contact" in smpl_params:
        smpl_params = {k: v for k, v in smpl_params.items() if k != "contact"}

    betas = smpl_params["betas"]
    body_pose = smpl_params["body_pose"]
    root_pose = smpl_params["root_pose"]
    trans = smpl_params["trans"]

    if betas.dim() != 3:
        raise ValueError(f"Expected betas shape [B,P,?], got {betas.shape}")

    bsize, npeople = betas.shape[:2]
    betas = _prepare_param_for_people(betas, bsize, npeople, is_pose=False, name="betas")
    body_pose = _prepare_param_for_people(body_pose, bsize, npeople, is_pose=True, name="body_pose")
    root_pose = _prepare_param_for_people(root_pose, bsize, npeople, is_pose=True, name="root_pose")
    trans = _prepare_param_for_people(trans, bsize, npeople, is_pose=False, name="trans")

    smpl_layers = _expand_body_model_layers(smpl_layers, npeople, "SMPL")
    verts_per_person = []
    for p, smpl_layer in enumerate(smpl_layers):
        output = smpl_layer(
            global_orient=root_pose[:, p],
            body_pose=body_pose[:, p],
            betas=betas[:, p],
            transl=trans[:, p],
        )
        verts_per_person.append(output.vertices)
    return torch.stack(verts_per_person, dim=1)

def _smplx_params_to_vertices(
    smplx_params: Dict[str, torch.Tensor],
    smplx_layers,
) -> torch.Tensor:
    if "contact" in smplx_params:
        smplx_params = {k: v for k, v in smplx_params.items() if k != "contact"}

    betas = smplx_params["betas"]
    body_pose = smplx_params["body_pose"]
    root_pose = smplx_params["root_pose"]
    trans = smplx_params["trans"]
    jaw_pose = smplx_params.get("jaw_pose")
    leye_pose = smplx_params.get("leye_pose")
    reye_pose = smplx_params.get("reye_pose")
    lhand_pose = smplx_params.get("lhand_pose")
    rhand_pose = smplx_params.get("rhand_pose")
    expr = smplx_params.get("expr", smplx_params.get("expression"))

    if betas.dim() != 3:
        raise ValueError(f"Expected betas shape [B,P,?], got {betas.shape}")

    bsize, npeople = betas.shape[:2]

    betas = _prepare_param_for_people(betas, bsize, npeople, is_pose=False, name="betas")
    body_pose = _prepare_param_for_people(body_pose, bsize, npeople, is_pose=True, name="body_pose")
    root_pose = _prepare_param_for_people(root_pose, bsize, npeople, is_pose=True, name="root_pose")
    trans = _prepare_param_for_people(trans, bsize, npeople, is_pose=False, name="trans")
    jaw_pose = _prepare_param_for_people(jaw_pose, bsize, npeople, is_pose=True, name="jaw_pose")
    leye_pose = _prepare_param_for_people(leye_pose, bsize, npeople, is_pose=True, name="leye_pose")
    reye_pose = _prepare_param_for_people(reye_pose, bsize, npeople, is_pose=True, name="reye_pose")
    lhand_pose = _prepare_param_for_people(lhand_pose, bsize, npeople, is_pose=True, name="lhand_pose")
    rhand_pose = _prepare_param_for_people(rhand_pose, bsize, npeople, is_pose=True, name="rhand_pose")
    expr = _prepare_param_for_people(expr, bsize, npeople, is_pose=False, name="expression")

    smplx_layers = _expand_body_model_layers(smplx_layers, npeople, "SMPL-X")
    verts_per_person = []
    for p, smplx_layer in enumerate(smplx_layers):
        expr_dim = int(
            getattr(
                smplx_layer,
                "num_expression_coeffs",
                getattr(smplx_layer, "num_expression", getattr(smplx_layer, "num_expr", 0)),
            )
        )
        if expr_dim > 0:
            if expr is None:
                expr_p = torch.zeros((bsize, expr_dim), device=betas.device, dtype=betas.dtype)
            else:
                expr_p = expr[:, p]
                if expr_p.shape[1] >= expr_dim:
                    expr_p = expr_p[:, :expr_dim]
                else:
                    pad = torch.zeros(
                        (expr_p.shape[0], expr_dim - expr_p.shape[1]),
                        device=expr_p.device,
                        dtype=expr_p.dtype,
                    )
                    expr_p = torch.cat([expr_p, pad], dim=1)
        else:
            expr_p = torch.zeros((bsize, 0), device=betas.device, dtype=betas.dtype)
        smplx_kwargs = {
            "global_orient": root_pose[:, p],
            "body_pose": body_pose[:, p],
            "jaw_pose": jaw_pose[:, p] if jaw_pose is not None else None,
            "leye_pose": leye_pose[:, p] if leye_pose is not None else None,
            "reye_pose": reye_pose[:, p] if reye_pose is not None else None,
            "left_hand_pose": lhand_pose[:, p] if lhand_pose is not None else None,
            "right_hand_pose": rhand_pose[:, p] if rhand_pose is not None else None,
            "betas": betas[:, p],
            "transl": trans[:, p],
            "expression": expr_p,
        }
        output = smplx_layer(**smplx_kwargs)
        verts_per_person.append(output.vertices)
    return torch.stack(verts_per_person, dim=1)

def compute_smpl_mve_per_frame(
    pred_smpl_params: Dict[str, torch.Tensor],
    gt_smpl_params: Dict[str, torch.Tensor],
    body_model_layer,
    unit: str = "mm",
) -> torch.Tensor:
    pred_layers = _select_layers_for_params(body_model_layer, pred_smpl_params, "SMPL", "pred")
    gt_layers = _select_layers_for_params(body_model_layer, gt_smpl_params, "SMPL", "gt")
    pred_verts = _smpl_params_to_vertices(pred_smpl_params, pred_layers)
    gt_verts = _smpl_params_to_vertices(gt_smpl_params, gt_layers)
    per_vertex = torch.linalg.norm(pred_verts - gt_verts, dim=-1)
    per_frame = per_vertex.mean(dim=(1, 2))

    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported MVE unit: {unit}")

    return per_frame * scale


def compute_smplx_mve_per_frame(
    pred_smplx_params: Dict[str, torch.Tensor],
    gt_smplx_params: Dict[str, torch.Tensor],
    body_model_layer,
    unit: str = "mm",
) -> torch.Tensor:
    pred_layers = _select_layers_for_params(body_model_layer, pred_smplx_params, "SMPL-X", "pred")
    gt_layers = _select_layers_for_params(body_model_layer, gt_smplx_params, "SMPL-X", "gt")
    pred_verts = _smplx_params_to_vertices(pred_smplx_params, pred_layers)
    gt_verts = _smplx_params_to_vertices(gt_smplx_params, gt_layers)
    per_vertex = torch.linalg.norm(pred_verts - gt_verts, dim=-1)
    per_frame = per_vertex.mean(dim=(1, 2))

    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported MVE unit: {unit}")

    return per_frame * scale


def compute_pose_mve_per_frame(
    pred_params: Dict[str, torch.Tensor],
    gt_params: Dict[str, torch.Tensor],
    body_model_layer,
    unit: str = "mm",
    pose_type: str = "smplx",
) -> torch.Tensor:
    if pose_type == "smplx":
        return compute_smplx_mve_per_frame(pred_params, gt_params, body_model_layer, unit=unit)
    if pose_type == "smpl":
        return compute_smpl_mve_per_frame(pred_params, gt_params, body_model_layer, unit=unit)
    raise ValueError(f"Unknown pose_type: {pose_type}")


# ------- Contact Distance
def _contact_distance_per_frame(
    pred_verts: torch.Tensor,
    gt_contact: torch.Tensor,
    unit: str = "mm",
    invalid_value: int = 0,
) -> torch.Tensor:
    if gt_contact.dim() == 2:
        gt_contact = gt_contact.unsqueeze(0)
    if gt_contact.dim() != 3:
        raise ValueError(f"Expected contact shape [B,P,V] or [P,V], got {gt_contact.shape}")

    if gt_contact.shape[:2] != pred_verts.shape[:2] or gt_contact.shape[2] != pred_verts.shape[2]:
        raise ValueError(
            f"Contact shape {gt_contact.shape} does not match verts {pred_verts.shape}"
        )

    if pred_verts.shape[1] != 2:
        raise ValueError(f"Contact distance expects 2 people, got {pred_verts.shape[1]}")

    device = pred_verts.device
    gt_contact = gt_contact.to(device)
    num_verts = pred_verts.shape[2]
    per_frame = []
    for b in range(pred_verts.shape[0]):
        verts_b = pred_verts[b]
        contact_b = gt_contact[b].long()
        cd_vals = []
        for p in range(2):
            other = 1 - p
            corr = contact_b[p]
            valid = (
                (corr != invalid_value)
                & (corr >= 0)
                & (corr < num_verts)
            )
            if not torch.any(valid):
                cd_vals.append(None)
                continue
            idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            src = verts_b[p, idx]
            dst = verts_b[other, corr[idx]]
            cd_vals.append(torch.linalg.norm(src - dst, dim=-1).mean())

        if cd_vals[0] is not None and cd_vals[1] is not None:
            cd = 0.5 * (cd_vals[0] + cd_vals[1])
        elif cd_vals[0] is not None:
            cd = cd_vals[0]
        elif cd_vals[1] is not None:
            cd = cd_vals[1]
        else:
            cd = torch.tensor(0.0, device=device)
        per_frame.append(cd)

    per_frame = torch.stack(per_frame, dim=0)

    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported CD unit: {unit}")

    return per_frame * scale


def compute_smpl_contact_distance_per_frame(
    pred_smpl_params: Dict[str, torch.Tensor],
    gt_contact: torch.Tensor,
    body_model_layer,
    unit: str = "mm",
    invalid_value: int = 0,
) -> torch.Tensor:
    pred_layers = _select_layers_for_params(body_model_layer, pred_smpl_params, "SMPL", "pred")
    pred_verts = _smpl_params_to_vertices(pred_smpl_params, pred_layers)
    return _contact_distance_per_frame(
        pred_verts, gt_contact, unit=unit, invalid_value=invalid_value
    )


def compute_smplx_contact_distance_per_frame(
    pred_smplx_params: Dict[str, torch.Tensor],
    gt_contact: torch.Tensor,
    body_model_layer,
    unit: str = "mm",
    invalid_value: int = 0,
) -> torch.Tensor:
    pred_layers = _select_layers_for_params(body_model_layer, pred_smplx_params, "SMPL-X", "pred")
    pred_verts = _smplx_params_to_vertices(pred_smplx_params, pred_layers)
    return _contact_distance_per_frame(
        pred_verts, gt_contact, unit=unit, invalid_value=invalid_value
    )


def compute_pose_contact_distance_per_frame(
    pred_params: Dict[str, torch.Tensor],
    gt_contact: torch.Tensor,
    body_model_layer,
    unit: str = "mm",
    invalid_value: int = 0,
    pose_type: str = "smplx",
) -> torch.Tensor:
    if pose_type == "smplx":
        return compute_smplx_contact_distance_per_frame(
            pred_params, gt_contact, body_model_layer, unit=unit, invalid_value=invalid_value
        )
    if pose_type == "smpl":
        return compute_smpl_contact_distance_per_frame(
            pred_params, gt_contact, body_model_layer, unit=unit, invalid_value=invalid_value
        )
    raise ValueError(f"Unknown pose_type: {pose_type}")

# ------- PCDR
def _points_world_to_cam(points_world: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    if c2w.dim() != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w shape [B,4,4] or [4,4], got {c2w.shape}")
    w2c = torch.inverse(c2w)
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3]
    points_cam = torch.einsum("bpj,bkj->bpk", points_world, rot)
    return points_cam + trans[:, None, :]


def compute_smpl_pcdr_per_frame(
    pred_smpl_params: Dict[str, torch.Tensor],
    gt_smpl_params: Dict[str, torch.Tensor],
    c2w: torch.Tensor,
    tau: float = 0.15,
    gamma: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """Compute BEV-style PCDR per frame in camera coordinates (range [0,1]).

    This follows BEV's Relative Human (RH) evaluation protocol:
    - Derive ordinal depth layers from GT depths (DLs) using the grouping threshold `gamma`.
    - For each person pair, assign the GT relation category: eq / closer / farther.
    - A relation is correct if the predicted depth difference satisfies the category rule
      under threshold `tau` (applied to predicted depth differences).

    Args:
        pred_smpl_params: Predicted SMPL params with shapes [B, P, ...].
        gt_smpl_params: GT SMPL params with shapes [B, P, ...].
        c2w: Camera-to-world matrices, shape [B, 4, 4] or [4, 4].
        tau: Depth-relation threshold in meters for correctness.
        gamma: Depth-layer grouping threshold in meters (people within gamma are "equal depth").

    Returns:
        Dict with key "pcdr" mapping to a tensor of per-frame PCDR values of shape [B]

    Notes:
        - PCDR is view-dependent; person positions are transformed from world to camera
          space using the provided c2w matrices.
        - Depths are derived from per-person translations in camera space.
        - Only unique person pairs (upper triangle) are evaluated.
    """
    pred_trans = pred_smpl_params["trans"]
    gt_trans = gt_smpl_params["trans"]
    if pred_trans.dim() == 2:
        pred_trans = pred_trans.unsqueeze(0)
    if gt_trans.dim() == 2:
        gt_trans = gt_trans.unsqueeze(0)
    if pred_trans.dim() != 3 or gt_trans.dim() != 3:
        raise ValueError(
            f"Expected trans shape [B,P,3], got pred {pred_trans.shape} and gt {gt_trans.shape}"
        )

    pred_cam = _points_world_to_cam(pred_trans, c2w)
    gt_cam = _points_world_to_cam(gt_trans, c2w)

    if pred_cam.shape != gt_cam.shape:
        raise ValueError(f"Pred/GT camera positions must match. Got {pred_cam.shape} vs {gt_cam.shape}")

    device = pred_cam.device
    bsize, npeople, _ = pred_cam.shape
    if npeople < 2:
        return {
            "pcdr": torch.zeros((bsize,), device=device)
        }

    z_pred = pred_cam[:, :, 2]
    z_gt = gt_cam[:, :, 2]

    idx_i, idx_j = torch.triu_indices(npeople, npeople, offset=1, device=device)
    total_pairs = idx_i.numel()
    per_frame_pcdr = []

    for b in range(bsize):
        z_gt_b = z_gt[b]
        z_pred_b = z_pred[b]

        # Derive ordinal depth layers (DLs) from GT using 1D clustering along z.
        # Closest person gets layer 0; a new layer starts if the depth gap exceeds gamma.
        sorted_z, sorted_idx = torch.sort(z_gt_b, dim=0)
        layer_ids = torch.empty((npeople,), device=device, dtype=torch.long)
        current_layer = 0
        layer_ids[sorted_idx[0]] = current_layer
        for k in range(1, npeople):
            if (sorted_z[k] - sorted_z[k - 1]) > gamma:
                current_layer += 1
            layer_ids[sorted_idx[k]] = current_layer

        li = layer_ids[idx_i]
        lj = layer_ids[idx_j]
        dz_pred = z_pred_b[idx_i] - z_pred_b[idx_j]

        eq_mask = li == lj
        cd_mask = li < lj  # i is closer than j
        fd_mask = li > lj  # i is farther than j

        correct_eq = (dz_pred.abs() < tau)[eq_mask]
        correct_cd = (dz_pred < -tau)[cd_mask]
        correct_fd = (dz_pred > tau)[fd_mask]

        correct_total = (
            correct_eq.sum()
            + correct_cd.sum()
            + correct_fd.sum()
        ).float()
        pcdr_val = correct_total / float(total_pairs)
        per_frame_pcdr.append(pcdr_val)

    return {
        "pcdr": torch.stack(per_frame_pcdr, dim=0),
    }


def compute_smplx_pcdr_per_frame(
    pred_smplx_params: Dict[str, torch.Tensor],
    gt_smplx_params: Dict[str, torch.Tensor],
    c2w: torch.Tensor,
    tau: float = 0.15,
    gamma: float = 0.3,
) -> Dict[str, torch.Tensor]:
    return compute_smpl_pcdr_per_frame(
        pred_smplx_params, gt_smplx_params, c2w, tau=tau, gamma=gamma
    )


def compute_pose_pcdr_per_frame(
    pred_params: Dict[str, torch.Tensor],
    gt_params: Dict[str, torch.Tensor],
    c2w: torch.Tensor,
    tau: float = 0.15,
    gamma: float = 0.3,
    pose_type: str = "smplx",
) -> Dict[str, torch.Tensor]:
    if pose_type == "smplx":
        return compute_smplx_pcdr_per_frame(pred_params, gt_params, c2w, tau=tau, gamma=gamma)
    if pose_type == "smpl":
        return compute_smpl_pcdr_per_frame(pred_params, gt_params, c2w, tau=tau, gamma=gamma)
    raise ValueError(f"Unknown pose_type: {pose_type}")

# ---------------------------------------------------------------------------
# Segmentation evaluation metrics
# ---------------------------------------------------------------------------

def segmentation_mask_metrics(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute IoU, F1, and Recall for each sample in a batch of boolean masks.

    Args:
        gt_masks: [B,H,W] ground truth binary masks
        pred_masks: [B,H,W] predicted binary masks

    Returns:
        Dict with keys "segm_iou", "segm_f1", "segm_recall" mapping to tensors of shape [B]
    """

    if gt_masks.shape != pred_masks.shape:
        raise ValueError(f"Mask shapes must match. Got {gt_masks.shape} vs {pred_masks.shape}.")

    gt_flat = gt_masks.reshape(gt_masks.shape[0], -1).float()
    pred_flat = pred_masks.reshape(pred_masks.shape[0], -1).float()

    tp = (pred_flat * gt_flat).sum(dim=1)
    fp = (pred_flat * (1 - gt_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * gt_flat).sum(dim=1)

    union = tp + fp + fn
    safe_union = union.clamp_min(1e-6)
    iou_vals = tp / safe_union

    denom_f1 = (2 * tp + fp + fn).clamp_min(1e-6)
    f1_vals = (2 * tp) / denom_f1

    denom_recall = (tp + fn).clamp_min(1e-6)
    recall_vals = tp / denom_recall

    zero_mask = (union < 1e-6)
    iou_vals = torch.where(zero_mask, torch.zeros_like(iou_vals), iou_vals)
    f1_vals = torch.where(zero_mask, torch.zeros_like(f1_vals), f1_vals)
    recall_vals = torch.where(zero_mask, torch.zeros_like(recall_vals), recall_vals)

    return {"segm_iou": iou_vals, "segm_f1": f1_vals, "segm_recall": recall_vals}


# ---------------------------------------------------------------------------
# Appearance evaluation metrics
# ---------------------------------------------------------------------------

# Cached LPIPS metric instance; built lazily on first use.
_LPIPS_METRIC: Optional[torch.nn.Module] = None


def _get_lpips_net(device: torch.device) -> torch.nn.Module:
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None:
        # Spatial LPIPS gives a per-pixel distance map (pyiqa handles input normalisation to [-1,1])
        _LPIPS_METRIC = pyiqa.create_metric(
            "lpips", device=device, net="vgg", spatial=True, as_loss=False
        ).eval()
    return _LPIPS_METRIC.to(device)


def _ensure_nchw(t: torch.Tensor) -> torch.Tensor:
    """Convert tensors from NHWC (renderer output) to NCHW for library calls."""
    return t.permute(0, 3, 1, 2).contiguous()


def _mask_sums(mask: torch.Tensor) -> torch.Tensor:
    """Sum mask activations per sample (expects shape [B,1,H,W])."""
    return mask.sum(dim=(2, 3))


def ssim(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked SSIM per sample.
    
    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]
    Returns: 
        ssim_vals: [B] masked SSIM values 
    """
    target = _ensure_nchw(images.float())
    preds = _ensure_nchw(renders.float())
    mask = masks.unsqueeze(1).float()

    # Kornia returns per-channel SSIM; average across channels before masking
    ssim_map = kornia.metrics.ssim(preds, target, window_size=11, max_val=1.0)
    ssim_map = ssim_map.mean(1, keepdim=True)

    # Reduce over the masked region for each batch element
    numerator = (ssim_map * mask).sum(dim=(2, 3))
    mask_sum = _mask_sums(mask)
    safe_mask_sum = mask_sum.clamp_min(1e-6)
    result = numerator / safe_mask_sum
    result = torch.where(mask_sum < 1e-5, torch.zeros_like(result), result)
    return result.squeeze(1)


def psnr(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Masked PSNR per sample.

    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]

    Returns:
        psnr_vals: [B] masked PSNR values 
    """
    target = images.float()
    preds = renders.float()
    mask = masks.unsqueeze(-1).float()

    diff2 = (preds - target) ** 2
    masked_diff2 = diff2 * mask
    # Compute masked MSE then convert to PSNR
    numerator = masked_diff2.sum(dim=(1, 2, 3))
    denom = mask.sum(dim=(1, 2, 3))
    safe_denom = denom.clamp_min(1e-6)
    mse = numerator / safe_denom
    mse = mse.clamp_min(1e-12)
    psnr_vals = 10.0 * torch.log10((max_val ** 2) / mse)
    psnr_vals = torch.where(denom < 1e-5, torch.zeros_like(psnr_vals), psnr_vals)
    return psnr_vals


def lpips(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked spatial LPIPS per sample using pyiqa.


    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]
    Returns:
        lpips_vals: [B] masked LPIPS values 
    """
    target = _ensure_nchw(images.float()).clamp(0.0, 1.0)
    preds = _ensure_nchw(renders.float()).clamp(0.0, 1.0)
    mask = masks.unsqueeze(1).float()

    net = _get_lpips_net(preds.device)
    with torch.no_grad():
        dmap = net(preds, target)

    # Match mask resolution to the LPIPS map and average within the mask
    if dmap.shape[-2:] != mask.shape[-2:]:
        mask_resized = F.interpolate(mask, size=dmap.shape[-2:], mode="nearest")
    else:
        mask_resized = mask

    numerator = (dmap * mask_resized).sum(dim=(1, 2, 3))
    denom = mask_resized.sum(dim=(1, 2, 3))
    safe_denom = denom.clamp_min(1e-6)
    lpips_vals = numerator / safe_denom
    lpips_vals = torch.where(denom < 1e-5, torch.zeros_like(lpips_vals), lpips_vals)
    return lpips_vals
