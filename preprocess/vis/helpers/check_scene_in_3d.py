from __future__ import annotations

import sys
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import viser
import viser.transforms as tf
import cv2
from tqdm import tqdm
import tyro
from PIL import Image
import pyrender
from pytorch3d.transforms import quaternion_to_matrix

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from submodules.smplx import smplx
from preprocess.vis.helpers.cano_3dgs_to_posed import get_posed_3dgs


@dataclass
class Config:
    scene_dir: Annotated[Path, tyro.conf.arg(aliases=["--scenes-dir"])]
    src_cam_id: int = 0
    device: str = "cuda"
    model_folder: Path = Path("/home/cizinsky/body_models")
    smpl_model_ext: str = "pkl"
    smplx_model_ext: str = "npz"
    gender: str = "neutral"
    max_3dgs_scale: float = 0.02
    port: int = 8080
    center_scene: bool = True
    frame_idx_range: Tuple[int, int] = (0, 10)
    vis_3dgs: bool = False

def _load_skip_frames(scene_dir: Path) -> set[int]:
    skip_path = scene_dir / "skip_frames.csv"
    if not skip_path.exists():
        return set()
    content = skip_path.read_text(encoding="utf-8").strip()
    if not content:
        return set()
    return {int(x.strip()) for x in content.split(",") if x.strip().isdigit()}


def _collect_frame_stems(root: Path, suffix: str) -> List[str]:
    if not root.exists():
        return []
    stems = [p.stem for p in root.glob(f"*{suffix}") if p.is_file()]
    stems = [s for s in stems if s.isdigit()]
    return sorted(stems, key=lambda s: int(s))


def _build_layer(model_folder: Path, model_type: str, gender: str, ext: str, device: torch.device):
    layer = smplx.create(
        str(model_folder),
        model_type=model_type,
        gender=gender,
        ext=ext,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
    )
    return layer.to(device)


def _color_for_person(base: np.ndarray, pid: int) -> Tuple[int, int, int]:
    scale = 0.6 + 0.4 * ((pid % 5) / 4.0)
    color = np.clip(base * scale, 0, 255)
    return tuple(int(c) for c in color)


def _normalize_gender(value: str) -> str:
    gender = value.strip().lower()
    if gender in ("m", "male", "man", "masc", "masculine"):
        return "male"
    if gender in ("f", "female", "woman", "fem", "feminine"):
        return "female"
    if gender in ("neutral", "n", "none", "unknown", "unk"):
        return "neutral"
    return "neutral"


def _load_scene_genders(scene_dir: Path) -> Optional[List[str]]:
    meta_path = scene_dir / "meta.npz"
    if not meta_path.exists():
        return None
    with np.load(meta_path, allow_pickle=True) as data:
        if "genders" not in data.files:
            raise KeyError(f"Missing 'genders' key in {meta_path}")
        genders = data["genders"]
    if isinstance(genders, np.ndarray):
        genders = genders.tolist()
    if isinstance(genders, (str, bytes)):
        genders = [genders]
    normalized: List[str] = []
    for g in genders:
        if isinstance(g, bytes):
            g = g.decode("utf-8", errors="ignore")
        normalized.append(_normalize_gender(str(g)))
    return normalized


def _pad_or_truncate(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(vec.shape[-1])
    if current_dim == target_dim:
        return vec
    if current_dim > target_dim:
        return vec[..., :target_dim]
    pad = torch.zeros((*vec.shape[:-1], target_dim - current_dim), device=vec.device, dtype=vec.dtype)
    return torch.cat([vec, pad], dim=-1)


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        if not data.files:
            return None
        return {k: np.asarray(data[k]) for k in data.files}

def _collect_camera_ids(root: Path) -> List[str]:
    if not root.exists():
        return []
    ids = [p.name for p in root.iterdir() if p.is_dir()]

    def _sort_key(val: str):
        return int(val) if val.isdigit() else val

    return sorted(ids, key=_sort_key)

def _find_image_hw(images_dir: Path, cam_id: str, frame_ids: List[str]) -> Optional[Tuple[int, int]]:
    cam_dir = images_dir / cam_id
    if not cam_dir.exists():
        return None
    for frame_id in frame_ids:
        for ext in (".jpg", ".png", ".jpeg"):
            path = cam_dir / f"{frame_id}{ext}"
            if path.exists():
                with Image.open(path) as img:
                    width, height = img.size
                return height, width
    return None

def _find_frame_image(images_dir: Path, cam_id: str, frame_id: str) -> Optional[Path]:
    cam_dir = images_dir / cam_id
    if not cam_dir.exists():
        return None
    for ext in (".jpg", ".png", ".jpeg"):
        path = cam_dir / f"{frame_id}{ext}"
        if path.exists():
            return path
    return None

def _load_union_mask(scene_dir: Path, cam_id: str, frame_id: str) -> Optional[np.ndarray]:
    mask_path = scene_dir / "seg" / "img_seg_mask" / cam_id / "all" / f"{frame_id}.png"
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask

def _load_camera_params(camera_dir: Path, cam_id: str, frame_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    cam_path = camera_dir / cam_id / f"{frame_id}.npz"
    cam_data = _load_npz(cam_path)
    if cam_data is None:
        return None
    intr = cam_data.get("intrinsics")
    extr = cam_data.get("extrinsics")
    if intr is None or extr is None:
        return None
    if intr.ndim == 3:
        intr = intr[0]
    if extr.ndim == 3:
        extr = extr[0]
    if intr.shape != (3, 3) or extr.shape != (3, 4):
        return None
    return intr.astype(np.float32), extr.astype(np.float32)

def _masked_rgb_for_frame(scene_dir: Path, images_dir: Path, cam_id: str, frame_id: str) -> Optional[np.ndarray]:
    img_path = _find_frame_image(images_dir, cam_id, frame_id)
    if img_path is None:
        return None
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = img[..., ::-1]

    mask = _load_union_mask(scene_dir, cam_id, frame_id)
    if mask is None:
        return img

    if mask.shape[:2] != img.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {img.shape} for frame {frame_id} cam {cam_id}.")
    keep = mask > 0
    masked = img.copy()
    masked[~keep] = 0
    return masked

def _set_gui_image(handle, image: np.ndarray) -> None:
    if handle is None or image is None:
        return
    if hasattr(handle, "image"):
        handle.image = image
    elif hasattr(handle, "value"):
        handle.value = image


def _w2c_to_c2w_gl(w2c_cv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cv_to_gl = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    w2c_gl = cv_to_gl @ w2c_cv
    c2w_gl = np.linalg.inv(w2c_gl)
    c2w_gl[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return w2c_gl, c2w_gl


def _render_colored_meshes(
    meshes: List[trimesh.Trimesh],
    colors: List[Tuple[int, int, int]],
    renderer: pyrender.OffscreenRenderer,
    intr: np.ndarray,
    c2w_gl: np.ndarray,
    mesh_alpha: float,
) -> Tuple[np.ndarray, np.ndarray] | None:
    if not meshes:
        return None
    fx, fy, cx, cy = (
        float(intr[0, 0]),
        float(intr[1, 1]),
        float(intr[0, 2]),
        float(intr[1, 2]),
    )
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(1.0, 1.0, 1.0))
    added = False
    for mesh, color in zip(meshes, colors):
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        base_color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, float(mesh_alpha))
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            roughnessFactor=0.8,
            alphaMode="OPAQUE",
            baseColorFactor=base_color,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene.add(pr_mesh)
        added = True
    if not added:
        return None
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)
    scene.add(camera, pose=c2w_gl)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
        pose=c2w_gl,
    )
    color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return color_rgba, depth


def _render_smplx_overlay(
    scene_dir: Path,
    images_dir: Path,
    camera_dir: Path,
    cam_id: str,
    frame_id: str,
    smplx_layer,
    smplx_faces: Optional[np.ndarray],
    device: torch.device,
    renderer_state: Dict[str, object],
) -> Optional[np.ndarray]:
    img_path = _find_frame_image(images_dir, cam_id, frame_id)
    if img_path is None:
        return None
    rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if rgb is None:
        return None
    rgb = rgb[..., ::-1]

    if smplx_layer is None or smplx_faces is None:
        return rgb

    smplx_data = _load_npz(scene_dir / "smplx" / f"{frame_id}.npz")
    if smplx_data is None:
        return rgb

    verts = _smplx_vertices(smplx_layer, smplx_data, device, genders=None, layer_by_gender=None)
    if verts is None or verts.size == 0:
        return rgb

    cam_params = _load_camera_params(camera_dir, cam_id, frame_id)
    if cam_params is None:
        return rgb
    intr, extr = cam_params

    height, width = rgb.shape[:2]
    if renderer_state.get("hw") != (height, width):
        if renderer_state.get("renderer") is not None:
            renderer_state["renderer"].delete()
        renderer_state["renderer"] = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        renderer_state["hw"] = (height, width)

    meshes: List[trimesh.Trimesh] = []
    colors: List[Tuple[int, int, int]] = []
    for pid in range(verts.shape[0]):
        mesh = trimesh.Trimesh(vertices=verts[pid], faces=smplx_faces, process=False)
        meshes.append(mesh)
        colors.append(_color_for_person(np.array([70, 130, 255], dtype=np.float32), pid))

    w2c_cv = np.eye(4, dtype=np.float32)
    w2c_cv[:3, :4] = extr
    _, c2w_gl = _w2c_to_c2w_gl(w2c_cv)

    render_out = _render_colored_meshes(
        meshes,
        colors,
        renderer_state["renderer"],
        intr,
        c2w_gl,
        mesh_alpha=0.8,
    )
    if render_out is None:
        return rgb
    smplx_rgba, _ = render_out
    smplx_rgb = smplx_rgba[..., :3].astype(np.float32)
    smplx_alpha = smplx_rgba[..., 3:4].astype(np.float32) / 255.0
    base = rgb.astype(np.float32)
    overlay = np.clip(smplx_rgb * smplx_alpha + base * (1.0 - smplx_alpha), 0, 255).astype(np.uint8)
    return overlay

def _fov_aspect_from_intrinsics(
    intr: np.ndarray, image_hw: Optional[Tuple[int, int]]
) -> Tuple[float, float]:
    fy = float(intr[1, 1])
    if image_hw is not None and fy > 0.0:
        height, width = image_hw
        if height > 0 and width > 0:
            fov = 2.0 * np.arctan2(height / 2.0, fy)
            return fov, width / float(height)

    cx = float(intr[0, 2])
    cy = float(intr[1, 2])
    if fy > 0.0 and cx > 0.0 and cy > 0.0:
        height = 2.0 * cy
        width = 2.0 * cx
        fov = 2.0 * np.arctan2(height / 2.0, fy)
        return fov, width / height

    return np.pi / 3.0, 1.0

def _center_offset_from_vertices(verts: np.ndarray) -> np.ndarray:
    if verts.ndim == 3:
        verts_flat = verts.reshape(-1, 3)
    else:
        verts_flat = verts
    vmin = verts_flat.min(axis=0)
    vmax = verts_flat.max(axis=0)
    return (vmin + vmax) * 0.5


def _ground_y_from_mesh(mesh_dir: Path, frame_id: str) -> Optional[float]:
    if not mesh_dir.exists():
        return None
    mesh_path = mesh_dir / f"{frame_id}.obj"
    if not mesh_path.exists():
        return None
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        return None
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    return float(verts[:, 1].min())


def _compute_center_offset(
    frames: List[str],
    smpl_dir: Path,
    smplx_dir: Path,
    mesh_dir: Path,
    smpl_layer,
    smplx_layer,
    device: torch.device,
) -> np.ndarray:
    frame_id = frames[0]
    if mesh_dir.exists():
        mesh_path = mesh_dir / f"{frame_id}.obj"
        if mesh_path.exists():
            mesh = trimesh.load_mesh(mesh_path, process=False)
            if isinstance(mesh, trimesh.Trimesh) and mesh.vertices.size:
                return _center_offset_from_vertices(np.asarray(mesh.vertices, dtype=np.float32))

    if smplx_layer is not None:
        smplx_data = _load_npz(smplx_dir / f"{frame_id}.npz")
        if smplx_data is not None:
            verts = _smplx_vertices(smplx_layer, smplx_data, device)
            if verts is not None and verts.size:
                return _center_offset_from_vertices(verts)

    if smpl_layer is not None:
        smpl_data = _load_npz(smpl_dir / f"{frame_id}.npz")
        if smpl_data is not None:
            verts = _smpl_vertices(smpl_layer, smpl_data, device)
            if verts is not None and verts.size:
                return _center_offset_from_vertices(verts)

    return np.zeros(3, dtype=np.float32)


def _prepare_posed_3dgs(posed_3dgs, max_scale: Optional[float]) -> Dict[str, np.ndarray]:
    xyz = posed_3dgs["xyz"].float()
    opacity = posed_3dgs["opacity"].float()
    rotation = posed_3dgs["rotation"].float()
    scaling = posed_3dgs["scaling"].float()
    shs = posed_3dgs["shs"].float()

    opacity = opacity.clamp(0.0, 1.0)
    rgb_coeff = shs.squeeze(1)
    rgb = rgb_coeff.clamp(0.0, 1.0)
    rotation = torch.nn.functional.normalize(rotation, dim=-1)
    R = quaternion_to_matrix(rotation)  # [N, 3, 3]
    scales = scaling.clamp(min=1e-8)
    if max_scale is not None:
        scales = scales.clamp(max=max_scale)


    cov = R @ torch.diag_embed(scales**2) @ R.transpose(-1, -2)  # [N, 3, 3]

    return {
        "centers": xyz.detach().cpu().numpy(),
        "opacities": opacity.detach().cpu().numpy(),
        "rgbs": rgb.detach().cpu().numpy(),
        "covariances": cov.detach().cpu().numpy(),
    }

def _smpl_vertices(
    layer,
    params: Dict[str, np.ndarray],
    device: torch.device,
    genders: Optional[List[str]] = None,
    layer_by_gender: Optional[Dict[str, object]] = None,
) -> Optional[np.ndarray]:
    required = {"betas", "global_orient", "body_pose", "transl"}
    if not required.issubset(params.keys()):
        return None
    betas = torch.tensor(params["betas"], dtype=torch.float32, device=device) # [P, 10]
    global_orient = torch.tensor(params["global_orient"], dtype=torch.float32, device=device) # [P, 3]
    body_pose = torch.tensor(params["body_pose"], dtype=torch.float32, device=device) # [P, 69]
    transl = torch.tensor(params["transl"], dtype=torch.float32, device=device) # [P, 3]

    betas = betas.reshape(-1, betas.shape[-1])
    global_orient = global_orient.reshape(-1, 3)
    body_pose = body_pose.reshape(-1, 69)
    transl = transl.reshape(-1, 3)

    verts_out = []
    num_people = global_orient.shape[0]
    for pid in range(num_people):
        gender = genders[pid] if genders is not None and pid < len(genders) else "neutral"
        use_layer = layer_by_gender.get(gender) if layer_by_gender is not None else None
        if use_layer is None:
            use_layer = layer
        if use_layer is None:
            continue

        expected_betas = int(getattr(use_layer, "num_betas", betas.shape[-1]))
        betas_pid = _pad_or_truncate(betas[pid : pid + 1], expected_betas)
        with torch.no_grad():
            output = use_layer(
                global_orient=global_orient[pid : pid + 1],
                body_pose=body_pose[pid : pid + 1],
                betas=betas_pid,
                transl=transl[pid : pid + 1],
            )
        verts_out.append(output.vertices.detach().cpu().numpy())
    if not verts_out:
        return None
    return np.concatenate(verts_out, axis=0)


def _smplx_vertices(
    layer,
    params: Dict[str, np.ndarray],
    device: torch.device,
    genders: Optional[List[str]] = None,
    layer_by_gender: Optional[Dict[str, object]] = None,
) -> Optional[np.ndarray]:
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

    verts_out = []
    num_people = root_pose.shape[0]
    for pid in range(num_people):
        gender = genders[pid] if genders is not None and pid < len(genders) else "neutral"
        use_layer = layer_by_gender.get(gender) if layer_by_gender is not None else None
        if use_layer is None:
            use_layer = layer
        if use_layer is None:
            continue

        expected_betas = int(getattr(use_layer, "num_betas", betas.shape[-1]))
        betas_pid = _pad_or_truncate(betas[pid : pid + 1], expected_betas)

        expr_dim = int(getattr(use_layer, "num_expression_coeffs", 0))
        if expr_dim > 0:
            expr = params.get("expression")
            if expr is None:
                expr_pid = torch.zeros((1, expr_dim), device=device, dtype=betas.dtype)
            else:
                expr_tensor = torch.tensor(expr, dtype=torch.float32, device=device)
                if expr_tensor.ndim == 1:
                    expr_tensor = expr_tensor[None, :]
                expr_pid = _pad_or_truncate(expr_tensor[pid : pid + 1], expr_dim)
        else:
            expr_pid = None

        call_args = dict(
            global_orient=root_pose[pid : pid + 1],
            body_pose=body_pose[pid : pid + 1],
            jaw_pose=jaw_pose[pid : pid + 1],
            leye_pose=leye_pose[pid : pid + 1],
            reye_pose=reye_pose[pid : pid + 1],
            left_hand_pose=lhand_pose[pid : pid + 1],
            right_hand_pose=rhand_pose[pid : pid + 1],
            betas=betas_pid,
            transl=trans[pid : pid + 1],
        )
        if expr_pid is not None:
            call_args["expression"] = expr_pid

        with torch.no_grad():
            output = use_layer(**call_args)
        verts_out.append(output.vertices.detach().cpu().numpy())
    if not verts_out:
        return None
    return np.concatenate(verts_out, axis=0)


def main() -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    cfg = tyro.cli(Config)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    seq_name = cfg.scene_dir.name

    # Resolve modality directories.
    smpl_dir = cfg.scene_dir / "smpl"
    smplx_dir = cfg.scene_dir / "smplx"
    mesh_dir = cfg.scene_dir / "meshes"
    camera_dir = cfg.scene_dir / "all_cameras"
    images_dir = cfg.scene_dir / "images"

    # Collect frames for each modality (if present).
    smpl_frames = _collect_frame_stems(smpl_dir, ".npz") if smpl_dir.exists() else []
    smplx_frames = _collect_frame_stems(smplx_dir, ".npz") if smplx_dir.exists() else []
    mesh_frames = _collect_frame_stems(mesh_dir, ".obj") if mesh_dir.exists() else []

    available_sets: List[set[str]] = []
    if smpl_frames:
        available_sets.append(set(smpl_frames))
    if smplx_frames:
        available_sets.append(set(smplx_frames))
    if mesh_frames:
        available_sets.append(set(mesh_frames))

    if not available_sets:
        raise FileNotFoundError("No SMPL, SMPL-X, or mesh data found in scene directory.")

    # Intersect frames across available modalities, then drop skipped frames.
    frames = sorted(set.intersection(*available_sets), key=lambda s: int(s))
    if not frames:
        raise FileNotFoundError("No common frames across available modalities.")

    skip_frames = _load_skip_frames(cfg.scene_dir)
    frames = [f for f in frames if int(f) not in skip_frames]
    if not frames:
        raise FileNotFoundError("No frames left after applying skip_frames.csv.")

    start_idx, end_idx = cfg.frame_idx_range
    if start_idx < 0 or end_idx < 0:
        raise ValueError("frame_idx_range must be non-negative.")
    if end_idx <= start_idx:
        raise ValueError("frame_idx_range end must be greater than start.")
    if start_idx >= len(frames):
        raise ValueError(
            f"frame_idx_range start {start_idx} is out of bounds for {len(frames)} frames."
        )
    end_idx = min(end_idx, len(frames))
    frames = frames[start_idx:end_idx]
    if not frames:
        raise FileNotFoundError("No frames left after applying frame_idx_range.")

    # Load and prepare posed 3D Gaussians (optional; can be heavy).
    all_posed_3dgs: List[Dict[str, np.ndarray]] = []
    if cfg.vis_3dgs:
        all_posed_3dgs_raw = get_posed_3dgs(cfg.scene_dir, frames=frames)
        all_posed_3dgs = [
            _prepare_posed_3dgs(p, max_scale=cfg.max_3dgs_scale) for p in all_posed_3dgs_raw
        ]
    has_3dgs = len(all_posed_3dgs) > 0

    camera_ids = _collect_camera_ids(camera_dir)
    image_camera_ids = _collect_camera_ids(images_dir)
    camera_image_hw = {
        cam_id: _find_image_hw(images_dir, cam_id, frames) for cam_id in camera_ids
    }

    mask_cam_id: Optional[str] = None
    if image_camera_ids:
        src_id = str(cfg.src_cam_id)
        mask_cam_id = src_id if src_id in image_camera_ids else image_camera_ids[0]

    overlay_cam_id: Optional[str] = None
    if mask_cam_id is not None and mask_cam_id in camera_ids:
        overlay_cam_id = mask_cam_id
    else:
        for cid in camera_ids:
            if cid in image_camera_ids:
                overlay_cam_id = cid
                break

    # Load per-person genders if available (meta.npz -> genders).
    scene_genders = _load_scene_genders(cfg.scene_dir)
    if scene_genders is None:
        print(f"[info] No meta.npz genders found in {cfg.scene_dir}; using default gender.")
    else:
        print(f"[info] Found meta.npz genders: {scene_genders}")

    # Build body model layers (only if the modality exists).
    smpl_layer_by_gender = None
    smpl_layer = None
    smplx_layer = None
    if smpl_frames:
        if scene_genders is None:
            smpl_layer = _build_layer(
                cfg.model_folder, "smpl", cfg.gender, cfg.smpl_model_ext, device
            )
        else:
            smpl_layer_by_gender = {
                g: _build_layer(cfg.model_folder, "smpl", g, cfg.smpl_model_ext, device)
                for g in sorted(set(scene_genders))
            }
    if smplx_frames:
        smplx_layer = _build_layer(
            cfg.model_folder, "smplx", cfg.gender, cfg.smplx_model_ext, device
        )

    if smpl_layer is not None:
        smpl_faces = np.asarray(smpl_layer.faces, dtype=np.int32)
    elif smpl_layer_by_gender:
        smpl_faces = np.asarray(next(iter(smpl_layer_by_gender.values())).faces, dtype=np.int32)
    else:
        smpl_faces = None
    smplx_faces = np.asarray(smplx_layer.faces, dtype=np.int32) if smplx_layer is not None else None

    # Compute a shared scene offset for visual centering.
    center_offset = (
        _compute_center_offset(
            frames,
            smpl_dir,
            smplx_dir,
            mesh_dir,
            smpl_layer or (next(iter(smpl_layer_by_gender.values())) if smpl_layer_by_gender else None),
            smplx_layer,
            device,
        )
        if cfg.center_scene
        else np.zeros(3, dtype=np.float32)
    )


    R_fix = tf.SO3.from_x_radians(np.pi / 2)

    # Create the Viser server and a centered root frame.
    server = viser.ViserServer(port=cfg.port)

    server.scene.add_frame(
        "/scene",
        show_axes=False,
    )
    smpl_root = (
        server.scene.add_frame(
            "/scene/smpl", 
            show_axes=False, 
            wxyz=tuple(R_fix.wxyz),
            position=tuple((-R_fix.apply(center_offset)).tolist()),
        ) 
        if smpl_layer is not None or smpl_layer_by_gender is not None
        else None
    )
    smplx_root = (
        server.scene.add_frame(
            "/scene/smplx", 
            show_axes=False, 
            wxyz=tuple(R_fix.wxyz),
            position=tuple((-R_fix.apply(center_offset)).tolist()),
        )
        if smplx_layer is not None
        else None
    )
    mesh_root = (
        server.scene.add_frame(
            "/scene/meshes", 
            show_axes=False,
            wxyz=tuple(R_fix.wxyz),
            position=tuple((-R_fix.apply(center_offset)).tolist()),
        ) 
        if mesh_frames else None
    )
    camera_root = (
        server.scene.add_frame(
            "/scene/cameras",
            show_axes=False,
            wxyz=tuple(R_fix.wxyz),
            position=tuple((-R_fix.apply(center_offset)).tolist()),
        )
        if camera_ids
        else None
    )
    gs_root = (
        server.scene.add_frame(
            "/scene/3dgs",
            show_axes=False,
            wxyz=tuple(R_fix.wxyz),
            position=tuple((-R_fix.apply(center_offset)).tolist()),
        )
        if has_3dgs
        else None
    )

    # GUI controls.
    show_smpl = None
    show_smplx = None
    show_mesh = None
    show_cameras = None
    show_3dgs = None

    with server.gui.add_folder("Visibility"):
        if smpl_root is not None:
            show_smpl = server.gui.add_checkbox("Show SMPL", True)
        if smplx_root is not None:
            show_smplx = server.gui.add_checkbox("Show SMPL-X", True)
        if mesh_root is not None:
            show_mesh = server.gui.add_checkbox("Show Mesh", True)
        if camera_root is not None:
            show_cameras = server.gui.add_checkbox("Show Cameras", True)
        if gs_root is not None:
            show_3dgs = server.gui.add_checkbox("Show 3DGS", True)

    with server.gui.add_folder("Frames"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(frames) - 1,
            step=1,
            initial_value=0,
        )
        frame_label = server.gui.add_text("File", frames[0])

    # 2D masks view.
    mask_image_handle = None
    with server.gui.add_folder("2D masks"):
        if mask_cam_id is None:
            server.gui.add_text("Masked RGB", "No images/seg found for masks.")
        else:
            masked_rgb = _masked_rgb_for_frame(cfg.scene_dir, images_dir, mask_cam_id, frames[0])
            if masked_rgb is None:
                masked_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
            mask_image_handle = server.gui.add_image(masked_rgb, label="Masked RGB")

    # Precompute SMPL-X overlays to avoid GL usage inside GUI callbacks.
    overlay_images: Dict[str, np.ndarray] = {}
    if overlay_cam_id is not None and smplx_layer is not None and smplx_faces is not None:
        overlay_renderer_state: Dict[str, object] = {"renderer": None, "hw": None}
        for frame_id in tqdm(frames, desc="Rendering SMPL-X overlays", total=len(frames)):
            try:
                overlay_rgb = _render_smplx_overlay(
                    cfg.scene_dir,
                    images_dir,
                    camera_dir,
                    overlay_cam_id,
                    frame_id,
                    smplx_layer,
                    smplx_faces,
                    device,
                    overlay_renderer_state,
                )
            except Exception as exc:
                print(f"[WARN] Failed to render overlay for frame {frame_id}: {exc}")
                overlay_rgb = None
            if overlay_rgb is None:
                overlay_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
            overlay_images[frame_id] = overlay_rgb
        if overlay_renderer_state.get("renderer") is not None:
            overlay_renderer_state["renderer"].delete()

    # 3D -> 2D view.
    overlay_image_handle = None
    with server.gui.add_folder("3D -> 2D"):
        if not overlay_images:
            server.gui.add_text("SMPL-X Overlay", "Missing camera/images/SMPL-X for overlay.")
        else:
            overlay_rgb = overlay_images.get(frames[0], np.zeros((1, 1, 3), dtype=np.uint8))
            overlay_image_handle = server.gui.add_image(overlay_rgb, label="SMPL-X Overlay")

    # Colors for different modalities.
    smpl_base = np.array([255, 140, 70], dtype=np.float32)
    smplx_base = np.array([70, 130, 255], dtype=np.float32)
    mesh_color = (200, 200, 200)

    # Preload meshes for each modality under per-frame nodes.
    smpl_nodes: List[viser.FrameHandle] = []
    smplx_nodes: List[viser.FrameHandle] = []
    mesh_nodes: List[viser.FrameHandle] = []
    camera_nodes: List[viser.FrameHandle] = []
    gs_nodes: List[viser.FrameHandle] = []

    if smpl_root is not None:
        for frame_id in tqdm(frames, desc="Loading SMPL"):
            node = server.scene.add_frame(f"/scene/smpl/f_{frame_id}", show_axes=False)
            smpl_nodes.append(node)
            smpl_data = _load_npz(smpl_dir / f"{frame_id}.npz")
            if smpl_data is not None:
                verts = _smpl_vertices(
                    smpl_layer,
                    smpl_data,
                    device,
                    genders=scene_genders,
                    layer_by_gender=smpl_layer_by_gender,
                )
                if verts is not None:
                    for pid in range(verts.shape[0]):
                        color = _color_for_person(smpl_base, pid)
                        server.scene.add_mesh_simple(
                            f"/scene/smpl/f_{frame_id}/person_{pid}",
                            vertices=verts[pid],
                            faces=smpl_faces,
                            color=color,
                        )
            node.visible = False

    if smplx_root is not None:
        for frame_id in tqdm(frames, desc="Loading SMPL-X"):
            node = server.scene.add_frame(f"/scene/smplx/f_{frame_id}", show_axes=False)
            smplx_nodes.append(node)
            smplx_data = _load_npz(smplx_dir / f"{frame_id}.npz")
            if smplx_data is not None:
                verts = _smplx_vertices(
                    smplx_layer,
                    smplx_data,
                    device,
                    genders=None,
                    layer_by_gender=None,
                )
                if verts is not None:
                    for pid in range(verts.shape[0]):
                        color = _color_for_person(smplx_base, pid)
                        server.scene.add_mesh_simple(
                            f"/scene/smplx/f_{frame_id}/person_{pid}",
                            vertices=verts[pid],
                            faces=smplx_faces,
                            color=color,
                        )
                        # debug - compute min of y axis for given person
#                        print(f"SMPL-X Frame {frame_id} Person {pid} min y: {verts[pid][:,1].min()}, max y: {verts[pid][:,1].max()}, delta y: {verts[pid][:,1].max() - verts[pid][:,1].min()}")
            node.visible = False

    if mesh_root is not None:
        for frame_id in tqdm(frames, desc="Loading Meshes"):
            node = server.scene.add_frame(f"/scene/meshes/f_{frame_id}", show_axes=False)
            mesh_nodes.append(node)
            mesh_path = mesh_dir / f"{frame_id}.obj"
            if mesh_path.exists():
                mesh = trimesh.load_mesh(mesh_path, process=False)
                if isinstance(mesh, trimesh.Trimesh) and mesh.vertices.size and mesh.faces.size:
                    server.scene.add_mesh_simple(
                        f"/scene/meshes/f_{frame_id}/mesh",
                        vertices=np.asarray(mesh.vertices, dtype=np.float32),
                        faces=np.asarray(mesh.faces, dtype=np.int32),
                        color=mesh_color,
                    )
            node.visible = False

    if camera_root is not None:
        for frame_id in tqdm(frames, desc="Loading Cameras"):
            node = server.scene.add_frame(f"/scene/cameras/f_{frame_id}", show_axes=False)
            camera_nodes.append(node)
            for cam_id in camera_ids:
                cam_path = camera_dir / cam_id / f"{frame_id}.npz"
                cam_data = _load_npz(cam_path)
                if cam_data is None:
                    continue
                intr = cam_data.get("intrinsics")
                extr = cam_data.get("extrinsics")
                if intr is None or extr is None:
                    continue
                if intr.ndim == 3:
                    intr = intr[0]
                if extr.ndim == 3:
                    extr = extr[0]
                if intr.shape != (3, 3) or extr.shape != (3, 4):
                    continue
                w2c = np.eye(4, dtype=np.float32)
                w2c[:3, :4] = extr.astype(np.float32)
                c2w = np.linalg.inv(w2c)
                fov, aspect = _fov_aspect_from_intrinsics(intr, camera_image_hw.get(cam_id))
                is_src = cam_id == str(cfg.src_cam_id)
                color = (30, 144, 255) if is_src else (255, 140, 0)
                server.scene.add_camera_frustum(
                    f"/scene/cameras/f_{frame_id}/{cam_id}",
                    fov=float(fov),
                    aspect=float(aspect),
                    scale=0.2,
                    wxyz=tuple(tf.SO3.from_matrix(c2w[:3, :3]).wxyz),
                    position=tuple(c2w[:3, 3].tolist()),
                    color=color,
                )
            node.visible = False
    if gs_root is not None:
        for frame_idx, frame_id in enumerate(tqdm(frames, desc="Loading 3DGS")):
            node = server.scene.add_frame(f"/scene/3dgs/f_{frame_id}", show_axes=False)
            gs_nodes.append(node)
            splat = all_posed_3dgs[frame_idx]
            centers = splat["centers"]
            if centers.size:
                server.scene.add_gaussian_splats(
                    f"/scene/3dgs/f_{frame_id}/splats",
                    centers=centers,
                    rgbs=splat["rgbs"],
                    opacities=splat["opacities"],
                    covariances=splat["covariances"],
                )
            node.visible = False

    current_idx = 0

    # Update visibility to show only the selected frame.
    def _apply_visibility(frame_idx: int) -> None:
        nonlocal current_idx
        if frame_idx == current_idx:
            return
        if smpl_nodes:
            smpl_nodes[current_idx].visible = False
        if smplx_nodes:
            smplx_nodes[current_idx].visible = False
        if mesh_nodes:
            mesh_nodes[current_idx].visible = False
        if camera_nodes:
            camera_nodes[current_idx].visible = False
        if gs_nodes:
            gs_nodes[current_idx].visible = False

        current_idx = frame_idx
        frame_label.value = frames[frame_idx]

        if smpl_nodes and show_smpl is not None:
            smpl_nodes[frame_idx].visible = show_smpl.value
        if smplx_nodes and show_smplx is not None:
            smplx_nodes[frame_idx].visible = show_smplx.value
        if mesh_nodes and show_mesh is not None:
            mesh_nodes[frame_idx].visible = show_mesh.value
        if camera_nodes and show_cameras is not None:
            camera_nodes[frame_idx].visible = show_cameras.value
        if gs_nodes and show_3dgs is not None:
            gs_nodes[frame_idx].visible = show_3dgs.value

        if mask_cam_id is not None and mask_image_handle is not None:
            masked_rgb = _masked_rgb_for_frame(cfg.scene_dir, images_dir, mask_cam_id, frames[frame_idx])
            if masked_rgb is None:
                masked_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
            _set_gui_image(mask_image_handle, masked_rgb)

        if overlay_image_handle is not None and overlay_images:
            overlay_rgb = overlay_images.get(frames[frame_idx])
            if overlay_rgb is None:
                overlay_rgb = np.zeros((1, 1, 3), dtype=np.uint8)
            _set_gui_image(overlay_image_handle, overlay_rgb)

    # Refresh visibility for the current frame when toggles change.
    def _refresh_current() -> None:
        if smpl_nodes and show_smpl is not None:
            smpl_nodes[current_idx].visible = show_smpl.value
        if smplx_nodes and show_smplx is not None:
            smplx_nodes[current_idx].visible = show_smplx.value
        if mesh_nodes and show_mesh is not None:
            mesh_nodes[current_idx].visible = show_mesh.value
        if camera_nodes and show_cameras is not None:
            camera_nodes[current_idx].visible = show_cameras.value
        if gs_nodes and show_3dgs is not None:
            gs_nodes[current_idx].visible = show_3dgs.value

    if smpl_nodes:
        smpl_nodes[0].visible = True if show_smpl is None else show_smpl.value
    if smplx_nodes:
        smplx_nodes[0].visible = True if show_smplx is None else show_smplx.value
    if mesh_nodes:
        mesh_nodes[0].visible = True if show_mesh is None else show_mesh.value
    if camera_nodes:
        camera_nodes[0].visible = True if show_cameras is None else show_cameras.value
    if gs_nodes:
        gs_nodes[0].visible = True if show_3dgs is None else show_3dgs.value

    @frame_slider.on_update
    def _(_) -> None:
        _apply_visibility(int(frame_slider.value))

    if show_smpl is not None:
        @show_smpl.on_update
        def _(_) -> None:
            _refresh_current()

    if show_smplx is not None:
        @show_smplx.on_update
        def _(_) -> None:
            _refresh_current()

    if show_mesh is not None:
        @show_mesh.on_update
        def _(_) -> None:
            _refresh_current()
    if show_cameras is not None:
        @show_cameras.on_update
        def _(_) -> None:
            _refresh_current()
    if show_3dgs is not None:
        @show_3dgs.on_update
        def _(_) -> None:
            _refresh_current()

    print("Viser server is running. Use the slider to change frames (Ctrl+C to exit).")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main()
