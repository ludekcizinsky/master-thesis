from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import viser
import viser.transforms as tf
from tqdm import tqdm
import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from submodules.smplx import smplx


@dataclass
class Config:
    scene_dir: Annotated[Path, tyro.conf.arg(aliases=["--scenes-dir"])]
    device: str = "cuda"
    model_folder: Path = Path("/home/cizinsky/body_models")
    smpl_model_ext: str = "pkl"
    smplx_model_ext: str = "npz"
    gender: str = "neutral"
    port: int = 8080
    center_scene: bool = True
    max_frames: int = 10

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

def _center_offset_from_vertices(verts: np.ndarray) -> np.ndarray:
    if verts.ndim == 3:
        verts_flat = verts.reshape(-1, 3)
    else:
        verts_flat = verts
    vmin = verts_flat.min(axis=0)
    vmax = verts_flat.max(axis=0)
    return (vmin + vmax) * 0.5


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


def _smpl_vertices(
    layer,
    params: Dict[str, np.ndarray],
    device: torch.device,
) -> Optional[np.ndarray]:
    required = {"betas", "global_orient", "body_pose", "transl"}
    if not required.issubset(params.keys()):
        return None
    betas = torch.tensor(params["betas"], dtype=torch.float32, device=device)
    global_orient = torch.tensor(params["global_orient"], dtype=torch.float32, device=device)
    body_pose = torch.tensor(params["body_pose"], dtype=torch.float32, device=device)
    transl = torch.tensor(params["transl"], dtype=torch.float32, device=device)

    if betas.ndim == 1:
        betas = betas[None, :]
    if global_orient.ndim == 1:
        global_orient = global_orient[None, :]
    if body_pose.ndim == 2:
        body_pose = body_pose[None, :, :]
    if transl.ndim == 1:
        transl = transl[None, :]
    if global_orient.ndim == 2 and body_pose.ndim == 3:
        global_orient = global_orient[:, None, :]
    if body_pose.ndim == 2 and global_orient.ndim == 3:
        body_pose = body_pose[:, None, :]

    expected_betas = int(getattr(layer, "num_betas", betas.shape[-1]))
    betas = _pad_or_truncate(betas, expected_betas)

    with torch.no_grad():
        output = layer(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            transl=transl,
        )
    return output.vertices.detach().cpu().numpy()


def _smplx_vertices(
    layer,
    params: Dict[str, np.ndarray],
    device: torch.device,
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
    return output.vertices.detach().cpu().numpy()


def main() -> None:
    cfg = tyro.cli(Config)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")
    seq_name = cfg.scene_dir.name

    # Resolve modality directories.
    smpl_dir = cfg.scene_dir / "smpl"
    smplx_dir = cfg.scene_dir / "smplx"
    mesh_dir = cfg.scene_dir / "meshes"

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

    frames = frames[: cfg.max_frames]
    if not frames:
        raise FileNotFoundError("No frames left after applying max_frames.")

    # Build body model layers (only if the modality exists).
    smpl_layer = _build_layer(cfg.model_folder, "smpl", cfg.gender, cfg.smpl_model_ext, device) if smpl_frames else None
    smplx_layer = _build_layer(cfg.model_folder, "smplx", cfg.gender, cfg.smplx_model_ext, device) if smplx_frames else None

    smpl_faces = np.asarray(smpl_layer.faces, dtype=np.int32) if smpl_layer is not None else None
    smplx_faces = np.asarray(smplx_layer.faces, dtype=np.int32) if smplx_layer is not None else None

    # Compute a shared scene offset for visual centering.
    center_offset = (
        _compute_center_offset(frames, smpl_dir, smplx_dir, mesh_dir, smpl_layer, smplx_layer, device)
        if cfg.center_scene
        else np.zeros(3, dtype=np.float32)
    )

    # Compute a fixed rotation to align the scene upright.
    is_y_up = 1 if "hi4d" in seq_name.lower() else -1
    R_fix = tf.SO3.from_x_radians(is_y_up*np.pi / 2)

    # Create the Viser server and a centered root frame.
    server = viser.ViserServer(port=cfg.port)

    server.scene.add_frame(
        "/scene",
        show_axes=True,
    )
    smpl_root = (
        server.scene.add_frame(
            "/scene/smpl", 
            show_axes=False, 
            wxyz=tuple(R_fix.wxyz),
            position=tuple((-R_fix.apply(center_offset)).tolist()),
        ) 
        if smpl_layer is not None else None
    )
    smplx_root = (
        server.scene.add_frame(
            "/scene/smplx", 
            show_axes=False, 
            wxyz=tuple(R_fix.wxyz),
            position=tuple((-R_fix.apply(center_offset)).tolist()),
        ) if smplx_layer is not None else None
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

    # GUI controls.
    show_smpl = None
    show_smplx = None
    show_mesh = None

    with server.gui.add_folder("Visibility"):
        if smpl_root is not None:
            show_smpl = server.gui.add_checkbox("Show SMPL", True)
        if smplx_root is not None:
            show_smplx = server.gui.add_checkbox("Show SMPL-X", True)
        if mesh_root is not None:
            show_mesh = server.gui.add_checkbox("Show Mesh", True)

    with server.gui.add_folder("Frames"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(frames) - 1,
            step=1,
            initial_value=0,
        )
        frame_label = server.gui.add_text("File", frames[0])

    # Colors for different modalities.
    smpl_base = np.array([255, 140, 70], dtype=np.float32)
    smplx_base = np.array([70, 130, 255], dtype=np.float32)
    mesh_color = (200, 200, 200)

    # Preload meshes for each modality under per-frame nodes.
    smpl_nodes: List[viser.FrameHandle] = []
    smplx_nodes: List[viser.FrameHandle] = []
    mesh_nodes: List[viser.FrameHandle] = []

    if smpl_root is not None:
        for frame_id in tqdm(frames, desc="Loading SMPL"):
            node = server.scene.add_frame(f"/scene/smpl/f_{frame_id}", show_axes=False)
            smpl_nodes.append(node)
            smpl_data = _load_npz(smpl_dir / f"{frame_id}.npz")
            if smpl_data is not None:
                verts = _smpl_vertices(smpl_layer, smpl_data, device)
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
                verts = _smplx_vertices(smplx_layer, smplx_data, device)
                if verts is not None:
                    for pid in range(verts.shape[0]):
                        color = _color_for_person(smplx_base, pid)
                        server.scene.add_mesh_simple(
                            f"/scene/smplx/f_{frame_id}/person_{pid}",
                            vertices=verts[pid],
                            faces=smplx_faces,
                            color=color,
                        )
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

        current_idx = frame_idx
        frame_label.value = frames[frame_idx]

        if smpl_nodes and show_smpl is not None:
            smpl_nodes[frame_idx].visible = show_smpl.value
        if smplx_nodes and show_smplx is not None:
            smplx_nodes[frame_idx].visible = show_smplx.value
        if mesh_nodes and show_mesh is not None:
            mesh_nodes[frame_idx].visible = show_mesh.value

    # Refresh visibility for the current frame when toggles change.
    def _refresh_current() -> None:
        if smpl_nodes and show_smpl is not None:
            smpl_nodes[current_idx].visible = show_smpl.value
        if smplx_nodes and show_smplx is not None:
            smplx_nodes[current_idx].visible = show_smplx.value
        if mesh_nodes and show_mesh is not None:
            mesh_nodes[current_idx].visible = show_mesh.value

    if smpl_nodes:
        smpl_nodes[0].visible = True if show_smpl is None else show_smpl.value
    if smplx_nodes:
        smplx_nodes[0].visible = True if show_smplx is None else show_smplx.value
    if mesh_nodes:
        mesh_nodes[0].visible = True if show_mesh is None else show_mesh.value

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

    print("Viser server is running. Use the slider to change frames (Ctrl+C to exit).")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main()
