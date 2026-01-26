from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tyro

import viser
import viser.transforms as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submodules.smplx import smplx  # noqa: E402


def _collect_frame_stems(root: Path, suffix: str) -> List[str]:
    if not root.exists():
        return []
    stems = [p.stem for p in root.glob(f"*{suffix}") if p.is_file()]
    stems = [s for s in stems if s.isdigit()]
    return sorted(stems, key=lambda s: int(s))


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        if not data.files:
            return None
        return {k: np.asarray(data[k]) for k in data.files}


def _pad_or_truncate(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
    current_dim = int(vec.shape[-1])
    if current_dim == target_dim:
        return vec
    if current_dim > target_dim:
        return vec[..., :target_dim]
    pad = torch.zeros((*vec.shape[:-1], target_dim - current_dim), device=vec.device, dtype=vec.dtype)
    return torch.cat([vec, pad], dim=-1)


def _reshape_pose(pose: torch.Tensor, joints: int) -> torch.Tensor:
    if pose.ndim == 3 and pose.shape[-2:] == (joints, 3):
        return pose[:1]
    if pose.ndim == 2 and pose.shape == (joints, 3):
        return pose.unsqueeze(0)
    flat = pose.reshape(-1)
    expected = joints * 3
    if flat.numel() != expected:
        raise ValueError(f"Expected pose with {expected} values, got {flat.numel()}")
    return flat.view(1, joints, 3)


def _reshape_body_pose_flat(pose: torch.Tensor, joints: int) -> torch.Tensor:
    if pose.ndim == 2 and pose.shape == (joints, 3):
        return pose.reshape(1, joints * 3)
    if pose.ndim == 3 and pose.shape[-2:] == (joints, 3):
        return pose.reshape(-1, joints * 3)
    if pose.ndim == 2 and pose.shape[1] == joints * 3:
        return pose[:1]
    flat = pose.reshape(-1)
    expected = joints * 3
    if flat.numel() != expected:
        raise ValueError(f"Expected body pose with {expected} values, got {flat.numel()}")
    return flat.view(1, expected)


def _reshape_vec3(vec: torch.Tensor) -> torch.Tensor:
    flat = vec.reshape(-1)
    if flat.numel() != 3:
        raise ValueError(f"Expected 3 values, got {flat.numel()}")
    return flat.view(1, 3)


def _extract_smpl_person_params(smpl_data: Dict[str, np.ndarray], pid: int) -> Optional[Dict[str, np.ndarray]]:
    required = ["betas", "global_orient", "body_pose"]
    if any(k not in smpl_data for k in required):
        return None
    transl_key = "transl" if "transl" in smpl_data else "trans" if "trans" in smpl_data else None
    if transl_key is None:
        return None
    try:
        params = {
            "betas": smpl_data["betas"][pid],
            "global_orient": smpl_data["global_orient"][pid],
            "body_pose": smpl_data["body_pose"][pid],
            "transl": smpl_data[transl_key][pid],
        }
    except IndexError:
        return None
    return params


def _extract_smplx_person_params(
    smplx_data: Dict[str, np.ndarray], pid: int
) -> Optional[Dict[str, np.ndarray]]:
    required = [
        "betas",
        "root_pose",
        "body_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "lhand_pose",
        "rhand_pose",
        "trans",
    ]
    if any(k not in smplx_data for k in required):
        return None
    try:
        params = {k: smplx_data[k][pid] for k in required}
    except IndexError:
        return None
    if "expression" in smplx_data:
        params["expression"] = smplx_data["expression"][pid]
    return params


def _smpl_vertices(layer, smpl_data: Dict[str, np.ndarray], device: torch.device) -> Optional[np.ndarray]:
    transl_key = "transl" if "transl" in smpl_data else "trans" if "trans" in smpl_data else None
    if transl_key is None or "betas" not in smpl_data:
        return None
    num_people = int(smpl_data[transl_key].shape[0])
    if num_people <= 0:
        return None

    expected_betas = int(getattr(layer, "num_betas", smpl_data["betas"].shape[-1]))

    verts_out = []
    for pid in range(num_people):
        params_np = _extract_smpl_person_params(smpl_data, pid)
        if params_np is None:
            continue
        params = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in params_np.items()}

        betas = _pad_or_truncate(params["betas"].view(1, -1), expected_betas)
        call_args = dict(
            global_orient=_reshape_vec3(params["global_orient"]),
            body_pose=_reshape_body_pose_flat(params["body_pose"], 23),
            betas=betas,
            transl=_reshape_vec3(params["transl"]),
        )
        with torch.no_grad():
            output = layer(**call_args)
        verts_out.append(output.vertices[0].detach().cpu().numpy())

    if not verts_out:
        return None
    return np.stack(verts_out, axis=0)


def _smplx_vertices(
    layer, smplx_data: Dict[str, np.ndarray], device: torch.device
) -> Optional[np.ndarray]:
    if "trans" not in smplx_data:
        return None
    num_people = int(smplx_data["trans"].shape[0])
    if num_people <= 0:
        return None

    expr_dim = int(getattr(layer, "num_expression_coeffs", 0))
    expected_betas = int(getattr(layer, "num_betas", smplx_data["betas"].shape[-1]))

    verts_out = []
    for pid in range(num_people):
        params_np = _extract_smplx_person_params(smplx_data, pid)
        if params_np is None:
            continue
        params = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in params_np.items()}

        betas = _pad_or_truncate(params["betas"].view(1, -1), expected_betas)
        call_args = dict(
            global_orient=_reshape_vec3(params["root_pose"]),
            body_pose=_reshape_pose(params["body_pose"], 21),
            jaw_pose=_reshape_vec3(params["jaw_pose"]),
            leye_pose=_reshape_vec3(params["leye_pose"]),
            reye_pose=_reshape_vec3(params["reye_pose"]),
            left_hand_pose=_reshape_pose(params["lhand_pose"], 15),
            right_hand_pose=_reshape_pose(params["rhand_pose"], 15),
            betas=betas,
            transl=_reshape_vec3(params["trans"]),
        )
        if expr_dim > 0:
            expr = params.get("expression")
            if expr is None:
                expr = torch.zeros((1, expr_dim), device=device, dtype=betas.dtype)
            else:
                expr = _pad_or_truncate(expr.view(1, -1), expr_dim)
            call_args["expression"] = expr

        with torch.no_grad():
            output = layer(**call_args)
        verts_out.append(output.vertices[0].detach().cpu().numpy())

    if not verts_out:
        return None
    return np.stack(verts_out, axis=0)


def _build_smpl_layer(model_folder: Path, gender: str, ext: str, device: torch.device):
    layer = smplx.create(
        str(model_folder),
        model_type="smpl",
        gender=gender,
        ext=ext,
        use_pca=False,
        use_face_contour=False,
        flat_hand_mean=True,
    )
    return layer.to(device)


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


def _color_for_person(base: np.ndarray, pid: int) -> Tuple[int, int, int]:
    scale = 0.6 + 0.4 * ((pid % 5) / 4.0)
    color = np.clip(base * scale, 0, 255)
    return tuple(int(c) for c in color)


def _update_bounds(
    verts: np.ndarray, bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if verts.size == 0:
        return bounds
    verts_flat = verts.reshape(-1, 3)
    vmin = verts_flat.min(axis=0)
    vmax = verts_flat.max(axis=0)
    min_bound, max_bound = bounds
    if min_bound is None:
        return vmin, vmax
    return np.minimum(min_bound, vmin), np.maximum(max_bound, vmax)


def _format_min_y_markdown(
    smpl_verts: Optional[np.ndarray], smplx_verts: Optional[np.ndarray]
) -> str:
    lines = ["**Min Y per person (world coords)**", ""]
    if smpl_verts is None:
        lines.append("- SMPL: n/a")
    else:
        for pid in range(smpl_verts.shape[0]):
            min_y = float(np.min(smpl_verts[pid][:, 1]))
            lines.append(f"- SMPL person {pid}: {min_y:.4f}")
    if smplx_verts is None:
        lines.append("- SMPL-X: n/a")
    else:
        for pid in range(smplx_verts.shape[0]):
            min_y = float(np.min(smplx_verts[pid][:, 1]))
            lines.append(f"- SMPL-X person {pid}: {min_y:.4f}")
    return "\n".join(lines)


@dataclass
class Args:
    scene_dir: Path
    port: int = 8080
    center_scene: bool = True
    is_minus_y_up: bool = True
    frame_index: int = 0
    frame_idx_range: Tuple[int, int] = (0, 10)
    device: str = "cuda"
    model_folder: Path = Path("/home/cizinsky/body_models")
    smpl_model_ext: str = "pkl"
    smplx_model_ext: str = "npz"
    gender: str = "neutral"
    smpl_pattern: str = "*.npz"
    smplx_pattern: str = "*.npz"
    mesh_opacity: float = 0.85


def main(args: Args) -> None:
    smpl_dir = args.scene_dir / "smpl"
    smplx_dir = args.scene_dir / "smplx"

    if not smpl_dir.exists():
        raise FileNotFoundError(f"SMPL dir not found: {smpl_dir}")
    if not smplx_dir.exists():
        raise FileNotFoundError(f"SMPL-X dir not found: {smplx_dir}")

    smpl_frames = _collect_frame_stems(smpl_dir, ".npz")
    smplx_frames = _collect_frame_stems(smplx_dir, ".npz")
    if not smpl_frames:
        raise FileNotFoundError(f"No SMPL frames found in {smpl_dir}")
    if not smplx_frames:
        raise FileNotFoundError(f"No SMPL-X frames found in {smplx_dir}")

    frames = sorted(set(smpl_frames) & set(smplx_frames), key=lambda s: int(s))
    if not frames:
        raise FileNotFoundError("No common frame ids between SMPL and SMPL-X.")

    start_idx, end_idx = args.frame_idx_range
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

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    smpl_layer = _build_smpl_layer(args.model_folder, args.gender, args.smpl_model_ext, device)
    smplx_layer = _build_smplx_layer(args.model_folder, args.gender, args.smplx_model_ext, device)
    smpl_faces = np.asarray(smpl_layer.faces, dtype=np.int32)
    smplx_faces = np.asarray(smplx_layer.faces, dtype=np.int32)

    smpl_verts_per_frame: List[Optional[np.ndarray]] = []
    smplx_verts_per_frame: List[Optional[np.ndarray]] = []

    bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
    for frame_id in frames:
        smpl_data = _load_npz(smpl_dir / f"{frame_id}.npz")
        smplx_data = _load_npz(smplx_dir / f"{frame_id}.npz")
        if smpl_data is None or smplx_data is None:
            raise FileNotFoundError(f"Missing SMPL/SMPL-X params for frame {frame_id}")

        smpl_verts = _smpl_vertices(smpl_layer, smpl_data, device)
        smplx_verts = _smplx_vertices(smplx_layer, smplx_data, device)

        smpl_verts_per_frame.append(smpl_verts)
        smplx_verts_per_frame.append(smplx_verts)

        if smpl_verts is not None:
            bounds = _update_bounds(smpl_verts, bounds)
        if smplx_verts is not None:
            bounds = _update_bounds(smplx_verts, bounds)

    center_offset = np.zeros(3, dtype=np.float32)
    if args.center_scene and bounds[0] is not None and bounds[1] is not None:
        center_offset = (bounds[0] + bounds[1]) * 0.5

    server = viser.ViserServer(port=args.port)
    angle = -np.pi / 2 if args.is_minus_y_up else np.pi / 2
    R_fix = tf.SO3.from_x_radians(angle)
    server.scene.add_frame(
        "/scene",
        show_axes=False,
        wxyz=tuple(R_fix.wxyz),
        position=tuple((-R_fix.apply(center_offset)).tolist()),
    )

    smpl_root = server.scene.add_frame("/scene/smpl", show_axes=False)
    smplx_root = server.scene.add_frame("/scene/smplx", show_axes=False)

    smpl_base = np.array([255, 140, 70], dtype=np.float32)
    smplx_base = np.array([70, 130, 255], dtype=np.float32)

    smpl_nodes: List[viser.FrameHandle] = []
    smplx_nodes: List[viser.FrameHandle] = []

    for idx, frame_id in enumerate(frames):
        smpl_node = server.scene.add_frame(f"/scene/smpl/f_{frame_id}", show_axes=False)
        smplx_node = server.scene.add_frame(f"/scene/smplx/f_{frame_id}", show_axes=False)
        smpl_nodes.append(smpl_node)
        smplx_nodes.append(smplx_node)

        smpl_verts = smpl_verts_per_frame[idx]
        smplx_verts = smplx_verts_per_frame[idx]

        if smpl_verts is not None:
            for pid in range(smpl_verts.shape[0]):
                color = _color_for_person(smpl_base, pid)
                handle = server.scene.add_mesh_simple(
                    f"/scene/smpl/f_{frame_id}/person_{pid}",
                    vertices=smpl_verts[pid],
                    faces=smpl_faces,
                    color=color,
                )
                if hasattr(handle, "opacity"):
                    handle.opacity = float(args.mesh_opacity)

        if smplx_verts is not None:
            for pid in range(smplx_verts.shape[0]):
                color = _color_for_person(smplx_base, pid)
                handle = server.scene.add_mesh_simple(
                    f"/scene/smplx/f_{frame_id}/person_{pid}",
                    vertices=smplx_verts[pid],
                    faces=smplx_faces,
                    color=color,
                )
                if hasattr(handle, "opacity"):
                    handle.opacity = float(args.mesh_opacity)

        smpl_node.visible = False
        smplx_node.visible = False

    with server.gui.add_folder("Visibility"):
        show_smpl = server.gui.add_checkbox("Show SMPL", True)
        show_smplx = server.gui.add_checkbox("Show SMPL-X", True)

    with server.gui.add_folder("Frames"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=len(frames) - 1,
            step=1,
            initial_value=min(max(args.frame_index, 0), len(frames) - 1),
        )
        frame_label = server.gui.add_text("File", frames[int(frame_slider.value)])
        min_y_markdown = server.gui.add_markdown(
            _format_min_y_markdown(
                smpl_verts_per_frame[int(frame_slider.value)],
                smplx_verts_per_frame[int(frame_slider.value)],
            )
        )

    current_idx = int(frame_slider.value)

    def _apply_visibility(frame_idx: int) -> None:
        nonlocal current_idx
        if frame_idx == current_idx:
            return
        smpl_nodes[current_idx].visible = False
        smplx_nodes[current_idx].visible = False

        current_idx = frame_idx
        frame_label.value = frames[frame_idx]
        min_y_markdown.content = _format_min_y_markdown(
            smpl_verts_per_frame[frame_idx],
            smplx_verts_per_frame[frame_idx],
        )

        smpl_nodes[frame_idx].visible = show_smpl.value
        smplx_nodes[frame_idx].visible = show_smplx.value

    def _refresh_current() -> None:
        smpl_nodes[current_idx].visible = show_smpl.value
        smplx_nodes[current_idx].visible = show_smplx.value

    smpl_nodes[current_idx].visible = show_smpl.value
    smplx_nodes[current_idx].visible = show_smplx.value

    @frame_slider.on_update
    def _(_event=None) -> None:
        _apply_visibility(int(frame_slider.value))

    @show_smpl.on_update
    def _(_event=None) -> None:
        _refresh_current()

    @show_smplx.on_update
    def _(_event=None) -> None:
        _refresh_current()

    print("Viser server running. Use the slider to change frames (Ctrl+C to exit).")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main(tyro.cli(Args))
