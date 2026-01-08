from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import tyro
from pytorch3d.transforms import quaternion_to_matrix

import viser
import viser.transforms as tf

def _sorted_frame_files(root: Path, pattern: str) -> List[Path]:
    paths = list(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {root}")

    def _key(p: Path) -> Tuple[int, str]:
        stem = p.stem
        if stem.isdigit():
            return (0, f"{int(stem):012d}")
        return (1, stem)

    return sorted(paths, key=_key)


def _axis_vector(axis: Literal["x", "y", "z", "-x", "-y", "-z"]) -> np.ndarray:
    base = {"x": 0, "y": 1, "z": 2}[axis.lstrip("-")]
    v = np.zeros(3, dtype=np.float32)
    v[base] = -1.0 if axis.startswith("-") else 1.0
    return v


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _look_at_wxyz(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Returns camera orientation (wxyz) for a camera-to-world rotation where +Z points forward.
    """
    forward = _normalize(target - position)
    if np.linalg.norm(forward) < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = _normalize(np.cross(up, forward))
    if np.linalg.norm(right) < 1e-8:
        # If up is parallel to forward, pick an arbitrary orthogonal up.
        tmp_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(tmp_up, forward))) > 0.9:
            tmp_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = _normalize(np.cross(tmp_up, forward))
    true_up = np.cross(forward, right)
    R_c2w = np.stack([right, true_up, forward], axis=1)  # columns are camera axes in world coords
    return tf.SO3.from_matrix(R_c2w).wxyz.astype(np.float32)


def _state_to_splat_arrays(
    state: Dict[str, Any],
    *,
    center: bool,
    max_scale: Optional[float],
    max_gaussians: Optional[int],
    seed: int,
) -> Dict[str, np.ndarray]:
    xyz = state["xyz"].float()
    opacity = state["opacity"].float()
    rotation = state["rotation"].float()
    scaling = state["scaling"].float()
    shs = state["shs"].float()

    if center:
        xyz = xyz - xyz.mean(dim=0, keepdim=True)

    opacity = opacity.squeeze(-1)
    opacity = opacity.clamp(0.0, 1.0)
    opacity = opacity.unsqueeze(-1)  # [N, 1]

    rgb_coeff = shs.squeeze(1)
    rgb = rgb_coeff.clamp(0.0, 1.0)

    rotation = torch.nn.functional.normalize(rotation, dim=-1)

    scales = scaling.clamp(min=1e-8)
    if max_scale is not None:
        scales = scales.clamp(max=max_scale)

    if max_gaussians is not None and xyz.shape[0] > max_gaussians:
        g = torch.Generator(device=xyz.device)
        g.manual_seed(seed)
        idx = torch.randperm(xyz.shape[0], generator=g)[:max_gaussians]
        xyz = xyz[idx]
        opacity = opacity[idx]
        rotation = rotation[idx]
        scales = scales[idx]
        rgb = rgb[idx]

    R = quaternion_to_matrix(rotation)  # [N, 3, 3]
    cov = R @ torch.diag_embed(scales**2) @ R.transpose(-1, -2)  # [N, 3, 3]

    return {
        "centers": xyz.detach().cpu().numpy().astype(np.float32),
        "opacities": opacity.detach().cpu().numpy().astype(np.float32),
        "rgbs": rgb.detach().cpu().numpy().astype(np.float32),
        "covariances": cov.detach().cpu().numpy().astype(np.float32),
    }

def _person_color(pid: int) -> Tuple[int, int, int]:
    palette = np.array(
        [
            [255, 80, 80],
            [80, 180, 255],
            [120, 255, 120],
            [255, 200, 80],
            [200, 120, 255],
            [255, 120, 200],
        ],
        dtype=np.int32,
    )
    c = palette[pid % len(palette)]
    return int(c[0]), int(c[1]), int(c[2])


def _load_mesh_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    npz = np.load(path)
    verts = npz["vertices"].astype(np.float32)
    faces = npz["faces"].astype(np.int32)
    return verts, faces


def _find_mesh_paths(mesh_root: Path, frame_name: str) -> List[Tuple[int, Path]]:
    if mesh_root.name == "instance":
        instance_root = mesh_root
    else:
        instance_root = mesh_root / "instance"
    if not instance_root.exists():
        return []
    paths: List[Tuple[int, Path]] = []
    if frame_name.isdigit():
        mesh_filename = f"mesh-f{int(frame_name):05d}.npz"
    else:
        mesh_filename = f"mesh-f{frame_name}.npz"
    for inst_dir in sorted(instance_root.iterdir()):
        if not inst_dir.is_dir() or not inst_dir.name.isdigit():
            continue
        pid = int(inst_dir.name)
        mesh_path = inst_dir / mesh_filename
        if mesh_path.exists():
            paths.append((pid, mesh_path))
    return paths


@dataclass
class Args:
    posed_3dgs_dir: Path
    posed_meshes_dir: Optional[Path] = None
    gt_meshes_dir: Optional[Path] = None
    pattern: str = "*.pt"
    port: int = 8080
    center: bool = False
    max_scale: Optional[float] = None
    max_gaussians: Optional[int] = None
    seed: int = 0
    debounce_ms: float = 10.0
    init_view_axis: Literal["x", "y", "z", "-x", "-y", "-z"] = "x"
    init_up_axis: Literal["x", "y", "z", "-x", "-y", "-z"] = "-y"
    init_distance_scale: float = 2.5
    init_fov_deg: float = 55.0
    mesh_opacity: float = 0.5
    gt_mesh_opacity: float = 0.7


def main(args: Args) -> None:


    frame_files = _sorted_frame_files(args.posed_3dgs_dir, args.pattern)
    num_frames = len(frame_files)

    def _torch_load(path: Path) -> Dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    initial_camera: Dict[str, np.ndarray] = {}
    first_state = _torch_load(frame_files[0])
    first_splat_data = _state_to_splat_arrays(
        first_state,
        center=args.center,
        max_scale=args.max_scale,
        max_gaussians=args.max_gaussians,
        seed=args.seed,
    )
    centers0 = first_splat_data["centers"]
    cmin0 = centers0.min(axis=0)
    cmax0 = centers0.max(axis=0)
    target0 = (cmin0 + cmax0) * 0.5
    radius0 = float(np.linalg.norm(cmax0 - cmin0)) * 0.5
    radius0 = max(radius0, 1e-3)
    forward_axis0 = _axis_vector(args.init_view_axis)
    up_axis0 = _axis_vector(args.init_up_axis)
    position0 = target0 - forward_axis0 * (radius0 * float(args.init_distance_scale))
    wxyz0 = _look_at_wxyz(position0, target0, up_axis0)
    initial_camera["position"] = position0.astype(np.float32)
    initial_camera["wxyz"] = wxyz0.astype(np.float32)
    initial_camera["fov"] = np.array([np.deg2rad(float(args.init_fov_deg))], dtype=np.float32)

    show_meshes = args.posed_meshes_dir is not None
    show_gt_meshes = args.gt_meshes_dir is not None
    show_3dgs = True

    server = viser.ViserServer(port=args.port)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )
        play_button = server.gui.add_button("Play")
        stop_button = server.gui.add_button("Stop", disabled=True)
        fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=0.5, initial_value=10)
    with server.gui.add_folder("Visibility"):
        show_3dgs_checkbox = server.gui.add_checkbox("Show 3DGS", True)
        show_meshes_checkbox = server.gui.add_checkbox("Show Meshes", show_meshes)
        show_gt_meshes_checkbox = server.gui.add_checkbox("Show GT Meshes", show_gt_meshes)
        if not show_meshes:
            show_meshes_checkbox.disabled = True
        if not show_gt_meshes:
            show_gt_meshes_checkbox.disabled = True
    frame_label = server.gui.add_text("File", frame_files[0].name)

    gs_handle: Optional[Any] = None
    last_frame_idx: Optional[int] = None
    pending_frame_idx: Optional[int] = None
    pending_since: Optional[float] = None
    ignore_slider_update: bool = False
    playing: bool = False
    last_play_step: float = time.monotonic()
    mesh_handles: Dict[int, Any] = {}
    gt_mesh_handles: Dict[int, Any] = {}

    def _set_initial_camera(client: viser.ClientHandle) -> None:
        if not initial_camera:
            return
        client.camera.position = initial_camera["position"]
        client.camera.wxyz = initial_camera["wxyz"]
        client.camera.fov = float(initial_camera["fov"][0])

    server.on_client_connect(_set_initial_camera)

    @show_3dgs_checkbox.on_update
    def _(_) -> None:
        nonlocal show_3dgs
        show_3dgs = bool(show_3dgs_checkbox.value)
        if gs_handle is not None:
            gs_handle.visible = show_3dgs

    @show_meshes_checkbox.on_update
    def _(_) -> None:
        nonlocal last_frame_idx
        if not show_meshes:
            return
        if not show_meshes_checkbox.value:
            for handle in mesh_handles.values():
                handle.visible = False
            return
        last_frame_idx = None
        _show_frame(int(frame_slider.value))

    @show_gt_meshes_checkbox.on_update
    def _(_) -> None:
        nonlocal last_frame_idx
        if not show_gt_meshes:
            return
        if not show_gt_meshes_checkbox.value:
            for handle in gt_mesh_handles.values():
                handle.visible = False
            return
        last_frame_idx = None
        _show_frame(int(frame_slider.value))

    @play_button.on_click
    def _(_) -> None:
        nonlocal playing, last_play_step
        playing = True
        last_play_step = time.monotonic()
        play_button.disabled = True
        stop_button.disabled = False

    @stop_button.on_click
    def _(_) -> None:
        nonlocal playing
        playing = False
        play_button.disabled = False
        stop_button.disabled = True

    def _show_frame(frame_idx: int) -> None:
        nonlocal gs_handle, last_frame_idx, mesh_handles, gt_mesh_handles
        if last_frame_idx == frame_idx:
            return

        frame_path = frame_files[frame_idx]
        state = first_state if frame_idx == 0 else _torch_load(frame_path)
        splat_data = _state_to_splat_arrays(
            state,
            center=args.center,
            max_scale=args.max_scale,
            max_gaussians=args.max_gaussians,
            seed=args.seed,
        )
        print(f"Showing frame {frame_idx}/{num_frames - 1}: {frame_path} with {splat_data['centers'].shape[0]} gaussians")

        with server.atomic():
            frame_label.value = frame_path.name
            if gs_handle is not None and hasattr(gs_handle, "update"):
                gs_handle.update(
                    centers=splat_data["centers"],
                    rgbs=splat_data["rgbs"],
                    opacities=splat_data["opacities"],
                    covariances=splat_data["covariances"],
                )
            else:
                if gs_handle is not None and hasattr(gs_handle, "remove"):
                    gs_handle.remove()
                gs_handle = server.scene.add_gaussian_splats(
                    "/gaussian_splats",
                    centers=splat_data["centers"],
                    rgbs=splat_data["rgbs"],
                    opacities=splat_data["opacities"],
                    covariances=splat_data["covariances"],
                )
            gs_handle.visible = show_3dgs

            if show_meshes and args.posed_meshes_dir is not None and show_meshes_checkbox.value:
                mesh_paths = _find_mesh_paths(args.posed_meshes_dir, frame_path.stem)
                active_pids = set()
                for pid, mesh_path in mesh_paths:
                    verts, faces = _load_mesh_npz(mesh_path)
                    active_pids.add(pid)
                    handle = mesh_handles.get(pid)
                    if handle is not None and hasattr(handle, "update"):
                        handle.update(vertices=verts, faces=faces, color=_person_color(pid))
                    else:
                        handle = server.scene.add_mesh_simple(
                            f"/meshes/person_{pid}",
                            vertices=verts,
                            faces=faces,
                            color=_person_color(pid),
                        )
                        mesh_handles[pid] = handle
                    handle.visible = True
                    if hasattr(handle, "opacity"):
                        handle.opacity = float(args.mesh_opacity)

                for pid, handle in mesh_handles.items():
                    if pid not in active_pids:
                        handle.visible = False
            elif show_meshes:
                for handle in mesh_handles.values():
                    handle.visible = False

            if show_gt_meshes and args.gt_meshes_dir is not None and show_gt_meshes_checkbox.value:
                mesh_paths = _find_mesh_paths(args.gt_meshes_dir, frame_path.stem)
                active_pids = set()
                for pid, mesh_path in mesh_paths:
                    verts, faces = _load_mesh_npz(mesh_path)
                    active_pids.add(pid)
                    handle = gt_mesh_handles.get(pid)
                    if handle is not None and hasattr(handle, "update"):
                        handle.update(vertices=verts, faces=faces, color=_person_color(pid))
                    else:
                        handle = server.scene.add_mesh_simple(
                            f"/gt_meshes/person_{pid}",
                            vertices=verts,
                            faces=faces,
                            color=_person_color(pid),
                        )
                        gt_mesh_handles[pid] = handle
                    handle.visible = True
                    if hasattr(handle, "opacity"):
                        handle.opacity = float(args.gt_mesh_opacity)

                for pid, handle in gt_mesh_handles.items():
                    if pid not in active_pids:
                        handle.visible = False
            elif show_gt_meshes:
                for handle in gt_mesh_handles.values():
                    handle.visible = False
            last_frame_idx = frame_idx

        server.flush()

    @frame_slider.on_update
    def _(_) -> None:
        nonlocal pending_frame_idx, pending_since, ignore_slider_update, last_play_step
        if ignore_slider_update:
            return
        pending_frame_idx = int(frame_slider.value)
        pending_since = time.monotonic()
        last_play_step = time.monotonic()

    _show_frame(0)

    try:
        while True:
            now = time.monotonic()

            # Debounced manual slider update.
            if pending_frame_idx is not None and pending_since is not None:
                if now - pending_since >= max(0.0, args.debounce_ms / 1000.0):
                    idx = pending_frame_idx
                    pending_frame_idx = None
                    pending_since = None
                    _show_frame(idx)
                    last_play_step = now

            # Auto-play loop.
            if playing:
                fps = float(fps_slider.value)
                step_s = 1.0 / max(fps, 1e-6)
                if now - last_play_step >= step_s:
                    next_idx = 0 if last_frame_idx is None else (last_frame_idx + 1) % num_frames
                    ignore_slider_update = True
                    frame_slider.value = next_idx
                    ignore_slider_update = False
                    pending_frame_idx = None
                    pending_since = None
                    _show_frame(next_idx)
                    last_play_step = now

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main(tyro.cli(Args))
