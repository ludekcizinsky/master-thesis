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


def _maybe_sigmoid_to_unit(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if x.min().item() < 0.0 or x.max().item() > 1.0:
        return torch.sigmoid(x)
    return x.clamp(0.0, 1.0)


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
    scaling_mode: Literal["auto", "log", "linear"],
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

    if opacity.ndim == 2 and opacity.shape[-1] == 1:
        opacity = opacity.squeeze(-1)
    opacity = _maybe_sigmoid_to_unit(opacity)
    opacity = opacity.unsqueeze(-1)  # [N, 1]

    if shs.ndim == 3:
        rgb = shs[:, 0, :] if shs.shape[1] > 1 else shs.squeeze(1)
    else:
        rgb = shs
    rgb = _maybe_sigmoid_to_unit(rgb)

    if rotation.shape[-1] != 4:
        raise ValueError(f"Expected rotation quaternions of shape [N,4], got {tuple(rotation.shape)}")
    rotation = torch.nn.functional.normalize(rotation, dim=-1)

    if scaling_mode == "auto":
        scaling_mode = "log" if scaling.min().item() < 0.0 else "linear"

    scales = torch.exp(scaling) if scaling_mode == "log" else scaling
    if scales.ndim == 2 and scales.shape[-1] == 3:
        scales = scales.clamp(min=1e-8)
        if max_scale is not None:
            scales = scales.clamp(max=max_scale)
    else:
        raise ValueError(f"Expected scaling of shape [N,3], got {tuple(scaling.shape)}")

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


@dataclass
class Args:
    posed_3dgs_dir: Path
    pattern: str = "*.pt"
    port: int = 8080
    scaling_mode: Literal["auto", "log", "linear"] = "auto"
    center: bool = False
    max_scale: Optional[float] = None
    max_gaussians: Optional[int] = None
    seed: int = 0
    debounce_ms: float = 150.0
    init_view_axis: Literal["x", "y", "z", "-x", "-y", "-z"] = "x"
    init_up_axis: Literal["x", "y", "z", "-x", "-y", "-z"] = "-y"
    init_distance_scale: float = 2.5
    init_fov_deg: float = 55.0


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
        scaling_mode=args.scaling_mode,
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

    server = viser.ViserServer(port=args.port)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )
    frame_label = server.gui.add_text("File", frame_files[0].name)

    gs_handle: Optional[Any] = None
    last_frame_idx: Optional[int] = None
    pending_frame_idx: Optional[int] = None
    pending_since: Optional[float] = None

    def _set_initial_camera(client: viser.ClientHandle) -> None:
        if not initial_camera:
            return
        client.camera.position = initial_camera["position"]
        client.camera.wxyz = initial_camera["wxyz"]
        client.camera.fov = float(initial_camera["fov"][0])

    server.on_client_connect(_set_initial_camera)

    def _show_frame(frame_idx: int) -> None:
        nonlocal gs_handle, last_frame_idx
        if last_frame_idx == frame_idx:
            return

        frame_path = frame_files[frame_idx]
        state = first_state if frame_idx == 0 else _torch_load(frame_path)
        splat_data = _state_to_splat_arrays(
            state,
            scaling_mode=args.scaling_mode,
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
            last_frame_idx = frame_idx

        server.flush()

    @frame_slider.on_update
    def _(_) -> None:
        nonlocal pending_frame_idx, pending_since
        pending_frame_idx = int(frame_slider.value)
        pending_since = time.monotonic()

    _show_frame(0)

    debounce_s = max(0.0, args.debounce_ms / 1000.0)
    try:
        while True:
            if pending_frame_idx is not None and pending_since is not None:
                if time.monotonic() - pending_since >= debounce_s:
                    idx = pending_frame_idx
                    pending_frame_idx = None
                    pending_since = None
                    _show_frame(idx)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main(tyro.cli(Args))
