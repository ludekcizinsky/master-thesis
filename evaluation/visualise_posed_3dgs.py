from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

def _state_to_splat_arrays(
    state: Dict[str, Any],
    *,
    max_scale: Optional[float],
    max_gaussians: Optional[int],
    seed: int,
) -> Dict[str, np.ndarray]:
    xyz = state["xyz"].float()
    opacity = state["opacity"].float()
    rotation = state["rotation"].float()
    scaling = state["scaling"].float()
    shs = state["shs"].float()

    opacity = opacity.squeeze(-1)
    opacity = opacity.clamp(0.0, 1.0)
    opacity = opacity.unsqueeze(-1)

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

    R = quaternion_to_matrix(rotation)
    cov = R @ torch.diag_embed(scales**2) @ R.transpose(-1, -2)

    return {
        "centers": xyz.detach().cpu().numpy().astype(np.float32),
        "opacities": opacity.detach().cpu().numpy().astype(np.float32),
        "rgbs": rgb.detach().cpu().numpy().astype(np.float32),
        "covariances": cov.detach().cpu().numpy().astype(np.float32),
    }


def _select_frame(frame_files: List[Path], frame_index: int, frame_name: Optional[str]) -> Path:
    if frame_name:
        if "." in frame_name:
            matches = [p for p in frame_files if p.name == frame_name]
        else:
            matches = [p for p in frame_files if p.stem == frame_name]
        if not matches:
            raise FileNotFoundError(f"No frame named {frame_name!r} found in {frame_files[0].parent}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple frames matched {frame_name!r}; please disambiguate.")
        return matches[0]
    if frame_index < 0 or frame_index >= len(frame_files):
        raise IndexError(f"frame_index {frame_index} is out of range [0, {len(frame_files) - 1}]")
    return frame_files[frame_index]


def _torch_load(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)

def _center_offset_from_centers(centers: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.zeros(3, dtype=np.float32)
    cmin = centers.min(axis=0)
    cmax = centers.max(axis=0)
    return (cmin + cmax) * 0.5

def main(args: Args) -> None:
    frame_files = _sorted_frame_files(args.posed_3dgs_dir, args.pattern)
    frame_path = _select_frame(frame_files, args.frame_index, args.frame_name)

    state = _torch_load(frame_path)
    raw_xyz = state.get("xyz")
    if isinstance(raw_xyz, torch.Tensor):
        raw_xyz = raw_xyz.detach().cpu().numpy()
    raw_xyz = np.asarray(raw_xyz, dtype=np.float32)
    splat_data = _state_to_splat_arrays(
        state,
        max_scale=args.max_scale,
        max_gaussians=args.max_gaussians,
        seed=args.seed,
    )

    server = viser.ViserServer(port=args.port)
    R_fix = tf.SO3.from_x_radians(-np.pi / 2)
    center_offset = (
        _center_offset_from_centers(raw_xyz)
        if args.center_scene
        else np.zeros(3, dtype=np.float32)
    )
    server.scene.add_frame(
        "/scene/3dgs",
        show_axes=False,
        wxyz=tuple(R_fix.wxyz),
        position=tuple((-R_fix.apply(center_offset)).tolist()),
    )
    server.scene.add_gaussian_splats(
        "/scene/3dgs/gaussian_splats",
        centers=splat_data["centers"],
        rgbs=splat_data["rgbs"],
        opacities=splat_data["opacities"],
        covariances=splat_data["covariances"],
    )

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")

@dataclass
class Args:
    posed_3dgs_dir: Path
    pattern: str = "*.pt"
    frame_index: int = 0
    frame_name: Optional[str] = None
    port: int = 8080
    center_scene: bool = True
    max_scale: Optional[float] = None
    max_gaussians: Optional[int] = None
    seed: int = 0


if __name__ == "__main__":
    main(tyro.cli(Args))
