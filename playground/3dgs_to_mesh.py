from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
import tyro
from skimage.measure import marching_cubes
from tqdm import tqdm


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


def _maybe_sigmoid_to_unit(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    if x.min() < 0.0 or x.max() > 1.0:
        return 1.0 / (1.0 + np.exp(-x))
    return np.clip(x, 0.0, 1.0)


def _torch_load(path: Path) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _split_person_ids(state: Dict[str, torch.Tensor]) -> np.ndarray:
    if "person_ids" in state:
        return state["person_ids"].detach().cpu().numpy().astype(np.int32)

    counts = state.get("person_gaussian_counts", None)
    if counts is not None:
        counts_np = counts.detach().cpu().numpy().astype(np.int32).tolist()
        ids = []
        for pid, c in enumerate(counts_np):
            ids.extend([pid] * int(c))
        return np.asarray(ids, dtype=np.int32)

    return np.zeros((state["xyz"].shape[0],), dtype=np.int32)


def _prepare_gaussians(
    state: Dict[str, torch.Tensor],
    mask: np.ndarray,
    *,
    scaling_mode: Literal["auto", "log", "linear"],
    sigma_scale: float,
    min_sigma: float,
    max_sigma: Optional[float],
    min_opacity: float,
    max_gaussians: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = state["xyz"].detach().cpu().numpy().astype(np.float32)[mask]
    scaling = state["scaling"].detach().cpu().numpy().astype(np.float32)[mask]
    opacity = state["opacity"].detach().cpu().numpy().astype(np.float32)[mask]

    if opacity.ndim == 2 and opacity.shape[-1] == 1:
        opacity = opacity[:, 0]
    opacity = _maybe_sigmoid_to_unit(opacity)

    if max_gaussians is not None and xyz.shape[0] > max_gaussians:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=max_gaussians, replace=False)
        xyz = xyz[idx]
        scaling = scaling[idx]
        opacity = opacity[idx]

    if scaling_mode == "auto":
        scaling_mode = "log" if scaling.min() < 0.0 else "linear"
    scales = np.exp(scaling) if scaling_mode == "log" else scaling
    if scales.ndim == 2 and scales.shape[1] == 3:
        sigma = scales.mean(axis=1)
    else:
        raise ValueError(f"Expected scaling shape [N,3], got {scaling.shape}")
    sigma = sigma * float(sigma_scale)
    sigma = np.clip(sigma, min_sigma, max_sigma if max_sigma is not None else np.inf)

    keep = opacity >= min_opacity
    return xyz[keep], sigma[keep], opacity[keep]


def _density_grid_from_gaussians(
    centers: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray,
    grid_size: Tuple[int, int, int],
    padding: float,
    truncation: float,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    if centers.shape[0] == 0:
        raise ValueError("No gaussians remaining after filtering.")

    min_corner = centers.min(axis=0) - padding
    max_corner = centers.max(axis=0) + padding

    nx, ny, nz = grid_size
    xs = np.linspace(min_corner[0], max_corner[0], nx, dtype=np.float32)
    ys = np.linspace(min_corner[1], max_corner[1], ny, dtype=np.float32)
    zs = np.linspace(min_corner[2], max_corner[2], nz, dtype=np.float32)
    dx = float(xs[1] - xs[0]) if nx > 1 else 1.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 1.0
    dz = float(zs[1] - zs[0]) if nz > 1 else 1.0

    density = np.zeros((nx, ny, nz), dtype=np.float32)

    for mu, sigma, w in zip(centers, sigmas, weights):
        radius = truncation * sigma
        if radius <= 0.0:
            continue
        ix0 = max(0, int(np.floor((mu[0] - radius - min_corner[0]) / dx)))
        ix1 = min(nx - 1, int(np.ceil((mu[0] + radius - min_corner[0]) / dx)))
        iy0 = max(0, int(np.floor((mu[1] - radius - min_corner[1]) / dy)))
        iy1 = min(ny - 1, int(np.ceil((mu[1] + radius - min_corner[1]) / dy)))
        iz0 = max(0, int(np.floor((mu[2] - radius - min_corner[2]) / dz)))
        iz1 = min(nz - 1, int(np.ceil((mu[2] + radius - min_corner[2]) / dz)))

        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            continue

        sigma2 = float(sigma * sigma)
        x = xs[ix0 : ix1 + 1] - mu[0]
        y = ys[iy0 : iy1 + 1] - mu[1]
        z = zs[iz0 : iz1 + 1] - mu[2]
        gx = np.exp(-(x * x) / (2.0 * sigma2)).astype(np.float32)
        gy = np.exp(-(y * y) / (2.0 * sigma2)).astype(np.float32)
        gz = np.exp(-(z * z) / (2.0 * sigma2)).astype(np.float32)

        density[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] += (
            w * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
        )

    return density, min_corner.astype(np.float32), (dx, dy, dz)


def _extract_mesh(
    density: np.ndarray,
    origin: np.ndarray,
    spacing: Tuple[float, float, float],
    iso_level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces, _, _ = marching_cubes(density, level=iso_level, spacing=spacing)
    verts = verts + origin[None, :]
    return verts.astype(np.float32), faces.astype(np.int32)


def _write_mesh_npz(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vertices=vertices, faces=faces)


@dataclass
class Args:
    posed_3dgs_dir: Path
    output_dir: Path
    pattern: str = "*.pt"
    grid_size: int = 96
    padding: float = 0.05
    truncation: float = 3.0
    iso_level: float = 0.05
    iso_percentile: Optional[float] = None
    scaling_mode: Literal["auto", "log", "linear"] = "auto"
    sigma_scale: float = 1.0
    min_sigma: float = 1e-4
    max_sigma: Optional[float] = None
    min_opacity: float = 0.01
    max_gaussians: Optional[int] = None
    seed: int = 0
    max_frames: Optional[int] = None
    overwrite: bool = False
    save_merged: bool = False


def _iter_frames(frame_files: Iterable[Path], max_frames: Optional[int]) -> Iterable[Path]:
    if max_frames is None:
        return frame_files
    return list(frame_files)[: max_frames]


def main(args: Args) -> None:
    frame_files = _sorted_frame_files(args.posed_3dgs_dir, args.pattern)
    frame_files = _iter_frames(frame_files, args.max_frames)
    grid_size = (args.grid_size, args.grid_size, args.grid_size)

    for frame_path in tqdm(frame_files, desc="3DGS -> mesh"):
        state = _torch_load(frame_path)
        person_ids = _split_person_ids(state)
        frame_name = frame_path.stem

        unique_persons = np.unique(person_ids)
        for pid in unique_persons:
            mask = person_ids == pid
            centers, sigmas, weights = _prepare_gaussians(
                state,
                mask,
                scaling_mode=args.scaling_mode,
                sigma_scale=args.sigma_scale,
                min_sigma=args.min_sigma,
                max_sigma=args.max_sigma,
                min_opacity=args.min_opacity,
                max_gaussians=args.max_gaussians,
                seed=args.seed,
            )
            if centers.shape[0] == 0:
                continue

            density, origin, spacing = _density_grid_from_gaussians(
                centers,
                sigmas,
                weights,
                grid_size=grid_size,
                padding=args.padding,
                truncation=args.truncation,
            )
            if args.iso_percentile is not None:
                positive = density[density > 0]
                if positive.size == 0:
                    continue
                iso_level = float(np.percentile(positive, args.iso_percentile))
            else:
                iso_level = args.iso_level

            try:
                vertices, faces = _extract_mesh(density, origin, spacing, iso_level)
            except ValueError:
                continue

            out_path = args.output_dir / "instance" / f"{pid}" / f"mesh-f{frame_name}.npz"
            if out_path.exists() and not args.overwrite:
                continue
            _write_mesh_npz(out_path, vertices, faces)

        if args.save_merged:
            centers, sigmas, weights = _prepare_gaussians(
                state,
                np.ones_like(person_ids, dtype=bool),
                scaling_mode=args.scaling_mode,
                sigma_scale=args.sigma_scale,
                min_sigma=args.min_sigma,
                max_sigma=args.max_sigma,
                min_opacity=args.min_opacity,
                max_gaussians=args.max_gaussians,
                seed=args.seed,
            )
            if centers.shape[0] == 0:
                continue
            density, origin, spacing = _density_grid_from_gaussians(
                centers,
                sigmas,
                weights,
                grid_size=grid_size,
                padding=args.padding,
                truncation=args.truncation,
            )
            if args.iso_percentile is not None:
                positive = density[density > 0]
                if positive.size == 0:
                    continue
                iso_level = float(np.percentile(positive, args.iso_percentile))
            else:
                iso_level = args.iso_level

            try:
                vertices, faces = _extract_mesh(density, origin, spacing, iso_level)
            except ValueError:
                continue

            out_path = args.output_dir / f"mesh-f{frame_name}.npz"
            if out_path.exists() and not args.overwrite:
                continue
            _write_mesh_npz(out_path, vertices, faces)


if __name__ == "__main__":
    main(tyro.cli(Args))
