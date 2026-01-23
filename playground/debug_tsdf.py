from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import List

import numpy as np
import open3d as o3d
import tyro
import viser
import viser.transforms as tf


@dataclass
class Args:
    tsdf_debug_dir: Path
    person_id: int = 0
    frame: str = "0"
    point_size: float = 0.01
    max_points: int = 200000
    host: str = "0.0.0.0"
    port: int = 8080


def _resolve_frame_dir(base_dir: Path, frame_arg: str) -> Path:
    frame_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir()])
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {base_dir}")

    direct = base_dir / frame_arg
    if direct.is_dir():
        return direct

    if frame_arg.isdigit():
        idx = int(frame_arg)
        if 0 <= idx < len(frame_dirs):
            return frame_dirs[idx]
        padded = f"{idx:06d}"
        padded_dir = base_dir / padded
        if padded_dir.is_dir():
            return padded_dir

    available = ", ".join([p.name for p in frame_dirs[:10]])
    raise FileNotFoundError(
        f"Frame '{frame_arg}' not found in {base_dir}. Available (first 10): {available}"
    )


def _load_point_cloud(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    return points


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(0)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _color_palette(n: int) -> List[np.ndarray]:
    base = np.array(
        [
            [231, 76, 60],
            [46, 204, 113],
            [52, 152, 219],
            [241, 196, 15],
            [155, 89, 182],
            [26, 188, 156],
            [230, 126, 34],
            [127, 140, 141],
        ],
        dtype=np.float32,
    )
    colors = []
    for i in range(n):
        colors.append(base[i % base.shape[0]] / 255.0)
    return colors


def main() -> None:
    args = tyro.cli(Args)
    base_dir = Path(args.tsdf_debug_dir)
    frame_dir = _resolve_frame_dir(base_dir, args.frame)

    cam_dirs = sorted([p for p in frame_dir.iterdir() if p.is_dir() and p.name.startswith("cam_")])
    if not cam_dirs:
        raise FileNotFoundError(f"No cam_* directories found in {frame_dir}")

    print(f"[INFO] Using frame dir: {frame_dir}")
    print(f"[INFO] Found {len(cam_dirs)} camera folders")

    colors = _color_palette(len(cam_dirs))
    entries = []
    bounds_min = None
    bounds_max = None

    for cam_idx, cam_dir in enumerate(cam_dirs):
        ply_path = cam_dir / f"person_{args.person_id:02d}.ply"
        if not ply_path.exists():
            print(f"[WARN] Missing {ply_path}, skipping.")
            continue
        points = _load_point_cloud(ply_path)
        if points.size == 0:
            print(f"[WARN] Empty point cloud {ply_path}, skipping.")
            continue
        points = _subsample(points, args.max_points)
        if points.size == 0:
            print(f"[WARN] Empty point cloud {ply_path}, skipping.")
            continue
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        bounds_min = pmin if bounds_min is None else np.minimum(bounds_min, pmin)
        bounds_max = pmax if bounds_max is None else np.maximum(bounds_max, pmax)
        name = cam_dir.name
        entries.append((name, points, colors[cam_idx]))

    if not entries:
        raise RuntimeError("No point clouds loaded for visualization.")

    center_offset = (bounds_min + bounds_max) * 0.5
    server = viser.ViserServer(host=args.host, port=args.port)
    R_fix = tf.SO3.from_x_radians(-np.pi / 2)
    server.scene.add_frame(
        "/scene",
        show_axes=False,
        wxyz=tuple(R_fix.wxyz),
        position=tuple((-R_fix.apply(center_offset)).tolist()),
    )

    for name, points, color in entries:
        server.scene.add_point_cloud(
            f"/scene/tsdf/{name}",
            points=points,
            colors=np.tile(color, (points.shape[0], 1)),
            point_size=args.point_size,
        )

    print("[INFO] Viser server running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[INFO] Exiting.")


if __name__ == "__main__":
    main()
