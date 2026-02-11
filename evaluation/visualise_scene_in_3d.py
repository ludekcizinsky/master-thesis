from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import tyro
from pytorch3d.transforms import quaternion_to_matrix
from PIL import Image

import viser
import viser.transforms as tf


def _sorted_frame_files(root: Path, pattern: str) -> List[Path]:
    paths = list(root.glob(pattern))
    if not paths:
        return []

    def _key(p: Path) -> Tuple[int, str]:
        stem = p.stem
        if stem.isdigit():
            return (0, f"{int(stem):012d}")
        return (1, stem)

    return sorted(paths, key=_key)


def _select_frame(frame_files: List[Path], frame_index: int, frame_name: Optional[str]) -> Path:
    if frame_name:
        if "." in frame_name:
            matches = [p for p in frame_files if p.name == frame_name]
        else:
            matches = [p for p in frame_files if p.stem == frame_name]
        if not matches and frame_name.isdigit():
            target = int(frame_name)
            matches = [p for p in frame_files if p.stem.isdigit() and int(p.stem) == target]
        if not matches:
            raise FileNotFoundError(f"No frame named {frame_name!r}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple frames matched {frame_name!r}")
        return matches[0]
    for stem in (str(frame_index), f"{frame_index:06d}", f"{frame_index:08d}"):
        for path in frame_files:
            if path.stem == stem:
                return path
    if frame_index < 0 or frame_index >= len(frame_files):
        raise IndexError(f"frame_index {frame_index} is out of range [0, {len(frame_files) - 1}]")
    return frame_files[frame_index]


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


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Failed to load mesh at {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if verts.size == 0 or faces.size == 0:
        raise RuntimeError(f"Mesh has no vertices or faces at {path}")
    return verts, faces


def _load_camera_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as cams:
        if "intrinsics" not in cams.files or "extrinsics" not in cams.files:
            raise KeyError(f"Missing intrinsics/extrinsics in {path}")
        intr = cams["intrinsics"]
        extr = cams["extrinsics"]
    if intr.ndim == 3:
        intr = intr[0]
    if extr.ndim == 3:
        extr = extr[0]
    if intr.shape != (3, 3) or extr.shape != (3, 4):
        raise ValueError(f"Unexpected camera shapes in {path}: {intr.shape}, {extr.shape}")
    return intr.astype(np.float32), extr.astype(np.float32)


def _fov_aspect_from_intrinsics(intr: np.ndarray) -> Tuple[float, float]:
    fy = float(intr[1, 1])
    if fy > 0.0:
        cy = float(intr[1, 2])
        cx = float(intr[0, 2])
        if cx > 0.0 and cy > 0.0:
            height = 2.0 * cy
            width = 2.0 * cx
            fov = 2.0 * np.arctan2(height / 2.0, fy)
            return fov, width / height
    return np.deg2rad(60.0), 1.0


def _load_rgb_image(scene_dir: Path, cam_id: int, frame_stem: str) -> np.ndarray:
    path = scene_dir / "images" / f"{cam_id}" / f"{frame_stem}.jpg"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing RGB frame at {path}. Expected images saved as "
            f"images/{cam_id}/{frame_stem}.jpg"
        )
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _depth_to_point_cloud(
    depth: np.ndarray,
    intr: np.ndarray,
    c2w: np.ndarray,
    *,
    stride: int,
    max_depth: float,
    rgb: Optional[np.ndarray],
    filter_sparse_points: bool = False,
    sparse_voxel_size: float = 0.05,
    sparse_min_neighbors: int = 12,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if depth.ndim != 2:
        raise ValueError(f"Expected depth map with shape (H, W), got {depth.shape}")
    h, w = depth.shape
    if rgb is not None and (rgb.shape[0] != h or rgb.shape[1] != w):
        rgb = np.asarray(
            Image.fromarray(rgb).resize((w, h), resample=Image.BILINEAR), dtype=np.uint8
        )
    stride = max(int(stride), 1)
    ys = np.arange(0, h, stride, dtype=np.float32)
    xs = np.arange(0, w, stride, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth[::stride, ::stride].astype(np.float32)
    valid = np.isfinite(z) & (z > 0.0) & (z <= float(max_depth))
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), None
    x = (grid_x - float(intr[0, 2])) / float(intr[0, 0]) * z
    y = (grid_y - float(intr[1, 2])) / float(intr[1, 1]) * z
    points_cam = np.stack([x, y, z], axis=-1)[valid]
    points_world = (c2w[:3, :3] @ points_cam.T).T + c2w[:3, 3]
    colors = None
    if rgb is not None:
        colors = rgb[::stride, ::stride][valid]
    points_world = points_world.astype(np.float32)
    if filter_sparse_points:
        points_world, colors = _remove_sparse_points(
            points_world,
            colors,
            voxel_size=float(sparse_voxel_size),
            min_neighbors=int(sparse_min_neighbors),
        )
    return points_world, colors


def _remove_sparse_points(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    *,
    voxel_size: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if points.size == 0:
        return points, colors
    if voxel_size <= 0.0 or min_neighbors <= 1:
        return points, colors

    voxel_coords = np.floor(points / float(voxel_size)).astype(np.int32)
    unique_voxels, inverse, counts = np.unique(
        voxel_coords, axis=0, return_inverse=True, return_counts=True
    )

    count_by_voxel: Dict[Tuple[int, int, int], int] = {}
    for voxel, count in zip(unique_voxels, counts):
        key = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
        count_by_voxel[key] = int(count)

    neighbor_counts = np.zeros(unique_voxels.shape[0], dtype=np.int32)
    for idx, voxel in enumerate(unique_voxels):
        vx, vy, vz = int(voxel[0]), int(voxel[1]), int(voxel[2])
        total = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    total += count_by_voxel.get((vx + dx, vy + dy, vz + dz), 0)
        neighbor_counts[idx] = total

    keep_mask = neighbor_counts[inverse] >= int(min_neighbors)
    filtered_points = points[keep_mask]
    filtered_colors = colors[keep_mask] if colors is not None else None
    return filtered_points, filtered_colors


def _color_for_track(track_id: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(track_id.encode("utf-8")).digest()
    return (60 + digest[0] % 160, 60 + digest[1] % 160, 60 + digest[2] % 160)


def _update_bounds(
    verts: np.ndarray, bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if verts.size == 0:
        return bounds
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    min_bound, max_bound = bounds
    if min_bound is None:
        return vmin, vmax
    return np.minimum(min_bound, vmin), np.maximum(max_bound, vmax)


@dataclass
class Args:
    eval_scene_dir: Path
    frame_index: int = 0
    frame_name: Optional[str] = None
    port: int = 8080
    center_scene: bool = True
    is_minus_y_up: bool = True
    source_camera_id: int = 0
    background_max_depth: float = 5.0
    depth_stride: int = 4
    background_point_size: float = 0.02
    background_filter_sparse_points: bool = True
    background_filter_voxel_size: float = 0.05
    background_filter_min_neighbors: int = 12
    max_scale: Optional[float] = 0.5
    max_gaussians: Optional[int] = 200000
    seed: int = 0
    mesh_opacity: float = 0.8
    posed_3dgs_pattern: str = "*.pt"
    posed_mesh_pattern: str = "*.obj"
    smplx_mesh_pattern: str = "*.obj"
    camera_pattern: str = "*.npz"
    camera_frustum_scale: float = 0.2
    depth_pattern: str = "*.npy"


def main(args: Args) -> None:
    posed_3dgs_dir = args.eval_scene_dir / "posed_3dgs_per_frame"
    posed_meshes_dir = args.eval_scene_dir / "posed_meshes_per_frame"
    posed_smplx_meshes_dir = args.eval_scene_dir / "posed_smplx_meshes_per_frame"
    gt_smplx_meshes_dir = args.eval_scene_dir / "gt_inputs" / "pose" / "gt_smplx_meshes_per_frame"
    cameras_dir = args.eval_scene_dir / "all_cameras"
    masked_depth_dir = args.eval_scene_dir / "masked_depth_maps"

    per_person_splats: List[Tuple[str, Dict[str, np.ndarray], Optional[np.ndarray]]] = []

    if posed_3dgs_dir.exists():
        person_dirs = sorted(
            [p for p in posed_3dgs_dir.iterdir() if p.is_dir()], key=lambda p: p.name
        )
        if person_dirs:
            for person_dir in person_dirs:
                frame_files = _sorted_frame_files(person_dir, args.posed_3dgs_pattern)
                if not frame_files:
                    print(
                        f"Skipping 3DGS {person_dir.name}: no files matching {args.posed_3dgs_pattern}"
                    )
                    continue
                try:
                    frame_path = _select_frame(frame_files, args.frame_index, args.frame_name)
                except (FileNotFoundError, IndexError, RuntimeError) as exc:
                    print(f"Skipping 3DGS {person_dir.name}: {exc}")
                    continue
                state = _torch_load(frame_path)
                person_xyz = state.get("xyz")
                if isinstance(person_xyz, torch.Tensor):
                    person_xyz = person_xyz.detach().cpu().numpy()
                person_xyz = None if person_xyz is None else np.asarray(person_xyz, dtype=np.float32)
                person_splat = _state_to_splat_arrays(
                    state,
                    max_scale=args.max_scale,
                    max_gaussians=args.max_gaussians,
                    seed=args.seed,
                )
                per_person_splats.append((person_dir.name, person_splat, person_xyz))
            if not per_person_splats:
                print(f"No per-person 3DGS frames loaded from {posed_3dgs_dir}")
        else:
            print(f"No per-person 3DGS subfolders found in {posed_3dgs_dir}")
    else:
        print(f"No posed 3DGS directory found at {posed_3dgs_dir}")

    mesh_entries: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]] = []
    if posed_meshes_dir.exists():
        track_dirs = sorted([p for p in posed_meshes_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        for track_dir in track_dirs:
            frame_files = _sorted_frame_files(track_dir, args.posed_mesh_pattern)
            if not frame_files:
                print(f"Skipping {track_dir.name}: no files matching {args.posed_mesh_pattern}")
                continue
            try:
                frame_path = _select_frame(frame_files, args.frame_index, args.frame_name)
            except (FileNotFoundError, IndexError, RuntimeError) as exc:
                print(f"Skipping {track_dir.name}: {exc}")
                continue
            try:
                verts, faces = _load_mesh(frame_path)
            except RuntimeError as exc:
                print(f"Skipping {track_dir.name}: {exc}")
                continue
            color = _color_for_track(track_dir.name)
            mesh_entries.append((track_dir.name, verts, faces, color))
    else:
        print(f"No posed meshes directory found at {posed_meshes_dir}")

    smplx_entries: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]] = []
    if posed_smplx_meshes_dir.exists():
        track_dirs = sorted(
            [p for p in posed_smplx_meshes_dir.iterdir() if p.is_dir()], key=lambda p: p.name
        )
        for track_dir in track_dirs:
            frame_files = _sorted_frame_files(track_dir, args.smplx_mesh_pattern)
            if not frame_files:
                print(
                    f"Skipping SMPL-X {track_dir.name}: no files matching {args.smplx_mesh_pattern}"
                )
                continue
            try:
                frame_path = _select_frame(frame_files, args.frame_index, args.frame_name)
            except (FileNotFoundError, IndexError, RuntimeError) as exc:
                print(f"Skipping SMPL-X {track_dir.name}: {exc}")
                continue
            try:
                verts, faces = _load_mesh(frame_path)
            except RuntimeError as exc:
                print(f"Skipping SMPL-X {track_dir.name}: {exc}")
                continue
            color = _color_for_track(f"smplx_{track_dir.name}")
            smplx_entries.append((track_dir.name, verts, faces, color))
    else:
        print(f"No posed SMPL-X meshes directory found at {posed_smplx_meshes_dir}")

    gt_smplx_entries: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]] = []
    if gt_smplx_meshes_dir.exists():
        track_dirs = sorted(
            [p for p in gt_smplx_meshes_dir.iterdir() if p.is_dir()], key=lambda p: p.name
        )
        for track_dir in track_dirs:
            frame_files = _sorted_frame_files(track_dir, args.smplx_mesh_pattern)
            if not frame_files:
                print(
                    f"Skipping GT SMPL-X {track_dir.name}: no files matching {args.smplx_mesh_pattern}"
                )
                continue
            try:
                frame_path = _select_frame(frame_files, args.frame_index, args.frame_name)
            except (FileNotFoundError, IndexError, RuntimeError) as exc:
                print(f"Skipping GT SMPL-X {track_dir.name}: {exc}")
                continue
            try:
                verts, faces = _load_mesh(frame_path)
            except RuntimeError as exc:
                print(f"Skipping GT SMPL-X {track_dir.name}: {exc}")
                continue
            color = _color_for_track(f"gt_smplx_{track_dir.name}")
            gt_smplx_entries.append((track_dir.name, verts, faces, color))
    else:
        print(f"No GT SMPL-X meshes directory found at {gt_smplx_meshes_dir}")

    camera_data: List[Tuple[str, np.ndarray, np.ndarray, float, float, Tuple[int, int, int]]] = []
    if cameras_dir.exists():
        camera_ids = sorted([p for p in cameras_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        for cam_dir in camera_ids:
            frame_files = _sorted_frame_files(cam_dir, args.camera_pattern)
            if not frame_files:
                print(f"Skipping camera {cam_dir.name}: no files matching {args.camera_pattern}")
                continue
            try:
                frame_path = _select_frame(frame_files, args.frame_index, args.frame_name)
            except (FileNotFoundError, IndexError, RuntimeError) as exc:
                print(f"Skipping camera {cam_dir.name}: {exc}")
                continue
            try:
                intr, extr = _load_camera_npz(frame_path)
            except (KeyError, ValueError) as exc:
                print(f"Skipping camera {cam_dir.name}: {exc}")
                continue
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :4] = extr
            c2w = np.linalg.inv(w2c)
            fov, aspect = _fov_aspect_from_intrinsics(intr)
            is_source = cam_dir.name == str(args.source_camera_id)
            color = (30, 144, 255) if is_source else (255, 140, 0)
            camera_data.append((cam_dir.name, c2w, fov, aspect, color, is_source))
    else:
        print(f"No cameras directory found at {cameras_dir}")

    background_depth: Optional[np.ndarray] = None
    background_intr: Optional[np.ndarray] = None
    background_c2w: Optional[np.ndarray] = None
    background_rgb: Optional[np.ndarray] = None
    if masked_depth_dir.exists():
        depth_files = _sorted_frame_files(masked_depth_dir, args.depth_pattern)
        if depth_files:
            try:
                depth_path = _select_frame(depth_files, args.frame_index, args.frame_name)
            except (FileNotFoundError, IndexError, RuntimeError) as exc:
                print(f"Skipping masked depth maps: {exc}")
                depth_path = None
            if depth_path is not None:
                depth = np.load(depth_path).astype(np.float32)
                cam_dir = cameras_dir / f"{args.source_camera_id}"
                cam_path = cam_dir / f"{depth_path.stem}.npz"
                if cam_dir.exists() and cam_path.exists():
                    intr, extr = _load_camera_npz(cam_path)
                    w2c = np.eye(4, dtype=np.float32)
                    w2c[:3, :4] = extr
                    c2w = np.linalg.inv(w2c)
                    background_depth = depth
                    background_intr = intr
                    background_c2w = c2w
                    background_rgb = _load_rgb_image(
                        args.eval_scene_dir, args.source_camera_id, depth_path.stem
                    )
                else:
                    if not cam_dir.exists():
                        print(f"Missing camera dir for source camera {args.source_camera_id}: {cam_dir}")
                    else:
                        print(f"Missing camera file for depth frame: {cam_path}")
        else:
            print(f"No masked depth maps found in {masked_depth_dir} with pattern {args.depth_pattern}")
    else:
        print(f"No masked depth maps directory found at {masked_depth_dir}")

    if (
        not per_person_splats
        and not mesh_entries
        and not smplx_entries
        and not gt_smplx_entries
        and not camera_data
        and background_depth is None
    ):
        raise FileNotFoundError(
            "No 3DGS, mesh, SMPL-X mesh, GT SMPL-X mesh, or camera data found to visualize."
        )

    bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
    for _, _, person_xyz in per_person_splats:
        if person_xyz is not None:
            bounds = _update_bounds(person_xyz, bounds)
    for _, verts, _, _ in mesh_entries:
        bounds = _update_bounds(verts, bounds)
    for _, verts, _, _ in smplx_entries:
        bounds = _update_bounds(verts, bounds)
    for _, verts, _, _ in gt_smplx_entries:
        bounds = _update_bounds(verts, bounds)

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

    gs_person_handles: List[Tuple[str, object]] = []
    for person_id, person_splat, _ in per_person_splats:
        handle = server.scene.add_gaussian_splats(
            f"/scene/3dgs/{person_id}",
            centers=person_splat["centers"],
            rgbs=person_splat["rgbs"],
            opacities=person_splat["opacities"],
            covariances=person_splat["covariances"],
        )
        gs_person_handles.append((person_id, handle))

    track_handles: List[Tuple[str, object]] = []
    for track_id, verts, faces, color in mesh_entries:
        handle = server.scene.add_mesh_simple(
            f"/scene/meshes/{track_id}",
            vertices=verts,
            faces=faces,
            color=color,
        )
        if hasattr(handle, "opacity"):
            handle.opacity = float(args.mesh_opacity)
        handle.visible = False
        track_handles.append((track_id, handle))

    smplx_handles: List[Tuple[str, object]] = []
    for track_id, verts, faces, color in smplx_entries:
        handle = server.scene.add_mesh_simple(
            f"/scene/smplx_meshes/{track_id}",
            vertices=verts,
            faces=faces,
            color=color,
        )
        if hasattr(handle, "opacity"):
            handle.opacity = float(args.mesh_opacity)
        handle.visible = False
        smplx_handles.append((track_id, handle))

    gt_smplx_handles: List[Tuple[str, object]] = []
    for track_id, verts, faces, color in gt_smplx_entries:
        handle = server.scene.add_mesh_simple(
            f"/scene/gt_smplx_meshes/{track_id}",
            vertices=verts,
            faces=faces,
            color=color,
        )
        if hasattr(handle, "opacity"):
            handle.opacity = float(args.mesh_opacity)
        handle.visible = False
        gt_smplx_handles.append((track_id, handle))

    camera_entries: List[Tuple[str, object]] = []
    for cam_id, c2w, fov, aspect, color, is_source in camera_data:
        handle = server.scene.add_camera_frustum(
            f"/scene/cameras/{cam_id}",
            fov=float(fov),
            aspect=float(aspect),
            scale=float(args.camera_frustum_scale),
            wxyz=tuple(tf.SO3.from_matrix(c2w[:3, :3]).wxyz),
            position=tuple(c2w[:3, 3].tolist()),
            color=color,
        )
        handle.visible = bool(is_source)
        camera_entries.append((cam_id, handle))

    background_handle = None
    if background_depth is not None and background_intr is not None and background_c2w is not None:
        background_points, background_colors = _depth_to_point_cloud(
            background_depth,
            background_intr,
            background_c2w,
            stride=args.depth_stride,
            max_depth=args.background_max_depth,
            rgb=background_rgb,
            filter_sparse_points=args.background_filter_sparse_points,
            sparse_voxel_size=args.background_filter_voxel_size,
            sparse_min_neighbors=args.background_filter_min_neighbors,
        )
        if background_colors is None:
            background_colors = np.full((background_points.shape[0], 3), 180, dtype=np.uint8)
        background_handle = server.scene.add_point_cloud(
            "/scene/background",
            points=background_points,
            colors=background_colors,
            point_size=float(args.background_point_size),
            point_shape="rounded",
        )
        background_handle.visible = False

    if camera_entries:
        with server.gui.add_folder("Cameras"):
            for cam_id, handle in camera_entries:
                checkbox = server.gui.add_checkbox(
                    f"Show {cam_id}", cam_id == str(args.source_camera_id)
                )

                @checkbox.on_update
                def _(_event=None, handle=handle, checkbox=checkbox) -> None:
                    handle.visible = bool(checkbox.value)

    if background_handle is not None:
        with server.gui.add_folder("Background"):
            bg_checkbox = server.gui.add_checkbox("Show Background", False)
            bg_sparse_checkbox = server.gui.add_checkbox(
                "Filter Sparse Points", bool(args.background_filter_sparse_points)
            )
            bg_depth_slider = server.gui.add_slider(
                "Max Depth (m)",
                min=0.5,
                max=20.0,
                step=0.1,
                initial_value=float(args.background_max_depth),
            )
            bg_voxel_slider = server.gui.add_slider(
                "Filter Radius (m)",
                min=0.01,
                max=0.5,
                step=0.01,
                initial_value=float(args.background_filter_voxel_size),
            )
            bg_neighbor_slider = server.gui.add_slider(
                "Min Neighbor Count",
                min=1,
                max=100,
                step=1,
                initial_value=int(args.background_filter_min_neighbors),
            )

            def _refresh_background_point_cloud() -> None:
                nonlocal background_handle
                points, colors = _depth_to_point_cloud(
                    background_depth,
                    background_intr,
                    background_c2w,
                    stride=args.depth_stride,
                    max_depth=float(bg_depth_slider.value),
                    rgb=background_rgb,
                    filter_sparse_points=bool(bg_sparse_checkbox.value),
                    sparse_voxel_size=float(bg_voxel_slider.value),
                    sparse_min_neighbors=int(bg_neighbor_slider.value),
                )
                if colors is None:
                    colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
                if background_handle is not None and hasattr(background_handle, "update"):
                    background_handle.update(points=points, colors=colors)
                else:
                    if background_handle is not None and hasattr(background_handle, "remove"):
                        background_handle.remove()
                    background_handle = server.scene.add_point_cloud(
                        "/scene/background",
                        points=points,
                        colors=colors,
                        point_size=float(args.background_point_size),
                        point_shape="rounded",
                    )
                background_handle.visible = bool(bg_checkbox.value)

            @bg_checkbox.on_update
            def _(_event=None, handle=background_handle, bg_checkbox=bg_checkbox) -> None:
                handle.visible = bool(bg_checkbox.value)

            @bg_depth_slider.on_update
            def _(_event=None) -> None:
                _refresh_background_point_cloud()

            @bg_sparse_checkbox.on_update
            def _(_event=None) -> None:
                _refresh_background_point_cloud()

            @bg_voxel_slider.on_update
            def _(_event=None) -> None:
                _refresh_background_point_cloud()

            @bg_neighbor_slider.on_update
            def _(_event=None) -> None:
                _refresh_background_point_cloud()

    if gs_person_handles:
        with server.gui.add_folder("3DGS"):
            for person_id, handle in gs_person_handles:
                checkbox = server.gui.add_checkbox(f"Show {person_id}", True)

                @checkbox.on_update
                def _(_event=None, handle=handle, checkbox=checkbox) -> None:
                    handle.visible = bool(checkbox.value)

    if track_handles:
        with server.gui.add_folder("Meshes"):
            for track_id, handle in track_handles:
                checkbox = server.gui.add_checkbox(f"Show {track_id}", False)

                @checkbox.on_update
                def _(_event=None, handle=handle, checkbox=checkbox) -> None:
                    handle.visible = bool(checkbox.value)

    if smplx_handles or gt_smplx_handles:
        with server.gui.add_folder("SMPL-X Meshes"):
            for track_id, handle in smplx_handles:
                checkbox = server.gui.add_checkbox(f"Show Pred {track_id}", False)

                @checkbox.on_update
                def _(_event=None, handle=handle, checkbox=checkbox) -> None:
                    handle.visible = bool(checkbox.value)

            for track_id, handle in gt_smplx_handles:
                checkbox = server.gui.add_checkbox(f"Show GT {track_id}", False)

                @checkbox.on_update
                def _(_event=None, handle=handle, checkbox=checkbox) -> None:
                    handle.visible = bool(checkbox.value)

    frame_desc = args.frame_name if args.frame_name is not None else str(args.frame_index)
    if camera_entries:
        print(f"Loaded cameras: {', '.join(cam_id for cam_id, _ in camera_entries)}")
    print(f"Viser server running. Showing frame {frame_desc} from {args.eval_scene_dir}.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main(tyro.cli(Args))
