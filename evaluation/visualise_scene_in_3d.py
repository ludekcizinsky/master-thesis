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
    max_scale: Optional[float] = None
    max_gaussians: Optional[int] = None
    seed: int = 0
    mesh_opacity: float = 0.8
    posed_3dgs_pattern: str = "*.pt"
    posed_mesh_pattern: str = "*.obj"
    smplx_mesh_pattern: str = "*.obj"
    camera_pattern: str = "*.npz"
    camera_frustum_scale: float = 0.2


def main(args: Args) -> None:
    posed_3dgs_dir = args.eval_scene_dir / "posed_3dgs_per_frame"
    posed_meshes_dir = args.eval_scene_dir / "posed_meshes_per_frame"
    posed_smplx_meshes_dir = args.eval_scene_dir / "posed_smplx_meshes_per_frame"
    cameras_dir = args.eval_scene_dir / "all_cameras"

    splat_data: Optional[Dict[str, np.ndarray]] = None
    raw_xyz: Optional[np.ndarray] = None
    frame_used_3dgs: Optional[Path] = None

    if posed_3dgs_dir.exists():
        frame_files = _sorted_frame_files(posed_3dgs_dir, args.posed_3dgs_pattern)
        if frame_files:
            frame_used_3dgs = _select_frame(frame_files, args.frame_index, args.frame_name)
            state = _torch_load(frame_used_3dgs)
            raw_xyz = state.get("xyz")
            if isinstance(raw_xyz, torch.Tensor):
                raw_xyz = raw_xyz.detach().cpu().numpy()
            raw_xyz = None if raw_xyz is None else np.asarray(raw_xyz, dtype=np.float32)
            splat_data = _state_to_splat_arrays(
                state,
                max_scale=args.max_scale,
                max_gaussians=args.max_gaussians,
                seed=args.seed,
            )
        else:
            print(
                f"No 3DGS frames found in {posed_3dgs_dir} with pattern {args.posed_3dgs_pattern}"
            )
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
            color = _color_for_track(f"cam_{cam_dir.name}")
            camera_data.append((cam_dir.name, c2w, fov, aspect, color))
    else:
        print(f"No cameras directory found at {cameras_dir}")

    if splat_data is None and not mesh_entries and not smplx_entries and not camera_data:
        raise FileNotFoundError("No 3DGS, mesh, SMPL-X mesh, or camera data found to visualize.")

    bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
    if raw_xyz is not None:
        bounds = _update_bounds(raw_xyz, bounds)
    for _, verts, _, _ in mesh_entries:
        bounds = _update_bounds(verts, bounds)
    for _, verts, _, _ in smplx_entries:
        bounds = _update_bounds(verts, bounds)

    center_offset = np.zeros(3, dtype=np.float32)
    if args.center_scene and bounds[0] is not None and bounds[1] is not None:
        center_offset = (bounds[0] + bounds[1]) * 0.5

    server = viser.ViserServer(port=args.port)
    R_fix = tf.SO3.from_x_radians(-np.pi / 2)
    server.scene.add_frame(
        "/scene",
        show_axes=False,
        wxyz=tuple(R_fix.wxyz),
        position=tuple((-R_fix.apply(center_offset)).tolist()),
    )

    gs_handle = None
    if splat_data is not None:
        gs_handle = server.scene.add_gaussian_splats(
            "/scene/3dgs/gaussian_splats",
            centers=splat_data["centers"],
            rgbs=splat_data["rgbs"],
            opacities=splat_data["opacities"],
            covariances=splat_data["covariances"],
        )

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

    camera_entries: List[Tuple[str, object]] = []
    for cam_id, c2w, fov, aspect, color in camera_data:
        handle = server.scene.add_camera_frustum(
            f"/scene/cameras/{cam_id}",
            fov=float(fov),
            aspect=float(aspect),
            scale=float(args.camera_frustum_scale),
            wxyz=tuple(tf.SO3.from_matrix(c2w[:3, :3]).wxyz),
            position=tuple(c2w[:3, 3].tolist()),
            color=color,
        )
        camera_entries.append((cam_id, handle))

    with server.gui.add_folder("Visibility"):
        if gs_handle is not None:
            show_3dgs = server.gui.add_checkbox("Show 3DGS", True)

            @show_3dgs.on_update
            def _(_event=None) -> None:
                gs_handle.visible = bool(show_3dgs.value)
        if camera_entries:
            show_cameras = server.gui.add_checkbox("Show Cameras", True)

            @show_cameras.on_update
            def _(_event=None) -> None:
                for _cam_id, handle in camera_entries:
                    handle.visible = bool(show_cameras.value)

    if track_handles:
        with server.gui.add_folder("Meshes"):
            for track_id, handle in track_handles:
                checkbox = server.gui.add_checkbox(f"Show {track_id}", False)

                @checkbox.on_update
                def _(_event=None, handle=handle, checkbox=checkbox) -> None:
                    handle.visible = bool(checkbox.value)

    if smplx_handles:
        with server.gui.add_folder("SMPL-X Meshes"):
            for track_id, handle in smplx_handles:
                checkbox = server.gui.add_checkbox(f"Show {track_id}", False)

                @checkbox.on_update
                def _(_event=None, handle=handle, checkbox=checkbox) -> None:
                    handle.visible = bool(checkbox.value)

    frame_desc = args.frame_name if args.frame_name is not None else str(args.frame_index)
    if frame_used_3dgs is not None:
        print(f"3DGS frame: {frame_used_3dgs.name}")
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
