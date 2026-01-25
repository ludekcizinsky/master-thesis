from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import tyro

import viser
import viser.transforms as tf


def _sorted_frame_files(root: Path, pattern: str) -> List[Path]:
    paths = [p for p in root.glob(pattern) if p.is_file()]
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
    pred_aligned_meshes_dir: Path
    gt_meshes_dir: Optional[Path] = None
    other_pred_aligned_meshes_dir: Optional[Path] = None
    frame_index: int = 0
    frame_name: Optional[str] = None
    port: int = 8080
    merged_mesh_pattern: str = "*.obj"
    gt_mesh_pattern: str = "*.obj"
    center_scene: bool = True
    mesh_opacity: float = 0.9


def main(args: Args) -> None:
    if not args.pred_aligned_meshes_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.pred_aligned_meshes_dir}")

    frame_files = _sorted_frame_files(args.pred_aligned_meshes_dir, args.merged_mesh_pattern)
    if not frame_files:
        raise FileNotFoundError(
            f"No merged mesh files found in {args.pred_aligned_meshes_dir} "
            f"matching {args.merged_mesh_pattern}. "
            "Note: per-person meshes are expected in subfolders."
        )

    pred_frame_path = _select_frame(frame_files, args.frame_index, args.frame_name)
    pred_verts, pred_faces = _load_mesh(pred_frame_path)

    other_pred_verts = None
    other_pred_faces = None
    other_pred_frame_path = None
    if args.other_pred_aligned_meshes_dir is not None:
        if not args.other_pred_aligned_meshes_dir.exists():
            raise FileNotFoundError(
                f"Other pred directory not found: {args.other_pred_aligned_meshes_dir}"
            )
        other_pred_files = _sorted_frame_files(
            args.other_pred_aligned_meshes_dir, args.merged_mesh_pattern
        )
        if not other_pred_files:
            raise FileNotFoundError(
                f"No meshes found in {args.other_pred_aligned_meshes_dir} "
                f"matching {args.merged_mesh_pattern}."
            )
        other_pred_frame_path = _select_frame(
            other_pred_files, args.frame_index, args.frame_name
        )
        other_pred_verts, other_pred_faces = _load_mesh(other_pred_frame_path)

    gt_verts = None
    gt_faces = None
    gt_frame_path = None
    if args.gt_meshes_dir is not None:
        if not args.gt_meshes_dir.exists():
            raise FileNotFoundError(f"GT directory not found: {args.gt_meshes_dir}")
        gt_frame_files = _sorted_frame_files(args.gt_meshes_dir, args.gt_mesh_pattern)
        if not gt_frame_files:
            raise FileNotFoundError(
                f"No GT mesh files found in {args.gt_meshes_dir} matching {args.gt_mesh_pattern}."
            )
        gt_frame_path = _select_frame(gt_frame_files, args.frame_index, args.frame_name)
        gt_verts, gt_faces = _load_mesh(gt_frame_path)

    bounds: Tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
    bounds = _update_bounds(pred_verts, bounds)
    if gt_verts is not None:
        bounds = _update_bounds(gt_verts, bounds)
    if other_pred_verts is not None:
        bounds = _update_bounds(other_pred_verts, bounds)

    center_offset = np.zeros(3, dtype=np.float32)
    if args.center_scene and bounds[0] is not None and bounds[1] is not None:
        center_offset = (bounds[0] + bounds[1]) * 0.5

    server = viser.ViserServer(port=args.port)
    R_fix = tf.SO3.from_x_radians(np.pi / 2)
    server.scene.add_frame(
        "/scene",
        show_axes=False,
        wxyz=tuple(R_fix.wxyz),
        position=tuple((-R_fix.apply(center_offset)).tolist()),
    )

    pred_handle = server.scene.add_mesh_simple(
        "/scene/pred_mesh",
        vertices=pred_verts,
        faces=pred_faces,
        color=(220, 220, 220),
    )
    if hasattr(pred_handle, "opacity"):
        pred_handle.opacity = float(args.mesh_opacity)

    other_pred_handle = None
    if other_pred_verts is not None and other_pred_faces is not None:
        other_pred_handle = server.scene.add_mesh_simple(
            "/scene/other_pred_mesh",
            vertices=other_pred_verts,
            faces=other_pred_faces,
            color=(70, 130, 180),
        )
        if hasattr(other_pred_handle, "opacity"):
            other_pred_handle.opacity = float(args.mesh_opacity)

    gt_handle = None
    if gt_verts is not None and gt_faces is not None:
        gt_handle = server.scene.add_mesh_simple(
            "/scene/gt_mesh",
            vertices=gt_verts,
            faces=gt_faces,
            color=(80, 200, 120),
        )
        if hasattr(gt_handle, "opacity"):
            gt_handle.opacity = float(args.mesh_opacity)

    frame_desc = args.frame_name if args.frame_name is not None else str(args.frame_index)
    if gt_frame_path is not None or other_pred_frame_path is not None:
        print(
            "Viser server running. "
            f"Showing frame {frame_desc} from pred {args.pred_aligned_meshes_dir}"
            f"{' and comparison pred ' + str(args.other_pred_aligned_meshes_dir) if other_pred_frame_path is not None else ''}"
            f"{' and gt ' + str(args.gt_meshes_dir) if gt_frame_path is not None else ''}."
        )
    else:
        print(
            "Viser server running. "
            f"Showing merged mesh frame {frame_desc} from {args.pred_aligned_meshes_dir}."
        )

    with server.gui.add_folder("Meshes"):
        pred_checkbox = server.gui.add_checkbox("Baseline Pred", True)

        @pred_checkbox.on_update
        def _(_event=None, handle=pred_handle, checkbox=pred_checkbox) -> None:
            handle.visible = bool(checkbox.value)

        if other_pred_handle is not None:
            other_pred_checkbox = server.gui.add_checkbox("Comparison Pred", True)

            @other_pred_checkbox.on_update
            def _(
                _event=None, handle=other_pred_handle, checkbox=other_pred_checkbox
            ) -> None:
                handle.visible = bool(checkbox.value)

        if gt_handle is not None:
            gt_checkbox = server.gui.add_checkbox("Show GT", True)

            @gt_checkbox.on_update
            def _(_event=None, handle=gt_handle, checkbox=gt_checkbox) -> None:
                handle.visible = bool(checkbox.value)

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main(tyro.cli(Args))
