from __future__ import annotations

import hashlib
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


def _color_for_track(track_id: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(track_id.encode("utf-8")).digest()
    return (60 + digest[0] % 160, 60 + digest[1] % 160, 60 + digest[2] % 160)


@dataclass
class Args:
    mesh_dir: Path
    pattern: str = "*.obj"
    frame_index: int = 0
    frame_name: Optional[str] = None
    port: int = 8080
    center_scene: bool = True
    mesh_opacity: float = 0.8


def main(args: Args) -> None:
    if not args.mesh_dir.exists():
        raise FileNotFoundError(f"Mesh dir not found: {args.mesh_dir}")

    track_dirs = sorted([p for p in args.mesh_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not track_dirs:
        raise FileNotFoundError(f"No track subfolders found in {args.mesh_dir}")

    server = viser.ViserServer(port=args.port)
    mesh_entries: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int]]] = []
    global_min: Optional[np.ndarray] = None
    global_max: Optional[np.ndarray] = None

    for track_dir in track_dirs:
        frame_files = _sorted_frame_files(track_dir, args.pattern)
        if not frame_files:
            print(f"Skipping {track_dir.name}: no files matching {args.pattern}")
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
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        global_min = vmin if global_min is None else np.minimum(global_min, vmin)
        global_max = vmax if global_max is None else np.maximum(global_max, vmax)

    if not mesh_entries:
        raise FileNotFoundError("No meshes loaded. Check frame selection and pattern.")

    center_offset = np.zeros(3, dtype=np.float32)
    if args.center_scene and global_min is not None and global_max is not None:
        center_offset = (global_min + global_max) * 0.5
    R_fix = tf.SO3.from_x_radians(-np.pi / 2)
    server.scene.add_frame(
        "/scene/meshes",
        show_axes=False,
        wxyz=tuple(R_fix.wxyz),
        position=tuple((-R_fix.apply(center_offset)).tolist()),
    )

    track_entries: List[Tuple[str, object]] = []
    for track_id, verts, faces, color in mesh_entries:
        handle = server.scene.add_mesh_simple(
            f"/scene/meshes/{track_id}",
            vertices=verts,
            faces=faces,
            color=color,
        )
        if hasattr(handle, "opacity"):
            handle.opacity = float(args.mesh_opacity)
        track_entries.append((track_id, handle))

    with server.gui.add_folder("Visibility"):
        for track_id, handle in track_entries:
            checkbox = server.gui.add_checkbox(f"Show {track_id}", True)

            @checkbox.on_update
            def _(_event=None, handle=handle, checkbox=checkbox) -> None:
                handle.visible = bool(checkbox.value)

    frame_desc = args.frame_name if args.frame_name is not None else str(args.frame_index)
    print(f"Viser server running. Showing frame {frame_desc} from {args.mesh_dir}.")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main(tyro.cli(Args))
