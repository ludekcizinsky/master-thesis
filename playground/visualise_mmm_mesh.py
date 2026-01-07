from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import trimesh

import numpy as np
import tyro
import viser
import viser.transforms as tf


def _sorted_obj_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"MMM mesh directory not found: {root}")
    obj_files = [p for p in root.glob("*.obj") if p.is_file()]
    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in {root}")

    def _key(path: Path) -> Tuple[int, int, str]:
        stem = path.stem
        digits = re.findall(r"\d+", stem)
        if digits:
            return (0, int(digits[-1]), stem)
        return (1, 0, stem)

    return sorted(obj_files, key=_key)


def _load_obj_mesh(
    path: Path,
    *,
    center_offset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"OBJ is not a single mesh: {path}")
    
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()

    if not vertices or not faces:
        raise ValueError(f"OBJ has no geometry: {path}")

    verts_np = np.asarray(vertices, dtype=np.float32)
#    x = verts_np[:, 0].copy()
    #y = verts_np[:, 1].copy()
    #z = verts_np[:, 2].copy()
    #verts_np[:, 0] = x
    #verts_np[:, 1] = z
    #verts_np[:, 2] = y

    if center_offset is not None:
        verts_np = verts_np - center_offset
    faces_np = np.asarray(faces, dtype=np.int32)
    return verts_np, faces_np

@dataclass
class Args:
    mmm_data_dir: Path
    port: int = 8080
    center_mesh: bool = True


def main(args: Args) -> None:
    obj_files = _sorted_obj_files(args.mmm_data_dir)
    num_frames = len(obj_files)

    first_verts_raw, first_faces = _load_obj_mesh(obj_files[0])
    if args.center_mesh:
        vmin_raw = first_verts_raw.min(axis=0)
        vmax_raw = first_verts_raw.max(axis=0)
        center_offset = (vmin_raw + vmax_raw) * 0.5
    else:
        center_offset = np.zeros(3, dtype=np.float32)

    server = viser.ViserServer(port=args.port)

    frame_slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=num_frames - 1,
        step=1,
        initial_value=0,
    )
    frame_label = server.gui.add_text("File", obj_files[0].name)

    mesh_handle: Optional[viser.MeshHandle] = None
    last_frame_idx: Optional[int] = None


    def _show_frame(frame_idx: int) -> None:
        nonlocal mesh_handle, last_frame_idx
        if last_frame_idx == frame_idx:
            return
        frame_path = obj_files[frame_idx]
        verts, faces = _load_obj_mesh(
            frame_path,
            center_offset=center_offset,
        )
        with server.atomic():
            frame_label.value = frame_path.name
            if mesh_handle is not None and hasattr(mesh_handle, "update"):
                mesh_handle.update(vertices=verts, faces=faces, color=(220, 220, 220))
            else:
                if mesh_handle is not None and hasattr(mesh_handle, "remove"):
                    mesh_handle.remove()
                mesh_handle = server.scene.add_mesh_simple(
                    "/mmm_mesh",
                    vertices=verts,
                    faces=faces,
                    color=(220, 220, 220),
                    wxyz=tf.SO3.from_x_radians(np.pi / 2).wxyz,
                    position=(0.0, 0.0, 0.0),
                )
            if mesh_handle is not None and hasattr(mesh_handle, "opacity"):
                mesh_handle.opacity = 1.0
        last_frame_idx = frame_idx
        server.flush()

    @frame_slider.on_update
    def _(_) -> None:
        _show_frame(int(frame_slider.value))

    _show_frame(0)

    while True:
        time.sleep(1.0)



if __name__ == "__main__":
    main(tyro.cli(Args))
