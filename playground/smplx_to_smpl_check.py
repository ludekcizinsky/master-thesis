from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import tyro
import viser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from submodules.smplx import smplx


@dataclass
class Config:
    scene_dir: Path
    smplx_folder: str = "smplx"
    smpl_folder: str = "smpl"
    model_folder: Path = Path("/home/cizinsky/body_models")
    smplx_model_ext: str = "npz"
    smpl_model_ext: str = "pkl"
    gender: str = "neutral"
    port: int = 8080


def _load_faces(model_folder: Path, model_type: str, gender: str, ext: str) -> np.ndarray:
    model = smplx.create(
        str(model_folder),
        model_type=model_type,
        gender=gender,
        ext=ext,
    )
    return np.asarray(model.faces, dtype=np.int32)


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def _collect_frames(smplx_dir: Path, smpl_dir: Path) -> List[str]:
    smplx_frames = {p.stem for p in smplx_dir.glob("*.npz")}
    smpl_frames = {p.stem for p in smpl_dir.glob("*.npz")}
    intersection = sorted(smplx_frames & smpl_frames)
    if intersection:
        return intersection
    return sorted(smplx_frames | smpl_frames)


def _color_for_person(base: np.ndarray, pid: int) -> Tuple[int, int, int]:
    scale = 0.6 + 0.4 * ((pid % 5) / 4.0)
    color = np.clip(base * scale, 0, 255)
    return tuple(int(c) for c in color)


def _update_mesh(
    server: viser.ViserServer,
    handle: Optional[Any],
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: Tuple[int, int, int],
) -> viser.MeshHandle:
    if handle is not None and hasattr(handle, "update"):
        handle.update(vertices=vertices, faces=faces, color=color)
        return handle
    return server.scene.add_mesh_simple(
        name,
        vertices=vertices,
        faces=faces,
        color=color,
    )


def main() -> None:
    cfg = tyro.cli(Config)

    smplx_dir = cfg.scene_dir / cfg.smplx_folder
    smpl_dir = cfg.scene_dir / cfg.smpl_folder
    frames = _collect_frames(smplx_dir, smpl_dir)
    if not frames:
        raise FileNotFoundError(f"No npz files found under {smplx_dir} or {smpl_dir}")

    smplx_faces = _load_faces(cfg.model_folder, "smplx", cfg.gender, cfg.smplx_model_ext)
    smpl_faces = _load_faces(cfg.model_folder, "smpl", cfg.gender, cfg.smpl_model_ext)

    server = viser.ViserServer(port=cfg.port)

    smplx_root = server.scene.add_frame("/smplx", show_axes=True)
    smpl_root = server.scene.add_frame("/smpl", show_axes=True)

    handles: Dict[str, List[Any]] = {"smplx": [], "smpl": []}

    smplx_base = np.array([70, 130, 255], dtype=np.float32)
    smpl_base = np.array([255, 140, 70], dtype=np.float32)

    with server.gui.add_folder("Visibility"):
        show_smplx = server.gui.add_checkbox("Show SMPL-X", True)
        show_smpl = server.gui.add_checkbox("Show SMPL", True)

    @show_smplx.on_update
    def _(_) -> None:
        smplx_root.visible = show_smplx.value

    @show_smpl.on_update
    def _(_) -> None:
        smpl_root.visible = show_smpl.value

    with server.gui.add_folder("Frames"):
        frame_slider = server.gui.add_slider(
            "Frame Index",
            min=0,
            max=len(frames) - 1,
            step=1,
            initial_value=0,
        )

    def render_frame(frame_idx: int) -> None:
        frame_id = frames[frame_idx]
        smplx_npz = _load_npz(smplx_dir / f"{frame_id}.npz")
        smpl_npz = _load_npz(smpl_dir / f"{frame_id}.npz")

        if smplx_npz is not None and "verts" in smplx_npz:
            verts = smplx_npz["verts"]
            for pid in range(verts.shape[0]):
                color = _color_for_person(smplx_base, pid)
                name = f"/smplx/person_{pid}"
                handle = handles["smplx"][pid] if pid < len(handles["smplx"]) else None
                new_handle = _update_mesh(server, handle, name, verts[pid], smplx_faces, color)
                if pid >= len(handles["smplx"]):
                    handles["smplx"].append(new_handle)
                handles["smplx"][pid].visible = True
            for pid in range(verts.shape[0], len(handles["smplx"])):
                handles["smplx"][pid].visible = False
        else:
            for handle in handles["smplx"]:
                handle.visible = False

        if smpl_npz is not None and "verts" in smpl_npz:
            verts = smpl_npz["verts"]
            for pid in range(verts.shape[0]):
                color = _color_for_person(smpl_base, pid)
                name = f"/smpl/person_{pid}"
                handle = handles["smpl"][pid] if pid < len(handles["smpl"]) else None
                new_handle = _update_mesh(server, handle, name, verts[pid], smpl_faces, color)
                if pid >= len(handles["smpl"]):
                    handles["smpl"].append(new_handle)
                handles["smpl"][pid].visible = True
            for pid in range(verts.shape[0], len(handles["smpl"])):
                handles["smpl"][pid].visible = False
        else:
            for handle in handles["smpl"]:
                handle.visible = False

        print(f"Showing frame {frame_id} ({frame_idx + 1}/{len(frames)})")

    @frame_slider.on_update
    def _(_) -> None:
        render_frame(int(frame_slider.value))

    render_frame(0)
    print("Viser server is running. Use the slider to change frames (Ctrl+C to exit).")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main()
