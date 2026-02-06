from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import torch
import tyro
import viser
from pytorch3d.transforms import matrix_to_axis_angle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from submodules.smplx import smplx


@dataclass
class Config:
    scene_dir: Path
    frame_idx: int = 0
    port: int = 8080
    device: str = "cuda"
    model_folder: Path = Path("/home/cizinsky/body_models")
    gender: str = "neutral"
    smplx_model_ext: str = "npz"


def _load_track_npzs(root: Path) -> Dict[int, Dict[str, np.ndarray]]:
    tracks: Dict[int, Dict[str, np.ndarray]] = {}
    if not root.exists():
        return tracks
    for npz_path in sorted(root.glob("track_*.npz")):
        try:
            tid = int(npz_path.stem.split("_")[-1])
        except ValueError:
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            tracks[tid] = {k: np.asarray(data[k]) for k in data.files}
    return tracks


def _split_smplx_pose(pose_aa: np.ndarray):
    root_pose = pose_aa[:, 0:3]
    body_pose = pose_aa[:, 3:66].reshape(-1, 21, 3)
    jaw_pose = pose_aa[:, 66:69]
    leye_pose = pose_aa[:, 69:72]
    reye_pose = pose_aa[:, 72:75]
    lhand_pose = pose_aa[:, 75:120].reshape(-1, 15, 3)
    rhand_pose = pose_aa[:, 120:165].reshape(-1, 15, 3)
    return root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose


def _pad_pose_aa(pose_aa: np.ndarray) -> np.ndarray:
    if pose_aa.shape[-1] >= 165:
        return pose_aa[..., :165]
    pad = np.zeros((*pose_aa.shape[:-1], 165 - pose_aa.shape[-1]), dtype=pose_aa.dtype)
    return np.concatenate([pose_aa, pad], axis=-1)


def _pose_from_track(track: Dict[str, np.ndarray], row_idx: int) -> Optional[np.ndarray]:
    if "pose" in track:
        pose = np.asarray(track["pose"])
        if pose.ndim == 1:
            pose = pose[None, :]
        return _pad_pose_aa(pose[row_idx : row_idx + 1])
    if "rotmat" in track:
        rotmat = np.asarray(track["rotmat"])
        if rotmat.ndim == 3:
            rotmat = rotmat[None, ...]
        aa = matrix_to_axis_angle(torch.from_numpy(rotmat[row_idx : row_idx + 1]).float())
        aa = aa.reshape(1, -1).cpu().numpy()
        return _pad_pose_aa(aa)
    return None


def _mesh_for_frame(
    layer,
    track: Dict[str, np.ndarray],
    frame_id: int,
    device: torch.device,
) -> Optional[np.ndarray]:
    if "frames" not in track or "shape" not in track or "trans" not in track:
        return None

    frames = np.asarray(track["frames"]).astype(int)
    row = np.where(frames == frame_id)[0]
    if row.size == 0:
        return None
    idx = int(row[0])

    pose = _pose_from_track(track, idx)
    if pose is None:
        return None

    betas = np.asarray(track["shape"])
    trans = np.asarray(track["trans"])
    if betas.ndim == 1:
        betas = betas[None, :]
    if trans.ndim == 1:
        trans = trans[None, :]

    betas_t = torch.tensor(betas[idx : idx + 1], dtype=torch.float32, device=device)
    expected_betas = int(getattr(layer, "num_betas", betas_t.shape[-1]))
    if betas_t.shape[-1] != expected_betas:
        if betas_t.shape[-1] > expected_betas:
            betas_t = betas_t[..., :expected_betas]
        else:
            pad = torch.zeros((1, expected_betas - betas_t.shape[-1]), device=device, dtype=betas_t.dtype)
            betas_t = torch.cat([betas_t, pad], dim=-1)

    expr_dim = int(getattr(layer, "num_expression_coeffs", 10))
    expression = torch.zeros((1, expr_dim), dtype=torch.float32, device=device)

    root_pose, body_pose, jaw_pose, leye_pose, reye_pose, lhand_pose, rhand_pose = _split_smplx_pose(pose)

    with torch.no_grad():
        out = layer(
            global_orient=torch.tensor(root_pose, dtype=torch.float32, device=device),
            body_pose=torch.tensor(body_pose.reshape(1, -1), dtype=torch.float32, device=device),
            jaw_pose=torch.tensor(jaw_pose, dtype=torch.float32, device=device),
            leye_pose=torch.tensor(leye_pose, dtype=torch.float32, device=device),
            reye_pose=torch.tensor(reye_pose, dtype=torch.float32, device=device),
            left_hand_pose=torch.tensor(lhand_pose.reshape(1, -1), dtype=torch.float32, device=device),
            right_hand_pose=torch.tensor(rhand_pose.reshape(1, -1), dtype=torch.float32, device=device),
            betas=betas_t,
            transl=torch.tensor(trans[idx : idx + 1], dtype=torch.float32, device=device),
            expression=expression,
        )
    return out.vertices[0].detach().cpu().numpy().astype(np.float32)


def _frame_ids(camera_tracks: Dict[int, Dict[str, np.ndarray]], world_tracks: Dict[int, Dict[str, np.ndarray]]) -> List[int]:
    ids = set()
    for tracks in (camera_tracks, world_tracks):
        for t in tracks.values():
            if "frames" in t:
                ids.update(np.asarray(t["frames"]).astype(int).tolist())
    return sorted(ids)


def main() -> None:
    cfg = tyro.cli(Config)
    device = torch.device(cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu")

    camera_root = cfg.scene_dir / "misc" / "prompthmr" / "camera_smplx"
    world_root = cfg.scene_dir / "misc" / "prompthmr" / "world_smplx"
    camera_tracks = _load_track_npzs(camera_root)
    world_tracks = _load_track_npzs(world_root)

    if not camera_tracks and not world_tracks:
        raise FileNotFoundError(
            f"No SMPL-X snapshots found. Expected at least one of: {camera_root}, {world_root}"
        )

    frame_ids = _frame_ids(camera_tracks, world_tracks)
    if not frame_ids:
        raise RuntimeError("No frame ids found in camera/world SMPL-X snapshots.")

    layer = smplx.create(
        str(cfg.model_folder),
        model_type="smplx",
        gender=cfg.gender,
        ext=cfg.smplx_model_ext,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
    ).to(device)
    faces = np.asarray(layer.faces, dtype=np.uint32)

    server = viser.ViserServer(port=cfg.port)
    server.scene.set_up_direction("+y")

    with server.gui.add_folder("Frames"):
        init_idx = min(max(cfg.frame_idx, 0), len(frame_ids) - 1)
        slider = server.gui.add_slider("Frame", min=0, max=len(frame_ids) - 1, step=1, initial_value=init_idx)
        label = server.gui.add_text("Frame id", str(frame_ids[init_idx]))

    with server.gui.add_folder("Visibility"):
        show_camera = server.gui.add_checkbox("Show camera_smplx", True)
        show_world = server.gui.add_checkbox("Show world_smplx", True)

    camera_nodes = []
    world_nodes = []
    camera_colors = [(80, 160, 255), (120, 190, 255), (160, 220, 255), (90, 140, 220)]
    world_colors = [(255, 150, 80), (255, 180, 120), (255, 210, 160), (220, 130, 90)]

    for frame_id in frame_ids:
        node_cam = server.scene.add_frame(f"/camera_smplx/f_{frame_id}", show_axes=False)
        node_world = server.scene.add_frame(f"/world_smplx/f_{frame_id}", show_axes=False)
        camera_nodes.append(node_cam)
        world_nodes.append(node_world)

        for tid, track in sorted(camera_tracks.items()):
            verts = _mesh_for_frame(layer, track, frame_id, device)
            if verts is None:
                continue
            color = camera_colors[tid % len(camera_colors)]
            server.scene.add_mesh_simple(
                f"/camera_smplx/f_{frame_id}/track_{tid}",
                vertices=verts,
                faces=faces,
                color=color,
            )

        for tid, track in sorted(world_tracks.items()):
            verts = _mesh_for_frame(layer, track, frame_id, device)
            if verts is None:
                continue
            color = world_colors[tid % len(world_colors)]
            server.scene.add_mesh_simple(
                f"/world_smplx/f_{frame_id}/track_{tid}",
                vertices=verts,
                faces=faces,
                color=color,
            )

        node_cam.visible = False
        node_world.visible = False

    current_idx = int(slider.value)

    def _set_visible(idx: int) -> None:
        nonlocal current_idx
        if idx == current_idx:
            return
        camera_nodes[current_idx].visible = False
        world_nodes[current_idx].visible = False
        current_idx = idx
        camera_nodes[current_idx].visible = show_camera.value
        world_nodes[current_idx].visible = show_world.value
        label.value = str(frame_ids[current_idx])

    camera_nodes[current_idx].visible = show_camera.value
    world_nodes[current_idx].visible = show_world.value

    @slider.on_update
    def _(_event) -> None:
        _set_visible(int(slider.value))

    @show_camera.on_update
    def _(_event) -> None:
        camera_nodes[current_idx].visible = show_camera.value

    @show_world.on_update
    def _(_event) -> None:
        world_nodes[current_idx].visible = show_world.value

    print("Viser server is running. Use the slider to inspect camera/world SMPL-X.")
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main()
