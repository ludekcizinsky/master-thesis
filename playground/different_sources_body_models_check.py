import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import tyro
import viser
import viser.transforms as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.helpers.dataset import SceneDataset, root_dir_to_skip_frames_path
from submodules.smplx import smplx


@dataclass
class DebugConfig:
    source_a_name: str = "estimated"
    source_a_scene_dir: Path = Path("/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17")
    source_b_name: str = "ground_truth"
    source_b_scene_dir: Path = Path("/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance")
    src_cam_id: int = 4
    device: str = "cuda"
    smplx_model_path: Path = Path("/scratch/izar/cizinsky/pretrained/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz")
    port: int = 8080
    initial_view: str = "estimated"  # source_a_name | source_b_name | none
    camera_backoff: float = 0.5

def load_skip_frames(scene_dir: Path) -> List[int]:
    """
    Load skip frames from skip_frames.csv in the scene directory.

    Note: the frame indicies are actual frame indexes and the frames dir may
    not always start with frame 0. Therefore, the returnd indices correspond to
    the actual frame indices to skip. e.g. if we have frames 10, 11, 12, 13 and skip_frames.csv contains "11,13",
    we will skip frames 11 and 13.

    Args:
        scene_dir: Path to the scene directory.
    Returns:
        List of frame indices to skip.
    """

    skip_frames_file = root_dir_to_skip_frames_path(scene_dir)
    if not skip_frames_file.exists():
        return []
    with open(skip_frames_file, "r") as f:
        line = f.readline().strip()
        skip_frames = [int(idx_str) for idx_str in line.split(",") if idx_str.isdigit()]
    return skip_frames


def resolve_human_model_root(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path.parent.parent
    return model_path


def select_frame_params(params: dict, frame_idx: int) -> dict:
    out = {}
    for key, value in params.items():
        if key == "expr" and value.dim() == 3:
            out[key] = value[:, 0]
        else:
            out[key] = value
    return out


def get_color(idx: int) -> np.ndarray:
    palette = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 128, 0],
            [128, 0, 255],
            [0, 255, 255],
            [255, 0, 255],
            [128, 128, 128],
            [0, 128, 255],
        ],
        dtype=np.int32,
    )
    return palette[idx % len(palette)]


def build_smplx_layer(model_root: Path, num_betas: int, num_expr: int, device: torch.device):
    layer = smplx.create(
        str(model_root),
        "smplx",
        gender="neutral",
        num_betas=num_betas,
        num_expression_coeffs=num_expr,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
    )
    return layer.to(device)


def smplx_vertices(smplx_layer, smplx_params: dict) -> torch.Tensor:
    params = {
        "global_orient": smplx_params["root_pose"],
        "body_pose": smplx_params["body_pose"],
        "jaw_pose": smplx_params["jaw_pose"],
        "leye_pose": smplx_params["leye_pose"],
        "reye_pose": smplx_params["reye_pose"],
        "left_hand_pose": smplx_params["lhand_pose"],
        "right_hand_pose": smplx_params["rhand_pose"],
        "betas": smplx_params["betas"],
        "transl": smplx_params["trans"],
        "expression": smplx_params["expr"],
    }
    with torch.no_grad():
        output = smplx_layer(**params)
    return output.vertices


def add_camera_frustum(server, name: str, image: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor):
    image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    K_np = K.detach().cpu().numpy()
    c2w_np = c2w.detach().cpu().numpy()

    H, W = image_np.shape[:2]
    fov = 2.0 * np.arctan2(H / 2.0, K_np[1, 1])
    aspect = W / float(H)

    wxyz = tf.SO3.from_matrix(c2w_np[:3, :3]).wxyz
    position = c2w_np[:3, 3]

    server.scene.add_camera_frustum(
        name=name,
        fov=fov,
        aspect=aspect,
        scale=0.2,
        wxyz=wxyz,
        position=position,
        image=image_np,
    )


def set_initial_camera(client, image: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor, backoff: float):
    image_np = image.detach().cpu().numpy()
    K_np = K.detach().cpu().numpy()
    c2w_np = c2w.detach().cpu().numpy()

    H, W = image_np.shape[:2]
    fov = 2.0 * np.arctan2(H / 2.0, K_np[1, 1])
    aspect = W / float(H)

    wxyz = tf.SO3.from_matrix(c2w_np[:3, :3]).wxyz
    position = c2w_np[:3, 3].copy()
    if backoff != 0.0:
        forward = c2w_np[:3, 2]
        position = position - backoff * forward

    client.camera.wxyz = wxyz
    client.camera.position = position
    client.camera.fov = fov


def update_meshes(
    server: viser.ViserServer,
    root_name: str,
    smplx_verts: torch.Tensor,
    smplx_faces: np.ndarray,
    handles: list,
    color: np.ndarray,
) -> None:
    color_tuple = tuple(int(c) for c in color.tolist())
    for pid in range(smplx_verts.shape[0]):
        name = f"/scene/{root_name}/person_{pid}"
        handle = handles[pid] if pid < len(handles) else None
        vertices = smplx_verts[pid].detach().cpu().numpy()
        if handle is not None and hasattr(handle, "update"):
            handle.update(vertices=vertices, faces=smplx_faces, color=color_tuple)
        else:
            handle = server.scene.add_mesh_simple(
                name,
                vertices=vertices,
                faces=smplx_faces,
                color=color_tuple,
            )
            if pid < len(handles):
                handles[pid] = handle
            else:
                handles.append(handle)
        handles[pid].visible = True
    for pid in range(smplx_verts.shape[0], len(handles)):
        handles[pid].visible = False


def update_camera_frustum(
    server: viser.ViserServer,
    handle,
    name: str,
    image: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
):
    image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    K_np = K.detach().cpu().numpy()
    c2w_np = c2w.detach().cpu().numpy()

    H, W = image_np.shape[:2]
    fov = 2.0 * np.arctan2(H / 2.0, K_np[1, 1])
    aspect = W / float(H)

    wxyz = tf.SO3.from_matrix(c2w_np[:3, :3]).wxyz
    position = c2w_np[:3, 3]

    if handle is None or not hasattr(handle, "_impl") or handle._impl.removed:
        return server.scene.add_camera_frustum(
            name=name,
            fov=fov,
            aspect=aspect,
            scale=0.2,
            wxyz=wxyz,
            position=position,
            image=image_np,
        )

    handle.image = image_np
    handle.fov = fov
    handle.aspect = aspect
    handle.scale = 0.2
    handle.wxyz = wxyz
    handle.position = position
    return handle


def add_scene(
    server,
    root_name: str,
    image: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    smplx_verts: torch.Tensor,
    smplx_faces: np.ndarray,
    color: np.ndarray,
):
    root = server.scene.add_frame(f"/scene/{root_name}", show_axes=True)
    add_camera_frustum(server, f"/scene/{root_name}/camera", image, K, c2w)
    color_tuple = tuple(int(c) for c in color.tolist())
    for pid in range(smplx_verts.shape[0]):
        server.scene.add_mesh_simple(
            f"/scene/{root_name}/person_{pid}",
            vertices=smplx_verts[pid].detach().cpu().numpy(),
            faces=smplx_faces,
            color=color_tuple,
        )
    return root


if __name__ == "__main__":
    cfg = tyro.cli(DebugConfig)
    device = torch.device(cfg.device)

    skip_frames = load_skip_frames(cfg.source_a_scene_dir)
    source_a_ds = SceneDataset(cfg.source_a_scene_dir, src_cam_id=cfg.src_cam_id, device=device, skip_frames=skip_frames)
    source_b_ds = SceneDataset(cfg.source_b_scene_dir, src_cam_id=cfg.src_cam_id, device=device, skip_frames=skip_frames)

    source_a_len = len(source_a_ds)
    source_b_len = len(source_b_ds)
    if source_a_len == 0 or source_b_len == 0:
        raise RuntimeError("Both sources must contain at least one frame.")
    if source_a_len != source_b_len:
        print(
            f"Warning: source lengths differ ({source_a_len} vs {source_b_len}); "
            "using the shorter length for the slider."
        )
    max_frames = min(source_a_len, source_b_len)

    source_a_sample = source_a_ds[0]
    source_b_sample = source_b_ds[0]

    source_a_smplx = select_frame_params(source_a_sample["smplx_params"], 0)
    source_b_smplx = select_frame_params(source_b_sample["smplx_params"], 0)

    model_root = resolve_human_model_root(cfg.smplx_model_path)
    source_a_layer = build_smplx_layer(
        model_root, source_a_smplx["betas"].shape[-1], source_a_smplx["expr"].shape[-1], device
    )
    source_b_layer = build_smplx_layer(
        model_root, source_b_smplx["betas"].shape[-1], source_b_smplx["expr"].shape[-1], device
    )

    smplx_faces = source_a_layer.faces.astype(np.int32)

    server = viser.ViserServer(port=cfg.port)

    source_a_root = server.scene.add_frame(f"/scene/{cfg.source_a_name}", show_axes=True)
    source_b_root = server.scene.add_frame(f"/scene/{cfg.source_b_name}", show_axes=True)

    source_a_handles = []
    source_b_handles = []
    camera_handles = {"a": None, "b": None}
    current_initial = {}

    initial_choice = cfg.initial_view.lower().strip()

    def _set_camera(client: viser.ClientHandle) -> None:
        if initial_choice in current_initial:
            image, K, c2w = current_initial[initial_choice]
            set_initial_camera(client, image, K, c2w, cfg.camera_backoff)

    server.on_client_connect(_set_camera)

    with server.gui.add_folder("Visibility"):
        gui_show_source_a = server.gui.add_checkbox(f"Show {cfg.source_a_name}", True)
        gui_show_source_b = server.gui.add_checkbox(f"Show {cfg.source_b_name}", True)

    @gui_show_source_a.on_update
    def _(_) -> None:
        source_a_root.visible = gui_show_source_a.value

    @gui_show_source_b.on_update
    def _(_) -> None:
        source_b_root.visible = gui_show_source_b.value

    with server.gui.add_folder("Frames"):
        frame_slider = server.gui.add_slider(
            "Frame Index",
            min=0,
            max=max_frames - 1,
            step=1,
            initial_value=0,
        )

    def render_frame(frame_idx: int) -> None:
        source_a_sample = source_a_ds[frame_idx]
        source_b_sample = source_b_ds[frame_idx]

        source_a_smplx = select_frame_params(source_a_sample["smplx_params"], frame_idx)
        source_b_smplx = select_frame_params(source_b_sample["smplx_params"], frame_idx)

        source_a_verts = smplx_vertices(source_a_layer, source_a_smplx)
        source_b_verts = smplx_vertices(source_b_layer, source_b_smplx)

        source_a_color = get_color(0)
        source_b_color = get_color(1)

        update_meshes(
            server,
            cfg.source_a_name,
            source_a_verts,
            smplx_faces,
            source_a_handles,
            source_a_color,
        )
        update_meshes(
            server,
            cfg.source_b_name,
            source_b_verts,
            smplx_faces,
            source_b_handles,
            source_b_color,
        )

        camera_handles["a"] = update_camera_frustum(
            server,
            camera_handles["a"],
            f"/scene/{cfg.source_a_name}/camera",
            source_a_sample["image"],
            source_a_sample["K"][:3, :3],
            source_a_sample["c2w"],
        )
        camera_handles["b"] = update_camera_frustum(
            server,
            camera_handles["b"],
            f"/scene/{cfg.source_b_name}/camera",
            source_b_sample["image"],
            source_b_sample["K"][:3, :3],
            source_b_sample["c2w"],
        )

        current_initial.clear()
        current_initial[cfg.source_a_name.lower()] = (
            source_a_sample["image"],
            source_a_sample["K"][:3, :3],
            source_a_sample["c2w"],
        )
        current_initial[cfg.source_b_name.lower()] = (
            source_b_sample["image"],
            source_b_sample["K"][:3, :3],
            source_b_sample["c2w"],
        )

    @frame_slider.on_update
    def _(_) -> None:
        render_frame(int(frame_slider.value))

    render_frame(0)

    print("Viser server is running. Toggle visibility in the GUI (Ctrl+C to exit).")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer...")
