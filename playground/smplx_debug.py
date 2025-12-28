import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
import viser.transforms as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.helpers.dataset import SceneDataset
from submodules.smplx import smplx


@dataclass
class DebugConfig:
    hi4d_scene_dir: Path = Path("/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17")
    human3r_scene_dir: Path = Path("/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance")
    src_cam_id: int = 28
    device: str = "cuda"
    selected_frame_idx: int = 0
    smplx_model_path: Path = Path("/scratch/izar/cizinsky/pretrained/pretrained_models/human_model_files/smplx/SMPLX_NEUTRAL.npz")
    port: int = 8080
    initial_view: str = "human3r"  # hi4d | human3r | none
    camera_backoff: float = 0.5


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


def add_scene(
    server,
    root_name: str,
    image: torch.Tensor,
    K: torch.Tensor,
    c2w: torch.Tensor,
    smplx_verts: torch.Tensor,
    smplx_faces: np.ndarray,
    color_offset: int = 0,
):
    root = server.scene.add_frame(f"/scene/{root_name}", show_axes=True)
    add_camera_frustum(server, f"/scene/{root_name}/camera", image, K, c2w)
    for pid in range(smplx_verts.shape[0]):
        color = get_color(pid + color_offset).tolist()
        server.scene.add_mesh_simple(
            f"/scene/{root_name}/person_{pid}",
            vertices=smplx_verts[pid].detach().cpu().numpy(),
            faces=smplx_faces,
            color=tuple(int(c) for c in color),
        )
    return root


if __name__ == "__main__":
    cfg = tyro.cli(DebugConfig)
    device = torch.device(cfg.device)

    hi4d_ds = SceneDataset(cfg.hi4d_scene_dir, src_cam_id=cfg.src_cam_id, device=device)
    human3r_ds = SceneDataset(cfg.human3r_scene_dir, src_cam_id=cfg.src_cam_id, device=device)

    hi4d_sample = hi4d_ds[cfg.selected_frame_idx]
    human3r_sample = human3r_ds[cfg.selected_frame_idx]

    hi4d_smplx = select_frame_params(hi4d_sample["smplx_params"], cfg.selected_frame_idx)
    human3r_smplx = select_frame_params(human3r_sample["smplx_params"], cfg.selected_frame_idx)

    model_root = resolve_human_model_root(cfg.smplx_model_path)
    hi4d_layer = build_smplx_layer(
        model_root, hi4d_smplx["betas"].shape[-1], hi4d_smplx["expr"].shape[-1], device
    )
    human3r_layer = build_smplx_layer(
        model_root, human3r_smplx["betas"].shape[-1], human3r_smplx["expr"].shape[-1], device
    )

    hi4d_verts = smplx_vertices(hi4d_layer, hi4d_smplx)
    human3r_verts = smplx_vertices(human3r_layer, human3r_smplx)
    smplx_faces = hi4d_layer.faces.astype(np.int32)

    server = viser.ViserServer(port=cfg.port)

    hi4d_root = add_scene(
        server,
        "hi4d",
        hi4d_sample["image"],
        hi4d_sample["K"][:3, :3],
        hi4d_sample["c2w"],
        hi4d_verts,
        smplx_faces,
        color_offset=0,
    )
    human3r_root = add_scene(
        server,
        "human3r",
        human3r_sample["image"],
        human3r_sample["K"][:3, :3],
        human3r_sample["c2w"],
        human3r_verts,
        smplx_faces,
        color_offset=10,
    )

    initial_choice = cfg.initial_view.lower()
    initial_map = {
        "hi4d": (hi4d_sample["image"], hi4d_sample["K"][:3, :3], hi4d_sample["c2w"]),
        "human3r": (human3r_sample["image"], human3r_sample["K"][:3, :3], human3r_sample["c2w"]),
    }
    if initial_choice in initial_map:
        image, K, c2w = initial_map[initial_choice]

        def _set_camera(client: viser.ClientHandle) -> None:
            set_initial_camera(client, image, K, c2w, cfg.camera_backoff)

        server.on_client_connect(_set_camera)

    with server.gui.add_folder("Visibility"):
        gui_show_hi4d = server.gui.add_checkbox("Show GT", True)
        gui_show_human3r = server.gui.add_checkbox("Show Human3R", True)

    @gui_show_hi4d.on_update
    def _(_) -> None:
        hi4d_root.visible = gui_show_hi4d.value

    @gui_show_human3r.on_update
    def _(_) -> None:
        human3r_root.visible = gui_show_human3r.value

    print("Viser server is running. Toggle visibility in the GUI (Ctrl+C to exit).")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer...")
