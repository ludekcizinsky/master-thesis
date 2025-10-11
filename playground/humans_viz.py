import sys
import os
import time

import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from training.helpers.dataset import FullSceneDataset
import viser.transforms as tf
import viser
import numpy as np
import torch.nn.functional as F
from utils.smpl_deformer.smpl_server import SMPLServer


device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Helpers
def canon_to_posed(smpl_server, smpl_params, verts_c, weights_c):
    """Transform vertices from canonical to posed space using LBS.

    Args:
        smpl_server: SMPLServer instance.
        smpl_params: SMPL parameters. Shape: [1, 86].
        verts_c: Canonical vertices. Shape: [M, 3].
        weights_c: Skinning weights for canonical vertices. Shape: [M, 24].
    """

    tsf = smpl_server(smpl_params.to(device), absolute=False)["smpl_tfs"]
    x_c = verts_c
    x_c_h = F.pad(x_c, (0, 1), value=1.0)
    w = weights_c
    x_p_h = torch.einsum("pn,bnij,pj->bpi", w, tsf, x_c_h)
    x_p = x_p_h[:, :, :3] / x_p_h[:, :, 3:4]
    verts_p = x_p.cpu().numpy()

    return verts_p

def load_unidepth_pointcloud(npz_path, downsample: int = 1):
    """
    Load UniDepth fused point cloud saved as .npz.

    Args:
        npz_path (str or Path): Path to .npz file (must contain 'points' and 'colors').
        downsample (int): Keep every k-th point. Default=1 (no downsampling).

    Returns:
        pts_world (N,3) float32: 3D points in Trace world coordinates
        colors    (N,3) uint8 : Corresponding RGB values
    """
    data = np.load(npz_path)
    pts = data["points"].astype(np.float32)
    cols = data["colors"].astype(np.uint8)

    if downsample > 1:
        pts = pts[::downsample]
        cols = cols[::downsample]

    return pts, cols

# ---- Load data
preprocess_dir = Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/initial_demo")
tids = [0]  # List of tids to include
ds = FullSceneDataset(preprocess_dir=preprocess_dir, tids=tids, train_bg=False)

# SMPL canonical data
smpl_server = SMPLServer().eval()
with torch.no_grad():
    verts_c = smpl_server.verts_c[0]
    weights_c = smpl_server.weights_c[0]
    face_indices = smpl_server.faces

# ---- Viser GUI setup
server = viser.ViserServer()

# Add playback UI
num_frames = len(ds)
with server.gui.add_folder("Playback"):
    gui_timestep = server.gui.add_slider(
        "Timestep",
        min=0,
        max=len(ds) - 1,
        step=1,
        initial_value=0,
        disabled=True,
    )
    gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
    gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
    gui_playing = server.gui.add_checkbox("Playing", True)
    gui_framerate = server.gui.add_slider(
        "FPS", min=1, max=60, step=0.1, initial_value=10
    )
    gui_framerate_options = server.gui.add_button_group(
        "FPS options", ("10", "20", "30", "60")
    )
    gui_show_all_frames = server.gui.add_checkbox("Show all frames", False)
    gui_stride = server.gui.add_slider(
        "Stride",
        min=1,
        max=len(ds),
        step=1,
        initial_value=1,
        disabled=True,  # Initially disabled
    )

# Frame step buttons.
@gui_next_frame.on_click
def _(_) -> None:
    gui_timestep.value = (gui_timestep.value + 1) % num_frames

@gui_prev_frame.on_click
def _(_) -> None:
    gui_timestep.value = (gui_timestep.value - 1) % num_frames

# Disable frame controls when we're playing.
@gui_playing.on_update
def _(_) -> None:
    gui_timestep.disabled = gui_playing.value or gui_show_all_frames.value
    gui_next_frame.disabled = gui_playing.value or gui_show_all_frames.value
    gui_prev_frame.disabled = gui_playing.value or gui_show_all_frames.value

# Toggle frame visibility when the timestep slider changes.
@gui_timestep.on_update
def _(_) -> None:
    global prev_timestep
    current_timestep = gui_timestep.value
    if not gui_show_all_frames.value:
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
    prev_timestep = current_timestep
    server.flush()  # Optional!

# Show or hide all frames based on the checkbox.
@gui_show_all_frames.on_update
def _(_) -> None:
    gui_stride.disabled = not gui_show_all_frames.value  # Enable/disable stride slider
    if gui_show_all_frames.value:
        # Show frames with stride
        stride = gui_stride.value
        with server.atomic():
            for i, frame_node in enumerate(frame_nodes):
                frame_node.visible = (i % stride == 0)
        # Disable playback controls
        gui_playing.disabled = True
        gui_timestep.disabled = True
        gui_next_frame.disabled = True
        gui_prev_frame.disabled = True
    else:
        # Show only the current frame
        current_timestep = gui_timestep.value
        with server.atomic():
            for i, frame_node in enumerate(frame_nodes):
                frame_node.visible = i == current_timestep
        # Re-enable playback controls
        gui_playing.disabled = False
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

# Update frame visibility when the stride changes.
@gui_stride.on_update
def _(_) -> None:
    if gui_show_all_frames.value:
        # Update frame visibility based on new stride
        stride = gui_stride.value
        with server.atomic():
            for i, frame_node in enumerate(frame_nodes):
                frame_node.visible = (i % stride == 0)

# ---- Add data to scene
server.scene.add_frame(
    "/scene",
    wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
    position=(0, 0, 0),
    show_axes=False,
)

frame_nodes: list[viser.FrameHandle] = []
for i in range(len(ds)):
    sample = ds[i]
    smpl_params = sample["smpl_param"]
    verts_p = canon_to_posed(smpl_server, smpl_params, verts_c, weights_c)

    # Add base frame
    frame_nodes.append(server.scene.add_frame(f"/scene/f{i}", show_axes=False))

    # Add posed mesh
    for tid in tids:
        server.scene.add_mesh_simple(
            f"/scene/f{i}/mesh_p_{tid}",
            vertices=verts_p[tid],
            faces=face_indices,
            color=(100, 100, 100),
        )

    # Add camera frustum
    image = sample["image"].numpy()
    w2c = sample["M_ext"].numpy()
    wxyz = tf.SO3.from_matrix(w2c[:3, :3]).wxyz
    position = w2c[:3, 3]
    aspect = image.shape[1] / image.shape[0]
    fov = (2 * np.arctan2(image.shape[0] / 2, sample["K"][0, 0])).item()

    server.scene.add_camera_frustum(
        f"/scene/f{i}/frustum",
        fov=fov,
        aspect=aspect,
        scale=0.2,
        image=image,
        wxyz=wxyz,
        position=position,
    )

# Initialize frame visibility.
for i, frame_node in enumerate(frame_nodes):
    if gui_show_all_frames.value:
        frame_node.visible = (i % gui_stride.value == 0)
    else:
        frame_node.visible = i == gui_timestep.value

prev_timestep = gui_timestep.value
while True:
    if gui_playing.value and not gui_show_all_frames.value:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames
    time.sleep(1.0 / gui_framerate.value)
