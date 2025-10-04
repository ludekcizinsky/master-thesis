import sys
import os

import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from training.helpers.dataset import FullSceneDataset
import viser.transforms as tf
import viser
import numpy as np
import torch.nn.functional as F
from utils.smpl_deformer.smpl_server import SMPLServer

from playground.alignment import align_megasam_to_trace

device = "cuda" if torch.cuda.is_available() else "cpu"


def canon_to_posed(smpl_server, smpl_params, verts_c, weights_c):
    """Transform vertices from canonical to posed space using LBS.

    Args:
        smpl_server: SMPLServer instance.
        smpl_params: SMPL parameters. Shape: [1, 86].
        verts_c: Canonical vertices. Shape: [M, 3].
        weights_c: Skinning weights for canonical vertices. Shape: [M, 24].
    """

    # smpl_params = smpl_params.unsqueeze(0)  # [1, 86]
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

# -------- Dataset loading
preprocess_dir = Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res")
# Load Trace dataset
tids = [0, 1]  # List of tids to include
ds = FullSceneDataset(preprocess_dir=preprocess_dir, tids=tids)

# load canonical
smpl_server = SMPLServer().eval()
with torch.no_grad():
    verts_c = smpl_server.verts_c[0]
    weights_c = smpl_server.weights_c[0]
    face_indices = smpl_server.faces

# canonical -> posed
sample = ds[37]
smpl_params = sample["smpl_param"]
verts_p = canon_to_posed(smpl_server, smpl_params, verts_c, weights_c)

# -- UniDepth
path_to_cloud_scaled = preprocess_dir / "unidepth_cloud_static_scaled.npz"
pts_world_scaled, colors_scaled = ds.point_cloud

# --------- Visualize interactively with Viser
server = viser.ViserServer()

# Add world frame
server.scene.add_frame(
    "/scene",
    wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
    position=(0, 0, 0),
    show_axes=False,
)

# --- Static
server.scene.add_point_cloud(
    "/scene/point_cloud_scaled", 
    points=pts_world_scaled,
    colors=colors_scaled,
    point_size=0.03,
    point_shape="rounded"
)

# --- Dynamic
# Add posed mesh
for tid in tids:
    server.scene.add_mesh_simple(
        f"/scene/mesh_p_{tid}", 
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
    f"/scene/frustum",
    fov=fov,
    aspect=aspect,
    scale=0.2,
    image=image,
    wxyz=wxyz,
    position=position,
)


print("Viser server running at http://localhost:8080")

# Keep the server running
import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Server stopped.")