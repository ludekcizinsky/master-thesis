import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import math
from pathlib import Path
from training.helpers.dataset import TraceDataset

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.manual_seed(1000)

import viser
import numpy as np
import viser.transforms as tf

from utils.smpl_deformer.smpl_server import SMPLServer

from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def canon_to_posed(smpl_server, smpl_params, verts_c, weights_c):
    """Transform vertices from canonical to posed space using LBS.

    Args:
        smpl_server: SMPLServer instance.
        smpl_params: SMPL parameters. Shape: [1, 86].
        verts_c: Canonical vertices. Shape: [M, 3].
        weights_c: Skinning weights for canonical vertices. Shape: [M, 24].
    """

    smpl_params = smpl_params.unsqueeze(0)  # [1, 86]
    tsf = smpl_server(smpl_params.to(device), absolute=False)["smpl_tfs"][0]
    x_c = verts_c
    x_c_h = F.pad(x_c, (0, 1), value=1.0)
    w = weights_c
    x_p_h = torch.einsum("pn,nij,pj->pi", w, tsf, x_c_h)
    x_p = x_p_h[:, :3] / x_p_h[:, 3:4]
    verts_p = x_p.cpu().numpy()

    return verts_p


# dataset
preprocess_dir = Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res")
tid = 0
downscale = 1
ds = TraceDataset(preprocess_dir, tid=tid, downscale=downscale)


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

# ---- Viser visualization
server = viser.ViserServer()

# Add world frame
server.scene.add_frame(
    "/scene",
    wxyz=tf.SO3.exp(np.array([np.pi / 2, 0.0, 0.0])).wxyz,
    position=(0, 0, 0),
    show_axes=False,
)

# Add posed mesh
server.scene.add_mesh_simple(
    "/scene/mesh_p", 
    vertices=verts_p.astype(np.float32),
    faces=face_indices.astype(np.int32),
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
