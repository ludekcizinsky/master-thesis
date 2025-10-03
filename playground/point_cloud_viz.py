import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from training.helpers.dataset import MegaSAMDataset
import viser.transforms as tf
import viser
import numpy as np
import matplotlib.pyplot as plt

preprocess_dir = Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res")
npz_path = str(preprocess_dir / "megasam" / "sgd_cvd_hr.npz")
ds = MegaSAMDataset(
    npz_path,
    normalize_images=True,                # images in [0,1]
    device="cpu",
)



# 2) Build static world point cloud (+ colors)
pixel_downsample = 10  # downsample image by this factor when building point cloud
pts_world, colors = ds.build_static_point_cloud(every_k=1, downsample=pixel_downsample, device="cpu")
pts_world, colors = pts_world.numpy().astype(np.float32), (colors.numpy() * 255).astype(np.uint8)  # to [0,255]

diag = np.linalg.norm(pts_world.max(0) - pts_world.min(0))
target_diag = 100.0  # pick something comfy for navigation
s = target_diag / max(diag, 1e-6)
pts_vis_world = pts_world * s

# 3) Visualize interactively with Viser
# Add point cloud (points in camera coords, camera at origin)
server = viser.ViserServer()
server.scene.set_up_direction('-z')

# Add world frame
server.scene.add_frame(
    "/scene",
    wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
    position=(0, 0, 0),
    show_axes=False,
)

# Add point cloud

server.scene.add_point_cloud(
    "/scene/point_cloud", 
    points=pts_world,
    colors=colors,
    point_size=0.001,
    point_shape="rounded"
)

# Add camera frustum
middle_frame_idx = len(ds) // 2
frame = ds[middle_frame_idx]
fov = (2 * np.arctan2(frame.H / 2, frame.K[0, 0])).item()
aspect = frame.W / frame.H
image = frame.image.numpy()
wxyz = tf.SO3.from_matrix(frame.w2c[:3, :3].numpy()).wxyz
position = frame.w2c[:3, 3].numpy()

server.scene.add_camera_frustum(
    f"/scene/frustum",
    fov=fov,
    aspect=aspect,
    scale=0.02,
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