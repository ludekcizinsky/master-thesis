import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import math
from pathlib import Path
from training.helpers.dataset import HumanOnlyDataset, TraceDataset
from training.helpers.trainer_init import lbs_apply

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.manual_seed(1000)

from gsplat import rasterization

from utils.smpl_deformer.smpl_server import SMPLServer

from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

# --- Dataset and DataLoader
# preprocess_dir = Path("/scratch/izar/cizinsky/thesis/output/modric_vs_ribberi/preprocess")
preprocess_dir = Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res")
tid = 0
downscale = 1

# ds = HumanOnlyDataset(preprocess_dir, tid=tid, split='train', downscale=downscale)
ds = TraceDataset(preprocess_dir, tid=tid, downscale=downscale)
dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

smpl_server = SMPLServer().to(device).eval()
with torch.no_grad():
    verts_c = smpl_server.verts_c[0].to(device)      # [6890,3]
    weights_c = smpl_server.weights_c[0].to(device)  # [6890,24]

M = verts_c.shape[0]

# --- Splats (the model)
# means 
smpl_vertices = verts_c.clone()                     # [M,3]

# anisotropic per-axis log-scales
init_sigma = 0.02
log_scales = torch.full((M, 3), math.log(init_sigma), device=device) # [M,3]

# rotations as quaternions [w,x,y,z], init to identity
quats_init = torch.zeros(M, 4, device=device)
quats_init[:, 0] = 1.0

# opacities
colors = torch.rand(M, 3, device=device)      # [M,3]
init_opacity = 0.1
opacity_logit = torch.full((M,), torch.logit(torch.tensor(init_opacity, device=device)))

params = [
    # name, value, lr
    ("means", torch.nn.Parameter(smpl_vertices)),
    ("scales", torch.nn.Parameter(log_scales)),
    ("quats", torch.nn.Parameter(quats_init)),
    ("opacities", torch.nn.Parameter(opacity_logit)),
]


# color is SH coefficients
sh_degree = 3
colors = torch.zeros((M, (sh_degree + 1) ** 2, 3), device=device)  # [M, K, 3]
colors[:, 0, :] = rgb_to_sh(torch.rand(M, 3))  # [M,3]
params.append(("sh0", torch.nn.Parameter(colors[:, :1, :])))
params.append(("shN", torch.nn.Parameter(colors[:, 1:, :])))

splats = torch.nn.ParameterDict({n: v for n, v in params}).to(device)

# --- Rendering setup
batch = next(iter(dl))
images = batch["image"].to(device)  # [B,H,W,3]
masks  = batch["mask"].to(device)   # [B,H,W]
K  = batch["K"].to(device) + 0      # [B, 3,3]
if "M_ext" in batch:
    M_ext = batch["M_ext"]
else:
    M_ext = torch.eye(4).unsqueeze(0).repeat(images.shape[0], 1, 1)  # [B,4,4]
smpl_param = batch["smpl_param"].to(device)  # [B,86]
H, W = images.shape[1:3]
print(f"K: {K[0]}")
print(f"M_ext: {M_ext[0]}")


# --- Forward pass (rendering)
tsf = smpl_server(smpl_param, absolute=False)["smpl_tfs"][0].to(device)  # [24,4,4]
x_c = splats["means"]
x_c_h = F.pad(x_c, (0, 1), value=1.0)
w = weights_c
x_p_h = torch.einsum("pn,nij,pj->pi", w, tsf, x_c_h)
means = x_p_h[:, :3] / x_p_h[:, 3:4]

# - Quats
q = splats["quats"]
quats = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
# - Scales
s = torch.exp(splats["scales"])
min_sigma = 1e-4
max_sigma = 1.0
scales = s.clamp(min_sigma, max_sigma)
# - Colours
colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]
# - Opacity
opacity = torch.sigmoid(splats["opacities"])

# Define cameras
dtype = torch.float32
viewmats = M_ext.to(device, dtype).contiguous()  # [1,4,4]
Ks = K.to(device, dtype).contiguous()                  # [1,3,3]

# Render
_, alphas, _ = rasterization(
    means, quats, scales, opacity, colors, viewmats, Ks, W, H, sh_degree=sh_degree, packed=False
)

# --- visualization
img_arr = images[0].cpu().numpy()
proj_arr = alphas[0].detach().cpu().numpy()
binary_mask_arr = (proj_arr > 0.75).astype(float) # note that higher the tr, the tigher the mask
masked_img = img_arr * binary_mask_arr

fig, ax = plt.subplots(1, 3, figsize=(15, 10))
ax[0].imshow(img_arr)
ax[0].set_title("Image")
ax[0].axis("off")

ax[1].imshow(proj_arr)
ax[1].set_title("Render")
ax[1].axis("off")

ax[2].imshow(masked_img)
ax[2].set_title("Masked Image")
ax[2].axis("off")

# save to disk
os.makedirs("playground/outputs", exist_ok=True)
plt.savefig(f"playground/outputs/render_trace_tid{tid:02d}.png")