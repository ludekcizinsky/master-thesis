import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
from typing import Optional, Tuple, Dict

import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

from pathlib import Path
import torch
from torch import nn
torch.manual_seed(1000)

from gsplat import rasterization
from matplotlib import pyplot as plt

from utils.smpl_deformer.smpl_server import SMPLServer
from training.helpers.dataset import FullSceneDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- utilities ----------

C0 = 0.28209479177387814

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)

def set_dc_from_rgb(splats, rgb_01, linearize=True):
    if linearize:
        rgb_01 = srgb_to_linear(rgb_01.clamp(0,1))
    dc = rgb_01 / C0
    # splats["sh0"]: [N,1,3]
    splats["sh0"].data.copy_(dc.unsqueeze(1))
    # keep higher orders zero; training will learn them

def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Map linear RGB in [0,1] to SH DC (l=0) coefficient convention (rgb-0.5)/C0."""
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)

def make_paramdict(means, log_scales, quats, opacity_logits, colors_sh, sh_degree) -> nn.ParameterDict:
    K = (sh_degree + 1) ** 2
    assert colors_sh.shape[1] == K and colors_sh.shape[2] == 3
    return nn.ParameterDict({
        "means":     nn.Parameter(means.contiguous()),
        "scales":    nn.Parameter(log_scales.contiguous()),   # log-scales
        "quats":     nn.Parameter(quats.contiguous()),
        "opacities": nn.Parameter(opacity_logits.contiguous()),
        "sh0":       nn.Parameter(colors_sh[:, :1, :].contiguous()),
        "shN":       nn.Parameter(colors_sh[:, 1:, :].contiguous()),
    })


def unit_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def prep_splats(splats: torch.nn.ParameterDict, clamp_sigma: tuple = (1e-4, 1.0), dtype: torch.dtype = torch.float32, device: str = "cuda") -> Tuple[torch.Tensor]:
    means = splats["means"].to(device, dtype)
    quats = unit_quat(splats["quats"].to(device, dtype))
    scales = torch.exp(splats["scales"].to(device, dtype)).clamp(*clamp_sigma)
    opacity = torch.sigmoid(splats["opacities"].to(device, dtype))
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device, dtype)

    return dict(
        means=means, quats=quats, scales=scales,
        opacity=opacity, colors=colors
    )

def init_sh_colors(color_mode: str, M: int, sh_degree: int = 3, given_colors: Optional[torch.Tensor] = None, device: str = "cuda") -> torch.Tensor:

    K = (sh_degree + 1) ** 2
    colors_sh = torch.zeros(M, K, 3, device=device)

    if color_mode == "random":
        base_rgb = torch.rand(M, 3, device=device)                  # sRGB in [0,1]
        base_rgb = srgb_to_linear(base_rgb)                         # -> linear
    elif color_mode == "gray":
        base_rgb = torch.full((M, 3), 0.5, device=device)           # sRGB mid-gray
        base_rgb = srgb_to_linear(base_rgb)                         # ≈ 0.214 in linear
    elif color_mode == "given":
        assert given_colors is not None and given_colors.shape == (M, 3)
        base_rgb = srgb_to_linear(given_colors.to(device).clamp(0, 1))
    else:
        raise ValueError("color_mode must be 'random'|'gray'|'given'")

    colors_sh[:, 0, :] = base_rgb / C0

    return colors_sh    


def init_isotropic_scales_from_knn(pts: np.array, k_for_scale: int = 16, scale_percentile: float = 50.0, device: str = "cuda") -> torch.Tensor:
    """Initialize isotropic scales from k-NN distances."""
    nbrs = NearestNeighbors(n_neighbors=k_for_scale + 1, algorithm='kd_tree').fit(pts)
    distances, _ = nbrs.kneighbors(pts)  # [N, k+1]

    # skip the first column (self-distance 0)
    distances = distances[:, 1:]                       # [N, k]
    kth_distances = distances[:, -1]                   # [N]
    sigma_init = np.percentile(kth_distances, scale_percentile)

    sigmas = torch.full((pts.shape[0],), sigma_init, device=device)
    return sigmas

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
    verts_p = x_p_h[:, :, :3] / x_p_h[:, :, 3:4]

    return verts_p


@torch.no_grad()
def init_3dgs_humans(
    smpl_server,
    n_humans,
    *,
    device: str = "cuda",
    sh_degree: int = 3,
    init_sigma: float = 0.02,
    init_opacity: float = 0.10,
    color_mode: str = "random",           # "random" | "gray" | "given"
    given_colors: Optional[torch.Tensor] = None,  # [M,3] in [0,1] if color_mode="given"
    seed: Optional[int] = 1000,
) -> Tuple[nn.ParameterDict, Dict]:


    smpl_server = smpl_server.to(device).eval()
    verts_c = smpl_server.verts_c[0].to(device)    
    weights_c = smpl_server.weights_c[0].to(device)
    M = verts_c.shape[0]

    # means (canonical)
    means = verts_c.clone()            

    # log-scales 
    log_scales = torch.full((M, 3), math.log(init_sigma), device=device)

    # quats (identity)
    quats = torch.zeros(M, 4, device=device); quats[:, 0] = 1.0

    # opacity logits
    opacity_logits = torch.full((M,), torch.logit(torch.tensor(init_opacity, device=device)))

    # SH colors
    colors_sh = init_sh_colors(color_mode, M, sh_degree, given_colors, device)

    # Load dataset
    splats = list()
    for _ in range(n_humans):
        splats.append(make_paramdict(means, log_scales, quats, opacity_logits, colors_sh, sh_degree))
    aux = {"verts_c": verts_c, "weights_c": weights_c, "smpl_server": smpl_server}
    return splats, aux
            

@torch.no_grad()
def init_3dgs_background(
    ds,
    *,
    device: str = "cuda",
    sh_degree: int = 3,
    k_for_scale: int = 16,
    scale_percentile: float = 50.0,
    init_opacity: float = 0.05,
) -> Tuple[nn.ParameterDict, Dict]:

    # init from point cloud
    pts, cols = ds.point_cloud

    # means
    means = torch.from_numpy(pts).to(device)

    # scales
    sigmas = init_isotropic_scales_from_knn(pts, k_for_scale=k_for_scale, scale_percentile=scale_percentile, device=device)
    sigmas = sigmas.clamp(1e-4, 1.0)
    log_scales = torch.log(sigmas.unsqueeze(1).expand(-1, 3)).to(device)   # [N,3]

    # quats (identity)
    quats = torch.zeros(pts.shape[0], 4, device=device)
    quats[:, 0] = 1.0

    # SH colors 
    col_mode = "given" if cols is not None else "random"
    sh_colors = init_sh_colors(col_mode, pts.shape[0], sh_degree, given_colors=torch.from_numpy(cols), device=device)
 
    # opacity logits
    opacity_logits = torch.full((pts.shape[0],), torch.logit(torch.tensor(init_opacity, device=device)))

    splats = make_paramdict(means, log_scales, quats, opacity_logits, sh_colors, sh_degree)

    return splats

def prep_splats_for_render(
    *,
    static_splats: torch.nn.ParameterDict,  
    dynamic_splats_per_human: torch.nn.ParameterDict, 
    dyn_aux: dict,         
    smpl_param: torch.Tensor = None, 
    clamp_sigma: tuple = (1e-4, 1.0),
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    # Static (background)
    static_pack = prep_splats(static_splats, clamp_sigma, dtype, device)

    # Dynamic (humans)
    smpl_server = dyn_aux.get("smpl_server", None)
    weights_c = dyn_aux.get("weights_c", None)
    dynamic_pack_all = dict()
    for i in range(len(dynamic_splats_per_human)):
        dynamic_splats = dynamic_splats_per_human[i]
        means_p = canon_to_posed(smpl_server, smpl_param[i:i+1], dynamic_splats["means"], weights_c)
        dynamic_splats["means"].data.copy_(means_p.squeeze(0))
        new_pack = prep_splats(dynamic_splats, clamp_sigma, dtype, device)
        if i == 0:
            dynamic_pack_all = {k: v for k, v in new_pack.items()}
        else:
            for k, v in new_pack.items():
                dynamic_pack_all[k] = torch.cat([dynamic_pack_all[k], v], dim=0)

    
    # Both
    all_pack = dict()
    for k in static_pack.keys():
        all_pack[k] = torch.cat([static_pack[k], dynamic_pack_all[k]], dim=0)

    return {"static": static_pack, "dynamic": dynamic_pack_all, "all": all_pack}


# ---------------- MAIN
# Settings
sh_degree = 3
cloud_downsample_factor = 30
preprocess_dir = Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res")
tids = [0, 1]
ds = FullSceneDataset(preprocess_dir, tids=tids, cloud_downsample=cloud_downsample_factor)
smpl_server = SMPLServer()

# Initialize splats
static_gs = init_3dgs_background(
    ds, device=device, sh_degree=sh_degree
)
print(f"Initialized {static_gs['means'].shape[0]} static splats from point cloud.")
dynamic_gs, dyn_aux = init_3dgs_humans(
    smpl_server, n_humans=len(tids), device=device, sh_degree=sh_degree,
    color_mode="random"
)
n_human_gs = [g["means"].shape[0] for g in dynamic_gs]
print(f"Initialized {sum(n_human_gs)} dynamic splats for {len(tids)} humans.")

# Sample a frame
frame_idx = torch.randint(0, len(ds), (1,)).item()
sample = ds[frame_idx]


# Init input for rendering
packs = prep_splats_for_render(
    static_splats=static_gs,
    dynamic_splats_per_human=dynamic_gs,
    dyn_aux=dyn_aux,
    smpl_param=sample["smpl_param"].to(device),
    clamp_sigma=(1e-4, 1.0),
    dtype=torch.float32,
    device=device
)

intrinsics = sample["K"].to(device).unsqueeze(0)
extrinsics = sample["M_ext"].to(device).unsqueeze(0)
H, W = sample["H"], sample["W"]

# Render
kind = "dynamic"   # try "static" or "all" too
p = packs[kind]   # try "static" or "dynamic" too
print(f"Input shapes: means {p['means'].shape}, scales {p['scales'].shape}, quats {p['quats'].shape}, opacity {p['opacity'].shape}, colors {p['colors'].shape}")
colors, alphas, info = rasterization(
    p["means"], p["quats"], p["scales"], p["opacity"], p["colors"],
    extrinsics, intrinsics, W, H, sh_degree=sh_degree, packed=False
)


# Visualize
img_arr = sample["image"].cpu().numpy()
proj_arr = alphas[0].detach().cpu().numpy()
binary_mask_arr = (proj_arr > 0.5).astype(float) # note that higher the tr, the tigher the mask
masked_img = img_arr * binary_mask_arr

fig, ax = plt.subplots(4, 1, figsize=(15, 10))
ax[0].imshow(img_arr)
ax[0].set_title("Image")
ax[0].axis("off")

ax[1].imshow(proj_arr)
ax[1].set_title("Alphas")
ax[1].axis("off")

ax[2].imshow(masked_img)
ax[2].set_title("Masked Image")
ax[2].axis("off")

ax[3].imshow(colors[0].detach().cpu().numpy())
ax[3].set_title("Colors")
ax[3].axis("off")

plt.savefig(f"playground/outputs/final_render.png", dpi=200)