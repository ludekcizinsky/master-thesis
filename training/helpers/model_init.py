import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils.smpl_deformer.smpl_server import SMPLServer

C0 = 0.28209479177387814

def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    a = 0.055
    return torch.where(x <= 0.04045, x/12.92, ((x + a)/(1+a))**2.4)

def unit_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


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


@torch.no_grad()
def init_3dgs_humans(
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

    smpl_server = SMPLServer().to(device).eval()
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
    opacity_logits = torch.full((M,), torch.logit(torch.tensor(init_opacity))).to(device)

    # SH colors
    colors_sh = init_sh_colors(color_mode, M, sh_degree, given_colors, device)

    # Load dataset
    splats = list()
    for _ in range(n_humans):
        splats.append(
            make_paramdict(
                means.clone(),
                log_scales.clone(),
                quats.clone(),
                opacity_logits.clone(),
                colors_sh.clone(),
                sh_degree,
            )
        )
    
    smpl_c_info = {"verts_c": verts_c, "weights_c": weights_c, "smpl_server": smpl_server}
    return splats, smpl_c_info 

@torch.no_grad()
def init_3dgs_background(
    ds,
    *,
    device: str = "cuda",
    sh_degree: int = 3,
    k_for_scale: int = 10,
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
    given_colors = None
    if cols is not None:
        given_colors = torch.from_numpy(cols).to(device=device, dtype=torch.float32) / 255.0
    sh_colors = init_sh_colors(col_mode, pts.shape[0], sh_degree, given_colors=given_colors, device=device)
 
    # opacity logits
    opacity_logits = torch.full((pts.shape[0],), torch.logit(torch.tensor(init_opacity))).to(device)

    splats = make_paramdict(means, log_scales, quats, opacity_logits, sh_colors, sh_degree)

    return splats


def init_optimizers(all_gs, cfg):

    all_optimizers = list()
    for splats in all_gs:
        BS = cfg.batch_size
        optimizers = {
            name: torch.optim.Adam(
                [{"params": params, "lr": cfg.get(f"{name}_lr", 1e-3) * math.sqrt(BS), "name": name}],
                eps=1e-15 / math.sqrt(BS),
                # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            )
            for name, params in splats.items()
        }
        all_optimizers.append(optimizers)

    return all_optimizers

def create_splats_with_optimizers(device, cfg, ds):

    # Static
    if cfg.train_bg:
        static_gs = init_3dgs_background(
            ds, device=device, sh_degree=cfg.sh_degree
        )
        print(f"--- FYI: Initialized {static_gs['means'].shape[0]} static splats from point cloud.")
    else:
        static_gs = None

    # Dynamic
    if len(cfg.tids) > 0:
        dynamic_gs_per_human, smpl_c_info = init_3dgs_humans(
            n_humans=len(cfg.tids), device=device, sh_degree=cfg.sh_degree,
            color_mode="random"
        )
        n_human_gs = [g["means"].shape[0] for g in dynamic_gs_per_human]
        print(f"--- FYI: Initialized {sum(n_human_gs)} dynamic splats for {len(cfg.tids)} humans.")
    else:
        dynamic_gs_per_human = None
        smpl_c_info = None

    # Combined
    if static_gs is None:
        all_gs = dynamic_gs_per_human
    elif dynamic_gs_per_human is None:
        all_gs = [static_gs]
    else:
        all_gs = [static_gs] + dynamic_gs_per_human

    # Optimizers
    all_optimizers = init_optimizers(all_gs, cfg)

    return all_gs, all_optimizers, smpl_c_info