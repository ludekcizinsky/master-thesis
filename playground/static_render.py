import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
from typing import Optional, Tuple, Dict

import torch.nn.functional as F


from pathlib import Path
import torch
from torch import nn
torch.manual_seed(1000)

from gsplat import rasterization
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.smpl_deformer.smpl_server import SMPLServer
from training.helpers.dataset import MegaSAMDataset, TraceDataset

from playground.alignment import align_megasam_to_trace_auto

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


def _unit_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

# ---------- dynamic (human) initializer ----------

@torch.no_grad()
def init_dynamic_splats_from_megasam(
    preprocess_dir: Path,
    tid,
    smpl_server,
    *,
    downscale: int = 1,
    device: str = "cuda",
    sh_degree: int = 3,
    init_sigma: float = 0.02,
    init_opacity: float = 0.10,
    color_mode: str = "random",           # "random" | "gray" | "given"
    given_colors: Optional[torch.Tensor] = None,  # [M,3] in [0,1] if color_mode="given"
    seed: Optional[int] = 1000,
) -> Tuple[nn.ParameterDict, Dict]:
    if seed is not None:
        torch.manual_seed(seed)

    smpl_server = smpl_server.to(device).eval()
    verts_c  = smpl_server.verts_c[0].to(device)      # [M,3]
    weights_c= smpl_server.weights_c[0].to(device)    # [M,24]
    M = verts_c.shape[0]

    # means (canonical)
    means = verts_c.clone()                           # [M,3]

    # log-scales (anisotropic, start iso)
    log_scales = torch.full((M, 3), math.log(init_sigma), device=device)

    # quats (identity)
    quats = torch.zeros(M, 4, device=device); quats[:, 0] = 1.0

    # opacity logits
    opacity_logits = torch.full((M,), torch.logit(torch.tensor(init_opacity, device=device)))

    # SH colors
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

    # DC coefficient (no offset):
    colors_sh[:, 0, :] = base_rgb / C0

    # Load dataset
    ds = TraceDataset(preprocess_dir, tid=tid, downscale=downscale)
    splats = make_paramdict(means, log_scales, quats, opacity_logits, colors_sh, sh_degree)
    aux = {"verts_c": verts_c, "weights_c": weights_c, "smpl_server": smpl_server, "dataset": ds}
    return splats, aux


# ---------- static (scene) initializer from MegaSAM ----------

@torch.no_grad()
def init_static_splats_from_megasam(
    npz_path: str,
    *,
    device: str = "cuda",
    sh_degree: int = 3,
    pixel_downsample: int = 4,        # back-projection stride
    max_points: int = 250_000,
    k_for_scale: int = 16,
    scale_percentile: float = 50.0,
    init_opacity: float = 0.05,
    use_linear_rgb: bool = True,
    align_poses_to: str = "median",   # 'none'|'first'|'median'
    static_mask_key: Optional[str] = None,
    motion_prob_key: Optional[str] = None,
    motion_static_threshold: float = 0.2,
    seed: Optional[int] = 1000,
) -> Tuple[nn.ParameterDict, Dict]:
    if seed is not None:
        torch.manual_seed(seed)

    # 1) Load dataset
    ds = MegaSAMDataset(
        npz_path,
        normalize_images=True,                # images in [0,1]
        static_mask_key=static_mask_key,
        motion_prob_key=motion_prob_key,
        motion_static_threshold=motion_static_threshold,
        align_poses_to=align_poses_to,
        device="cpu",
    )

    # 2) Build static world point cloud (+ colors)
    pts, cols = ds.build_static_point_cloud(every_k=1, downsample=pixel_downsample, device="cpu")
    if pts.numel() == 0:
        raise RuntimeError("MegaSaM produced no static points. Check masks/depths.")

    # Optional thinning for memory
    if pts.shape[0] > max_points:
        idx = torch.randperm(pts.shape[0])[:max_points]
        pts, cols = pts[idx], (cols[idx] if cols is not None else None)

    # 3) Scales from kNN spacing (isotropic -> copy to xyz)
    sigmas = MegaSAMDataset.estimate_scales_from_knn(pts, k=k_for_scale, percentile=scale_percentile)  # [N,1]
    sigmas = sigmas.clamp_(1e-4, 1.0)
    log_scales = torch.log(sigmas.repeat(1, 3))    # store **log-scales**

    # 4) Quats identity
    quats = torch.zeros(pts.shape[0], 4); quats[:, 0] = 1.0

    # 5) SH colors (DC from image color)
    K = (sh_degree + 1) ** 2
    colors_sh = torch.zeros(pts.shape[0], K, 3, device=pts.device)

    if cols is not None:
        base_rgb = cols.float().clamp(0,1)               # [N,3], normalized to [0,1]
        if use_linear_rgb:
            base_rgb = srgb_to_linear(base_rgb)          # convert sRGB -> linear
        colors_sh[:, 0, :] = base_rgb / C0               # ✅ no -0.5 here
    else:
        gray = torch.full((pts.shape[0], 3), 0.5, device=pts.device)
        gray = srgb_to_linear(gray) if use_linear_rgb else gray
        colors_sh[:, 0, :] = gray / C0

    
    # 6) Opacity logits
    opacity_logits = torch.full((pts.shape[0],), torch.logit(torch.tensor(init_opacity)))

    # 7) Move to device and pack as ParameterDict
    means = pts.to(device)
    log_scales = log_scales.to(device)
    quats = quats.to(device)
    colors_sh = colors_sh.to(device)
    opacity_logits = opacity_logits.to(device)

    splats = make_paramdict(means, log_scales, quats, opacity_logits, colors_sh, sh_degree)

    aux = {"dataset": ds}
    return splats, aux

# ---------- Prepare rasterization inputs for static + dynamic ----------
def _unit_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def prepare_raster_inputs_joint(
    *,
    static_splats: torch.nn.ParameterDict,          # from init_static_splats_from_megasam
    dynamic_splats: torch.nn.ParameterDict,         # from init_dynamic_splats_from_megasam
    dyn_aux: dict,                                  # must contain: weights_c [Nd,24], and smpl_server
    K: torch.Tensor,                                # [B,3,3]
    M_ext: torch.Tensor,                            # [B,4,4]  world->camera (w2c)
    H: int, W: int,
    sh_degree: int = 3,
    smpl_param: torch.Tensor = None,                # [B,86]
    absolute: bool = False,
    clamp_sigma: tuple = (1e-4, 1.0),
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    """
    Returns:
      dict with three entries: 'static', 'dynamic', 'both'.
      Each entry is a dict with keys:
        means [N,3], quats [N,4], scales [N,3], opacity [N], colors [N,K,3],
        viewmats [B,4,4], Ks [B,3,3], H, W
    All tensors are ready for gsplat.rasterization(...).
    """
    # Resolve device
    if device is None:
        device = K.device if K.device.type != "cpu" else next(iter(static_splats.values())).device

    # Cameras
    viewmats = M_ext.to(device, dtype).contiguous()
    Ks = K.to(device, dtype).contiguous()
    K_expected = (sh_degree + 1) ** 2

    def _prep_one(splats: torch.nn.ParameterDict):
        # quats, scales, opacity, colors (SH)
        quats = _unit_quat(splats["quats"].to(device, dtype))
        scales = torch.exp(splats["scales"].to(device, dtype)).clamp(*clamp_sigma)
        opacity = torch.sigmoid(splats["opacities"].to(device, dtype))
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device, dtype)
        # pad/truncate SH if degree changed
        if colors.shape[1] != K_expected:
            if colors.shape[1] < K_expected:
                pad = K_expected - colors.shape[1]
                pad_zeros = torch.zeros(colors.shape[0], pad, 3, device=device, dtype=dtype)
                colors = torch.cat([colors, pad_zeros], dim=1)
            else:
                colors = colors[:, :K_expected, :]
        return quats, scales, opacity, colors

    # --- Static ---
    quats_s, scales_s, opacity_s, colors_s = _prep_one(static_splats)
    means_s = static_splats["means"].to(device, dtype)

    static_pack = dict(
        means=means_s, quats=quats_s, scales=scales_s,
        opacity=opacity_s, colors=colors_s,
        viewmats=viewmats, Ks=Ks, H=H, W=W
    )

    C0 = 0.28209479177387814
    c0 = static_splats["sh0"][:, 0, :]           # [N,3]
    rgb_implied_if_no_offset = (c0 * C0).clamp(0,1)
    rgb_implied_if_offset    = (0.5 + c0 * C0).clamp(0,1)
    print("no-offset mean:", rgb_implied_if_no_offset.mean(0))
    print("with-offset mean:", rgb_implied_if_offset.mean(0))

    # --- Dynamic (pose with SMPL) ---
    assert smpl_param is not None, "smpl_param [B,86] required for dynamic posing."
    smpl_server = dyn_aux.get("smpl_server", None)
    weights_c = dyn_aux.get("weights_c", None)
    assert smpl_server is not None and weights_c is not None, \
        "dyn_aux must contain 'smpl_server' and 'weights_c' [Nd,24]."

    x_c = dynamic_splats["means"].to(device, dtype)               # [Nd,3] canonical
    x_c_h = F.pad(x_c, (0, 1), value=1.0)                         # [Nd,4]
    tsf = smpl_server(smpl_param.to(device), absolute=absolute)["smpl_tfs"][0].to(device)  # [24,4,4]
    w = weights_c.to(device, dtype)                               # [Nd,24]
    x_p_h = torch.einsum("pn,nij,pj->pi", w, tsf, x_c_h)          # [Nd,4]
    means_d = x_p_h[:, :3] / x_p_h[:, 3:4]                        # [Nd,3] world

    quats_d, scales_d, opacity_d, colors_d = _prep_one(dynamic_splats)

    dynamic_pack = dict(
        means=means_d, quats=quats_d, scales=scales_d,
        opacity=opacity_d, colors=colors_d,
        viewmats=viewmats, Ks=Ks, H=H, W=W
    )

    # --- Both (concat fields) ---
    means_b   = torch.cat([means_s,   means_d],   dim=0)
    quats_b   = torch.cat([quats_s,   quats_d],   dim=0)
    scales_b  = torch.cat([scales_s,  scales_d],  dim=0)
    opacity_b = torch.cat([opacity_s, opacity_d], dim=0)
    colors_b  = torch.cat([colors_s,  colors_d],  dim=0)

    both_pack = dict(
        means=means_b, quats=quats_b, scales=scales_b,
        opacity=opacity_b, colors=colors_b,
        viewmats=viewmats, Ks=Ks, H=H, W=W
    )

    return {"static": static_pack, "dynamic": dynamic_pack, "both": both_pack}


# ---------------- MAIN: static + dynamic rendering ----------------

sh_degree = 3

# STATIC
preprocess_dir = Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res")
static_splats, static_aux = init_static_splats_from_megasam(
    str(preprocess_dir / "megasam" / "sgd_cvd_hr.npz"),
    device=device,
    sh_degree=sh_degree,
    pixel_downsample=30,
)

print("Static N:", static_splats["means"].shape[0])

# DYNAMIC (SMPL-based)
smpl_server = SMPLServer().to(device).eval()

dynamic_splats, dyn_aux = init_dynamic_splats_from_megasam(
    preprocess_dir=preprocess_dir,
    tid=0,
    smpl_server=smpl_server,
    device=device,
    sh_degree=sh_degree,
    init_sigma=0.02,
    init_opacity=0.10,
    color_mode="random",
)

print("Dynamic N:", dynamic_splats["means"].shape[0])

# Load single sample
# - dynamic 
fid = 0
dynamic_ds = dyn_aux["dataset"]
dynamic_sample = dynamic_ds[fid]
smpl_param = dynamic_sample["smpl_param"].unsqueeze(0).to(device)  # [1,86]
H_dyn, W_dyn = dynamic_sample["image"].shape[:2]
K_dyn = dynamic_sample["K"].unsqueeze(0).to(device)                      # [1,3,3]
M_ext_dyn = dynamic_sample["M_ext"].unsqueeze(0).to(device)                # [1,4,4]
print(f"--- FYI: Dynamic intrinsics:\n{K_dyn[0]}")

# - static
static_ds = static_aux["dataset"]
static_sample = static_ds[fid]
K_static = static_sample.K.unsqueeze(0).to(device)                      # [1,3,3]
M_ext_static = static_sample.w2c.unsqueeze(0).to(device)                # [1,4,4]
H_static, W_static = static_sample.H, static_sample.W
print(f"--- FYI: Static intrinsics:\n{K_static[0]}")


# Alignment
# Grab MegSaM / TRACE poses
ms_c2w_all = torch.from_numpy(static_ds.c2w.numpy())   # MegaSaM c2w
trace_poses_all = torch.stack(dynamic_ds.pose_all, 0)  # UNKNOWN (could be c2w or w2c)

# DEBUG: Camera centers
# ms_centers = -ms_c2w_all[:, :3, :3].transpose(1,2) @ ms_c2w_all[:, :3, 3]
# print("DEBUG: MegaSaM camera centers sample:", ms_centers[:5])
# trace_centers = -trace_poses_all[:, :3, :3].transpose(1,2) @ trace_poses_all[:, :3, 3]
# print("DEBUG: Trace camera centers sample:", trace_centers[:5])

# DEBUG: Print before alignment
print("DEBUG: Before alignment - MegaSaM points sample:", static_splats["means"][:5])

# DEBUG: Visualize points before alignment
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
pts_cpu_before = static_splats["means"][:5000].detach().cpu().numpy()  # Sample points before
cam_center = -M_ext_dyn[0][:3, :3].T @ M_ext_dyn[0][:3, 3]  # Camera center
cam_center_cpu = cam_center.detach().cpu().numpy()
ax.scatter(pts_cpu_before[:, 0], pts_cpu_before[:, 1], pts_cpu_before[:, 2], c='r', s=1, label='Points Before Alignment')
ax.scatter(cam_center_cpu[0], cam_center_cpu[1], cam_center_cpu[2], c='b', s=100, label='Camera Center')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Points Before Alignment and Camera Center')
plt.savefig('playground/debug_points_before.png', dpi=150)
print("DEBUG: Saved 3D plot before alignment to playground/debug_points_before.png")

# Run auto-disambiguating alignment, scoring with the very frame you render (fid)
align_out = align_megasam_to_trace_auto(
    static_splats=static_splats,
    ms_c2w_all=ms_c2w_all,
    trace_poses_all=trace_poses_all,
    K_render=K_dyn, w2c_render=M_ext_dyn, W=W_dyn, H=H_dyn,
    max_pairs=75,
)


# Prepare for rasterization
# packs = prepare_raster_inputs_joint(
    # static_splats=static_splats,
    # dynamic_splats=dynamic_splats,
    # dyn_aux=dyn_aux,
    # K=K_dyn, M_ext=M_ext_dyn, H=H_dyn, W=W_dyn, sh_degree=sh_degree,
    # smpl_param=smpl_param, absolute=False
# )

# # Render
# kind = "static"
# p = packs[kind]   # try "static" or "dynamic" too
# colors, alphas, info = rasterization(
    # p["means"], p["quats"], p["scales"], p["opacity"], p["colors"],
    # p["viewmats"], p["Ks"], p["W"], p["H"], sh_degree=sh_degree, packed=False
# )
# print(f"opacity range {p['opacity'].min().item():.3f} .. {p['opacity'].max().item():.3f}")
# print(f"scales range {p['scales'].min().item():.4f} .. {p['scales'].max().item():.4f}")


# # Visualize
# img_arr = dynamic_sample["image"].cpu().numpy()
# proj_arr = alphas[0].detach().cpu().numpy()
# binary_mask_arr = (proj_arr > 0.5).astype(float) # note that higher the tr, the tigher the mask
# masked_img = img_arr * binary_mask_arr

# fig, ax = plt.subplots(1, 4, figsize=(15, 10))
# ax[0].imshow(img_arr)
# ax[0].set_title("Image")
# ax[0].axis("off")

# ax[1].imshow(proj_arr)
# ax[1].set_title("Alphas")
# ax[1].axis("off")

# ax[2].imshow(masked_img)
# ax[2].set_title("Masked Image")
# ax[2].axis("off")

# ax[3].imshow(colors[0].detach().cpu().numpy())
# ax[3].set_title("Colors")
# ax[3].axis("off")

# plt.savefig(f"playground/outputs/final_render.png", dpi=200)