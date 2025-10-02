import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import torch
torch.manual_seed(1000)

from gsplat import rasterization
from matplotlib import pyplot as plt


from training.helpers.dataset import MegaSAMDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

# --- Dataset and DataLoader
preprocess_dir = Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/football_high_res")


def init_static_splats_from_megasam(npz_path: str, device: str, sh_degree: int = 3,
                                    downsample_px: int = 4, max_points: int = 250_000):
    """
    Build static Gaussians from MegaSaM: backproject static pixels across frames,
    estimate scales from local spacing, set SH0 colors, unit quats, small opacity.
    """
    ds_ms = MegaSAMDataset(
        npz_path,
        normalize_images=True,
        static_mask_key=None,       # set if your .npz has a static/dynamic mask
        motion_prob_key=None,       # or use a motion prob key + threshold
        align_poses_to="median",
        device="cpu",
    )

    # Aggregate a static point cloud across frames (downsample pixels for speed)
    pts, cols = ds_ms.build_static_point_cloud(every_k=1, downsample=downsample_px, device="cpu")
    if pts.numel() == 0:
        raise RuntimeError("MegaSaM produced no static points. Check masks/depths.")

    # Optional random thinning to keep memory/rendering reasonable
    if pts.shape[0] > max_points:
        idx = torch.randperm(pts.shape[0])[:max_points]
        pts, cols = pts[idx], (cols[idx] if cols is not None else None)

    # Heuristic isotropic scales from kNN spacing
    sigmas = MegaSAMDataset.estimate_scales_from_knn(pts, k=16, percentile=50.0)  # [N,1]
    sigmas = sigmas.clamp_(1e-4, 1.0)
    scales = sigmas.repeat(1, 3).to(device)  # anisotropic per-axis, same sigma

    # Quats (identity)
    quats = torch.zeros(pts.shape[0], 4)
    quats[:, 0] = 1.0

    # SH colors: use only l=0 band to start (constant color)
    def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0

    K = (sh_degree + 1) ** 2
    colors_sh = torch.zeros(pts.shape[0], K, 3)
    if cols is not None:
        colors_sh[:, 0, :] = rgb_to_sh(cols.float())  # cols in [0,1]
    else:
        colors_sh[:, 0, :] = rgb_to_sh(torch.ones_like(colors_sh[:, 0, :]) * 0.5)

    # Opacity (fixed small to start)
    opacity = torch.full((pts.shape[0],), 0.05, dtype=torch.float32)

    # Push to device
    return {
        "means": pts.to(device).contiguous(),
        "quats": quats.to(device).contiguous(),
        "scales": scales.to(device).contiguous(),
        "colors": colors_sh.to(device).contiguous(),
        "opacity": opacity.to(device).contiguous(),
        "dataset": ds_ms,  # for depth/camera access below
    }

# Point this to your MegaSaM npz
sh_degree = 3
megasam_npz = str(preprocess_dir / "megasam" / "sgd_cvd_hr.npz")
static_splats = init_static_splats_from_megasam(megasam_npz, device=device, sh_degree=sh_degree, downsample_px=30)
print(f"Initialized {static_splats['means'].shape[0]} static splats from MegaSaM.")

# If MegaSaM and Trace world frames are different, either:
#   (A) use MegaSaM camera for the static renders, or
#   (B) estimate a rigid (similarity) alignment between the two worlds and map static_splats["means"].
use_megasam_camera = True  # set True if you see misalignment

# Choose a frame index to grab MegaSaM depth (match your tid if sequences align)
ms_idx = 40
ms_frame = static_splats["dataset"][ms_idx]   # MegaSaMFrame

# --- Render static-only and human+static -------------------------------------
# Cameras for the second row:
if use_megasam_camera:
    viewmats_ms = ms_frame.w2c.unsqueeze(0).to(device)        # [1,4,4]
    Ks_ms = ms_frame.K.unsqueeze(0).to(device)                 # [1,3,3]
    H_ms, W_ms = ms_frame.H, ms_frame.W
else:
    raise NotImplementedError("Implement alignment if using Trace cameras for static splats.")
#     viewmats_ms = viewmats                                     # reuse your current [1,4,4]
    # Ks_ms = Ks                                                 # reuse your current [1,3,3]
    # H_ms, W_ms = H, W

# Static-only render
colors_static, alphas_static, _ = rasterization(
    static_splats["means"], static_splats["quats"], static_splats["scales"],
    static_splats["opacity"], static_splats["colors"],
    viewmats_ms, Ks_ms, W_ms, H_ms, sh_degree=sh_degree, packed=False
)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(ms_frame.image.cpu().numpy())
axs[0].set_title("MegaSaM RGB")
axs[0].axis("off")
axs[1].imshow(colors_static[0].cpu().numpy())
axs[1].set_title("Static-only color")
axs[1].axis("off")
plt.tight_layout()

# save figure
plt.savefig("playground/outputs/static_only_render.png")