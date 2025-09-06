# file: train_3dgs_human_only.py
# Minimal human-only 3DGS prototype (no SfM), optimizing in SMPL canonical space.
# Renders via a simple 2D Gaussian splatter (good enough to see losses decrease).

import hydra
import os
import sys
from omegaconf import DictConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../submodules/humans4d")))
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

from utils.smpl_deformer.smpl_server import SMPLServer
from utils.io import load_frame_map_jsonl_restore
from preprocess.helpers.cameras import load_camdicts_json

from tqdm import tqdm

from training.helpers.utils import init_logging

# -----------------------------
# Utility: image & mask loading & Visualization
# -----------------------------
def load_image(path: Path, downscale: int = 1) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if downscale > 1:
        w, h = img.size
        img = img.resize((w // downscale, h // downscale), Image.BILINEAR)
    im = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3]
    return im.permute(2, 0, 1).contiguous()  # [3,H,W]

def load_mask(path: Path, downscale: int = 1) -> torch.Tensor:
    m = Image.open(path).convert("L")
    if downscale > 1:
        w, h = m.size
        m = m.resize((w // downscale, h // downscale), Image.NEAREST)
    m = torch.from_numpy(np.array(m)).float() / 255.0  # [H,W]
    return m.clamp(0, 1)

@torch.no_grad()
def debug_projection_stats(uv, Z, H, W, tag=""):
    u, v = uv[:,0], uv[:,1]
    in_u = (u >= 0) & (u < W)
    in_v = (v >= 0) & (v < H)
    in_img = in_u & in_v
    pct_in = 100.0 * in_img.float().mean().item()
    pct_zpos = 100.0 * (Z > 0).float().mean().item()
    print(f"[{tag}] uv range: u=({u.min():.1f},{u.max():.1f}) v=({v.min():.1f},{v.max():.1f}) | "
          f"in-img: {pct_in:.1f}% | Z>0: {pct_zpos:.1f}% | Z range=({Z.min():.3f},{Z.max():.3f})")

@torch.no_grad()
def save_loss_visualization(
    image: torch.Tensor,       # [3,H,W], GT image in [0,1]
    mask: torch.Tensor,        # [H,W], 0–1
    rgb_pred: torch.Tensor,    # [3,H,W], predicted image in [0,1]
    out_path: str,
):
    """
    Saves a side-by-side visualization of:
    - original image
    - masked image (image * mask)
    - predicted image
    """
    # Ensure all are 3×H×W tensors
    H, W = image.shape[-2:]
    mask3 = mask.expand_as(image)  # [3,H,W]
    masked_img = image * mask3

    # Stack [3, H, W] tensors into [3, H, 3*W]
    comparison = torch.cat([image, masked_img, rgb_pred.clamp(0,1)], dim=-1)

    # Convert to uint8 for saving
    img = (comparison.permute(1,2,0).cpu().numpy().clip(0,1) * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)

    return out_path

@torch.no_grad()
def save_canonical_preview(
    gaus,          # CanonicalGaussians module (self.gaus)
    out_path: str,
    H: int = 512,
    W: int = 384,
    fov_deg: float = 18.0,     # ~tight portrait; smaller => more zoom
    bg_color=(1.0, 1.0, 1.0),  # white background
    flip_z: bool = False,      # set True if your renderer expects -Z forward (OpenGL)
    color_mode: str = "learned", # "learned" | "bone"
    sigma_mult: float = 15.0,  # multiplier for scale -> pixel sigma heuristic
):
    """
    Render the current canonical Gaussians from a virtual front-view camera and save as PNG.

    NOTES / ASSUMPTIONS:
    - Assumes canonical SMPL faces +Y. We rotate canonical by +90° around X so that +Y -> +Z,
      i.e., the person faces the camera (+Z out of image in PyTorch3D convention).
      If you discover your canonical faces another axis, tweak the rotation (see NOTE A below).
    - Uses the lightweight 2D splatter you already have.
    - Scales are mapped to pixel sigmas with a heuristic; adjust 'sigma_mult' if blobs look off.
    """
    device = gaus.means_c.device
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    means_c = gaus.means_c.detach()          # [M,3] in canonical space
    colors  = gaus.colors.detach().clamp(0,1) # [M,3]
    opac    = gaus.opacity().detach().clamp(0,1) # [M]
    scales_w = gaus.scales().detach()        # [M] (world/canonical units)

    # --- Virtual camera setup ---
    # Focal from FOV (pin-hole): f = 0.5*W / tan(FOV/2)
    f_pix = 0.5 * W / math.tan(math.radians(fov_deg) * 0.5)
    fx = fy = torch.tensor(float(f_pix), device=device)
    cx = torch.tensor(W * 0.5, device=device)
    cy = torch.tensor(H * 0.5, device=device)

    # NOTE A (Facing): rotate canonical so front faces +Z (camera looks down +Z in PyTorch3D).
    # If your canonical faces +Y, Rx(+90°) maps +Y -> +Z. If you find it's off, try Rx(-90°) or add a Ry(±90°).
    # R = _rx(-90.0, device=device)  # [3,3]
    R = torch.eye(3, device=device)  # assume canonical already faces +Z
    R[:2, :] *= -1.0
    X = (R @ means_c.T).T  # [M,3]

    # Centering: make sure pelvis/centroid is roughly at origin before placing at distance.
    center = X.mean(dim=0, keepdim=True)
    X = X - center

    # Choose camera distance so the person fits nicely:
    # Compute XY extent and back-compute a distance from FOV (with small margin).
    ext = (X[:, :2].abs().max(dim=0).values).max().item()  # rough radius in XY
    margin = 1.2
    dist = margin * (ext / math.tan(math.radians(fov_deg) * 0.5) + 1e-6)

    # Camera coordinates: translate along +Z so the subject is in front
    X_cam = X.clone()
    X_cam[:, 2] = X_cam[:, 2] + dist  # [M,3], Z positive

    # Optional z-flip if your renderer expects -Z forward (OpenGL-like):
    if flip_z:
        X_cam[:, 2] = -X_cam[:, 2]

    # --- Project to pixels ---
    Z = X_cam[:, 2].clamp(min=1e-6)            # [M]
    u = fx * (X_cam[:, 0] / Z) + cx
    v = fy * (X_cam[:, 1] / Z) + cy
    uv = torch.stack([u, v], dim=-1)           # [M,2]

    # --- Convert canonical scales to pixel sigmas (heuristic) ---
    # A simple linear map using focal and depth. Tweak sigma_mult to control blob size.
    scales_pix = (scales_w * (fx / max(H, W)) * sigma_mult).clamp(0.5, 50.0)

    # (Optional) color by bone for debugging skinning weights
    if color_mode == "bone":
        # Argmax over weights to assign a pseudo-color per bone id
        # (needs weights_c handy; else skip)
        if hasattr(gaus, "weights_c"):
            bone_id = gaus.weights_c.argmax(dim=1)  # [M]
            # simple color table
            torch.manual_seed(0)
            palette = torch.rand(24, 3, device=device)
            colors = palette[bone_id]
        else:
            # fallback: keep learned colors
            pass

    # --- Render ---
    rgb, alpha = render_gaussians_2d(
        uv=uv, z=Z,
        scales=scales_pix, colors=colors, opacity=opac,
        H=H, W=W, soft_z_temp=1e-2, truncate=3.0
    )

    # Composite over bg
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device).view(3,1,1)
    out = rgb + (1.0 - alpha).unsqueeze(0) * bg  # [3,H,W]

    # Save
    import numpy as np
    from PIL import Image
    img = (out.clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)
    return out_path


# -----------------------------
# Dataset (one track id per run)
# -----------------------------
class HumanOnlyDataset(Dataset):
    """
    Exposes (frame_id, image, mask, cam_intrinsics, smpl_param) for a chosen track id.
    """

    def __init__(
        self,
        scene_root: Path,
        tid: int,
        downscale: int = 2,
        mask_folder: str = "masks",  # masks/<tid>/<fid>.png
    ):
        self.scene_root = Path(scene_root)
        self.tid = int(tid)
        self.downscale = downscale

        # Load cam dicts (by frame_id)
        cam_dicts_path = self.scene_root / "preprocess" / "cam_dicts.json"
        self.cam_dicts = load_camdicts_json(cam_dicts_path)

        # Load frame map jsonl
        preprocess_dir = self.scene_root / "preprocess"
        frame_map_path = self.scene_root / "preprocess" / "frame_map.jsonl"
        self.frame_map = load_frame_map_jsonl_restore(frame_map_path, preprocess_dir)

        # Collect frame_ids where this tid exists
        self.samples: List[int] = []
        for fid_str, tracks in self.frame_map.items():
            if self.tid in tracks:
                self.samples.append(int(fid_str))
        self.samples.sort()
        print(f"--- FYI: found {len(self.samples)} frames for tid={self.tid}")

        # Paths
        self.images_dir = self.scene_root / "preprocess" / "images"
        self.masks_dir = self.scene_root / "preprocess" / mask_folder / str(self.tid)
        assert self.images_dir.exists(), f"Images dir not found: {self.images_dir}"
        assert self.masks_dir.exists(), f"Masks dir not found: {self.masks_dir}"

        # Sanity: check at least one frame
        if not self.samples:
            raise ValueError(f"No frames found for tid={self.tid}")

        # Cache H,W from cam_dicts (assuming constant across frames; we still read per-frame K)
        first_fid = self.samples[0]
        c0 = self.cam_dicts[first_fid]
        self.W_full, self.H_full = int(c0["W"]), int(c0["H"])
        self.W = self.W_full // self.downscale
        self.H = self.H_full // self.downscale

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        fid = self.samples[idx]

        # Load image & mask
        img_path = self.images_dir / f"frame_{fid:05d}.jpg"

        image = load_image(img_path, self.downscale)  # [3,H,W]

        mask_path = self.masks_dir / f"{fid:05d}.png"
        if mask_path.exists():
            mask = load_mask(mask_path, self.downscale)  # [H,W]
        else:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Camera intrinsics
        cam = self.cam_dicts[fid]
        fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
        # Adjust intrinsics for downscale
        fx /= self.downscale
        fy /= self.downscale
        cx /= self.downscale
        cy /= self.downscale
        K = torch.tensor([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)

        # SMPL params (86D) for this tid
        smpl_param = np.array(self.frame_map[fid][self.tid]["smpl_param"], dtype=np.float32)
        smpl_param = torch.from_numpy(smpl_param)  # [86]

        return {
            "fid": fid,
            "image": image,     # [3,H,W]
            "mask": mask,       # [H,W]
            "K": K,             # [3,3]
            "smpl_param": smpl_param  # [86]
        }


# ---------------------------------
# Gaussian model (canonical params)
# ---------------------------------
class CanonicalGaussians(nn.Module):
    def __init__(self, verts_c: torch.Tensor, weights_c: torch.Tensor, n_keep: int = 3000):
        """
        verts_c: [N,3] canonical SMPL vertices (from SMPLServer)
        weights_c: [N,24] SMPL skinning weights aligned to verts_c
        n_keep: randomly subsample Gaussians to keep training fast
        """
        super().__init__()
        device = verts_c.device
        N = verts_c.shape[0]

        if n_keep < N:
            idx = torch.randperm(N, device=device)[:n_keep]
            verts_c = verts_c[idx]
            weights_c = weights_c[idx]
        else:
            idx = torch.arange(N, device=device)

        self.register_buffer("idx", idx, persistent=False)
        self.register_buffer("weights_c", weights_c, persistent=False)  # [M,24]

        # Learnable canonical means, scales, colors, opacity
        self.means_c = nn.Parameter(verts_c.clone())                    # [M,3]
        # isotropic log-scale (start small relative to body size)
        init_sigma = 0.02  # ~2 cm if your canonical is in meters; TODO: adjust if your scale differs
        self.log_scales = nn.Parameter(torch.full((verts_c.shape[0],), math.log(init_sigma), device=device))
        # colors (initialized later from first frame), range [0,1]
        self.colors = nn.Parameter(torch.rand(verts_c.shape[0], 3, device=device))
        # opacities in (0,1) (we store in logit space for stability)
        self.opacity_logit = nn.Parameter(torch.full((verts_c.shape[0],), torch.logit(torch.tensor(0.1)), device=device, requires_grad=True))

    def opacity(self):
        return torch.sigmoid(self.opacity_logit)  # [M]

    def scales(self):
        return torch.exp(self.log_scales).clamp(1e-4, 1.0)  # [M]


# ---------------------------------
# LBS: canonical -> posed (camera)
# ---------------------------------
def lbs_apply(means_c: torch.Tensor, weights_c: torch.Tensor, T_rel: torch.Tensor) -> torch.Tensor:
    """
    means_c:  [M,3] canonical means
    weights_c:[M,24]
    T_rel:    [24,4,4] bone transforms canonical->current pose
    returns:  [M,3] posed in camera coordinates
    """
    M = means_c.shape[0]
    device = means_c.device
    xyz1 = torch.cat([means_c, torch.ones(M, 1, device=device)], dim=1)  # [M,4]

    # Expand for bones
    # xyz1_exp: [M,24,4,1], T_rel: [24,4,4]
    xyz1_exp = xyz1[:, None, :, None]            # [M,1,4,1]
    T_rel_exp = T_rel[None, :, :, :]             # [1,24,4,4]
    out = (T_rel_exp @ xyz1_exp).squeeze(-1)     # [M,24,4]
    out = out[..., :3]                           # [M,24,3]

    # Blend by weights
    posed = (weights_c.unsqueeze(-1) * out).sum(dim=1)  # [M,3]
    return posed


# ------------------------------------------------
# Simple differentiable 2D Gaussian "splat" render
# ------------------------------------------------
def render_gaussians_2d(
    uv: torch.Tensor,            # [M,2]
    z: torch.Tensor,             # [M]
    scales: torch.Tensor,        # [M] sigma in pixels
    colors: torch.Tensor,        # [M,3] in [0,1]
    opacity: torch.Tensor,       # [M] in (0,1)
    H: int, W: int,
    soft_z_temp: float = 0.0,    # 0 => no depth fading; start with 0
    truncate: float = 2.5,       # tighter footprint
    z_sort: bool = True,         # painter’s algorithm (back-to-front)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterize isotropic 2D Gaussians with correct 'over' compositing:
        A_new = A + (1 - A) * a_i
        C_new = C + (1 - A) * (a_i * c_i)
    where a_i = kernel * per-splat opacity * visibility
    """
    device = uv.device
    img  = torch.zeros(3, H, W, device=device)
    A    = torch.zeros(H, W, device=device)

    # Visibility weight (keep simple first)
    vis_w = torch.ones_like(z)
    if soft_z_temp and soft_z_temp > 0:
        vis_w = torch.exp(-soft_z_temp * z).clamp(0.0, 1.0)

    # Optional painter’s algorithm: draw far → near
    if z_sort:
        order = torch.argsort(z, descending=True)  # far first
    else:
        order = torch.arange(uv.shape[0], device=device)

    for i in order.tolist():
        cx, cy = uv[i]
        s = float(scales[i].item())
        if s < 1e-3:
            continue

        rad = int(truncate * s) + 1
        u0, u1 = int(cx.item()) - rad, int(cx.item()) + rad + 1
        v0, v1 = int(cy.item()) - rad, int(cy.item()) + rad + 1
        if u1 <= 0 or v1 <= 0 or u0 >= W or v0 >= H:
            continue

        uu = torch.arange(max(0, u0), min(W, u1), device=device)
        vv = torch.arange(max(0, v0), min(H, v1), device=device)
        if uu.numel() == 0 or vv.numel() == 0:
            continue

        U, V = torch.meshgrid(uu, vv, indexing="xy")
        du2 = (U.float() - cx) ** 2
        dv2 = (V.float() - cy) ** 2
        s2  = 2.0 * (s ** 2)

        kernel = torch.exp(-(du2 + dv2) / s2)              # [w,h]
        a_i    = (kernel * opacity[i] * vis_w[i]).clamp(0, 1)

        # Pull current alpha
        A_patch = A[vv[:, None], uu[None, :]]

        # Over compositing
        one_mA  = (1.0 - A_patch)
        A_new   = A_patch + one_mA * a_i
        C_add   = (one_mA * a_i)[None, :, :] * colors[i][:, None, None]

        img[:, vv[:, None], uu[None, :]] += C_add
        A[vv[:, None], uu[None, :]] = A_new

    return img.clamp(0, 1), A.clamp(0, 1)


# -----------------------------
# Projection: camera -> pixels
# -----------------------------
def project_points(
    X_cam: torch.Tensor,
    K: torch.Tensor,
    flip_z: bool = False,
    rz180: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X_cam: [M,3] in camera-view coordinates.
    K: [3,3] intrinsics.
    flip_z: set True if your renderer uses -Z forward (OpenGL-style).
    rz180: set True to apply a global 180° rotation around Z (x->-x, y->-y).
           This matches the canonical preview fix you discovered.
    """
    Xc = X_cam.clone()

    if rz180:
        # 180° rotation around Z: (x,y,z) -> (-x,-y,z)
        Xc[:, 0] = -Xc[:, 0]
        Xc[:, 1] = -Xc[:, 1]

    if flip_z:
        Xc[:, 2] = -Xc[:, 2]

    Z = Xc[:, 2].clamp(min=1e-6)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * (Xc[:, 0] / Z) + cx
    v = fy * (Xc[:, 1] / Z) + cy
    uv = torch.stack([u, v], dim=-1)
    return uv, Z


# -----------------------------------
# Trainer (one tid, no SfM)
# -----------------------------------
class Trainer:
    def __init__(
        self,
        scene_root: str,
        tid: int,
        device: str = "cuda",
        downscale: int = 2,
        n_gauss_keep: int = 3000,
        sigma_mult: float = 15.0,  # multiplier for scale -> pixel sigma heuristic
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sigma_mult = sigma_mult
        print(f"--- FYI: using {self.device}")
        self.dataset = HumanOnlyDataset(scene_root, tid, downscale=downscale)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        print(f"--- FYI: dataset has {len(self.dataset)} samples and using batch size 1")
        self.trn_dir = Path(scene_root) / "training"
        self.trn_dir.mkdir(parents=True, exist_ok=True)
        self.trn_viz_dir = self.trn_dir / "visualizations"
        self.trn_viz_dir.mkdir(parents=True, exist_ok=True)
        self.trn_viz_canon_dir = self.trn_viz_dir / "canonical"
        self.trn_viz_canon_dir.mkdir(parents=True, exist_ok=True)

        # --- SMPL server init & canonical cache ---
        self.smpl_server = SMPLServer().to(self.device).eval()
        with torch.no_grad():
            # use canonical cached from server
            verts_c = self.smpl_server.verts_c[0].to(self.device)      # [6890,3]
            weights_c = self.smpl_server.weights_c[0].to(self.device)  # [6890,24]

        # Gaussians in canonical space
        self.gaus = CanonicalGaussians(verts_c, weights_c, n_keep=n_gauss_keep).to(self.device)
        param_groups = [
            {"params": [self.gaus.means_c],       "lr": 1e-4},
            {"params": [self.gaus.log_scales],    "lr": 1e-4},
            {"params": [self.gaus.opacity_logit], "lr": 1e-4},
            {"params": [self.gaus.colors],        "lr": 2e-3},
        ]

        # Optimizer: Gaussians + (optional) focal
        self.opt = torch.optim.Adam(param_groups)
        self.grad_clip = 1.0  # store a value
        params = list(self.gaus.parameters())
        print(f"--- FYI: training {sum(p.numel() for p in params)} parameters")

        # Warm-start colors from first available frame (project once)
        self._init_colors()

    @torch.no_grad()
    def _init_colors(self):
        # TODO: I need to understand what on earth is going on in here
        # Use first frame in dataset to initialize colors by sampling image at projections
        sample = self.dataset[0]
        image = sample["image"].to(self.device)  # [3,H,W]
        K = sample["K"].to(self.device)
        smpl_param = sample["smpl_param"].unsqueeze(0).to(self.device)  # [1,86]

        out = self.smpl_server(smpl_param, absolute=False)
        T_rel = out["smpl_tfs"][0].to(self.device)       # [24,4,4]
        means_cam = lbs_apply(self.gaus.means_c, self.gaus.weights_c, T_rel)  # [M,3]

        uv, Z = project_points(means_cam, K)  # [M,2], [M]
        # debug_projection_stats(uv, Z, image.shape[-2], image.shape[-1], tag="init_colors")

        # Bilinear sample colors (ignore out-of-bounds)
        H, W = image.shape[-2:]
        u = uv[:, 0].clamp(0, W - 1 - 1e-3)
        v = uv[:, 1].clamp(0, H - 1 - 1e-3)
        grid = torch.stack([(u / (W - 1)) * 2 - 1, (v / (H - 1)) * 2 - 1], dim=-1).view(1, -1, 1, 2)  # [1,M,1,2]
        # grid_sample expects [N,C,H,W] and grid [N,H_out,W_out,2]
        sampled = F.grid_sample(image.unsqueeze(0), grid, align_corners=True, mode="bilinear")  # [1,3,M,1]
        colors = sampled.squeeze(0).squeeze(-1).permute(1, 0)  # [M,3]
        self.gaus.colors.data.copy_(colors.clamp(0, 1))

        save_path = self.trn_viz_canon_dir / f"preview_init.png"
        save_canonical_preview(
            gaus=self.gaus,
            out_path=save_path,
            H=640, W=480,
            fov_deg=18.0,
            flip_z=False,         # set True if your renderer uses -Z forward
            color_mode="learned",  # or "bone" to visualize by SMPL bone IDs
            sigma_mult=self.sigma_mult
        )


    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        image = batch["image"].squeeze(0).to(self.device)  # [3,H,W]
        mask  = batch["mask"].squeeze(0).to(self.device)   # [H,W]
        K  = batch["K"].squeeze(0).to(self.device)      # [3,3]
        smpl_param = batch["smpl_param"].to(self.device)  # [1,86]
        H, W = image.shape[-2:]

#        with torch.no_grad():
            #self.gaus.opacity_logit.data.clamp_(min=torch.logit(torch.tensor(0.02)),
                                                #max=torch.logit(torch.tensor(0.6)))
            #self.gaus.log_scales.data.clamp_(math.log(1e-3), math.log(0.08))

        out = self.smpl_server(smpl_param, absolute=False)
        T_rel = out["smpl_tfs"][0]   # [24,4,4]
        means_cam = lbs_apply(self.gaus.means_c, self.gaus.weights_c, T_rel)  # [M,3]

        uv, Z = project_points(means_cam, K)  # [M,2], [M]
        # debug_projection_stats(uv, Z, H, W, tag=f"fid{fid}")
        # scales_pix = self.gaus.scales() * (K[0,0] / max(H, W)) * self.sigma_mult    
        eps = 1e-6
        sigma_mult = 8.0          # try 8 first; you can go down to 6
        scales_pix = (self.gaus.scales() * K[0,0] / (Z + eps) * sigma_mult)
        scales_pix = scales_pix.clamp(0.6, 7.0)   # was wider; tighten upper clamp

        # NOTE: crude heuristic scale -> pixels; adjust multiplier if splats look too big/small.

        rgb_pred, alpha_pred = render_gaussians_2d(
            uv=uv, z=Z, scales=scales_pix, colors=self.gaus.colors,
            opacity=self.gaus.opacity(), H=H, W=W, soft_z_temp=1e-2, truncate=3.0
        )  # [3,H,W], [H,W]

        # Losses
        mask3 = mask.expand_as(rgb_pred)
        l_rgb = (mask3 * (rgb_pred - image).abs()).mean()

        if it_number % 20 == 0:
            save_loss_visualization(
                image=image,
                mask=mask,
                rgb_pred=rgb_pred,
                out_path=self.trn_viz_dir / "debug" / f"lossviz_it{it_number:05d}.png"
            )

        # Silhouette BCE
        eps = 1e-6
        l_sil = F.binary_cross_entropy(alpha_pred.clamp(eps, 1 - eps), mask)

        # Regularizers
        l_scale_reg = self.gaus.scales().mean() * 1e-3
        l_opacity_reg = (self.gaus.opacity() ** 2).mean() * 1e-4

        loss = l_rgb + 1.0 * l_sil + l_scale_reg + l_opacity_reg 

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.gaus.parameters(), self.grad_clip)
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "l_rgb": float(l_rgb.item()),
            "l_sil": float(l_sil.item()),
        }

    def train_loop(self, iters: int = 2000):
        it = 0
        with tqdm(total=iters, desc="Training Progress", dynamic_ncols=True) as pbar:
            while it < iters:
                for batch in self.loader:
                    it += 1
                    logs = self.step(batch, it)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{logs['loss']:.4f}",
                        "rgb": f"{logs['l_rgb']:.4f}",
                        "sil": f"{logs['l_sil']:.4f}",
                    })

                    # Log to wandb
                    wandb.log({
                        "loss": logs["loss"],
                        "l_rgb": logs["l_rgb"],
                        "l_sil": logs["l_sil"],
                        "iteration": it,
                    })

                    # Periodic canonical preview
                    if it % 50 == 0 or it == iters:
                        save_path = self.trn_viz_canon_dir / f"preview_it{it:05d}.png"
                        save_canonical_preview(
                            gaus=self.gaus,
                            out_path=save_path,
                            H=640, W=480,
                            fov_deg=18.0,
                            flip_z=False,         # set True if your renderer uses -Z forward
                            color_mode="learned",  # or "bone" to visualize by SMPL bone IDs
                            sigma_mult=self.sigma_mult
                        )

                    if it >= iters:
                        break


@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):

    print("ℹ️ Initializing Trainer")
    init_logging(cfg)
    trainer = Trainer(
        scene_root=cfg.output_dir,
        tid=cfg.tid,
        device=cfg.device,
        downscale=cfg.downscale,
        n_gauss_keep=cfg.n_gauss,
        sigma_mult=cfg.sigma_mult,
    )
    print("✅ Trainer initialized.\n")

    print("ℹ️ Starting training")
    trainer.train_loop(iters=cfg.iters)
    print("✅ Training completed.")

if __name__ == "__main__":
    main()
