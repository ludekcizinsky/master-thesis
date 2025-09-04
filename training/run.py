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

from utils.smpl_deformer.smpl_server import SMPLServer
from utils.io import load_frame_map_jsonl_restore
from preprocess.helpers.cameras import load_camdicts_json

from tqdm import tqdm

# -----------------------------
# Utility: image & mask loading
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
        self.opacity_logit = nn.Parameter(torch.full((verts_c.shape[0],), torch.logit(torch.tensor(0.1)), device=device))

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
    uv: torch.Tensor,            # [M,2] projected centers in pixels
    z: torch.Tensor,             # [M] depth (used only for soft visibility sort)
    scales: torch.Tensor,        # [M] isotropic sigma in pixels
    colors: torch.Tensor,        # [M,3] in [0,1]
    opacity: torch.Tensor,       # [M] in (0,1)
    H: int, W: int,
    soft_z_temp: float = 1e-2,   # smaller → sharper z-occlusion weighting
    truncate: float = 3.0        # window = +/- truncate*sigma
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Super-simple splatter: accumulates per-Gaussian kernels onto the canvas.
    Visibility: soft z-weight using exp(-temp * z). (Not physically correct; good enough to start.)
    Returns: rgb [3,H,W], alpha [H,W], both in [0,1].
    """
    device = uv.device
    img = torch.zeros(3, H, W, device=device)
    alpha = torch.zeros(H, W, device=device)

    # Windowed per-splat loop. This is O(M * window^2).
    # Keep M small (e.g., 2-4k) and downscale images for a first test.
    # TODO: replace with diff-gaussian-rasterization for speed/quality later.
    vis_w = torch.exp(-soft_z_temp * z).clamp(0.0, 1.0)  # [M]

    for i in range(uv.shape[0]):
        cx, cy = uv[i]  # pixel center (u,v)
        s = scales[i]
        if s.item() < 1e-3:
            continue  # too tiny
        rad = int(truncate * s.item()) + 1
        u0, u1 = int(cx.item()) - rad, int(cx.item()) + rad + 1
        v0, v1 = int(cy.item()) - rad, int(cy.item()) + rad + 1
        if u1 < 0 or v1 < 0 or u0 >= W or v0 >= H:
            continue
        uu = torch.arange(max(0, u0), min(W, u1), device=device)
        vv = torch.arange(max(0, v0), min(H, v1), device=device)
        if uu.numel() == 0 or vv.numel() == 0:
            continue
        U, V = torch.meshgrid(uu, vv, indexing="xy")  # [w,h]
        du2 = (U - cx) ** 2
        dv2 = (V - cy) ** 2
        s2 = (2.0 * (s ** 2))
        kernel = torch.exp(-(du2 + dv2) / s2)  # [w,h]
        k = kernel * opacity[i] * vis_w[i]     # weight

        # alpha accumulation (1 - prod(1-a_i)) approximated by additive with clamp
        alpha_patch = alpha[vv[:, None], uu[None, :]]
        new_alpha = (alpha_patch + k).clamp(0, 1)

        # RGB accumulate (pre-multiplied alpha style)
        col = colors[i][:, None, None]  # [3,1,1]
        img_patch = img[:, vv[:, None], uu[None, :]]
        img[:, vv[:, None], uu[None, :]] = img_patch * (alpha_patch / (new_alpha + 1e-8)) * (new_alpha > 1e-5) + col * k

        # Write alpha
        alpha[vv[:, None], uu[None, :]] = new_alpha

    return img.clamp(0, 1), alpha.clamp(0, 1)


# -----------------------------
# Projection: camera -> pixels
# -----------------------------
def project_points(X_cam: torch.Tensor, K: torch.Tensor, flip_z: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X_cam: [M,3] in camera-view coordinates. Assumes Z>0 means in front of camera.
    K: [3,3] intrinsics.
    flip_z: set True if your coordinates are OpenGL-style (camera looking down -Z).
    Returns: (uv [M,2], z [M])
    """
    if flip_z:
        X_cam = X_cam.clone()
        X_cam[:, 2] = -X_cam[:, 2]

    Z = X_cam[:, 2].clamp(min=1e-6)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * (X_cam[:, 0] / Z) + cx
    v = fy * (X_cam[:, 1] / Z) + cy
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
        lr: float = 1e-2,
        flip_z: bool = False,
        learn_per_frame_f: bool = True,
        focal_prior_weight: float = 1e-4,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"--- FYI: using {self.device}")
        self.dataset = HumanOnlyDataset(scene_root, tid, downscale=downscale)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        print(f"--- FYI: dataset has {len(self.dataset)} samples and using batch size 1")

        # --- SMPL server init & canonical cache ---
        self.smpl_server = SMPLServer().to(self.device).eval()
        with torch.no_grad():
            # use canonical cached from server
            verts_c = self.smpl_server.verts_c[0].to(self.device)      # [6890,3]
            weights_c = self.smpl_server.weights_c[0].to(self.device)  # [6890,24]

        # Gaussians in canonical space
        self.gaus = CanonicalGaussians(verts_c, weights_c, n_keep=n_gauss_keep).to(self.device)

        # Intrinsics (per-frame focal learnable option)
        # We'll keep cx,cy fixed; optionally learn fx,fy per frame around their initial values.
        self.learn_per_frame_f = learn_per_frame_f
        self.focal_prior_weight = focal_prior_weight
        self.flip_z = flip_z

        if self.learn_per_frame_f:
            # Map fid->Parameter
            self.fx_params: Dict[int, nn.Parameter] = {}
            self.fy_params: Dict[int, nn.Parameter] = {}
            for fid in self.dataset.samples:
                # Capture per-frame K to seed
                K = self.dataset.cam_dicts[fid]
                fx0, fy0 = float(K["fx"]) / self.dataset.downscale, float(K["fy"]) / self.dataset.downscale
                self.fx_params[fid] = nn.Parameter(torch.tensor(fx0, dtype=torch.float32, device=self.device))
                self.fy_params[fid] = nn.Parameter(torch.tensor(fy0, dtype=torch.float32, device=self.device))
            self.f_params = nn.ParameterList(list(self.fx_params.values()) + list(self.fy_params.values()))
        else:
            self.f_params = nn.ParameterList([])
        print("--- FYI: learning per frame focal is enabled" if self.learn_per_frame_f else "--- FYI: keeping focal fixed")

        # Optimizer: Gaussians + (optional) focal
        params = list(self.gaus.parameters()) + list(self.f_params.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)

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

        uv, Z = project_points(means_cam, K, flip_z=self.flip_z)  # [M,2], [M]
        # Bilinear sample colors (ignore out-of-bounds)
        H, W = image.shape[-2:]
        u = uv[:, 0].clamp(0, W - 1 - 1e-3)
        v = uv[:, 1].clamp(0, H - 1 - 1e-3)
        grid = torch.stack([(u / (W - 1)) * 2 - 1, (v / (H - 1)) * 2 - 1], dim=-1).view(1, -1, 1, 2)  # [1,M,1,2]
        # grid_sample expects [N,C,H,W] and grid [N,H_out,W_out,2]
        sampled = F.grid_sample(image.unsqueeze(0), grid, align_corners=True, mode="bilinear")  # [1,3,M,1]
        colors = sampled.squeeze(0).squeeze(-1).permute(1, 0)  # [M,3]
        self.gaus.colors.data.copy_(colors.clamp(0, 1))

    def step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        image = batch["image"].to(self.device)  # [3,H,W]
        mask  = batch["mask"].squeeze(0).to(self.device)   # [H,W]
        K_in  = batch["K"].squeeze(0).to(self.device)      # [3,3]
        smpl_param = batch["smpl_param"].to(self.device)  # [1,86]
        fid = batch["fid"].item()
        H, W = image.shape[-2:]

        # Optionally replace fx,fy with learnable per-frame values (keep cx,cy fixed)
        if self.learn_per_frame_f:
            K = K_in.clone()
            K[0, 0] = self.fx_params[fid]
            K[1, 1] = self.fy_params[fid]
        else:
            K = K_in

        out = self.smpl_server(smpl_param, absolute=False)
        T_rel = out["smpl_tfs"][0]   # [24,4,4]
        means_cam = lbs_apply(self.gaus.means_c, self.gaus.weights_c, T_rel)  # [M,3]

        uv, Z = project_points(means_cam, K, flip_z=self.flip_z)  # [M,2], [M]
        scales_pix = self.gaus.scales() * (K[0,0] / max(H, W)) * 60.0
        # NOTE: crude heuristic scale -> pixels; adjust multiplier if splats look too big/small.

        rgb_pred, alpha_pred = render_gaussians_2d(
            uv=uv, z=Z, scales=scales_pix, colors=self.gaus.colors,
            opacity=self.gaus.opacity(), H=H, W=W, soft_z_temp=1e-2, truncate=3.0
        )  # [3,H,W], [H,W]

        # Losses
        mask3 = mask.expand_as(rgb_pred)
        l_rgb = (mask3 * (rgb_pred - image).abs()).mean()

        # Silhouette BCE
        eps = 1e-6
        l_sil = F.binary_cross_entropy(alpha_pred.clamp(eps, 1 - eps), mask)

        # Regularizers
        l_scale_reg = self.gaus.scales().mean() * 1e-3
        l_opacity_reg = (self.gaus.opacity() ** 2).mean() * 1e-4

        # Focal prior
        if self.learn_per_frame_f:
            fx0 = K_in[0,0].detach()
            fy0 = K_in[1,1].detach()
            l_focal = (self.fx_params[fid] - fx0).pow(2) + (self.fy_params[fid] - fy0).pow(2)
            l_focal = l_focal * self.focal_prior_weight
        else:
            l_focal = torch.tensor(0.0, device=self.device)

        loss = l_rgb + 0.5 * l_sil + l_scale_reg + l_opacity_reg + l_focal

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return {
            "loss": float(loss.item()),
            "l_rgb": float(l_rgb.item()),
            "l_sil": float(l_sil.item()),
            "l_focal": float(l_focal.item()) if self.learn_per_frame_f else 0.0,
        }

    def train_loop(self, iters: int = 2000):
        it = 0
        with tqdm(total=iters, desc="Training Progress", dynamic_ncols=True) as pbar:
            while it < iters:
                for batch in self.loader:
                    it += 1
                    logs = self.step(batch)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{logs['loss']:.4f}",
                        "rgb": f"{logs['l_rgb']:.4f}",
                        "sil": f"{logs['l_sil']:.4f}",
                        "focal": f"{logs['l_focal']:.6f}"
                    })

                    if it >= iters:
                        break



@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):
    print("ℹ️ Initializing Trainer")
    trainer = Trainer(
        scene_root=cfg.output_dir,
        tid=cfg.tid,
        device=cfg.device,
        downscale=cfg.downscale,
        n_gauss_keep=cfg.n_gauss,
        lr=cfg.lr,
        flip_z=cfg.flip_z,
        learn_per_frame_f=not cfg.no_learn_f,
    )
    print("✅ Trainer initialized.\n")

    print("ℹ️ Starting training")
    trainer.train_loop(iters=cfg.iters)
    print("✅ Training completed.")

if __name__ == "__main__":
    main()
