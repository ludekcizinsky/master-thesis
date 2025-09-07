from typing import Tuple

import torch
import wandb
from omegaconf import OmegaConf
import os

from pathlib import Path
from PIL import Image
import numpy as np

def init_logging(cfg):     
    wandb_path = Path("/scratch/izar/cizinsky/thesis/")
    os.makedirs(wandb_path, exist_ok=True)
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.logger.tags,
        dir=wandb_path,
        group=cfg.scene_name,
        mode="online" if not cfg.debug else "disabled",
    )
    if cfg.debug:
        print("--- FYI: Running in debug mode, wandb logging is disabled.")
    else:
        print(f"--- FYI: Logging to wandb project {cfg.logger.project}, entity {cfg.logger.entity}, group {cfg.scene_name}.")

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