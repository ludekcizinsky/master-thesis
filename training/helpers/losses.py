import torch
import torch.nn.functional as F

def _chunked_cdist_min(X, Y, chunk=20000):
    M = X.shape[0]
    mins = torch.empty(M, device=X.device, dtype=X.dtype)
    for i in range(0, M, chunk):
        d = torch.cdist(X[i:i+chunk], Y)        # [m,V]
        mins[i:i+chunk] = d.min(dim=1).values   # [m]
    return mins

def anchor_to_smpl_surface(
    means_c: torch.Tensor,        # [M,3]  (ALL splats, not just first m0)
    smpl_verts_c: torch.Tensor,   # [V,3]
    free_radius: float = 0.001,    # ~1 mm
) -> torch.Tensor:
    d = _chunked_cdist_min(means_c, smpl_verts_c)   # [M]
    excess = (d - free_radius).clamp_min(0.0)
    per = excess**2
    return per.mean()


def opacity_distance_penalty(
    opa_logits: torch.Tensor,       # [M] raw opacity params; alpha = sigmoid(opa_logits)
    means_c: torch.Tensor,          # [M,3] canonical means
    smpl_verts_c: torch.Tensor,     # [V,3]
    free_radius: float = 0.01,      # meters (e.g., 5–10 mm)
    smooth: float = 0.01,           # meters: how soft the ramp grows
    stop_means_grad: bool = True,
):
    # alpha in (0,1)
    alpha = torch.sigmoid(opa_logits)        # [M]

    # distance to SMPL
    d = _chunked_cdist_min(means_c, smpl_verts_c)     # [M]
    if stop_means_grad:
        d = d.detach()  # regulate only opacity; use a separate anchor for means

    # soft hinge: weight ~ softplus((d - r)/s)
    w = F.softplus((d - free_radius) / smooth)         # [M], ~0 inside band, grows outside

    # penalize opacity outside the body band
    loss = (w * alpha).mean()

    return loss


def prepare_input_for_loss(gt_imgs: torch.Tensor, renders: torch.Tensor, masks: torch.Tensor, kind="bbox_crop"):
    """
    Args:
        gt_imgs: [B,H,W,3] in [0,1]
        renders: [B,H,W,3] in [0,1]
        masks:   [B,H,W] in {0,1}

    Returns:
        gt_crop: [B,h,w,3] in [0,1]
        pr_crop: [B,h,w,3] in [0,1]
    """

    gt_imgs = gt_imgs.clamp(0, 1).float()          # [B,H,W,3]
    renders = renders.clamp(0, 1).float()

    if kind == "bbox_crop":
        # Apply mask to the ground truth (keep only target person pixels)
        gt_imgs *= masks.unsqueeze(-1)

        # bbox crop around the mask to avoid huge easy background
        ys, xs = torch.where(masks[0] > 0.5)
        if ys.numel() > 0:
            pad = 8
            y0 = max(int(ys.min().item()) - pad, 0)
            y1 = min(int(ys.max().item()) + pad + 1, gt_imgs.shape[1])
            x0 = max(int(xs.min().item()) - pad, 0)
            x1 = min(int(xs.max().item()) + pad + 1, gt_imgs.shape[2])

            gt_crop  = gt_imgs[:, y0:y1, x0:x1, :]
            pr_crop  = renders[:, y0:y1, x0:x1, :]
        else:
            # fallback if mask empty
            gt_crop, pr_crop = gt_imgs, renders

        return gt_crop, pr_crop
    elif kind == "bbox":
        # Apply mask to the ground truth (keep only target person pixels)
        gt_imgs *= masks.unsqueeze(-1)

        # Mask out the render outside the bbox of the mask (keep full GT)
        pad = 8
        ys, xs = torch.where(masks[0] > 0.5)
        y0 = max(int(ys.min().item()) - pad, 0)
        y1 = min(int(ys.max().item()) + pad + 1, renders.shape[1])
        x0 = max(int(xs.min().item()) - pad, 0)
        x1 = min(int(xs.max().item()) + pad + 1, renders.shape[2])

        pr_bbox = torch.zeros_like(renders)
        pr_bbox[:, y0:y1, x0:x1, :] = renders[:, y0:y1, x0:x1, :]

        return gt_imgs, pr_bbox

    elif kind == "tight_crop":
        # Apply tight mask (keep only target person pixels)
        gt_imgs *= masks.unsqueeze(-1)
        renders *= masks.unsqueeze(-1)

        # bbox crop around the mask to avoid huge easy background
        ys, xs = torch.where(masks[0] > 0.5)
        if ys.numel() > 0:
            y0 = int(ys.min().item())
            y1 = int(ys.max().item()) + 1
            x0 = int(xs.min().item())
            x1 = int(xs.max().item()) + 1

            gt_crop  = gt_imgs[:, y0:y1, x0:x1, :]
            pr_crop  = renders[:, y0:y1, x0:x1, :]
        else:
            # fallback if mask empty
            gt_crop, pr_crop = gt_imgs, renders
        
        return gt_crop, pr_crop

    elif kind == "tight":
        # Apply tight mask (keep only target person pixels)
        gt_imgs *= masks.unsqueeze(-1)
        renders *= masks.unsqueeze(-1)

        return gt_imgs, renders

    else:
        raise ValueError(f"Unknown prepare_input_for_loss kind: {kind}")