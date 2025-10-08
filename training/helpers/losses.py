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



def bbox_img_crop(gt_imgs: torch.Tensor, renders: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # gt_imgs, renders: [B, H, W, 3]; masks: [B, H, W] (bool or float)
    B = masks.shape[0]
    crops_gt = []
    crops_pr = []

    for b in range(B):
        mask = masks[b] > 0.5 if masks.dtype != torch.bool else masks[b]
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            crops_gt.append(gt_imgs[b:b+1])
            crops_pr.append(renders[b:b+1])
            continue

        y0 = ys.min().item()
        y1 = ys.max().item() + 1
        x0 = xs.min().item()
        x1 = xs.max().item() + 1

        crops_gt.append(gt_imgs[b:b+1, y0:y1, x0:x1, :])
        crops_pr.append(renders[b:b+1, y0:y1, x0:x1, :])

    return torch.cat(crops_gt, dim=0), torch.cat(crops_pr, dim=0)



def prepare_input_for_loss(gt_imgs: torch.Tensor, renders: torch.Tensor, human_masks: torch.Tensor, cfg):

    # input shapes: [B,H,W,3], [B,H,W,3], [B,P,H,W]
    # case 1: full image supervision
    if len(cfg.tids) > 0 and cfg.train_bg:
        return gt_imgs, renders
    # case 2: static bg only - mask out all humans and keep only background
    elif len(cfg.tids) == 0 and cfg.train_bg:
        bg_mask = (1.0 - human_masks.sum(dim=1)).clamp(0.0, 1.0)  # [B,H,W]
        gt_imgs *= bg_mask.unsqueeze(-1)
        renders *= bg_mask.unsqueeze(-1)
        return gt_imgs, renders
    # case 3: dynamic only (keep only target people, mask out rest)
    elif len(cfg.tids) > 0 and not cfg.train_bg: 
        # keep only target people and mask out rest
        joined_human_masks = human_masks.sum(dim=1).clamp(0.0, 1.0)  # [B,H,W]
        gt_imgs *= joined_human_masks.unsqueeze(-1)
        renders *= joined_human_masks.unsqueeze(-1)
        # crop to bbox of target people
        gt_imgs, renders = bbox_img_crop(gt_imgs, renders, joined_human_masks)
        return gt_imgs, renders
    else:
        raise ValueError("Invalid cfg: cannot train without bg and without humans.")

