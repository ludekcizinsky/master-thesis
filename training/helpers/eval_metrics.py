from typing import Optional, Dict

import torch
import torch.nn.functional as F

import kornia
import pyiqa


# ---------------------------------------------------------------------------
# Pose evaluation metrics
# ---------------------------------------------------------------------------

def _flatten_smpl_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.dim() == 4:
        return pose.reshape(pose.shape[0], pose.shape[1], -1)
    if pose.dim() == 3:
        return pose
    raise ValueError(f"Unexpected pose shape: {pose.shape}")


def _smpl_params_to_joints(
    smpl_params: Dict[str, torch.Tensor],
    smpl_layer,
) -> torch.Tensor:
    if "contact" in smpl_params:
        smpl_params = {k: v for k, v in smpl_params.items() if k != "contact"}

    betas = smpl_params["betas"]
    body_pose = smpl_params["body_pose"]
    root_pose = smpl_params["root_pose"]
    trans = smpl_params["trans"]

    if betas.dim() != 3:
        raise ValueError(f"Expected betas shape [B,P,?], got {betas.shape}")

    bsize, npeople = betas.shape[:2]
    betas = betas.reshape(bsize * npeople, -1)
    body_pose = _flatten_smpl_pose(body_pose).reshape(bsize * npeople, -1)
    root_pose = _flatten_smpl_pose(root_pose).reshape(bsize * npeople, -1)
    trans = trans.reshape(bsize * npeople, -1)

    output = smpl_layer(
        global_orient=root_pose,
        body_pose=body_pose,
        betas=betas,
        transl=trans,
    )
    joints = output.joints
    return joints.reshape(bsize, npeople, joints.shape[1], 3)


def _smpl_params_to_vertices(
    smpl_params: Dict[str, torch.Tensor],
    smpl_layer,
) -> torch.Tensor:
    if "contact" in smpl_params:
        smpl_params = {k: v for k, v in smpl_params.items() if k != "contact"}

    betas = smpl_params["betas"]
    body_pose = smpl_params["body_pose"]
    root_pose = smpl_params["root_pose"]
    trans = smpl_params["trans"]

    if betas.dim() != 3:
        raise ValueError(f"Expected betas shape [B,P,?], got {betas.shape}")

    bsize, npeople = betas.shape[:2]
    betas = betas.reshape(bsize * npeople, -1)
    body_pose = _flatten_smpl_pose(body_pose).reshape(bsize * npeople, -1)
    root_pose = _flatten_smpl_pose(root_pose).reshape(bsize * npeople, -1)
    trans = trans.reshape(bsize * npeople, -1)

    output = smpl_layer(
        global_orient=root_pose,
        body_pose=body_pose,
        betas=betas,
        transl=trans,
    )
    vertices = output.vertices
    return vertices.reshape(bsize, npeople, vertices.shape[1], 3)


def compute_smpl_mpjpe_per_frame(
    pred_smpl_params: Dict[str, torch.Tensor],
    gt_smpl_params: Dict[str, torch.Tensor],
    smpl_layer,
    unit: str = "mm",
) -> torch.Tensor:
    pred_joints = _smpl_params_to_joints(pred_smpl_params, smpl_layer)
    gt_joints = _smpl_params_to_joints(gt_smpl_params, smpl_layer)
    per_joint = torch.linalg.norm(pred_joints - gt_joints, dim=-1)
    per_frame = per_joint.mean(dim=(1, 2))

    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported MPJPE unit: {unit}")

    return per_frame * scale


def compute_smpl_mve_per_frame(
    pred_smpl_params: Dict[str, torch.Tensor],
    gt_smpl_params: Dict[str, torch.Tensor],
    smpl_layer,
    unit: str = "mm",
) -> torch.Tensor:
    pred_verts = _smpl_params_to_vertices(pred_smpl_params, smpl_layer)
    gt_verts = _smpl_params_to_vertices(gt_smpl_params, smpl_layer)
    per_vertex = torch.linalg.norm(pred_verts - gt_verts, dim=-1)
    per_frame = per_vertex.mean(dim=(1, 2))

    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported MVE unit: {unit}")

    return per_frame * scale


def compute_smpl_contact_distance_per_frame(
    pred_smpl_params: Dict[str, torch.Tensor],
    gt_contact: torch.Tensor,
    smpl_layer,
    unit: str = "mm",
    invalid_value: int = 0,
) -> torch.Tensor:
    pred_verts = _smpl_params_to_vertices(pred_smpl_params, smpl_layer)
    if gt_contact.dim() == 2:
        gt_contact = gt_contact.unsqueeze(0)
    if gt_contact.dim() != 3:
        raise ValueError(f"Expected contact shape [B,P,V] or [P,V], got {gt_contact.shape}")

    if gt_contact.shape[:2] != pred_verts.shape[:2] or gt_contact.shape[2] != pred_verts.shape[2]:
        raise ValueError(
            f"Contact shape {gt_contact.shape} does not match verts {pred_verts.shape}"
        )

    if pred_verts.shape[1] != 2:
        raise ValueError(f"Contact distance expects 2 people, got {pred_verts.shape[1]}")

    device = pred_verts.device
    gt_contact = gt_contact.to(device)
    num_verts = pred_verts.shape[2]
    per_frame = []
    for b in range(pred_verts.shape[0]):
        verts_b = pred_verts[b]
        contact_b = gt_contact[b].long()
        cd_vals = []
        for p in range(2):
            other = 1 - p
            corr = contact_b[p]
            valid = (
                (corr != invalid_value)
                & (corr >= 0)
                & (corr < num_verts)
            )
            if not torch.any(valid):
                cd_vals.append(None)
                continue
            idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            src = verts_b[p, idx]
            dst = verts_b[other, corr[idx]]
            cd_vals.append(torch.linalg.norm(src - dst, dim=-1).mean())

        if cd_vals[0] is not None and cd_vals[1] is not None:
            cd = 0.5 * (cd_vals[0] + cd_vals[1])
        elif cd_vals[0] is not None:
            cd = cd_vals[0]
        elif cd_vals[1] is not None:
            cd = cd_vals[1]
        else:
            cd = torch.tensor(0.0, device=device)
        per_frame.append(cd)

    per_frame = torch.stack(per_frame, dim=0)

    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 100.0
    elif unit == "mm":
        scale = 1000.0
    else:
        raise ValueError(f"Unsupported CD unit: {unit}")

    return per_frame * scale


def _points_world_to_cam(points_world: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    if c2w.dim() == 2:
        c2w = c2w.unsqueeze(0)
    if c2w.dim() != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w shape [B,4,4] or [4,4], got {c2w.shape}")
    w2c = torch.inverse(c2w)
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3]
    points_cam = torch.einsum("bpj,bkj->bpk", points_world, rot)
    return points_cam + trans[:, None, :]


def compute_smpl_pcdr_per_frame(
    pred_smpl_params: Dict[str, torch.Tensor],
    gt_smpl_params: Dict[str, torch.Tensor],
    c2w: torch.Tensor,
    tau: float = 0.15,
    gamma: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """Compute BEV-style PCDR per frame in camera coordinates (range [0,1]).

    This follows BEV's Relative Human (RH) evaluation protocol:
    - Derive ordinal depth layers from GT depths (DLs) using the grouping threshold `gamma`.
    - For each person pair, assign the GT relation category: eq / closer / farther.
    - A relation is correct if the predicted depth difference satisfies the category rule
      under threshold `tau` (applied to predicted depth differences).

    Args:
        pred_smpl_params: Predicted SMPL params with shapes [B, P, ...].
        gt_smpl_params: GT SMPL params with shapes [B, P, ...].
        c2w: Camera-to-world matrices, shape [B, 4, 4] or [4, 4].
        tau: Depth-relation threshold in meters for correctness.
        gamma: Depth-layer grouping threshold in meters (people within gamma are "equal depth").

    Returns:
        Dict with key "pcdr" mapping to a tensor of per-frame PCDR values of shape [B]

    Notes:
        - PCDR is view-dependent; person positions are transformed from world to camera
          space using the provided c2w matrices.
        - Depths are derived from per-person translations in camera space.
        - Only unique person pairs (upper triangle) are evaluated.
    """
    pred_trans = pred_smpl_params["trans"]
    gt_trans = gt_smpl_params["trans"]
    if pred_trans.dim() == 2:
        pred_trans = pred_trans.unsqueeze(0)
    if gt_trans.dim() == 2:
        gt_trans = gt_trans.unsqueeze(0)
    if pred_trans.dim() != 3 or gt_trans.dim() != 3:
        raise ValueError(
            f"Expected trans shape [B,P,3], got pred {pred_trans.shape} and gt {gt_trans.shape}"
        )

    pred_cam = _points_world_to_cam(pred_trans, c2w)
    gt_cam = _points_world_to_cam(gt_trans, c2w)

    if pred_cam.shape != gt_cam.shape:
        raise ValueError(f"Pred/GT camera positions must match. Got {pred_cam.shape} vs {gt_cam.shape}")

    device = pred_cam.device
    bsize, npeople, _ = pred_cam.shape
    if npeople < 2:
        return {
            "pcdr": torch.zeros((bsize,), device=device)
        }

    z_pred = pred_cam[:, :, 2]
    z_gt = gt_cam[:, :, 2]

    idx_i, idx_j = torch.triu_indices(npeople, npeople, offset=1, device=device)
    total_pairs = idx_i.numel()
    per_frame_pcdr = []

    for b in range(bsize):
        z_gt_b = z_gt[b]
        z_pred_b = z_pred[b]

        # Derive ordinal depth layers (DLs) from GT using 1D clustering along z.
        # Closest person gets layer 0; a new layer starts if the depth gap exceeds gamma.
        sorted_z, sorted_idx = torch.sort(z_gt_b, dim=0)
        layer_ids = torch.empty((npeople,), device=device, dtype=torch.long)
        current_layer = 0
        layer_ids[sorted_idx[0]] = current_layer
        for k in range(1, npeople):
            if (sorted_z[k] - sorted_z[k - 1]) > gamma:
                current_layer += 1
            layer_ids[sorted_idx[k]] = current_layer

        li = layer_ids[idx_i]
        lj = layer_ids[idx_j]
        dz_pred = z_pred_b[idx_i] - z_pred_b[idx_j]

        eq_mask = li == lj
        cd_mask = li < lj  # i is closer than j
        fd_mask = li > lj  # i is farther than j

        correct_eq = (dz_pred.abs() < tau)[eq_mask]
        correct_cd = (dz_pred < -tau)[cd_mask]
        correct_fd = (dz_pred > tau)[fd_mask]

        correct_total = (
            correct_eq.sum()
            + correct_cd.sum()
            + correct_fd.sum()
        ).float()
        pcdr_val = correct_total / float(total_pairs)
        per_frame_pcdr.append(pcdr_val)

    return {
        "pcdr": torch.stack(per_frame_pcdr, dim=0),
    }

# ---------------------------------------------------------------------------
# Segmentation evaluation metrics
# ---------------------------------------------------------------------------

def segmentation_mask_metrics(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute IoU, F1, and Recall for each sample in a batch of boolean masks.

    Args:
        gt_masks: [B,H,W] ground truth binary masks
        pred_masks: [B,H,W] predicted binary masks

    Returns:
        Dict with keys "segm_iou", "segm_f1", "segm_recall" mapping to tensors of shape [B]
    """

    if gt_masks.shape != pred_masks.shape:
        raise ValueError(f"Mask shapes must match. Got {gt_masks.shape} vs {pred_masks.shape}.")

    gt_flat = gt_masks.reshape(gt_masks.shape[0], -1).float()
    pred_flat = pred_masks.reshape(pred_masks.shape[0], -1).float()

    tp = (pred_flat * gt_flat).sum(dim=1)
    fp = (pred_flat * (1 - gt_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * gt_flat).sum(dim=1)

    union = tp + fp + fn
    safe_union = union.clamp_min(1e-6)
    iou_vals = tp / safe_union

    denom_f1 = (2 * tp + fp + fn).clamp_min(1e-6)
    f1_vals = (2 * tp) / denom_f1

    denom_recall = (tp + fn).clamp_min(1e-6)
    recall_vals = tp / denom_recall

    zero_mask = (union < 1e-6)
    iou_vals = torch.where(zero_mask, torch.zeros_like(iou_vals), iou_vals)
    f1_vals = torch.where(zero_mask, torch.zeros_like(f1_vals), f1_vals)
    recall_vals = torch.where(zero_mask, torch.zeros_like(recall_vals), recall_vals)

    return {"segm_iou": iou_vals, "segm_f1": f1_vals, "segm_recall": recall_vals}


# ---------------------------------------------------------------------------
# Appearance evaluation metrics
# ---------------------------------------------------------------------------

# Cached LPIPS metric instance; built lazily on first use.
_LPIPS_METRIC: Optional[torch.nn.Module] = None


def _get_lpips_net(device: torch.device) -> torch.nn.Module:
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None:
        # Spatial LPIPS gives a per-pixel distance map (pyiqa handles input normalisation to [-1,1])
        _LPIPS_METRIC = pyiqa.create_metric(
            "lpips", device=device, net="vgg", spatial=True, as_loss=False
        ).eval()
    return _LPIPS_METRIC.to(device)


def _ensure_nchw(t: torch.Tensor) -> torch.Tensor:
    """Convert tensors from NHWC (renderer output) to NCHW for library calls."""
    return t.permute(0, 3, 1, 2).contiguous()


def _mask_sums(mask: torch.Tensor) -> torch.Tensor:
    """Sum mask activations per sample (expects shape [B,1,H,W])."""
    return mask.sum(dim=(2, 3))


def ssim(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked SSIM per sample.
    
    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]
    Returns: 
        ssim_vals: [B] masked SSIM values 
    """
    target = _ensure_nchw(images.float())
    preds = _ensure_nchw(renders.float())
    mask = masks.unsqueeze(1).float()

    # Kornia returns per-channel SSIM; average across channels before masking
    ssim_map = kornia.metrics.ssim(preds, target, window_size=11, max_val=1.0)
    ssim_map = ssim_map.mean(1, keepdim=True)

    # Reduce over the masked region for each batch element
    numerator = (ssim_map * mask).sum(dim=(2, 3))
    mask_sum = _mask_sums(mask)
    safe_mask_sum = mask_sum.clamp_min(1e-6)
    result = numerator / safe_mask_sum
    result = torch.where(mask_sum < 1e-5, torch.zeros_like(result), result)
    return result.squeeze(1)


def psnr(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Masked PSNR per sample.

    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]

    Returns:
        psnr_vals: [B] masked PSNR values 
    """
    target = images.float()
    preds = renders.float()
    mask = masks.unsqueeze(-1).float()

    diff2 = (preds - target) ** 2
    masked_diff2 = diff2 * mask
    # Compute masked MSE then convert to PSNR
    numerator = masked_diff2.sum(dim=(1, 2, 3))
    denom = mask.sum(dim=(1, 2, 3))
    safe_denom = denom.clamp_min(1e-6)
    mse = numerator / safe_denom
    mse = mse.clamp_min(1e-12)
    psnr_vals = 10.0 * torch.log10((max_val ** 2) / mse)
    psnr_vals = torch.where(denom < 1e-5, torch.zeros_like(psnr_vals), psnr_vals)
    return psnr_vals


def lpips(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked spatial LPIPS per sample using pyiqa.


    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]
    Returns:
        lpips_vals: [B] masked LPIPS values 
    """
    target = _ensure_nchw(images.float()).clamp(0.0, 1.0)
    preds = _ensure_nchw(renders.float()).clamp(0.0, 1.0)
    mask = masks.unsqueeze(1).float()

    net = _get_lpips_net(preds.device)
    with torch.no_grad():
        dmap = net(preds, target)

    # Match mask resolution to the LPIPS map and average within the mask
    if dmap.shape[-2:] != mask.shape[-2:]:
        mask_resized = F.interpolate(mask, size=dmap.shape[-2:], mode="nearest")
    else:
        mask_resized = mask

    numerator = (dmap * mask_resized).sum(dim=(1, 2, 3))
    denom = mask_resized.sum(dim=(1, 2, 3))
    safe_denom = denom.clamp_min(1e-6)
    lpips_vals = numerator / safe_denom
    lpips_vals = torch.where(denom < 1e-5, torch.zeros_like(lpips_vals), lpips_vals)
    return lpips_vals



