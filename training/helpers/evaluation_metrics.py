from typing import List, Optional, Dict

import torch
import torch.nn.functional as F
import kornia
import pyiqa

from training.smpl_deformer.smpl_server import SMPLServer


# Cache the LPIPS metric instance so we initialise it only once.
# Cached LPIPS metric instance; built lazily on first use.
_LPIPS_METRIC: Optional[torch.nn.Module] = None
_SMPL_METRIC_SERVER: Optional[SMPLServer] = None


def _get_lpips_net(device: torch.device) -> torch.nn.Module:
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None:
        # Spatial LPIPS gives a per-pixel distance map (pyiqa handles input normalisation to [-1,1])
        _LPIPS_METRIC = pyiqa.create_metric(
            "lpips", device=device, net="vgg", spatial=True, as_loss=False
        ).eval()
    return _LPIPS_METRIC.to(device)


def _get_smpl_metric_server() -> SMPLServer:
    global _SMPL_METRIC_SERVER
    if _SMPL_METRIC_SERVER is None:
        _SMPL_METRIC_SERVER = SMPLServer().eval()
    return _SMPL_METRIC_SERVER


def _ensure_nchw(t: torch.Tensor) -> torch.Tensor:
    """Convert tensors from NHWC (renderer output) to NCHW for library calls."""
    return t.permute(0, 3, 1, 2).contiguous()


def _mask_sums(mask: torch.Tensor) -> torch.Tensor:
    """Sum mask activations per sample (expects shape [B,1,H,W])."""
    return mask.sum(dim=(2, 3))


def ssim(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked SSIM per sample."""
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
    """Masked PSNR per sample."""
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
    """Masked spatial LPIPS per sample using pyiqa."""
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

def segmentation_mask_metrics(
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute IoU, F1, and Recall for each sample in a batch of boolean masks.
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


def _smpl_params_to_joints(params: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Convert SMPL parameter tensor [B, P, 86] to joint tensor [B, P, J, 3].
    """

    b, p, _ = params.shape
    flat = params.reshape(-1, params.shape[-1]).to(device=device, dtype=torch.float32)
    smpl_server = _get_smpl_metric_server()

    with torch.no_grad():
        outputs = smpl_server(flat, absolute=True)
        joints = outputs["smpl_jnts"]

    return joints.reshape(b, p, joints.shape[1], 3)


def mpjpe_from_smpl_params(
    gt_params: torch.Tensor,
    pred_params: torch.Tensor,
    normalization_factor: float = 1.0,
    output_units: str = "m",
) -> torch.Tensor:
    """
    Compute MPJPE per (frame, person) pair given SMPL parameter tensors of shape (B, P, 86).
    Returns a tensor of shape (B*P,) in the same units as the inputs.
    """

    if gt_params.shape != pred_params.shape:
        raise ValueError(f"SMPL parameter shapes must match. Got {gt_params.shape} vs {pred_params.shape}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_joints = _smpl_params_to_joints(gt_params, device)
    pred_joints = _smpl_params_to_joints(pred_params, device)

    diff = pred_joints - gt_joints
    per_sample = diff.norm(dim=-1).mean(dim=-1)  # [B, P]
    values = per_sample.reshape(-1) * float(normalization_factor)

    if output_units not in {"m", "mm"}:
        raise ValueError("output_units must be either 'm' or 'mm'")
    if output_units == "mm":
        values = values * 1000.0

    return values

def compute_all_metrics(
    images: torch.Tensor,
    masks: torch.Tensor,
    renders: torch.Tensor,
    pred_masks: Optional[torch.Tensor] = None,
    gt_smpl: Optional[torch.Tensor] = None,
    pred_smpl: Optional[torch.Tensor] = None,
    gt_smpl_normalization_factor: float = 1.0,
    mpjpe_units: str = "mm",
) -> dict[str, torch.Tensor]:
    
    all_metrics = dict() 
    rendering_metrics = {
        "ssim": ssim(images, masks, renders),
        "psnr": psnr(images, masks, renders),
        "lpips": lpips(images, masks, renders),
    }
    all_metrics.update(rendering_metrics)

    if pred_masks is not None:
        segmentation_metrics = segmentation_mask_metrics(masks, pred_masks)
        all_metrics.update(segmentation_metrics)

    if gt_smpl is not None and pred_smpl is not None:
        mpjpe_values = mpjpe_from_smpl_params(
            gt_smpl,
            pred_smpl,
            normalization_factor=gt_smpl_normalization_factor,
            output_units=mpjpe_units,
        )
        all_metrics["mpjpe"] = mpjpe_values
    
    return all_metrics

def aggregate_batch_tid_metric_dicts(metric_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concat per-sample metrics across calls, then average each metric."""

    collected: dict[str, list[torch.Tensor]] = {}
    for metrics in metric_dicts:
        for name, values in metrics.items():
            collected.setdefault(name, []).append(values)

    aggregated: dict[str, torch.Tensor] = {}
    for name, tensor_list in collected.items():
        concatenated = torch.cat(tensor_list, dim=0)
        aggregated[name] = concatenated.mean().item()

    return aggregated

def aggregate_global_tids_metric_dicts(tid_video_metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    """Aggregate metrics across all TIDs by averaging each metric."""

    final_metrics_list: Dict[str, List[float]] = dict()
    for _, metrics in tid_video_metrics.items():
        for metric_name, value in metrics.items():
            final_metrics_list.setdefault(metric_name, []).append(value)

    # Average the metrics across TIDs
    final_metrics: Dict[str, float] = dict()
    for metric_name, values in final_metrics_list.items():
        final_metrics[metric_name] = sum(values) / len(values)
    
    return final_metrics
