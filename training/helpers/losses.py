import torch

def _anchor_penalty(dists: torch.Tensor, cfg) -> torch.Tensor:
    """
    dists: [N] Euclidean distances (meters)
    Uses cfg.ma_* (free_radius, robust, huber_delta).
    Returns a scalar tensor (mean over N).
    """
    if dists.numel() == 0:
        return torch.zeros((), device=dists.device)

    excess = torch.clamp(dists - cfg.ma_free_radius, min=0.0)

    if cfg.ma_robust == "huber":
        delta = cfg.ma_huber_delta
        quad = 0.5 * (excess**2)
        lin  = delta * (excess - 0.5 * delta)
        return torch.where(excess <= delta, quad, lin).mean()
    else:  # L2
        return (excess**2).mean()


def _per_splat_weight_from_log_scales(log_scales: torch.Tensor) -> torch.Tensor:
    """
    log_scales: [M,3] (per-axis log-sigma for each splat)
    Returns normalized weights [M] ~ 1/mean(sigma).
    If M==0, returns an empty tensor on the same device.
    """
    if log_scales.numel() == 0:
        return torch.empty((0,), device=log_scales.device)

    sigmas = torch.exp(log_scales)                  # [M,3]
    w = 1.0 / sigmas.mean(dim=-1).clamp_min(1e-5)  # [M]
    return (w / w.mean().clamp_min(1e-5)).detach()


def anchor_means_loss(
    canon_means_info,
    smpl_verts_c: torch.Tensor,
    cfg,
    log_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Standalone anchor loss.

    Args:
      canon_means_info:
        (means_c, m0, init_means_c)
          - means_c: [Mtot,3] current canonical means
          - m0: int, number of original seed splats
          - init_means_c: [M0,3] initial canonical means (frozen copy)
      smpl_verts_c: [V,3] canonical SMPL vertices
      cfg: contains ma_* keys
      log_scales: [Mtot,3] log-sigmas for scale-aware weighting (optional)

    Returns:
      scalar tensor (loss)
    """
    means_c, m0, init_means_c = canon_means_info
    device = means_c.device
    loss = torch.zeros((), device=device)

    # device safety
    if smpl_verts_c.device != device:
        smpl_verts_c = smpl_verts_c.to(device)

    mode = cfg.ma_mode

    if mode == "init":
        M_tot = means_c.shape[0]
        M = min(m0, M_tot)
        if M > 0:
            d = (means_c[:M] - init_means_c[:M].to(device)).norm(dim=-1)  # [M]
            if cfg.ma_scale_aware and (log_scales is not None):
                w = _per_splat_weight_from_log_scales(log_scales[:M])       # [M]
                excess = torch.clamp(d - cfg.ma_free_radius, min=0.0)
                if cfg.ma_robust == "huber":
                    delta = cfg.ma_huber_delta
                    quad = 0.5 * (excess**2)
                    lin  = delta * (excess - 0.5 * delta)
                    per = torch.where(excess <= delta, quad, lin)          # [M]
                else:
                    per = excess**2
                anchor = (w * per).mean()
            else:
                anchor = _anchor_penalty(d, cfg)
            loss = cfg.ma_lambda * anchor

    elif mode == "nn":
        Mtot = means_c.shape[0]
        V = smpl_verts_c.shape[0]
        if Mtot > 0 and V > 0:
            # If this is heavy, you can subsample verts or chunk cdist.
            nn_d = torch.cdist(means_c, smpl_verts_c).min(dim=1).values  # [Mtot]
            if cfg.ma_scale_aware and (log_scales is not None):
                w = _per_splat_weight_from_log_scales(log_scales)         # [Mtot]
                excess = torch.clamp(nn_d - cfg.ma_free_radius, min=0.0)
                if cfg.ma_robust == "huber":
                    delta = cfg.ma_huber_delta
                    quad = 0.5 * (excess**2)
                    lin  = delta * (excess - 0.5 * delta)
                    per = torch.where(excess <= delta, quad, lin)
                else:
                    per = excess**2
                anchor = (w * per).mean()
            else:
                anchor = _anchor_penalty(nn_d, cfg)
            loss = cfg.ma_lambda * anchor

    else:
        raise ValueError(f"Unknown ma_mode: {mode}")

    return loss
