import torch
from gsplat import rasterization


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

def gsplat_render(trainer, smpl_param, K, img_wh):

    device, dtype = trainer.device, torch.float32
    dev_width, dev_height = img_wh

    # Prepare gaussians
    # - Means
    out = trainer.smpl_server(smpl_param, absolute=False)
    T_rel = out["smpl_tfs"][0]   # [24,4,4]
    dev_means = lbs_apply(trainer.gaus.means_c, trainer.gaus.weights_c, T_rel)  # [M,3]
    # dev_means[:, 0] *= -1.0
    # dev_means[:, 1] *= -1.0
    # dev_means[:, 2] *= -1.0

    # - Quats
    M = dev_means.shape[0]
    dev_quats = torch.zeros(M, 4, device=device, dtype=dtype)
    dev_quats[:, 0] = 1.0
    # - Scales
    dev_scales = trainer.gaus.scales().unsqueeze(1).expand(-1, 3).contiguous() # isotropic scales
    # - Colours
    dev_colors = trainer.gaus.colors.clamp(0, 1)
    # - Opacity
    dev_opacity = trainer.gaus.opacity()

    # Define cameras
    dev_viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)   # [1,4,4]
    dev_Ks = K.to(device, dtype).unsqueeze(0).contiguous()                  # [1,3,3]

    # Render
    colors, _, _ = rasterization(
        dev_means, dev_quats, dev_scales, dev_opacity, dev_colors, dev_viewmats, dev_Ks, dev_width, dev_height
    )
    rgb_pred = colors.squeeze().permute(2, 0, 1)

    return rgb_pred