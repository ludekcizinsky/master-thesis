import torch
from gsplat import rasterization


def rasterize_splats(trainer, smpl_param, K, img_wh, sh_degree, masks=None):

    device, dtype = trainer.device, torch.float32
    dev_width, dev_height = img_wh

    # Prepare gaussians
    # - Means
    dev_means = trainer.means_can_to_cam(smpl_param)  # [M,3]
    # - Quats
    dev_quats = trainer.rotations()  # [M,4]
    # - Scales
    dev_scales = trainer.scales() # [M,3]
    # - Colours
    dev_colors = trainer.get_colors()
    # - Opacity
    dev_opacity = trainer.opacity()

    # Define cameras
    dev_viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)   # [1,4,4]
    dev_Ks = K.to(device, dtype).contiguous()                  # [1,3,3]

    # Render
    render_colors, render_alphas, info = rasterization(
        dev_means, dev_quats, dev_scales, dev_opacity, dev_colors, dev_viewmats, dev_Ks, dev_width, dev_height, sh_degree=sh_degree
    )

    if masks is not None:
        render_colors[~masks] = 0
    return render_colors, render_alphas, info