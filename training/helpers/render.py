from typing import Tuple, List
from gsplat import rasterization
import torch
from training.helpers.smpl_utils import canon_to_posed

def _unit_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def _prep_splats(splats: torch.nn.ParameterDict, clamp_sigma: tuple = (1e-4, 1.0), dtype: torch.dtype = torch.float32, device: str = "cuda") -> Tuple[torch.Tensor]:
    means = splats["means"].to(device, dtype)
    quats = _unit_quat(splats["quats"].to(device, dtype))
    scales = torch.exp(splats["scales"].to(device, dtype)).clamp(*clamp_sigma)
    opacity = torch.sigmoid(splats["opacities"].to(device, dtype))
    colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device, dtype)

    return dict(
        means=means, quats=quats, scales=scales,
        opacity=opacity, colors=colors
    )

def _prep_splats_for_render(
    *,
    all_gs: List[torch.nn.ParameterDict],
    smpl_c_info: dict,         
    smpl_param: torch.Tensor = None, # [P, 86] 
    lbs_weights: torch.Tensor = None, # [P, M, 24]
    clamp_sigma: tuple = (1e-4, 1.0),
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
):
    # Unpack splats
    static_splats = all_gs[0]
    dynamic_splats_per_human = all_gs[1:]

    # Static (background)
    static_pack = _prep_splats(static_splats, clamp_sigma, dtype, device)

    # Dynamic (humans)
    smpl_server = smpl_c_info["smpl_server"]
    dynamic_pack_all = dict()
    for i in range(len(dynamic_splats_per_human)):
        dynamic_splats = dynamic_splats_per_human[i]
        means_p = canon_to_posed(smpl_server, smpl_param[i:i+1], dynamic_splats["means"], lbs_weights[i], device=device)
        dynamic_splats["means"].data.copy_(means_p.squeeze(0))
        new_pack = _prep_splats(dynamic_splats, clamp_sigma, dtype, device)
        if i == 0:
            dynamic_pack_all = {k: v for k, v in new_pack.items()}
        else:
            for k, v in new_pack.items():
                dynamic_pack_all[k] = torch.cat([dynamic_pack_all[k], v], dim=0)

    
    # Both
    all_pack = dict()
    for k in static_pack.keys():
        all_pack[k] = torch.cat([static_pack[k], dynamic_pack_all[k]], dim=0)

    return {"static": static_pack, "dynamic": dynamic_pack_all, "all": all_pack}

def _register_subset_grad_hook(parent: torch.Tensor, child: torch.Tensor, start: int, end: int) -> None:
    """Mirror gradients from `parent` into the derived `child` slice.

    The densification strategy expects gradients on the per-model slices produced by
    `_parse_info`. These slices are created after the forward pass and are therefore
    not part of the autograd graph. By registering a hook on the parent tensor we can
    copy the relevant slice of the parent's gradient into the child tensor after the
    backward pass finishes.
    """
    if not (isinstance(parent, torch.Tensor) and isinstance(child, torch.Tensor)):
        return
    if not (parent.requires_grad and child.requires_grad):
        return

    slice_spec = (slice(None), slice(start, end))

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        if grad is not None:
            grad_slice = grad[slice_spec + (Ellipsis,)].clone()
            if child.grad is None:
                child.grad = grad_slice
            else:
                child.grad = child.grad + grad_slice
        return grad

    parent.register_hook(_hook)

def _parse_info(info: dict, n_gaussians_per_model) -> List[dict]:
    selected_keys_tensors = ["radii", "gaussian_ids", "means2d", "gradient_2dgs"]
    selected_keys_ints = ["width", "height", "n_cameras"]
    prev_n_gs = 0
    parsed = []
    for i, curr_n_gs in enumerate(n_gaussians_per_model):
        new_info = {}
        for key in selected_keys_tensors:
            value = info.get(key, None)
            if value is not None:
                start_idx, end_idx = prev_n_gs, prev_n_gs + curr_n_gs
                subset = value[:, start_idx:end_idx]
                _register_subset_grad_hook(value, subset, start_idx, end_idx)
                new_info[key] = subset
            else:
                new_info[key] = None

        for key in selected_keys_ints:
            new_info[key] = info.get(key, None)
        parsed.append(new_info)
        prev_n_gs += curr_n_gs
    return parsed

def render_splats(all_gs, smpl_info, smpl_param, lbs_weights, w2c, K, H, W, sh_degree, kind="all"):

    # Prep splats
    packs = _prep_splats_for_render(
        all_gs=all_gs,
        smpl_c_info=smpl_info,
        smpl_param=smpl_param[0], # assume batch size 1 for now
        lbs_weights=lbs_weights,
        clamp_sigma=(1e-4, 1.0),
        dtype=torch.float32,
        device=smpl_param.device,
    )

    # Render
    p = packs[kind]
    colors, alphas, info = rasterization(
        p["means"], p["quats"], p["scales"], p["opacity"], p["colors"],
        w2c, K, W, H, sh_degree=sh_degree, packed=False
    )
    n_gs = [g["means"].shape[0] for g in all_gs]
    new_info = _parse_info(info, n_gs)

    return colors, alphas, new_info
