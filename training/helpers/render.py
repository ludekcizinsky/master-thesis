from typing import Tuple, List
from gsplat import rasterization
import torch
from training.helpers.smpl_utils import canon_to_posed
from training.helpers.model_init import SceneSplats

def _unit_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def _prep_splats(
    splats: torch.nn.ParameterDict,
    clamp_sigma: tuple = (1e-4, 1.0),
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    means_override: torch.Tensor | None = None,
) -> Tuple[torch.Tensor]:
    means_src = means_override if means_override is not None else splats["means"]
    means = means_src.to(device, dtype)
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
    all_gs: SceneSplats,
    smpl_param: torch.Tensor = None, # [P, 86] 
    lbs_weights: torch.Tensor = None, # [P, M, 24]
    clamp_sigma: tuple = (1e-4, 1.0),
    dtype: torch.dtype = torch.float32,
    device: str = "cuda"
):

    # Static (background)
    if all_gs.static is not None:
        static_pack = _prep_splats(all_gs.static, clamp_sigma, dtype, device)
    else:
        static_pack = None

    # Dynamic (humans)
    if len(all_gs.dynamic) > 0:
        smpl_server = all_gs.smpl_c_info["smpl_server"]
        dynamic_pack_all = dict()
        for i in range(len(all_gs.dynamic)):
            dynamic_splats = all_gs.dynamic[i]
            means_p = canon_to_posed(
                smpl_server,
                smpl_param[i:i+1],
                dynamic_splats["means"],
                lbs_weights[i],
                device=device,
            ).squeeze(0)
            new_pack = _prep_splats(
                dynamic_splats,
                clamp_sigma,
                dtype,
                device,
                means_override=means_p,
            )
            if i == 0:
                dynamic_pack_all = {k: v for k, v in new_pack.items()}
            else:
                for k, v in new_pack.items():
                    dynamic_pack_all[k] = torch.cat([dynamic_pack_all[k], v], dim=0)
    else:
        dynamic_pack_all = None

    # Assert at least one of static or dynamic splats is present
    assert static_pack is not None or dynamic_pack_all is not None, "No splats to render."
 
    # Merge static and dynamic packs
    all_pack = dict()
    if static_pack is not None and dynamic_pack_all is not None:
        for k in static_pack.keys():
            all_pack[k] = torch.cat([static_pack[k], dynamic_pack_all[k]], dim=0)
    elif static_pack is not None:
        all_pack = static_pack
    elif dynamic_pack_all is not None:
        all_pack = dynamic_pack_all

    return all_pack

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

def _parse_info(info: dict, all_gs: SceneSplats) -> List[dict]:
    # TODO: polish this function, currently a bit hacky

    # Get number of gaussians per model
    n_gaussians_per_model = []
    includes_static = False
    if all_gs.static is not None:
        n_gaussians_per_model.append(all_gs.static["means"].shape[0])
        includes_static = True
    if len(all_gs.dynamic) > 0:
        for splats in all_gs.dynamic:
            n_gaussians_per_model.append(splats["means"].shape[0])

    selected_keys_tensors = ["radii", "gaussian_ids", "means2d", "gradient_2dgs"]
    selected_keys_ints = ["width", "height", "n_cameras"]
    prev_n_gs = 0
    parsed = []
    for curr_n_gs in n_gaussians_per_model:
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

    # static is always first if present
    if includes_static:
        static_info = parsed[0]
        dynamic_info = parsed[1:]
    else:
        static_info = None
        dynamic_info = parsed

    return static_info, dynamic_info

def render_splats(all_gs, smpl_param, lbs_weights, w2c, K, H, W, sh_degree):

    # Prep splats
    p = _prep_splats_for_render(
        all_gs=all_gs,
        smpl_param=smpl_param, 
        lbs_weights=lbs_weights,
        clamp_sigma=(1e-4, 1.0),
        dtype=torch.float32,
    )

    # Render
    colors, alphas, info = rasterization(
        p["means"], p["quats"], p["scales"], p["opacity"], p["colors"],
        w2c, K, W, H, sh_degree=sh_degree, packed=False
    )

    # Parse info per model
    new_info = _parse_info(info, all_gs)

    return colors, alphas, new_info
