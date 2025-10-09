import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from training.helpers.model_init import SceneSplats
from training.helpers.render import render_splats
from training.helpers.smpl_utils import canon_to_posed, update_skinning_weights
from training.helpers.utils import project_points


@dataclass
class SamMaskResult:
    mask: Tensor
    alpha: Tensor
    positive_points: Tensor
    negative_points: Tensor


def _infer_sh_degree(scene_splats: SceneSplats) -> int:
    if not scene_splats.dynamic:
        raise ValueError("Cannot infer spherical harmonics degree without dynamic splats.")
    sh_coeffs = scene_splats.dynamic[0]["shN"]
    total_coeffs = 1 + int(sh_coeffs.shape[1])
    degree = int(round(math.sqrt(total_coeffs) - 1))
    return max(degree, 0)


def _ensure_lbs_weights(
    all_gs: SceneSplats,
    lbs_weights: Optional[Sequence[Tensor]],
    device: torch.device,
    lbs_knn: int,
) -> List[Tensor]:
    if lbs_weights is not None:
        return [w.to(device) for w in lbs_weights]

    computed = update_skinning_weights(all_gs, k=lbs_knn, device=device)
    return [w.to(device) for w in computed]


def _render_alpha_mask(
    dynamic_index: int,
    all_gs: SceneSplats,
    smpl_params: Tensor,
    lbs_weights: Sequence[Tensor],
    w2c: Tensor,
    K: Tensor,
    image_size: Tuple[int, int],
    alpha_threshold: float,
    sh_degree: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    H, W = image_size

    single_scene = SceneSplats(
        static=None,
        dynamic=[all_gs.dynamic[dynamic_index]],
        smpl_c_info=all_gs.smpl_c_info,
    )
    smpl_param = smpl_params[dynamic_index].unsqueeze(0).unsqueeze(0).to(device)
    lbs_weight = [lbs_weights[dynamic_index]]

    colors, alphas, _ = render_splats(
        single_scene,
        smpl_param,
        lbs_weight,
        w2c.unsqueeze(0).to(device),
        K.unsqueeze(0).to(device),
        H,
        W,
        sh_degree=sh_degree,
    )

    alpha_map = alphas[0].detach()
    if alpha_map.ndim == 3 and alpha_map.shape[-1] == 1:
        alpha_map = alpha_map[..., 0]
    mask = (alpha_map > alpha_threshold).to(dtype=torch.bool)
    return mask.cpu(), alpha_map.cpu()


def _pose_body_vertices(
    all_gs: SceneSplats,
    smpl_params: Tensor,
    device: torch.device,
) -> List[Tensor]:
    if all_gs.smpl_c_info is None:
        raise ValueError("SceneSplats missing SMPL canonical information.")

    smpl_server = all_gs.smpl_c_info["smpl_server"].to(device)
    verts_c = all_gs.smpl_c_info["verts_c"].to(device)
    weights_c = all_gs.smpl_c_info["weights_c"].to(device)

    posed_vertices: List[Tensor] = []
    for params in smpl_params:
        verts = canon_to_posed(
            smpl_server,
            params.unsqueeze(0).to(device),
            verts_c,
            weights_c,
            device=device,
        )
        posed_vertices.append(verts.squeeze(0))
    return posed_vertices


def _project_vertices(
    verts_world: Tensor,
    w2c: Tensor,
    K: Tensor,
) -> Tensor:
    device = verts_world.device
    w2c = w2c.to(device)
    K = K.to(device)

    ones = torch.ones((verts_world.shape[0], 1), device=device)
    verts_h = torch.cat([verts_world, ones], dim=1)
    verts_cam_h = (w2c @ verts_h.T).T
    verts_cam = verts_cam_h[:, :3]
    uv, _ = project_points(verts_cam, K)
    return uv


def _point_in_mask(u: int, v: int, mask: Tensor) -> bool:
    if v < 0 or v >= mask.shape[0] or u < 0 or u >= mask.shape[1]:
        return False
    return bool(mask[v, u].item())


def _collect_prompt_points(
    idx: int,
    keypoints: Sequence[Tensor],
    masks: Sequence[Tensor],
    image_size: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:
    H, W = image_size
    target_kp = keypoints[idx]
    target_mask = masks[idx]

    other_masks = [masks[j] for j in range(len(masks)) if j != idx]

    positive_pts = []
    for uv in target_kp:
        u = int(round(float(uv[0].item())))
        v = int(round(float(uv[1].item())))
        if not _point_in_mask(u, v, target_mask):
            continue
        if any(_point_in_mask(u, v, m) for m in other_masks):
            continue
        positive_pts.append([u, v])

    negative_pts = []
    for j, kp in enumerate(keypoints):
        if j == idx:
            continue
        for uv in kp:
            u = int(round(float(uv[0].item())))
            v = int(round(float(uv[1].item())))
            if u < 0 or u >= W or v < 0 or v >= H:
                continue
            if _point_in_mask(u, v, target_mask):
                continue
            negative_pts.append([u, v])

    pos_tensor = (
        torch.tensor(positive_pts, dtype=torch.float32)
        if positive_pts
        else torch.empty((0, 2), dtype=torch.float32)
    )
    neg_tensor = (
        torch.tensor(negative_pts, dtype=torch.float32)
        if negative_pts
        else torch.empty((0, 2), dtype=torch.float32)
    )
    return pos_tensor, neg_tensor


def get_sam_masks(
    all_gs: SceneSplats,
    smpl_params: Tensor,
    w2c: Tensor,
    K: Tensor,
    image_size: Tuple[int, int],
    *,
    alpha_threshold: float,
    lbs_weights: Optional[Sequence[Tensor]] = None,
    lbs_knn: int = 30,
    device: Optional[torch.device | str] = None,
) -> List[SamMaskResult]:
    if not all_gs.dynamic:
        raise ValueError("SceneSplats contains no dynamic components to process.")

    if smpl_params.ndim != 2:
        raise ValueError(f"Expected smpl_params with shape [num_humans, 86], got {smpl_params.shape}.")
    if smpl_params.shape[0] != len(all_gs.dynamic):
        raise ValueError(
            f"SMPL params count ({smpl_params.shape[0]}) does not match dynamic splats ({len(all_gs.dynamic)})."
        )

    render_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    sh_degree = _infer_sh_degree(all_gs)
    lbs_weights_prepared = _ensure_lbs_weights(all_gs, lbs_weights, render_device, lbs_knn)

    masks: List[Tensor] = []
    alphas: List[Tensor] = []

    H, W = image_size
    K_3x3 = K[:3, :3] if K.shape[0] == 4 else K
    w2c_4x4 = w2c if w2c.shape == (4, 4) else w2c[:4, :4]

    with torch.no_grad():
        for idx in range(len(all_gs.dynamic)):
            mask, alpha_map = _render_alpha_mask(
                idx,
                all_gs,
                smpl_params,
                lbs_weights_prepared,
                w2c_4x4,
                K_3x3,
                (H, W),
                alpha_threshold,
                sh_degree,
                render_device,
            )
            masks.append(mask)
            alphas.append(alpha_map)

        posed_vertices = _pose_body_vertices(all_gs, smpl_params.to(render_device), render_device)
        keypoints = [
            _project_vertices(verts, w2c_4x4.to(render_device), K_3x3.to(render_device)).cpu()
            for verts in posed_vertices
        ]

        results: List[SamMaskResult] = []
        for idx in range(len(all_gs.dynamic)):
            pos_pts, neg_pts = _collect_prompt_points(idx, keypoints, masks, (H, W))
            results.append(
                SamMaskResult(
                    mask=masks[idx],
                    alpha=alphas[idx],
                    positive_points=pos_pts,
                    negative_points=neg_pts,
                )
            )

    return results
