import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import logging
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from hydra.core.global_hydra import GlobalHydra
import pyrender
import trimesh

RNG = np.random.default_rng(42)

from training.helpers.model_init import SceneSplats
from training.helpers.render import render_splats
from training.helpers.smpl_utils import canon_to_posed, update_skinning_weights
from training.helpers.geom_utils import project_points
from training.smpl_deformer.smpl_server import SMPLServer



@dataclass
class SamInputPrompt:
    mask: Tensor
    alpha: Tensor
    positive_points: Tensor
    negative_points: Tensor


@dataclass
class RefinedMaskResult:
    initial_mask: Tensor
    refined_mask: Tensor
    alpha: Tensor
    positive_points: Tensor
    negative_points: Tensor
    point_coords: Optional[np.ndarray]
    point_labels: Optional[np.ndarray]
    vis_positive_points: Optional[np.ndarray]
    vis_negative_points: Optional[np.ndarray]


if TYPE_CHECKING:
    from sam2.sam2_image_predictor import SAM2ImagePredictor


@dataclass
class SamMaskEntry:
    refined: Tensor
    alpha: Tensor
    initial: Tensor
    vis_pos: List[Optional[np.ndarray]]
    vis_neg: List[Optional[np.ndarray]]
    iou_scores: List[float]

@contextmanager
def suppress_sam_logging(level: int = logging.WARNING):
    logger = logging.getLogger()
    prev_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(prev_level)


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
) -> Tuple[Tensor, Tensor, Tensor]:
    H, W = image_size

    single_scene = SceneSplats(
        static=None,
        dynamic=[all_gs.dynamic[dynamic_index]],
        smpl_c_info=all_gs.smpl_c_info,
    )
    smpl_param = smpl_params[dynamic_index].unsqueeze(0).to(device)
    lbs_weight = [lbs_weights[dynamic_index]]

    renders, alphas, _ = render_splats(
        single_scene,
        smpl_param,
        lbs_weight,
        w2c.unsqueeze(0).to(device),
        K.unsqueeze(0).to(device),
        H,
        W,
        sh_degree=sh_degree,
        render_mode="RGB+ED",
    )

    render_map = renders[0].detach()
    if render_map.shape[-1] < 4:
        raise RuntimeError("Depth channel missing from render output; ensure render_mode includes depth.")
    depth_map = render_map[..., 3].to(dtype=torch.float32)

    alpha_map = alphas[0].detach()
    if alpha_map.ndim == 3 and alpha_map.shape[-1] == 1:
        alpha_map = alpha_map[..., 0]
    mask = (alpha_map > alpha_threshold).to(dtype=torch.bool)
    return mask.cpu(), alpha_map.cpu(), depth_map.cpu()


def _render_raw_smpl_mask(
    smpl_server: SMPLServer,
    smpl_param: Tensor,
    w2c: Tensor,
    K: Tensor,
    image_size: Tuple[int, int],
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    H, W = image_size
    smpl_output = smpl_server(smpl_param.unsqueeze(0).to(device), absolute=False)
    verts = smpl_output["smpl_verts"][0].detach().cpu().numpy()
    faces = smpl_server.smpl.faces.astype(np.int32)

    tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.2, 0.2, 0.2])
    scene.add(mesh)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=np.eye(4))

    w2c_np = w2c.detach().cpu().numpy()
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    w2c_gl = cv_to_gl @ w2c_np

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
    scene.add(cam, pose=np.linalg.inv(w2c_gl))

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color_rgb, depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.FLAT)
    renderer.delete()

    render_rgb = color_rgb.astype(np.float32) / 255.0
    mask_np = (depth > 0).astype(np.float32)
    if mask_np.max() <= 0.0:
        mask_np = (np.linalg.norm(render_rgb, axis=-1) > 1e-6).astype(np.float32)

    mask_tensor = torch.from_numpy(mask_np.astype(np.bool_))
    alpha_tensor = torch.from_numpy(mask_np.astype(np.float32))
    depth_tensor = torch.from_numpy(depth.astype(np.float32))
    return mask_tensor, alpha_tensor, depth_tensor


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


def _sample_from_category(
    indices: np.ndarray,
    uv_coords: Tensor,
    include_mask: Optional[Tensor],
    exclude_masks: Sequence[Tensor],
) -> List[List[int]]:
    candidates: List[List[int]] = []
    if indices.size == 0:
        return candidates

    for vid in indices.tolist():
        if vid >= uv_coords.shape[0]:
            continue
        uv = uv_coords[vid]
        u = int(round(float(uv[0].item())))
        v = int(round(float(uv[1].item())))
        if include_mask is not None and not _point_in_mask(u, v, include_mask):
            continue
        if any(_point_in_mask(u, v, m) for m in exclude_masks):
            continue
        candidates.append([u, v])

    return candidates


def _default_positive_points(
    uv_coords: Tensor,
    target_mask: Tensor,
    other_masks: Sequence[Tensor],
) -> List[List[int]]:
    fallback: List[List[int]] = []
    for uv in uv_coords:
        u = int(round(float(uv[0].item())))
        v = int(round(float(uv[1].item())))
        if not _point_in_mask(u, v, target_mask):
            continue
        if any(_point_in_mask(u, v, m) for m in other_masks):
            continue
        fallback.append([u, v])
    return fallback


def _default_negative_points(
    uv_coords: Tensor,
    target_mask: Tensor,
    own_mask: Tensor,
) -> List[List[int]]:
    fallback: List[List[int]] = []
    for uv in uv_coords:
        u = int(round(float(uv[0].item())))
        v = int(round(float(uv[1].item())))
        if not _point_in_mask(u, v, own_mask):
            continue
        if _point_in_mask(u, v, target_mask):
            continue
        fallback.append([u, v])
    return fallback


def _collect_prompt_points(
    idx: int,
    keypoints: Sequence[Tensor],
    masks: Sequence[Tensor],
    *,
    positive_joint_indices: Sequence[int],
    negative_joint_indices: Optional[Sequence[int]] = None,
) -> Tuple[Tensor, Tensor]:
    max_joints_available = keypoints[0].shape[0]
    positive_joint_indices = list(positive_joint_indices)[: min(27, max_joints_available)]
    negative_joint_indices = (
        list(negative_joint_indices)[: min(27, max_joints_available)]
        if negative_joint_indices is not None
        else list(range(min(27, max_joints_available)))
    )

    target_kp = keypoints[idx]
    target_mask = masks[idx]

    other_masks = [masks[j] for j in range(len(masks)) if j != idx]

    positive_pts: List[List[int]] = []
    positive_pts.extend(
        _sample_from_category(
            np.asarray(positive_joint_indices, dtype=np.int64),
            target_kp,
            target_mask,
            other_masks,
        )
    )
    if not positive_pts:
        positive_pts = _default_positive_points(target_kp, target_mask, other_masks)

    negative_pts: List[List[int]] = []

    # random background pixels as initial negatives
    mask_np = target_mask.cpu().numpy().astype(np.float32)
    neg_background: List[List[int]] = []
    attempts = 0
    max_attempts = 200
    while len(neg_background) < 10 and attempts < max_attempts:
        attempts += 1
        u = int(RNG.integers(0, mask_np.shape[1]))
        v = int(RNG.integers(0, mask_np.shape[0]))
        if mask_np[v, u] > 0.5:
            continue
        if any(_point_in_mask(u, v, other) for other in other_masks):
            continue
        neg_background.append([u, v])
    negative_pts.extend(neg_background)

    for j, kp in enumerate(keypoints):
        if j == idx:
            continue
        own_mask = masks[j]
        negative_pts.extend(
            _sample_from_category(
                np.asarray(negative_joint_indices, dtype=np.int64),
                kp,
                own_mask,
                [target_mask],
            )
        )
        if not negative_pts:
            negative_pts.extend(_default_negative_points(kp, target_mask, own_mask))

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


def _downsample_points(points: Optional[np.ndarray], max_points: int) -> Optional[np.ndarray]:
    if points is None:
        return None
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=int)
    return points[idx]


def _prepare_point_arrays(
    result: SamInputPrompt,
    max_pos_points: int,
    max_neg_points: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    pos = result.positive_points.cpu().numpy() if result.positive_points.numel() > 0 else None
    neg = result.negative_points.cpu().numpy() if result.negative_points.numel() > 0 else None

    pos_ds = _downsample_points(pos, max_pos_points)
    neg_ds = _downsample_points(neg, max_neg_points)

    if pos_ds is None and neg_ds is None:
        return None, None, pos_ds, neg_ds

    coords: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    if pos_ds is not None:
        coords.append(pos_ds.astype(np.float32))
        labels.append(np.ones(pos_ds.shape[0], dtype=np.int32))
    if neg_ds is not None:
        coords.append(neg_ds.astype(np.float32))
        labels.append(np.zeros(neg_ds.shape[0], dtype=np.int32))

    point_coords = np.concatenate(coords, axis=0).astype(np.float32) if coords else None
    point_labels = np.concatenate(labels, axis=0) if labels else None
    return point_coords, point_labels, pos_ds, neg_ds


def _compute_mask_iou(initial_mask: Tensor, refined_mask: Tensor) -> float:
    if initial_mask.dtype != torch.bool:
        initial_bool = initial_mask > 0.5
    else:
        initial_bool = initial_mask

    if refined_mask.dtype != torch.bool:
        refined_bool = refined_mask > 0.5
    else:
        refined_bool = refined_mask

    intersection = torch.logical_and(initial_bool, refined_bool).float().sum()
    union = torch.logical_or(initial_bool, refined_bool).float().sum()

    if union.item() == 0.0:
        return 1.0

    return float((intersection / union).item())


def get_sam_input_prompts(
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
    use_raw_smpl: bool = False,
) -> List[SamInputPrompt]:
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
    if use_raw_smpl:
        lbs_weights_prepared = None
    else:
        lbs_weights_prepared = _ensure_lbs_weights(all_gs, lbs_weights, render_device, lbs_knn)

    masks: List[Tensor] = []
    alphas: List[Tensor] = []
    depths: List[Tensor] = []

    H, W = image_size
    K_3x3 = K[:3, :3] if K.shape[0] == 4 else K
    w2c_4x4 = w2c if w2c.shape == (4, 4) else w2c[:4, :4]
    smpl_server = all_gs.smpl_c_info["smpl_server"].to(render_device)

    with torch.no_grad():
        for idx in range(len(all_gs.dynamic)):
            if use_raw_smpl:
                mask, alpha_map, depth_map = _render_raw_smpl_mask(
                    smpl_server,
                    smpl_params[idx],
                    w2c_4x4,
                    K_3x3,
                    (H, W),
                    render_device,
                )
            else:
                mask, alpha_map, depth_map = _render_alpha_mask(
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
            depths.append(depth_map)

        mask_stack = torch.stack([m.to(dtype=torch.bool) for m in masks], dim=0)
        depth_stack = torch.stack([d.to(dtype=torch.float32) for d in depths], dim=0)
        depth_stack = torch.where(
            torch.isnan(depth_stack),
            torch.full_like(depth_stack, float("inf")),
            depth_stack,
        )
        depth_with_background = torch.where(
            mask_stack,
            depth_stack,
            torch.full_like(depth_stack, float("inf")),
        )
        front_depth, _ = depth_with_background.min(dim=0)
        valid_front = torch.isfinite(front_depth)
        depth_epsilon = 5e-3
        for idx in range(len(masks)):
            depth_map = depths[idx]
            visible_mask = masks[idx] & valid_front & (depth_map <= front_depth + depth_epsilon)
            masks[idx] = visible_mask

        smpl_server = all_gs.smpl_c_info["smpl_server"].to(render_device)
        w2c_device = w2c_4x4.to(render_device)
        K_device = K_3x3.to(render_device)
        joint_projections: List[Tensor] = []
        joint_count: Optional[int] = None
        for idx in range(len(all_gs.dynamic)):
            smpl_output = smpl_server(
                smpl_params[idx].unsqueeze(0).to(render_device),
                absolute=False,
            )
            joints_world = smpl_output["smpl_jnts"][0]
            if joint_count is None:
                joint_count = int(joints_world.shape[0])
            joint_uv = _project_vertices(joints_world, w2c_device, K_device).cpu()
            joint_projections.append(joint_uv)

        if joint_count is not None:
            positive_joint_indices = list(range(min(27, joint_count)))
        else:
            positive_joint_indices = []

        results: List[SamInputPrompt] = []
        for idx in range(len(all_gs.dynamic)):
            pos_pts, neg_pts = _collect_prompt_points(
                idx,
                joint_projections,
                masks,
                positive_joint_indices=positive_joint_indices,
                negative_joint_indices=positive_joint_indices,
            )
            results.append(
                SamInputPrompt(
                    mask=masks[idx],
                    alpha=alphas[idx],
                    positive_points=pos_pts,
                    negative_points=neg_pts,
                )
            )

    return results


def refine_masks_with_predictor(
    sam_results: Sequence[SamInputPrompt],
    predictor: "SAM2ImagePredictor",
    *,
    multimask_output: bool,
    use_initial_mask: bool,
    max_pos_points: int,
    max_neg_points: int,
    n_iterations: int,
) -> List[RefinedMaskResult]:
    refined: List[RefinedMaskResult] = []
    for result in sam_results:
        point_coords, point_labels, pos_vis, neg_vis = _prepare_point_arrays(result, max_pos_points, max_neg_points)

        mask_input = None
        if use_initial_mask:
            mask_np = result.mask.numpy().astype(np.float32)
            nonzero = np.argwhere(mask_np > 0.5)
            if nonzero.size > 0:
                y_min = int(nonzero[:, 0].min())
                y_max = int(nonzero[:, 0].max())
                x_min = int(nonzero[:, 1].min())
                x_max = int(nonzero[:, 1].max())
                height = max(y_max - y_min + 1, 1)
                width = max(x_max - x_min + 1, 1)
                margin_y = max(int(0.03 * height), 1)
                margin_x = max(int(0.03 * width), 1)
                y_min = max(0, y_min - margin_y)
                x_min = max(0, x_min - margin_x)
                y_max = min(mask_np.shape[0] - 1, y_max + margin_y)
                x_max = min(mask_np.shape[1] - 1, x_max + margin_x)
                bounding_box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
            else:
                bounding_box = None

            height, width = mask_np.shape
            max_dim = max(height, width)
            canvas = np.zeros((max_dim, max_dim), dtype=np.float32)
            if height >= width:
                canvas[:height, :width] = mask_np
            else:
                offset = max_dim - width
                canvas[:height, offset:offset + width] = mask_np
            mask_tensor = torch.from_numpy(canvas)[None, None, :, :]
            target_size = predictor.model.sam_prompt_encoder.mask_input_size
            if mask_tensor.shape[-2:] != target_size:
                mask_tensor = F.interpolate(
                    mask_tensor,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            mask_input = torch.special.logit(mask_tensor.clamp(1e-6, 1 - 1e-6))
            mask_input = mask_input.cpu().numpy()
        else:
            bounding_box = None

        box_input = bounding_box[None, :] if bounding_box is not None else None

        refined_masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            box=box_input,
            multimask_output=multimask_output,
            )

        def _select_best(mask_arr, score_arr, logit_arr):
            mask_np_local = np.asarray(mask_arr)
            score_np_local = np.asarray(score_arr) if score_arr is not None else None
            logit_np_local = np.asarray(logit_arr) if logit_arr is not None else None

            if mask_np_local.ndim == 3:
                if score_np_local is not None and score_np_local.size > 0:
                    order = np.argsort(score_np_local)[::-1]
                    best_idx = int(order[0])
                else:
                    best_idx = 0
                best_mask = mask_np_local[best_idx]
                best_logit = (
                    logit_np_local[best_idx]
                    if logit_np_local is not None and logit_np_local.ndim >= 3
                    else None
                )
            else:
                best_mask = mask_np_local
                best_logit = logit_np_local

            return best_mask, best_logit

        best_mask_np, best_logit_np = _select_best(refined_masks, scores, logits)

        if point_coords is not None and n_iterations > 1:
            for _ in range(max(n_iterations - 1, 0)):
                if best_logit_np is None:
                    break
                if best_logit_np.ndim == 2:
                    mask_input_iter = best_logit_np[None, None, ...].astype(np.float32)
                elif best_logit_np.ndim == 3:
                    if best_logit_np.shape[0] == 1:
                        mask_input_iter = best_logit_np.astype(np.float32)
                    else:
                        mask_input_iter = best_logit_np[:1].astype(np.float32)
                else:
                    break

                refined_masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=mask_input_iter,
                    box=box_input,
                    multimask_output=multimask_output,
                )
                best_mask_np, best_logit_np = _select_best(refined_masks, scores, logits)

        refined_mask_tensor = torch.from_numpy(best_mask_np).float()

        refined.append(
            RefinedMaskResult(
                initial_mask=result.mask.clone(),
                refined_mask=refined_mask_tensor,
                alpha=result.alpha.clone(),
                positive_points=result.positive_points.clone(),
                negative_points=result.negative_points.clone(),
                point_coords=point_coords,
                point_labels=point_labels,
                vis_positive_points=pos_vis,
                vis_negative_points=neg_vis,
            )
        )

    predictor.reset_predictor()

    return refined


def compute_refined_masks(
    *,
    scene_splats: SceneSplats,
    smpl_params: Tensor,
    w2c: Tensor,
    K: Tensor,
    image_size: Tuple[int, int],
    alpha_threshold: float,
    predictor: "SAM2ImagePredictor",
    predictor_cfg: Dict[str, Any],
    lbs_weights: Optional[Sequence[Tensor]] = None,
    lbs_knn: int = 30,
    device: Optional[torch.device | str] = None,
    use_raw_smpl: bool = False,
) -> List[RefinedMaskResult]:
    sam_input_prompts = get_sam_input_prompts(
        scene_splats,
        smpl_params,
        w2c,
        K,
        image_size,
        alpha_threshold=alpha_threshold,
        lbs_weights=lbs_weights,
        lbs_knn=lbs_knn,
        device=device,
        use_raw_smpl=use_raw_smpl,
    )

    max_pos_points = int(predictor_cfg.max_pos_points)
    max_neg_points = int(predictor_cfg.max_neg_points)
    refined = refine_masks_with_predictor(
        sam_input_prompts,
        predictor,
        multimask_output=bool(predictor_cfg.multimask_output),
        use_initial_mask=bool(predictor_cfg.use_initial_mask),
        max_pos_points=max_pos_points,
        max_neg_points=max_neg_points,
        n_iterations=int(predictor_cfg.get("n_iterations", 1)),
    )
    return refined


class ProgressiveSAMManager:
    def __init__(
        self,
        mask_cfg: Optional[Dict[str, Any]],
        tids: Sequence[int],
        device: torch.device,
        default_lbs_knn: int,
        checkpoint_dir: Path,
        training_dir: Path,
        preprocessing_dir: Path,
        is_preprocessing: bool = False, 
    ) -> None:
        # cfg
        mask_cfg = mask_cfg or {}
        self.tids = list(tids)
        self.device = device
        self.default_lbs_knn = int(default_lbs_knn)
        self.alpha_threshold = mask_cfg.alpha_threshold
        self.alpha_threshold_warmup = max(int(mask_cfg.alpha_threshold_warmup_refinements), 0)
        self.rebuild_every_epochs = max(int(mask_cfg.rebuild_every_epochs), 1)
        self.rebuild_max_epoch = max(int(mask_cfg.rebuild_max_epoch), 1)
        self.predictor_cfg = mask_cfg.sam2
        self.preprocessing_dir = preprocessing_dir
        self.is_preprocessing = is_preprocessing
        self.use_raw_smpl_until_epoch = int(mask_cfg.use_raw_smpl_until_epoch)

        # paths
        if is_preprocessing:
            self.checkpoint_dir = preprocessing_dir / "sam2_masks"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = checkpoint_dir / "progressive_sam"
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.vis_dir = training_dir / "visualizations" / "progressive_sam"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # sam2
        self.predictor: Optional["SAM2ImagePredictor"] = None
        self.sam_device: Optional[str] = None

        # reliability stats
        self.last_update_epoch: int = -1
        self.frame_iou_scores: Dict[int, List[float]] = {}
        self.reliable_frames: List[int] = []
        self.unreliable_frames: List[int] = []
        self.iou_threshold: Optional[float] = None
        self._reliable_frame_set: set[int] = set()
        self._unreliable_frame_set: set[int] = set()
        self.base_iteration: int = 0
        self.rebuild_iteration: int = 0

    def should_update(self, epoch: int) -> bool:
        return (epoch - self.last_update_epoch) >= self.rebuild_every_epochs and epoch <= self.rebuild_max_epoch

    def _current_alpha_threshold(self) -> float:
        if self.alpha_threshold_warmup <= 0:
            return self.alpha_threshold
        if self.rebuild_iteration >= self.alpha_threshold_warmup:
            return self.alpha_threshold
        fraction = (self.rebuild_iteration + 1) / float(self.alpha_threshold_warmup)
        return float(self.alpha_threshold) * max(min(fraction, 1.0), 0.0)

    def _save_entry_to_disk(self, entry: SamMaskEntry, fid: int) -> None:
        payload = {
            "fid": int(fid),
            "refined": entry.refined.detach().cpu(),
            "alpha": entry.alpha.detach().cpu(),
            "initial": entry.initial.detach().cpu(),
            "vis_pos": entry.vis_pos,
            "vis_neg": entry.vis_neg,
            "iou_scores": entry.iou_scores,
        }

        entry_path = self.checkpoint_dir / f"mask_{fid:04d}.pt"
        torch.save(payload, entry_path)

    @staticmethod
    def _load_entry_from_disk(ckpt_dir: Path, fid: int, device: torch.device) -> SamMaskEntry:
        entry_path = ckpt_dir / f"mask_{fid:04d}.pt"
        if not entry_path.exists():
            return None
        data = torch.load(entry_path, map_location=device, weights_only=False)

        refined = data.get("refined")
        alpha = data.get("alpha")
        initial = data.get("initial")

        iou_scores_raw = data.get("iou_scores", [])
        if isinstance(iou_scores_raw, torch.Tensor):
            iou_scores = [float(v) for v in iou_scores_raw.flatten().tolist()]
        else:
            iou_scores = [float(v) for v in iou_scores_raw]

        entry = SamMaskEntry(
            refined=refined.detach().to(device),
            alpha=alpha.detach().to(device),
            initial=initial.detach().to(device),
            vis_pos=data.get("vis_pos", []),
            vis_neg=data.get("vis_neg", []),
            iou_scores=iou_scores,
        )
        return entry

    def _update_reliability_stats(self) -> None:
        all_scores = [score for scores in self.frame_iou_scores.values() for score in scores]
        if not all_scores:
            print("--- FYI: No IoU scores available to update reliability stats.")
            self.iou_threshold = None
            self.reliable_frames = []
            self.unreliable_frames = []
            return

        alpha = float(np.median(np.asarray(all_scores, dtype=np.float32)))
        self.iou_threshold = alpha

        reliable: List[int] = []
        unreliable: List[int] = []
        for fid, scores in self.frame_iou_scores.items():
            if scores:
                avg_iou = float(np.mean(np.asarray(scores, dtype=np.float32)))
            else:
                print(f"--- FYI: No IoU scores for frame {fid}, marking as unreliable.")
                avg_iou = 0.0
            if avg_iou >= alpha:
                reliable.append(int(fid))
            else:
                unreliable.append(int(fid))

        reliable.sort()
        unreliable.sort()
        self.reliable_frames = reliable
        self.unreliable_frames = unreliable
        self._reliable_frame_set = set(reliable)
        self._unreliable_frame_set = set(unreliable)

        print(f"--- FYI: SAM mask reliability updated: {len(reliable)} reliable, {len(unreliable)} unreliable frames. Threshold: {self.iou_threshold:.4f}")

    def _save_visualization_of_entry(self, image: np.ndarray, entry: SamMaskEntry, fid: int, epoch: int) -> None:
        vis_dir_epoch = self.vis_dir / f"epoch_{epoch:04d}"
        vis_dir_epoch.mkdir(parents=True, exist_ok=True)
        scores_array = np.asarray(entry.iou_scores, dtype=np.float32) if entry.iou_scores else np.asarray([])
        if scores_array.size > 0:
            avg_iou = float(np.mean(scores_array))
            avg_iou_str = f"{avg_iou:.2f}"
        else:
            avg_iou = float("nan")
            avg_iou_str = "nan"

        out_path = vis_dir_epoch / f"avg_{avg_iou_str}_{fid:04d}.png"

        num_tracks = int(entry.refined.shape[0])
        if num_tracks == 0:
            return

        fig_height = max(4.0, 5.0 * num_tracks)
        fig, axes = plt.subplots(num_tracks, 2, figsize=(12, fig_height), squeeze=False)

        for idx_h in range(num_tracks):
            initial_mask = entry.initial[idx_h].cpu().numpy()
            refined_mask = entry.refined[idx_h].cpu().numpy()
            positive_pts = entry.vis_pos[idx_h]
            negative_pts = entry.vis_neg[idx_h]

            iou_score: Optional[float]
            if idx_h < len(entry.iou_scores) and entry.iou_scores[idx_h] is not None:
                iou_score = float(entry.iou_scores[idx_h])
            else:
                iou_score = None
            title_suffix = ""
            if iou_score is not None and not math.isnan(iou_score):
                title_suffix = f" (IoU: {iou_score:.2f})"
            else:
                title_suffix = " (IoU: N/A)"

            ax_prompts = axes[idx_h, 0]
            ax_overlay = axes[idx_h, 1]

            ax_prompts.imshow(image)
            ax_prompts.imshow(initial_mask.astype(float), cmap="Greens", alpha=0.35)
            legend_items: List[str] = []
            if positive_pts is not None and positive_pts.size > 0:
                ax_prompts.scatter(
                    positive_pts[:, 0],
                    positive_pts[:, 1],
                    s=45,
                    c="lime",
                    marker="o",
                    edgecolors="black",
                    linewidths=0.6,
                    label="positive",
                )
                legend_items.append("positive")
            if negative_pts is not None and negative_pts.size > 0:
                ax_prompts.scatter(
                    negative_pts[:, 0],
                    negative_pts[:, 1],
                    s=45,
                    c="red",
                    marker="x",
                    linewidths=1.2,
                    label="negative",
                )
                legend_items.append("negative")
            if legend_items:
                ax_prompts.legend(loc="upper right")
            ax_prompts.set_title(f"SAM2 Input Prompts{title_suffix}")
            ax_prompts.set_axis_off()
            ax_prompts.text(
                0.02,
                0.98,
                f"tid {idx_h:02d}",
                transform=ax_prompts.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 2},
            )

            ax_overlay.imshow(image)
            ax_overlay.imshow(initial_mask.astype(float), cmap="Reds", alpha=0.4)
            ax_overlay.imshow(refined_mask.astype(float), cmap="Blues", alpha=0.6)
            ax_overlay.set_title(f"Refined Mask Overlay{title_suffix}")
            ax_overlay.set_axis_off()

            red_patch = matplotlib.patches.Patch(color="red", alpha=0.4, label="Initial Mask")
            blue_patch = matplotlib.patches.Patch(color="blue", alpha=0.6, label="Refined Mask")
            ax_overlay.legend(handles=[red_patch, blue_patch], loc="upper right")

        fig.tight_layout(pad=0.3, h_pad=0.7)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _save_epoch_iou_plot(self, processed_fids: Sequence[int], epoch: int) -> None:
        if not processed_fids:
            return

        frame_ids = sorted({int(fid) for fid in processed_fids})
        frame_scores = [self.frame_iou_scores.get(fid, []) for fid in frame_ids]
        max_tracks = max((len(scores) for scores in frame_scores), default=0)
        if max_tracks == 0:
            return

        frame_averages: List[float] = []
        pooled_scores: List[float] = []
        for scores in frame_scores:
            valid_scores = [
                float(score)
                for score in scores
                if score is not None and not math.isnan(float(score))
            ]
            if valid_scores:
                frame_averages.append(float(np.mean(valid_scores)))
                pooled_scores.extend(valid_scores)
            else:
                frame_averages.append(float("nan"))

        vis_dir_epoch = self.vis_dir / f"epoch_{epoch:04d}"
        vis_dir_epoch.mkdir(parents=True, exist_ok=True)
        out_path = vis_dir_epoch / "iou_scores.png"
        hist_path = vis_dir_epoch / "iou_histograms.png"

        x = np.arange(len(frame_ids), dtype=np.float32)
        bar_width = 0.8 / max_tracks

        fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(frame_ids)), 5.0))

        for track_idx in range(max_tracks):
            track_scores = []
            for scores in frame_scores:
                if track_idx < len(scores):
                    score = scores[track_idx]
                    track_scores.append(float(score) if score is not None else np.nan)
                else:
                    track_scores.append(np.nan)
            scores_array = np.asarray(track_scores, dtype=np.float32)
            offsets = x + (track_idx - (max_tracks - 1) / 2.0) * bar_width
            ax.bar(offsets, scores_array, width=bar_width, label=f"tid {track_idx:02d}")

        avg_array = np.asarray(frame_averages, dtype=np.float32)
        if avg_array.size > 0 and np.any(~np.isnan(avg_array)):
            ax.plot(
                x,
                avg_array,
                color="black",
                marker="o",
                linewidth=1.5,
                label="Frame Avg IoU",
            )

        if pooled_scores:
            global_median = float(np.median(np.asarray(pooled_scores, dtype=np.float32)))
            ax.axhline(
                global_median,
                color="orange",
                linestyle="--",
                linewidth=1.2,
                label=f"Global Median {global_median:.2f}",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{fid:04d}" for fid in frame_ids], rotation=45, ha="right")
        ax.set_xlabel("Frame ID")
        ax.set_ylabel("IoU Score")
        ax.set_title("Per-frame IoU Scores")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", ncol=min(4, max_tracks))

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        # Build per-track histograms
        track_score_lists: List[List[float]] = [[] for _ in range(max_tracks)]
        for scores in frame_scores:
            for track_idx in range(max_tracks):
                if track_idx < len(scores):
                    value = scores[track_idx]
                    if value is not None and not math.isnan(float(value)):
                        track_score_lists[track_idx].append(float(value))

        if any(track_score_lists):
            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            bins = np.linspace(0.0, 1.0, 21)
            for track_idx, track_values in enumerate(track_score_lists):
                if track_values:
                    ax.hist(
                        track_values,
                        bins=bins,
                        alpha=0.55,
                        label=f"tid {track_idx:02d}",
                        edgecolor="black",
                        linewidth=0.4,
                    )
            ax.set_xlabel("IoU Score")
            ax.set_ylabel("Frequency")
            ax.set_title("IoU Distribution per Track")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(hist_path, dpi=150)
            plt.close(fig)

    def update_masks(
        self,
        dataset,
        scene_splats: SceneSplats,
        lbs_weights: Optional[Sequence[Tensor]],
        epoch: int,
    ) -> None:

        predictor = self._ensure_predictor()

        current_alpha_threshold = self._current_alpha_threshold()

        n_samples = len(dataset)
        processed_fids: List[int] = []
        with torch.no_grad():
            for idx in tqdm(range(n_samples), desc=f"Rebuilding SAM mask cache (epoch {epoch})"):
                sample = dataset[idx]
                fid = int(sample["fid"])
                image_tensor = sample["image"]
                image_np = np.clip(image_tensor.detach().cpu().numpy(), 0.0, 1.0)
                image_uint8 = (image_np * 255.0).round().astype(np.uint8)

                with suppress_sam_logging():
                    predictor.set_image(image_uint8)
                    refined_results = compute_refined_masks(
                        scene_splats=scene_splats,
                        smpl_params=sample["smpl_param"].to(self.device),
                        w2c=sample["M_ext"].to(self.device),
                        K=sample["K"].to(self.device),
                        image_size=(int(sample["H"]), int(sample["W"])),
                        alpha_threshold=current_alpha_threshold,
                        predictor=predictor,
                        predictor_cfg=self.predictor_cfg,
                        lbs_weights=lbs_weights,
                        lbs_knn=self.default_lbs_knn,
                        device=self.device,
                        use_raw_smpl=epoch <= self.use_raw_smpl_until_epoch,
                    )

                entry, iou_scores = self._build_mask_entry(refined_results, self.device)
                self.frame_iou_scores[fid] = iou_scores
                processed_fids.append(fid)
                self._save_entry_to_disk(entry, fid)
                self._save_visualization_of_entry(image_np, entry, fid, epoch)

        self._save_epoch_iou_plot(processed_fids, epoch)
        self.last_update_epoch = epoch
        self._update_reliability_stats()
        self.rebuild_iteration += 1
        self._save_state()

    def is_frame_reliable(self, fid: int) -> bool:
        return int(fid) in self._reliable_frame_set

    def get_reliability_ratio(self) -> float:
        total = len(self._reliable_frame_set) + len(self._unreliable_frame_set)
        if total == 0:
            return 0.0
        return len(self._reliable_frame_set) / total

    def get_lowest_iou_reliable_frame(self) -> Optional[int]:
        if not self.reliable_frames:
            return None
        lowest_fid = None
        lowest_value = float("inf")
        for fid in self.reliable_frames:
            scores = self.frame_iou_scores.get(fid)
            if not scores:
                continue
            avg_iou = float(np.mean(np.asarray(scores, dtype=np.float32)))
            if avg_iou < lowest_value:
                lowest_value = avg_iou
                lowest_fid = fid
        return lowest_fid

    def _save_state(self) -> None:
        self.base_iteration += 1
        payload = {
            "iteration": self.base_iteration,
            "frame_iou_scores": self.frame_iou_scores,
            "reliable_frames": self.reliable_frames,
            "unreliable_frames": self.unreliable_frames,
            "iou_threshold": self.iou_threshold,
            "rebuild_iteration": self.rebuild_iteration,
        }
        path = self.checkpoint_dir / f"progressive_sam.pt"
        torch.save(payload, path)
        print(f"--- FYI: Saved Progressive SAM checkpoint to {path} with iteration {self.base_iteration}.")

    def _load_state(self, path: Path) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.frame_iou_scores = {int(k): list(map(float, v)) for k, v in data.get("frame_iou_scores", {}).items()}
        self.reliable_frames = [int(v) for v in data.get("reliable_frames", [])]
        self.unreliable_frames = [int(v) for v in data.get("unreliable_frames", [])]
        self.iou_threshold = data.get("iou_threshold", None)
        self.last_update_epoch = 0
        self._reliable_frame_set = set(self.reliable_frames)
        self._unreliable_frame_set = set(self.unreliable_frames)
        self.base_iteration = max(int(data.get("iteration", 0)), 0)
        self.rebuild_iteration = int(data.get("rebuild_iteration", 0))
        print(f"--- FYI: Loaded Progressive SAM checkpoint from {path} and with base iteration {self.base_iteration}. Reliable frames: {len(self.reliable_frames)}, Unreliable frames: {len(self.unreliable_frames)}. And IoU threshold: {self.iou_threshold}.")

    def clear_ckpt_dir(self) -> None:
        """
        Remove the content of the checkpoint directory only.
        """
        if not self.checkpoint_dir.exists():
            return
        files = list(self.checkpoint_dir.glob("*.pt"))
        for file_path in files:
            try:
                file_path.unlink()
            except Exception:
                pass
        print(f"--- FYI: Cleared Progressive SAM checkpoints in {self.checkpoint_dir} since resume is disabled and we are optimising humans (non-empty tids).")


    def init_state(self, resume: bool, dataset, scene_splats: SceneSplats, lbs_weights, epoch: int) -> None:
        if self.is_preprocessing:
            print(f"--- FYI: Initializing Progressive SAM preprocessing state in {self.checkpoint_dir}.")
            self.update_masks(dataset, scene_splats, lbs_weights, epoch)
        elif resume:
            ckpt_path = self.checkpoint_dir / "progressive_sam.pt"
            self._load_state(ckpt_path)
        else:
            if len(self.tids) > 0:
                self.clear_ckpt_dir()
            shutil.copytree(self.preprocessing_dir / "sam2_masks", self.checkpoint_dir, dirs_exist_ok=True)
            ckpt_path = self.checkpoint_dir / "progressive_sam.pt"
            self._load_state(ckpt_path)

    def _build_mask_entry(
        self,
        refined_results: Sequence[RefinedMaskResult],
        device: torch.device,
    ) -> Tuple[Optional[SamMaskEntry], List[float]]:
        if not refined_results:
            return None, []

        refined_tensor = torch.stack([res.refined_mask for res in refined_results], dim=0).to(device)
        alpha_tensor = torch.stack([res.alpha for res in refined_results], dim=0).to(device)
        initial_tensor = torch.stack([res.initial_mask for res in refined_results], dim=0).to(device)
        vis_pos = [res.vis_positive_points for res in refined_results]
        vis_neg = [res.vis_negative_points for res in refined_results]

        iou_scores: List[float] = []
        for initial_mask, refined_mask in zip(initial_tensor, refined_tensor):
            iou_scores.append(_compute_mask_iou(initial_mask, refined_mask))

        entry = SamMaskEntry(
            refined=refined_tensor,
            alpha=alpha_tensor,
            initial=initial_tensor,
            vis_pos=vis_pos,
            vis_neg=vis_neg,
            iou_scores=iou_scores,
        )
        return entry, iou_scores

    def _ensure_predictor(self) -> Optional["SAM2ImagePredictor"]:
        if self.predictor is not None:
            return self.predictor

        gh = GlobalHydra.instance()
        if gh.is_initialized():
            gh.clear()

        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam_cfg = self.predictor_cfg
        model_id = sam_cfg.model_id
        device_str = sam_cfg.device
        sam_model = build_sam2_hf(model_id, device=device_str)
        predictor = SAM2ImagePredictor(
            sam_model,
            mask_threshold=float(sam_cfg.mask_threshold),
        )

        logging.getLogger("sam2").setLevel(logging.ERROR)
        logging.getLogger("sam2").propagate = False

        self.sam_device = device_str
        self.predictor = predictor
        return self.predictor
