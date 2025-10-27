import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import logging
from tqdm import tqdm

from hydra.core.global_hydra import GlobalHydra

from training.helpers.model_init import SceneSplats
from training.helpers.render import render_splats
from training.helpers.smpl_utils import canon_to_posed, update_skinning_weights
from training.helpers.geom_utils import project_points


@dataclass
class SamMaskResult:
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
) -> Tuple[Tensor, Tensor]:
    H, W = image_size

    single_scene = SceneSplats(
        static=None,
        dynamic=[all_gs.dynamic[dynamic_index]],
        smpl_c_info=all_gs.smpl_c_info,
    )
    smpl_param = smpl_params[dynamic_index].unsqueeze(0).to(device)
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


CATEGORY_JOINTS = {
    "feet": {7, 8, 10, 11},
    "legs": {1, 2, 4, 5, 7, 8},
    "hands": {20, 21, 22, 23},
    "arms": {16, 17, 18, 19, 20, 21},
    "chest": {0, 3, 6, 9},
    "head": {12, 15},
}
POINTS_PER_CATEGORY = 10


def _build_category_indices(joint_labels: np.ndarray) -> dict[str, np.ndarray]:
    category_indices: dict[str, np.ndarray] = {}
    for name, joint_ids in CATEGORY_JOINTS.items():
        mask = np.isin(joint_labels, list(joint_ids))
        category_indices[name] = np.where(mask)[0]
    return category_indices


def _sample_from_category(
    indices: np.ndarray,
    uv_coords: Tensor,
    include_mask: Optional[Tensor],
    exclude_masks: Sequence[Tensor],
    count: int,
) -> List[List[int]]:
    selected: List[List[int]] = []
    if indices.size == 0:
        return selected
    H = include_mask.shape[0] if include_mask is not None else None
    W = include_mask.shape[1] if include_mask is not None else None
    candidates: List[List[int]] = []
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

    if not candidates:
        return selected

    if len(candidates) <= count:
        return candidates

    chosen_idx = np.linspace(0, len(candidates) - 1, num=count, dtype=int)
    for idx in chosen_idx:
        selected.append(candidates[idx])
    return selected


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
    image_size: Tuple[int, int],
    category_vertices: dict[str, np.ndarray],
) -> Tuple[Tensor, Tensor]:
    H, W = image_size
    target_kp = keypoints[idx]
    target_mask = masks[idx]

    other_masks = [masks[j] for j in range(len(masks)) if j != idx]

    positive_pts: List[List[int]] = []
    for indices in category_vertices.values():
        positive_pts.extend(
            _sample_from_category(
                indices,
                target_kp,
                target_mask,
                other_masks,
                POINTS_PER_CATEGORY,
            )
        )
    if not positive_pts:
        positive_pts = _default_positive_points(target_kp, target_mask, other_masks)

    negative_pts: List[List[int]] = []
    for j, kp in enumerate(keypoints):
        if j == idx:
            continue
        own_mask = masks[j]
        for indices in category_vertices.values():
            negative_pts.extend(
                _sample_from_category(
                    indices,
                    kp,
                    own_mask,
                    [target_mask],
                    POINTS_PER_CATEGORY,
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
    result: SamMaskResult,
    max_points: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    pos = result.positive_points.cpu().numpy() if result.positive_points.numel() > 0 else None
    neg = result.negative_points.cpu().numpy() if result.negative_points.numel() > 0 else None

    pos_ds = _downsample_points(pos, max_points)
    neg_ds = _downsample_points(neg, max_points)

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

        joint_labels = torch.argmax(all_gs.smpl_c_info["weights_c"], dim=1).cpu().numpy()
        category_vertices = _build_category_indices(joint_labels)

        results: List[SamMaskResult] = []
        for idx in range(len(all_gs.dynamic)):
            pos_pts, neg_pts = _collect_prompt_points(
                idx,
                keypoints,
                masks,
                (H, W),
                category_vertices,
            )
            results.append(
                SamMaskResult(
                    mask=masks[idx],
                    alpha=alphas[idx],
                    positive_points=pos_pts,
                    negative_points=neg_pts,
                )
            )

    return results


def refine_masks_with_predictor(
    sam_results: Sequence[SamMaskResult],
    predictor: "SAM2ImagePredictor",
    *,
    multimask_output: bool,
    use_initial_mask: bool,
    max_points: int,
) -> List[RefinedMaskResult]:
    refined: List[RefinedMaskResult] = []
    for result in sam_results:
        point_coords, point_labels, pos_vis, neg_vis = _prepare_point_arrays(result, max_points)

        mask_input = None
        if use_initial_mask:
            mask_np = result.mask.numpy().astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np)[None, None, :, :]
            target_size = predictor.model.sam_prompt_encoder.mask_input_size
            if mask_tensor.shape[-2:] != target_size:
                mask_tensor = F.interpolate(
                    mask_tensor,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            mask_input = mask_tensor.cpu().numpy()

        refined_masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=multimask_output,
            )

        if refined_masks.ndim == 3:
            idx = 0
            if refined_masks.shape[0] > 1:
                idx = int(np.argmax(scores))
            refined_mask_np = refined_masks[idx]
        else:
            refined_mask_np = refined_masks

        refined_mask_tensor = torch.from_numpy(refined_mask_np).float()

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
) -> List[RefinedMaskResult]:
    sam_results = get_sam_masks(
        scene_splats,
        smpl_params,
        w2c,
        K,
        image_size,
        alpha_threshold=alpha_threshold,
        lbs_weights=lbs_weights,
        lbs_knn=lbs_knn,
        device=device,
    )

    refined = refine_masks_with_predictor(
        sam_results,
        predictor,
        multimask_output=bool(predictor_cfg.get("multimask_output", False)),
        use_initial_mask=bool(predictor_cfg.get("use_initial_mask", True)),
        max_points=int(predictor_cfg.get("max_points", 50)),
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
    ) -> None:
        # cfg
        mask_cfg = mask_cfg or {}
        self.tids = list(tids)
        self.device = device
        self.default_lbs_knn = int(default_lbs_knn)
        self.loss_weight = float(mask_cfg.get("loss_weight", 0.0))
        self.alpha_threshold = float(mask_cfg.get("alpha_threshold", 0.3))
        self.rebuild_every_epochs = max(int(mask_cfg.get("rebuild_every_epochs", 10)), 1)
        self.rebuild_max_epoch = max(int(mask_cfg.get("rebuild_max_epoch", 20)), 1)
        self.predictor_cfg = dict(mask_cfg.get("sam2", {}))

        # paths
        self.checkpoint_dir = checkpoint_dir / "progressive_sam"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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

    def should_update(self, epoch: int) -> bool:
        return (epoch - self.last_update_epoch) >= self.rebuild_every_epochs and epoch <= self.rebuild_max_epoch

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

    def update_masks(
        self,
        dataset,
        scene_splats: SceneSplats,
        lbs_weights: Optional[Sequence[Tensor]],
        epoch: int,
    ) -> None:

        predictor = self._ensure_predictor()

        n_samples = len(dataset)
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
                        alpha_threshold=self.alpha_threshold,
                        predictor=predictor,
                        predictor_cfg=self.predictor_cfg,
                        lbs_weights=lbs_weights,
                        lbs_knn=self.default_lbs_knn,
                        device=self.device,
                    )

                entry, iou_scores = self._build_mask_entry(refined_results, self.device)
                self.frame_iou_scores[fid] = iou_scores
                self._save_entry_to_disk(entry, fid)

        self.last_update_epoch = epoch
        self._update_reliability_stats()
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
        if resume:
            ckpt_path = self.checkpoint_dir / "progressive_sam.pt"
            self._load_state(ckpt_path)
        else:
            if self.tids:
                self.clear_ckpt_dir()
            self.update_masks(dataset, scene_splats, lbs_weights, epoch)

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
        model_id = sam_cfg.get("model_id", "facebook/sam2.1-hiera-large")
        device_str = sam_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        sam_model = build_sam2_hf(model_id, device=device_str)
        predictor = SAM2ImagePredictor(
            sam_model,
            mask_threshold=float(sam_cfg.get("mask_threshold", 0.0)),
        )

        logging.getLogger("sam2").setLevel(logging.ERROR)
        logging.getLogger("sam2").propagate = False

        self.sam_device = device_str
        self.predictor = predictor
        return self.predictor
