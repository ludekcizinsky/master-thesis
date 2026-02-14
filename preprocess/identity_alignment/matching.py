from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    linear_sum_assignment = None


MaskGetter = Callable[[int, str], np.ndarray]


@dataclass
class IdentityMatchResult:
    preproc_person_ids: List[int]
    gt_person_ids: List[int]
    preproc_to_gt: List[int]
    gt_to_preproc: Dict[int, int]
    pair_confidences: List[float]
    confidence_min: float
    confidence_mean: float
    n_frames_used: int
    used_manual_override: bool
    method: str
    similarity_matrix: List[List[float]]


def _format_similarity_matrix(
    similarity: np.ndarray,
    preproc_ids: Sequence[int],
    gt_ids: Sequence[int],
) -> str:
    matrix_str = np.array2string(similarity, precision=4, suppress_small=False)
    return (
        "similarity_matrix (rows=preproc_person_ids, cols=gt_person_ids):\n"
        f"preproc_person_ids={list(preproc_ids)}\n"
        f"gt_person_ids={list(gt_ids)}\n"
        f"{matrix_str}"
    )


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(union)


def build_iou_similarity_matrix(
    preproc_person_ids: Sequence[int],
    gt_person_ids: Sequence[int],
    frame_names: Sequence[str],
    get_preproc_mask: MaskGetter,
    get_gt_mask: MaskGetter,
) -> np.ndarray:
    if len(frame_names) == 0:
        raise ValueError("Identity matching requires at least one frame.")
    sim = np.zeros((len(preproc_person_ids), len(gt_person_ids)), dtype=np.float64)
    for i, preproc_pid in enumerate(preproc_person_ids):
        for j, gt_pid in enumerate(gt_person_ids):
            ious: List[float] = []
            for frame_name in frame_names:
                preproc_mask = get_preproc_mask(int(preproc_pid), frame_name)
                gt_mask = get_gt_mask(int(gt_pid), frame_name)
                ious.append(_mask_iou(preproc_mask, gt_mask))
            sim[i, j] = float(np.mean(ious)) if ious else 0.0
    return sim


def _solve_max_assignment(similarity: np.ndarray) -> List[int]:
    n_rows, n_cols = similarity.shape
    if n_rows != n_cols:
        raise ValueError(
            f"Identity assignment currently requires equal counts; got preproc={n_rows}, gt={n_cols}."
        )
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(-similarity)
        assigned = [-1] * n_rows
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            assigned[r] = int(c)
        if any(idx < 0 for idx in assigned):
            raise RuntimeError("Failed to recover a complete identity assignment.")
        return assigned

    # Small fallback when scipy is unavailable.
    best_perm: Optional[Tuple[int, ...]] = None
    best_score = -float("inf")
    for perm in itertools.permutations(range(n_cols)):
        score = float(sum(similarity[r, c] for r, c in enumerate(perm)))
        if score > best_score:
            best_score = score
            best_perm = perm
    if best_perm is None:
        raise RuntimeError("Unable to compute identity assignment.")
    return [int(c) for c in best_perm]


def align_identities_from_masks(
    *,
    preproc_person_ids: Sequence[int],
    gt_person_ids: Sequence[int],
    frame_names: Sequence[str],
    get_preproc_mask: MaskGetter,
    get_gt_mask: MaskGetter,
    min_confidence: float,
    manual_preproc_to_gt: Optional[Sequence[int]] = None,
    raise_on_low_confidence: bool = True,
) -> IdentityMatchResult:
    preproc_ids = [int(v) for v in preproc_person_ids]
    gt_ids = [int(v) for v in gt_person_ids]
    if len(preproc_ids) == 0 or len(gt_ids) == 0:
        raise ValueError("Identity matching requires non-empty person id lists.")
    if len(preproc_ids) != len(gt_ids):
        raise ValueError(
            f"Identity matching requires equal person counts; got preproc={len(preproc_ids)}, gt={len(gt_ids)}."
        )
    if len(frame_names) == 0:
        raise ValueError("Identity matching requires at least one frame.")

    similarity = build_iou_similarity_matrix(
        preproc_person_ids=preproc_ids,
        gt_person_ids=gt_ids,
        frame_names=frame_names,
        get_preproc_mask=get_preproc_mask,
        get_gt_mask=get_gt_mask,
    )
    gt_id_to_col = {gt_id: col for col, gt_id in enumerate(gt_ids)}

    used_manual_override = manual_preproc_to_gt is not None
    if manual_preproc_to_gt is not None:
        override = [int(v) for v in manual_preproc_to_gt]
        if len(override) != len(preproc_ids):
            raise ValueError(
                "Manual identity override length must equal number of preprocessed people: "
                f"{len(override)} != {len(preproc_ids)}."
            )
        if set(override) != set(gt_ids):
            raise ValueError(
                "Manual identity override must be a permutation of GT person ids. "
                f"override={override}, gt_ids={gt_ids}."
            )
        preproc_to_gt = override
        method = "manual_override"
    else:
        assigned_cols = _solve_max_assignment(similarity)
        preproc_to_gt = [gt_ids[col] for col in assigned_cols]
        method = "auto_iou_hungarian"

    pair_confidences = []
    for row_idx, gt_id in enumerate(preproc_to_gt):
        col_idx = gt_id_to_col[gt_id]
        pair_confidences.append(float(similarity[row_idx, col_idx]))
    confidence_min = float(min(pair_confidences)) if pair_confidences else 0.0
    confidence_mean = float(np.mean(pair_confidences)) if pair_confidences else 0.0

    if (
        raise_on_low_confidence
        and manual_preproc_to_gt is None
        and confidence_min < float(min_confidence)
    ):
        sim_debug = _format_similarity_matrix(similarity, preproc_ids, gt_ids)
        raise RuntimeError(
            "Automatic identity matching confidence below threshold. "
            f"confidence_min={confidence_min:.4f} < min_confidence={float(min_confidence):.4f}. "
            f"Suggested manual_preproc_to_gt={preproc_to_gt}\n"
            f"{sim_debug}"
        )

    gt_to_preproc = {gt_id: pre_idx for pre_idx, gt_id in enumerate(preproc_to_gt)}
    return IdentityMatchResult(
        preproc_person_ids=preproc_ids,
        gt_person_ids=gt_ids,
        preproc_to_gt=preproc_to_gt,
        gt_to_preproc=gt_to_preproc,
        pair_confidences=pair_confidences,
        confidence_min=confidence_min,
        confidence_mean=confidence_mean,
        n_frames_used=len(frame_names),
        used_manual_override=used_manual_override,
        method=method,
        similarity_matrix=similarity.tolist(),
    )
