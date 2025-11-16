from pathlib import Path
from typing import List, Optional, Sequence, Union, Tuple

import numpy as np
import torch
from PIL import Image

def _load_multiply_sam_masks(
    visualisation_output_dir: Union[str, Path],
    epoch: Optional[int] = None,
    binarize: bool = True,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Load the SAM-refined masks dumped by Multiply's training loop.

    Parameters
    ----------
    visualisation_output_dir : str or Path
        Either the root visualisation directory (the one passed to `SAMServer`)
        or a direct path to the ``sam_opt_mask.npy`` file.
    epoch : int, optional
        Epoch number whose masks should be loaded. Can be omitted when
        ``visualisation_output_dir`` already points to the ``.npy`` file.
    binarize : bool, optional
        When True, convert the masks to boolean masks using `threshold`.
    threshold : float, optional
        Threshold used during binarisation. Ignored if `binarize` is False.

    Returns
    -------
    np.ndarray
        Array shaped (num_frames, num_person, H, W). If `binarize` is True
        the dtype is bool, otherwise float32. Raises FileNotFoundError if
        the expected `.npy` file is missing.
    """

    vis_path = Path(visualisation_output_dir)
    if vis_path.is_file() and vis_path.suffix == ".npy":
        mask_path = vis_path
    else:
        if epoch is None:
            raise ValueError("`epoch` must be provided when only the root directory is given.")
        mask_path = vis_path / "stage_sam_mask" / f"{epoch:05d}" / "sam_opt_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Could not find SAM masks for epoch {epoch} at {mask_path}"
        )

    masks = np.load(mask_path, allow_pickle=False)

    # SAM dumps singleton channel dimensions; squeeze them for convenience.
    if masks.ndim == 5:
        masks = np.squeeze(masks, axis=-3)

    if binarize:
        return masks >= threshold

    return masks.astype(np.float32, copy=False)


def _load_progressive_sam_masks(
    mask_dir: Union[str, Path],
    frame_ids: Optional[Sequence[int]] = None,
    component: str = "refined",
    binarize: bool = True,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Load Progressive SAM masks saved as ``mask_XXXX.pt`` files and return them as
    a stacked numpy array with shape (num_frames, num_tracks, H, W).

    Parameters
    ----------
    mask_dir : str or Path
        Directory containing the per-frame ``mask_*.pt`` payloads written by
        :class:`ProgressiveSAMManager`.
    frame_ids : sequence of int, optional
        Specific frame ids to load. When omitted, all ``mask_*.pt`` files in the
        directory are used in ascending order.
    component : {'refined', 'initial', 'alpha'}
        Which tensor in the payload to return.
    binarize : bool
        If True (default) threshold the masks using ``threshold`` before
        returning.
    threshold : float
        Value used for binarisation. Ignored when ``binarize`` is False.

    Returns
    -------
    np.ndarray
        Mask array ready for evaluation pipelines. Defaults to boolean dtype
        when ``binarize`` is True, otherwise float32.
    """

    mask_root = Path(mask_dir)
    if frame_ids is None:
        mask_paths = sorted(mask_root.glob("mask_*.pt"))
    else:
        mask_paths = [mask_root / f"mask_{int(fid):04d}.pt" for fid in frame_ids]

    if not mask_paths:
        raise FileNotFoundError(f"No mask_*.pt files found in {mask_root}")

    mask_arrays: List[np.ndarray] = []
    for path in mask_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing Progressive SAM mask file: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensor = payload.get(component)
        if tensor is None:
            raise KeyError(f"Component '{component}' not present in {path.name}")
        mask_arrays.append(tensor.detach().cpu().numpy())

    stacked = np.stack(mask_arrays, axis=0)
    if binarize:
        return stacked >= threshold
    return stacked.astype(np.float32, copy=False)

def _load_binary_masks_from_dir(
    mask_dir: Union[str, Path],
    frame_ids: Optional[Sequence[int]] = None,
    binarize: bool = True,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Load binary masks stored as PNG files in a directory.

    Args:
        mask_dir : str or Path
            Directory containing the per-frame mask PNG files.
        frame_ids : sequence of int, optional
            Specific frame ids to load. When omitted, all PNG files in the
            directory are used in ascending order.
        binarize : bool
            If True (default) threshold the masks using `threshold` before
            returning.
        threshold : float
            Value used for binarisation. Ignored when `binarize` is False.

    Returns:
        np.ndarray
            Mask array shaped (num_frames, 1, H, W) with boolean dtype.
    """


    mask_root = Path(mask_dir)
    if frame_ids is None:
        mask_paths = sorted(mask_root.glob("*.png"))
    else:
        mask_paths = [mask_root / f"mask_{int(fid):04d}.png" for fid in frame_ids]

    if not mask_paths:
        raise FileNotFoundError(f"No PNG mask files found in {mask_root}")

    mask_arrays: List[np.ndarray] = []
    for path in mask_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing binary mask file: {path}")
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        arr = (arr > 0.5).astype(np.float32) # Shape (H, W)
        arr = np.expand_dims(arr, axis=0)  # Shape (1, H, W)       
        mask_arrays.append(arr)

    stacked = np.stack(mask_arrays, axis=0) # Shape (num_frames, 1, H, W)
    if binarize:
        return stacked >= threshold
    return stacked.astype(np.float32, copy=False)


def _load_multicolor_masks_from_dir(
    mask_dir: Union[str, Path],
    frame_ids: Optional[Sequence[int]] = None,
    binarize: bool = True,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Load color-coded masks stored as PNGs and merge all foreground colors into one mask.

    Args:
        mask_dir : str or Path
            Directory containing per-frame color mask PNG files.
        frame_ids : sequence of int, optional
            Specific frame ids to load. When omitted, all PNG files in the
            directory are used in ascending order.
        binarize : bool
            If True (default) threshold the merged masks using `threshold`
            before returning.
        threshold : float
            Value used for binarisation. Ignored when `binarize` is False.

    Returns:
        np.ndarray
            Mask array shaped (num_frames, 1, H, W) with boolean dtype.
    """

    mask_root = Path(mask_dir)
    if frame_ids is None:
        mask_paths = sorted(mask_root.glob("*.png"))
    else:
        mask_paths = [mask_root / f"mask_{int(fid):04d}.png" for fid in frame_ids]

    if not mask_paths:
        raise FileNotFoundError(f"No PNG mask files found in {mask_root}")

    mask_arrays: List[np.ndarray] = []
    for path in mask_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing multicolor mask file: {path}")
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        mask = np.any(img > 0, axis=-1).astype(np.float32)  # Shape (H, W)
        mask = np.expand_dims(mask, axis=0)  # Shape (1, H, W)
        mask_arrays.append(mask)

    stacked = np.stack(mask_arrays, axis=0)  # Shape (num_frames, 1, H, W)
    if binarize:
        return stacked >= threshold
    return stacked.astype(np.float32, copy=False)


def demo_load_mask_shapes() -> None:
    """
    Small smoke test that loads both SAM mask variants and prints their shapes.
    """

    multiply_mask_path = (
        "/scratch/izar/cizinsky/multiply-output/training/football_high_res/"
        "visualisations/stage_sam_mask/02850/sam_opt_mask.npy"
    )
    multiply_masks = _load_multiply_sam_masks(multiply_mask_path)
    print(f"Multiply SAM masks shape: {multiply_masks.shape}")

    progressive_mask_dir = (
        "/scratch/izar/cizinsky/thesis/output/football_high_res/checkpoints/"
        "v7_football/progressive_sam"
    )
    progressive_masks = _load_progressive_sam_masks(progressive_mask_dir)
    print(f"Progressive SAM masks shape: {progressive_masks.shape}")


def get_mask_load_function(ds: str):
    if ds == "multiply":
        return _load_multiply_sam_masks
    elif ds == "progressive_sam":
        return _load_progressive_sam_masks
    elif ds == "binary_png":
        return _load_binary_masks_from_dir
    elif ds == "multicolor_png":
        return _load_multicolor_masks_from_dir
    else:
        raise ValueError(f"Unsupported dataset for mask loading: {ds}")

def load_masks_for_evaluation(gt_masks_dir_path: Path, gt_ds: str, pred_masks_dir_path: Optional[Path], pred_ds: Optional[str], device: str = "cuda") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    gt_mask_loader = get_mask_load_function(gt_ds)
    gt_masks = gt_mask_loader(gt_masks_dir_path, binarize=True) # Shape (num_frames, num_person, H, W)
    gt_masks = torch.from_numpy(gt_masks.sum(axis=1, keepdims=False)).to(device) #  Shape (num_frames, H, W); combine all persons into one mask

    if pred_masks_dir_path is not None and pred_ds is not None:
        pred_mask_loader = get_mask_load_function(pred_ds)
        pred_masks = pred_mask_loader(pred_masks_dir_path, binarize=True)
        pred_masks = torch.from_numpy(pred_masks.sum(axis=1, keepdims=False)).to(device) # Shape (num_frames, H, W)
    else:
        pred_masks = None

    return gt_masks, pred_masks


if __name__ == "__main__":
    demo_load_mask_shapes()
