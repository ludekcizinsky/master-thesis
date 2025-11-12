from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import torch


def _resolve_latest_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path

    latest_txt = path / "latest.txt"
    if latest_txt.exists():
        candidate = path / latest_txt.read_text().strip()
        if candidate.exists():
            return candidate

    epoch_files = sorted(path.glob("epoch_*.pt"))
    if epoch_files:
        return epoch_files[-1]

    iter_files = sorted(path.glob("iter_*.pt"))
    if iter_files:
        return iter_files[-1]

    raise FileNotFoundError(f"No checkpoint files found in {path}")


def _extract_params_dict(payload: Any) -> Dict[int, torch.Tensor]:
    """
    Return the SMPL params dict {frame_id: tensor} from a checkpoint payload.
    """
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict.")

    params = payload.get("params", None)
    if params is None:
        raise ValueError("Checkpoint payload missing 'params' entry.")

    smpl_params: Dict[int, torch.Tensor] = {}
    for fid_key, tensor in params.items():
        fid_int = int(fid_key)
        smpl_params[fid_int] = torch.as_tensor(tensor).detach().cpu()
    return smpl_params


def _stack_frame_params(params: Dict[int, torch.Tensor]) -> Tuple[np.ndarray, Iterable[int]]:
    if not params:
        raise ValueError("No SMPL parameters stored in checkpoint.")

    sorted_items = sorted(params.items(), key=lambda kv: kv[0])
    first_tensor = sorted_items[0][1]
    if first_tensor.ndim < 2:
        raise ValueError(f"Expected frame tensor with shape (P, D), got {first_tensor.shape}")

    num_persons = first_tensor.shape[0]
    frame_arrays = []
    for fid, tensor in sorted_items:
        if tensor.shape[0] != num_persons:
            raise AssertionError(
                f"Inconsistent number of persons across frames: expected {num_persons}, got {tensor.shape[0]} for frame {fid}"
            )
        frame_arrays.append(tensor.numpy())

    stacked = np.stack(frame_arrays, axis=0)
    assert stacked.shape[1] == num_persons
    return stacked, [fid for fid, _ in sorted_items]


def load_latest_smpl_checkpoint(checkpoint_dir: Union[str, Path]) -> np.ndarray:
    """
    Load the latest SMPL checkpoint and return an array with shape (F, P, D).

    F = number of frames, P = number of persons, D = parameter dimension (typically 86).
    """

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_dir}")

    ckpt_path = _resolve_latest_checkpoint_path(ckpt_dir)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    params_dict = _extract_params_dict(payload)
    stacked, _ = _stack_frame_params(params_dict)
    return stacked


def load_multiply_smpl_checkpoint(ckpt_path: Union[str, Path]) -> np.ndarray:
    """
    Load SMPL parameters from a Multiply Lightning checkpoint and return (F, P, 86).
    """

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if state is None:
        raise ValueError(f"Checkpoint {ckpt_path} missing 'state_dict'")

    person_ids = sorted({int(key.split(".")[1]) for key in state if key.startswith("body_model_list.")})
    if not person_ids:
        raise ValueError(f"No SMPL parameters found inside checkpoint {ckpt_path}")

    frame_tensors = []
    num_frames = None

    def _fetch(person_id: int, name: str) -> torch.Tensor:
        key = f"body_model_list.{person_id}.{name}.weight"
        if key not in state:
            raise KeyError(f"Missing key '{key}' in checkpoint {ckpt_path}")
        return state[key].detach().cpu()

    for pid in person_ids:
        transl = _fetch(pid, "transl")
        orient = _fetch(pid, "global_orient")
        body_pose = _fetch(pid, "body_pose")
        betas = _fetch(pid, "betas")

        frames = transl.shape[0]
        if num_frames is None:
            num_frames = frames
        elif frames != num_frames:
            raise AssertionError(
                f"Inconsistent frame counts across persons: {frames} vs expected {num_frames}"
            )

        if orient.shape[0] != frames or body_pose.shape[0] != frames:
            raise AssertionError("Orientation/body pose tensors do not share frame count with translations.")

        if betas.shape[0] == 1:
            betas = betas.repeat(frames, 1)
        elif betas.shape[0] != frames:
            raise AssertionError("Betas must either be per-frame or frame-invariant.")

        scale = torch.ones(frames, 1, dtype=transl.dtype)
        person_tensor = torch.cat([scale, transl, orient, body_pose, betas], dim=1)
        if person_tensor.shape[1] != 86:
            raise AssertionError(f"Unexpected SMPL parameter dimension: {person_tensor.shape}")
        frame_tensors.append(person_tensor)

    stacked = torch.stack(frame_tensors, dim=1).numpy()
    return stacked


def load_gt_smpl_params(
    gt_dir: Union[str, Path],
    preprocess_dir: Union[str, Path],
) -> np.ndarray:
    """
    Load ground-truth SMPL params stored as per-frame .npz files, apply preprocessing alignments,
    and return (F, P, 86).
    """

    gt_root = Path(gt_dir)
    if not gt_root.exists():
        raise FileNotFoundError(f"Ground-truth SMPL directory not found: {gt_root}")

    frame_files = sorted(gt_root.glob("*.npz"))
    if not frame_files:
        raise FileNotFoundError(f"No .npz frames found in {gt_root}")

    preprocess_root = Path(preprocess_dir)
    if not preprocess_root.exists():
        raise FileNotFoundError(f"Preprocessing directory not found: {preprocess_root}")

    poses = np.load(preprocess_root / "poses.npy")
    trans = np.load(preprocess_root / "normalize_trans.npy")
    shape = np.load(preprocess_root / "mean_shape.npy")
    if poses.shape[0] != len(frame_files):
        raise ValueError("Mismatch between poses.npy frames and raw SMPL frames.")

    cam_norm_path = preprocess_root / "cameras_normalize.npz"
    if not cam_norm_path.exists():
        raise FileNotFoundError(f"{cam_norm_path} not found. Cannot determine scale.")
    cam_norm = np.load(cam_norm_path)
    scale_mat = cam_norm["scale_mat_0"]
    scale_value = 1.0 / float(scale_mat[0, 0])

    frame_tensors = []
    num_persons = poses.shape[1]
    for idx in range(len(frame_files)):
        frame_pose = poses[idx]
        frame_trans = trans[idx] * scale_value
        scale = np.full((num_persons, 1), scale_value, dtype=np.float32)
        frame_tensor = np.concatenate(
            [scale, frame_trans, frame_pose[:, :3], frame_pose[:, 3:], shape],
            axis=1,
        )
        if frame_tensor.shape[1] != 86:
            raise ValueError(f"Unexpected SMPL parameter dimension {frame_tensor.shape[1]} at frame {idx}")
        frame_tensors.append(frame_tensor)

    stacked = np.stack(frame_tensors, axis=0)
    return stacked


def demo_predictions_vs_gt(
    prediction_checkpoint: Union[str, Path],
    gt_raw_dir: Union[str, Path],
    preprocess_dir: Union[str, Path],
) -> None:
    pred_array = load_latest_smpl_checkpoint(prediction_checkpoint)
    p_frames, p_persons = pred_array.shape[:2]
    print(f"Prediction checkpoint shape: {pred_array.shape} (frames={p_frames}, persons={p_persons})")

    gt_array = load_gt_smpl_params(gt_raw_dir, preprocess_dir)
    g_frames, g_persons = gt_array.shape[:2]
    print(f"GT params shape: {gt_array.shape} (frames={g_frames}, persons={g_persons})")

    if pred_array.shape != gt_array.shape:
        print("WARNING: predicted and GT shapes differ.")
    else:
        print("SUCCESS: predicted and GT tensors share the same shape.")


if __name__ == "__main__":
    PREDICTION_CKPT = "/scratch/izar/cizinsky/thesis/output/hi4d_pair00_dance00_cam76/checkpoints/v7_hi4d_pair00_dance00_cam76/smpl/iter_005000.pt"
    GT_SMPL_DIR = "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00/smpl"
    PREPROCESS_DIR = "/scratch/izar/cizinsky/multiply-output/preprocessing/data/hi4d_pair00_dance00_cam76"
    demo_predictions_vs_gt(PREDICTION_CKPT, GT_SMPL_DIR, PREPROCESS_DIR)
