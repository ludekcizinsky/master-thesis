from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import torch


def _resolve_latest_checkpoint_path(ckpt_dir: Path) -> Path:
    latest_txt = ckpt_dir / "latest.txt"
    if latest_txt.exists():
        candidate = ckpt_dir / latest_txt.read_text().strip()
        if candidate.exists():
            return candidate

    pt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    return pt_files[-1]


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
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")

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


def demo_latest_checkpoint_shape(
    internal_checkpoint_dir: Union[str, Path],
    multiply_checkpoint_path: Union[str, Path],
) -> None:
    """
    Helper used in __main__ to demonstrate successful loading.
    """

    array = load_latest_smpl_checkpoint(internal_checkpoint_dir)
    num_frames, num_persons = array.shape[:2]
    print(f"Internal checkpoint shape: {array.shape} (frames={num_frames}, persons={num_persons})")

    multiply_array = load_multiply_smpl_checkpoint(multiply_checkpoint_path)
    m_frames, m_persons = multiply_array.shape[:2]
    print(f"Multiply checkpoint shape: {multiply_array.shape} (frames={m_frames}, persons={m_persons})")


if __name__ == "__main__":
    INTERNAL_PATH = "/scratch/izar/cizinsky/thesis/output/taichi/checkpoints/v6_default/smpl"
    MULTIPLY_CKPT = "/scratch/izar/cizinsky/multiply-output/training/taichi01/checkpoints/epoch=2699-loss=0.024870239198207855.ckpt"
    demo_latest_checkpoint_shape(INTERNAL_PATH, MULTIPLY_CKPT)
