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

    pt_files = sorted(ckpt_dir.glob("iter_*.pt"))
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
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    params_dict = _extract_params_dict(payload)
    stacked, _ = _stack_frame_params(params_dict)
    return stacked


def demo_latest_checkpoint_shape(checkpoint_dir: Union[str, Path]) -> None:
    """
    Helper used in __main__ to demonstrate successful loading.
    """

    array = load_latest_smpl_checkpoint(checkpoint_dir)
    num_frames, num_persons = array.shape[:2]
    print(f"Loaded checkpoint shape: {array.shape} (frames={num_frames}, persons={num_persons})")


if __name__ == "__main__":
    DEMO_PATH = "/scratch/izar/cizinsky/thesis/output/taichi/checkpoints/v6_default/smpl"
    demo_latest_checkpoint_shape(DEMO_PATH)
