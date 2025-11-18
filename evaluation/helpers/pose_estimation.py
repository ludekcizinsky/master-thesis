from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union, Optional

import numpy as np
import torch

from training.helpers.smpl_utils import get_joints_from_pose_params


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

def _load_multiply_smpl_checkpoint(ckpt_dir_path: Path) -> np.ndarray:
    """
    Load SMPL parameters from a Multiply Lightning checkpoint and return (F, P, 86).
    """

    ckpt_path = ckpt_dir_path / "last.ckpt"
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

def _load_latest_smpl_checkpoint(checkpoint_dir: Union[str, Path]) -> np.ndarray:
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

def _load_metric_alignment(preprocess_dir: Path, pred_method: str, tgt_ds_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alignment_path = preprocess_dir / "smpl_joints_alignment_transforms" / pred_method / f"{tgt_ds_name}.npz"
    if not alignment_path.exists():
        raise FileNotFoundError(f"SMPL joints alignment transforms not found at {alignment_path}")
    data = np.load(alignment_path, allow_pickle=False)
    rotations = torch.from_numpy(data["rotations"]).float()
    translations = torch.from_numpy(data["translations"]).float()
    scales = torch.from_numpy(data["scales"]).float()

    return rotations, translations, scales

def align_input_smpl_joints(src_joints: torch.Tensor, frame_idx: int, person_idx: int, transformations: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Apply precomputed similarity transform to align SMPL joints to metric space.

    Args:
        src_joints: Source SMPL joints of shape (J, 3) in canonical space (arbitrary units).
        frame_idx: Frame index to select the transform.
        person_idx: Person index to select the transform.
        transformations: Tuple of (rotations, translations, scales) where:
            - rotations: Tensor of shape (F, P, 3, 3)
            - translations: Tensor of shape (F, P, 3)
            - scales: Tensor of shape (F, P, 1)

    Returns:
        dst_joints: Transformed SMPL joints of shape (J, 3) in metric space. 
    """

    # parse the transforms
    rotations, translations, scales = transformations
    rot = rotations[frame_idx, person_idx]
    trans_vec = translations[frame_idx, person_idx]
    scale_val = scales[frame_idx, person_idx]

    # apply the similarity transform
    original_device = src_joints.device
    device = rot.device
    dst_joints = (rot @ src_joints.to(device).T).T * scale_val + trans_vec
    dst_joints = dst_joints.to(original_device)

    return dst_joints 

def load_ours_pred_joints_canonical(pred_joints_dir_path: Path) -> torch.Tensor:
    """
    Load predicted 3D joints from our method's SMPL checkpoints in canonical space.
    Args:
        pred_joints_dir_path: Path
            Directory containing SMPL parameter checkpoints.
    Returns:
        torch.Tensor
            Tensor of shape (num_frames, num_persons, 24, 3) with 3D joint positions.
    """

    # Load the unaligned smpl joints
    smpl_params = _load_latest_smpl_checkpoint(pred_joints_dir_path)  # Shape (F, P, 86)
    num_frames, num_persons, _ = smpl_params.shape
    smpl_params_reshaped = smpl_params.reshape(num_frames * num_persons, 86)
    smpl_joints = get_joints_from_pose_params(torch.as_tensor(smpl_params_reshaped, device="cpu"))  # Shape (F*P, 24, 3)
    smpl_joints = smpl_joints.reshape(num_frames, num_persons, 24, 3)  # Shape (F, P, 24, 3)

    return smpl_joints

def load_ours_pred_joints(pred_joints_dir_path: Path, transformations_dir_path: Path, pred_method: str) -> torch.Tensor:
    """
    Load predicted 3D joints from our method's SMPL checkpoints.
    Args:
        pred_joints_dir_path: Path
            Directory containing SMPL parameter checkpoints.
        transformations_dir_path: Path
            Directory containing precomputed alignment transforms.
    Returns:
        torch.Tensor
            Tensor of shape (num_frames, num_persons, 24, 3) with 3D joint positions.
    """

    # Load the unaligned smpl joints
    smpl_joints = load_ours_pred_joints_canonical(pred_joints_dir_path)  # Shape (F, P, 24, 3)
    num_frames, num_persons, _, _ = smpl_joints.shape

    # Align the joints to the gt dataset
    transformations = _load_metric_alignment(transformations_dir_path, tgt_ds_name="hi4d", pred_method=pred_method)
    aligned_smpl_joints = torch.zeros_like(smpl_joints)
    for fidx in range(num_frames):
        for pidx in range(num_persons):
            smpl_joints[fidx, pidx] = align_input_smpl_joints(
                src_joints=smpl_joints[fidx, pidx],
                frame_idx=fidx,
                person_idx=pidx,
                transformations=transformations,
            )
            aligned_smpl_joints[fidx, pidx] = smpl_joints[fidx, pidx]


    return aligned_smpl_joints


def load_multiply_pred_joints(pred_joints_dir_path: Path, transformations_dir_path: Path) -> torch.Tensor:

    # Load the unaligned smpl joints
    smpl_params = _load_multiply_smpl_checkpoint(pred_joints_dir_path)  # Shape (F, P, 86)
    num_frames, num_persons, _ = smpl_params.shape
    smpl_params_reshaped = smpl_params.reshape(num_frames * num_persons, 86)
    smpl_joints = get_joints_from_pose_params(torch.as_tensor(smpl_params_reshaped, device="cpu"))  # Shape (F*P, 24, 3)
    smpl_joints = smpl_joints.reshape(num_frames, num_persons, 24, 3)  # Shape (F, P, 24, 3)


    # Align the joints to the gt dataset
    transformations = _load_metric_alignment(transformations_dir_path, tgt_ds_name="hi4d")
    aligned_smpl_joints = torch.zeros_like(smpl_joints)
    for fidx in range(num_frames):
        for pidx in range(num_persons):
            smpl_joints[fidx, pidx] = align_input_smpl_joints(
                src_joints=smpl_joints[fidx, pidx],
                frame_idx=fidx,
                person_idx=pidx,
                transformations=transformations,
            )
            aligned_smpl_joints[fidx, pidx] = smpl_joints[fidx, pidx]


    return aligned_smpl_joints


def load_hi4d_gt_joints(gt_joints_dir_path: Path) -> torch.Tensor:
    """
    Load ground-truth 3D joints from HI4D dataset.

    Args:
        gt_joints_dir_path: Path
            Directory containing per-frame ground-truth joint .npz files.
    Returns:
        torch.Tensor
            Tensor of shape (num_frames, num_persons, 24, 3) with 3D joint positions. 
    """
    frame_files = sorted(gt_joints_dir_path.glob("*.npz"))
    frame_smpl_joints: list[np.ndarray] = []

    for idx, path in enumerate(frame_files):
        data = np.load(path, allow_pickle=False)
        joints = data["joints_3d"].astype(np.float32) # [P, 24, 3]
        frame_smpl_joints.append(joints)

    smpl_joints = np.stack(frame_smpl_joints, axis=0)  # [F, P, 24, 3] 
    return torch.from_numpy(smpl_joints).float()

def get_3d_joint_load_function(ds: str):
    if ds == "hi4d":
        return load_hi4d_gt_joints
    elif ds == "ours":
        return load_ours_pred_joints
    elif ds == "ours_canonical":
        return load_ours_pred_joints_canonical
    elif ds == "multiply":
        return load_multiply_pred_joints
    else:
        raise ValueError(f"Unsupported dataset for mask loading: {ds}")

def load_3d_joints_for_evaluation(gt_joints_dir_path: Path, gt_ds: str, pred_joints_dir_path: Optional[Path], pred_ds: Optional[str], trnsfm_dir_path: Optional[Path], pred_method: Optional[str], device: str = "cuda") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    gt_joint_loader = get_3d_joint_load_function(gt_ds)
    gt_joints = gt_joint_loader(gt_joints_dir_path).to(device) # Shape (num_frames, num_person, J, 3)

    if pred_joints_dir_path is not None and pred_ds is not None:
        pred_joint_loader = get_3d_joint_load_function(pred_ds)
        pred_joints = pred_joint_loader(pred_joints_dir_path, trnsfm_dir_path, pred_method).to(device) # Shape (num_frames, num_person, J, 3)
    else:
        pred_joints = None
    return gt_joints, pred_joints