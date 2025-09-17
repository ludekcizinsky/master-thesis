from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np

from utils.io import load_frame_map_jsonl_restore
from preprocess.helpers.cameras import load_camdicts_json
from training.helpers.utils import load_image, load_mask


class HumanOnlyDataset(Dataset):
    """
    Exposes (frame_id, image, mask, cam_intrinsics, smpl_param) for a chosen track id.
    """

    def __init__(
        self,
        preprocess_dir: Path,
        tid: int,
        split: str = "train",
        downscale: int = 2,
        val_fids: List[int] = []  # Only used if split=='val'
    ):
        self.preprocess_dir = preprocess_dir
        self.tid = int(tid)
        self.split = split
        self.downscale = downscale

        # Load cam dicts (by frame_id)
        cam_dicts_path = self.preprocess_dir / "cam_dicts.json"
        self.cam_dicts = load_camdicts_json(cam_dicts_path)

        # Load frame map jsonl
        frame_map_path = self.preprocess_dir / "frame_map.jsonl"
        self.frame_map = load_frame_map_jsonl_restore(frame_map_path, self.preprocess_dir)

        # Collect frame_ids where this tid exists
        self.samples: List[int] = []
        for fid_str, tracks in self.frame_map.items():
            if self.tid in tracks:
                self.samples.append(int(fid_str))
        self.samples.sort()
        print(f"--- FYI: found {len(self.samples)} frames for tid={self.tid}")

        # Define the split
        if self.split == "train":
            self.samples = self.samples  # Start with all samples
        elif self.split == "val":
            self.samples = [fid for fid in self.samples if fid in val_fids]
        else:
            raise ValueError(f"Unknown split: {self.split}")
        print(f"--- FYI: split '{self.split}' has {len(self.samples)} frames for tid={self.tid}")

        # Paths
        self.images_dir = self.preprocess_dir / "images"
        self.masks_dir = self.preprocess_dir / "masks" / str(self.tid)
        assert self.images_dir.exists(), f"Images dir not found: {self.images_dir}"
        assert self.masks_dir.exists(), f"Masks dir not found: {self.masks_dir}"

        # Sanity: check at least one frame
        if not self.samples:
            raise ValueError(f"No frames found for tid={self.tid}")

        # Cache H,W from cam_dicts (assuming constant across frames; we still read per-frame K)
        first_fid = self.samples[0]
        c0 = self.cam_dicts[first_fid]
        self.W_full, self.H_full = int(c0["W"]), int(c0["H"])
        self.W = self.W_full // self.downscale
        self.H = self.H_full // self.downscale

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        fid = self.samples[idx]

        # Load image & mask
        img_path = self.images_dir / f"frame_{fid:05d}.jpg"

        image = load_image(img_path, self.downscale)  # [H,W,3]

        mask_path = self.masks_dir / f"{fid:05d}.png"
        if mask_path.exists():
            mask = load_mask(mask_path, self.downscale)  # [H,W]
        else:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Camera intrinsics
        cam = self.cam_dicts[fid]
        fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
        # Adjust intrinsics for downscale
        fx /= self.downscale
        fy /= self.downscale
        cx /= self.downscale
        cy /= self.downscale
        K = torch.tensor([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)

        # SMPL params (86D) for this tid
        smpl_param = np.array(self.frame_map[fid][self.tid]["smpl_param"], dtype=np.float32)
        smpl_param = torch.from_numpy(smpl_param)  # [86]

        return {
            "fid": fid,
            "image": image,     # [3,H,W]
            "mask": mask,       # [H,W]
            "K": K,             # [3,3]
            "smpl_param": smpl_param,  # [86]
            "confidence": self.frame_map[fid][self.tid]["confidence"]  # float
        }
