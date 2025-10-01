import os
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np

from utils.io import load_frame_map_jsonl_restore
from preprocess.helpers.cameras import load_camdicts_json
from training.helpers.utils import load_image, load_mask
from training.helpers.geom_utils import load_K_Rt_from_P

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


class TraceDataset(Dataset):
    """
    Exposes (frame_id, image, mask, cam_intrinsics, smpl_param) for a chosen track id.

    Simplifying assumptions:
    1. All persons are on all frames
    """

    def __init__(
        self,
        preprocess_dir: Path,
        tid: int,
        downscale: int = 2,
    ):
        self.preprocess_dir = preprocess_dir
        self.tid = int(tid)
        self.downscale = downscale

        # Paths
        self.images_dir = self.preprocess_dir / "image"
        self.masks_dir = self.preprocess_dir / "mask" / str(self.tid)
        assert self.images_dir.exists(), f"Images dir not found: {self.images_dir}"
        assert self.masks_dir.exists(), f"Masks dir not found: {self.masks_dir}"

        # Cache H,W from cam_dicts (assuming constant across frames; we still read per-frame K)
        fid = 0
        first_mask = load_mask(self.masks_dir / f"{fid:04d}.png", 1)
        self.W_full, self.H_full = first_mask.shape[1], first_mask.shape[0]
        self.W = self.W_full // self.downscale
        self.H = self.H_full // self.downscale
        print(f"--- FYI: image size {self.W_full}x{self.H_full} downscaled to {self.W}x{self.H}")

        self.n_samples = len(list(self.images_dir.glob("*.png")))
        print(f"--- FYI: found {self.n_samples} frames in {self.images_dir}")

        # Cameras
        self._load_cameras()

        # SMPL params
        self._load_smpl_params()


    def _load_cameras(self):
        # TODO: polish this function

        camera_dict = np.load(os.path.join(self.preprocess_dir, "cameras_normalize.npz"))
        scale_mats = [camera_dict[f'scale_mat_{idx}'].astype(np.float32) for idx in range(self.n_samples)]
        world_mats = [camera_dict[f'world_mat_{idx}'].astype(np.float32) for idx in range(self.n_samples)]

        self.scale = 1 / scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        self.Ps = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            self.Ps.append(torch.from_numpy(P).float())

            intrinsics, pose = load_K_Rt_from_P(P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        assert len(self.intrinsics_all) == len(self.pose_all), f"Len of intrinsics {len(self.intrinsics_all)} != len of extrinsics {len(self.pose_all)} matrices"

    def _load_smpl_params(self):
        self.shape = np.load(self.preprocess_dir / 'mean_shape.npy')
        self.poses = np.load(self.preprocess_dir / 'poses.npy')
        self.trans = np.load(self.preprocess_dir / 'normalize_trans.npy')


    def __len__(self):
        return len(self.pose_all)

    def __getitem__(self, idx: int):

        # Image
        img_path = self.images_dir / f"{idx:04d}.png"
        image = load_image(img_path, self.downscale)  # [H,W,3]
        assert image.shape == (self.H, self.W, 3), f"Image shape mismatch: {image.shape} vs {(self.H, self.W, 3)}"

        # Mask
        mask_path = self.masks_dir / f"{idx:04d}.png"
        mask = load_mask(mask_path, self.downscale)  # [H,W]
        assert mask.shape == (self.H, self.W), f"Mask shape mismatch: {mask.shape} vs {(self.H, self.W)}"

        # Camera 
        # - Extrinsics
        M_ext = self.pose_all[idx]  # [4,4]
        assert M_ext.shape == (4, 4), f"Extrinsics shape mismatch: {M_ext.shape} vs {(4, 4)}"

        # - Intrinsics
        intrinsics = self.intrinsics_all[idx][:3, :3]  # [3,3]
        K = intrinsics.clone()
        assert K.shape == (3, 3), f"K shape mismatch: {K.shape} vs {(3, 3)}"

        # Adjust intrinsics for downscale
        K[0,0] /= self.downscale  # fx
        K[1,1] /= self.downscale  # fy
        K[0,2] /= self.downscale  # cx
        K[1,2] /= self.downscale  # cy

        # SMPL params (86D) for this tid
        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float()
        smpl_params[1:4] = torch.from_numpy(self.trans[idx][self.tid]*self.scale).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx][self.tid]).float()
        smpl_params[76:] = torch.from_numpy(self.shape[self.tid]).float()

        # s = float(self.scale)
        # t = self.trans[idx][self.tid] * self.scale
        # smpl_params[0] = torch.tensor(s, dtype=torch.float32)
        # smpl_params[1:4] = torch.from_numpy(t).float()

        return {
            "fid": idx,
            "image": image,     # [3,H,W]
            "mask": mask,       # [H,W]
            "K": K,             # [3,3]
            "smpl_param": smpl_params,  # [86]
            "M_ext": M_ext,         # [4,4]
            "P": self.Ps[idx],     # [3,4]
        }