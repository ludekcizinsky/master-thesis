import os
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np

from training.helpers.utils import load_image, load_mask
from training.helpers.geom_utils import load_K_Rt_from_P


class FullSceneDataset(Dataset):
    """
    Exposes (frame_id, image, mask, cam_intrinsics, smpl_param) for a chosen track id.

    Simplifying assumptions:
    1. All persons are on all frames
    """

    def __init__(self, preprocess_dir: Path, tids: List[int], cloud_downsample: int = 10, train_bg: bool = False):
        self.preprocess_dir = preprocess_dir
        self.tids = tids
        self.train_bg = train_bg

        # Paths
        self.images_dir = self.preprocess_dir / "image"
        self.mask_dir = self.preprocess_dir / "mask"

        # Cache H,W from cam_dicts (assuming constant across frames; we still read per-frame K)
        first_image = load_image(self.images_dir / "0000.png")
        self.H, self.W = first_image.shape[:2]
        print(f"--- FYI: operating on images of size {self.W}x{self.H}.")

        self.n_samples = len(list(self.images_dir.glob("*.png")))
        print(f"--- FYI: found {self.n_samples} frames in {self.images_dir}")

        # Cameras
        self._load_cameras()

        # SMPL params
        self._load_smpl_params()

        # Point cloud (for static background)
        if self.train_bg:
            self.load_unidepth_pointcloud(downsample=cloud_downsample)


    def _load_cameras(self):

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


    def load_unidepth_pointcloud(self, downsample: int = 1):
        """
        Load UniDepth fused point cloud saved as .npz.

        Args:
            npz_path (str or Path): Path to .npz file (must contain 'points' and 'colors').
            downsample (int): Keep every k-th point. Default=1 (no downsampling).

        Returns:
            pts_world (N,3) float32: 3D points in Trace world coordinates
            colors    (N,3) uint8 : Corresponding RGB values
        """
        npz_path = self.preprocess_dir / "unidepth_cloud_static_scaled.npz"
        data = np.load(npz_path)
        pts = data["points"].astype(np.float32)
        cols = data["colors"].astype(np.uint8)

        if downsample > 1:
            pts = pts[::downsample]
            cols = cols[::downsample]

        self.point_cloud = (pts, cols)
        print(f"--- FYI: loaded POINT CLOUD from {npz_path} with {pts.shape[0]} points (downsample={downsample}).")

    def _load_masks(self, tids: List[int], idx: int):
        masks = []
        for tid in tids:
            mask_path = self.mask_dir / str(tid) / f"{idx:04d}.png"
            if mask_path.exists():
                mask = load_mask(mask_path)  # [H,W]
            else:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            masks.append(mask)
        mask = torch.stack(masks, dim=0)  # [P,H,W] combined mask
        assert mask.shape[0] == len(tids), f"Mask shape {mask.shape} does not match number of tids {len(tids)}"
        return mask

    def __len__(self):
        return len(self.pose_all)

    def __getitem__(self, idx: int):

        # Image
        img_path = self.images_dir / f"{idx:04d}.png"
        image = load_image(img_path)  # [H,W,3]
        assert image.shape[0] == self.H and image.shape[1] == self.W, f"Image shape mismatch: {image.shape} vs ({self.H},{self.W})"

        # Masks for each tid
        # - load masks only for selected tids
        if len(self.tids) > 0:
            mask = self._load_masks(self.tids, idx)  # [P,H,W]
        # - if no tids -> static background training -> load all masks and combine
        else:
            tids = os.listdir(self.mask_dir)
            mask = self._load_masks(tids, idx)  # [P,H,W]

        # Camera 
        # - Extrinsics
        M_ext = self.pose_all[idx]  # [4,4]
        assert M_ext.shape == (4, 4), f"Extrinsics shape mismatch: {M_ext.shape} vs {(4, 4)}"

        # - Intrinsics
        intrinsics = self.intrinsics_all[idx][:3, :3]  # [3,3]
        K = intrinsics.clone()
        assert K.shape == (3, 3), f"K shape mismatch: {K.shape} vs {(3, 3)}"

        # SMPL params (86D)
        if len(self.tids) > 0:
            all_smpl_params = []
            for tid in self.tids:
                smpl_params = torch.zeros([86]).float()
                smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float()
                smpl_params[1:4] = torch.from_numpy(self.trans[idx][tid]*self.scale).float()
                smpl_params[4:76] = torch.from_numpy(self.poses[idx][tid]).float()
                smpl_params[76:] = torch.from_numpy(self.shape[tid]).float()
                all_smpl_params.append(smpl_params)
            smpl_params = torch.stack(all_smpl_params, dim=0)  # [P,86]
            assert smpl_params.shape[0] == len(self.tids) and smpl_params.shape[1] == 86, f"SMPL params shape mismatch: {smpl_params.shape} vs ({len(self.tids)}, 86)"
        else:
            smpl_params = torch.zeros([1, 86]).float()  # dummy

        return {
            "fid": idx,
            "image": image,     # [H,W,3]
            "mask": mask,       # [P,H,W]
            "K": K,             # [3,3]
            "smpl_param": smpl_params,  # [P,86]
            "M_ext": M_ext,         # [4,4]
            "W": self.W,
            "H": self.H,
        }
