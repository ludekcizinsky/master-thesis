import os
from pathlib import Path
from typing import List, Sequence, Union, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from training.helpers.geom_utils import load_K_Rt_from_P
from training.helpers.progressive_sam import ProgressiveSAMManager


class Hi4DDataset:
    def __init__(self, mask_root: Optional[Union[str, Path]], smpl_root: Optional[Union[str, Path]], preprocess_dir: Optional[Union[str, Path]], pred_method: str = "ours"):

        # Segmentation masks
        self.mask_root = Path(mask_root) if mask_root is not None else None
        if self.mask_root is not None and not self.mask_root.exists():
            raise FileNotFoundError(f"Mask root '{self.mask_root}' does not exist.")
        elif self.mask_root is not None:
            self.person_dirs: Sequence[Path] = sorted(
                [d for d in self.mask_root.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda d: int(d.name),
            )
            if not self.person_dirs:
                raise FileNotFoundError(f"No person-id subdirectories found under '{self.mask_root}'.")
            person_files_sorted_ascending = sorted(list(self.person_dirs[0].glob("*.png")))
            sample_file = person_files_sorted_ascending[0] if person_files_sorted_ascending else None
            if sample_file is None:
                raise FileNotFoundError(f"No PNG files found in '{self.person_dirs[0]}'.")
            self.frame_name_width = len(sample_file.stem)
            self.seg_mask_dir_offset = int(sample_file.stem)

        # SMPL parameters
        self.smpl_root = Path(smpl_root) if smpl_root is not None else None
        if self.smpl_root is not None and not self.smpl_root.exists():
            raise FileNotFoundError(f"SMPL root '{self.smpl_root}' does not exist.")

        # Preprocessing directory -> needed for SMPL alignment
        self.preprocess_dir = Path(preprocess_dir) if preprocess_dir is not None else None
        if self.preprocess_dir is not None and not self.preprocess_dir.exists():
            raise FileNotFoundError(f"Preprocessing directory '{self.preprocess_dir}' does not exist.")
        
        self.pred_method = pred_method
        
        # Load SMPL params
        self._load_gt_smpl_params() 
        self._load_metric_alignment()


    def load_segmentation_masks(self, frame_id: int) -> torch.Tensor:
        """
        Load per-person segmentation masks for a given frame, ordered by person id.
        Returns a tensor shaped [P, H, W] with boolean masks.
        """
        if self.mask_root is None:
            raise ValueError("Mask root directory not specified for segmentation mask loading.")
    
        masks = []
        target_frame = self.seg_mask_dir_offset + frame_id
        for person_dir in self.person_dirs:
            frame_path = person_dir / f"{target_frame:0{self.frame_name_width}d}.png"
            if not frame_path.exists():
                raise FileNotFoundError(f"Mask for frame {target_frame} not found in '{person_dir}'.")

            mask_img = Image.open(frame_path).convert("L")
            mask_np = np.array(mask_img) > 127
            masks.append(torch.from_numpy(mask_np.astype(np.bool_)))

        return torch.stack(masks, dim=0)
    
    def is_seg_mask_loading_available(self) -> bool:
        """Check if segmentation mask loading is available."""
        return self.mask_root is not None 

    def _load_gt_smpl_params(self) -> np.ndarray:
        """
        Load ground-truth SMPL params stored as per-frame .npz files (already in metric units)
        and return (F, P, 86).
        """
        if self.smpl_root is None:
            self.smpl_params = None
            self.smpl_joints = None
            return 

        gt_root = self.smpl_root
        frame_files = sorted(gt_root.glob("*.npz"))
        if not frame_files:
            raise FileNotFoundError(f"No .npz frames found in {gt_root}")

        frame_smpl_params: list[np.ndarray] = []
        frame_smpl_joints: list[np.ndarray] = []
        for idx, path in enumerate(frame_files):
            data = np.load(path, allow_pickle=False)
            betas = data["betas"].astype(np.float32)
            global_orient = data["global_orient"].astype(np.float32)
            body_pose = data["body_pose"].astype(np.float32)
            transl = data["transl"].astype(np.float32)
            joints = data["joints_3d"].astype(np.float32) # [P, 24, 3]
            num_persons = joints.shape[0] 

            scale = np.ones((num_persons, 1), dtype=np.float32)
            frame_tensor = np.concatenate(
                [scale, transl, global_orient, body_pose, betas],
                axis=1,
            )
            if frame_tensor.shape[1] != 86:
                raise ValueError(
                    f"Unexpected SMPL parameter dimension {frame_tensor.shape[1]} at frame {idx}"
                )
            frame_smpl_params.append(frame_tensor)
            frame_smpl_joints.append(joints)

        self.smpl_params = torch.from_numpy(np.stack(frame_smpl_params, axis=0)).float()  # [F, P, 86]
        self.smpl_joints = torch.from_numpy(np.stack(frame_smpl_joints, axis=0)).float()  # [F, P, 24, 3]
        print(f"--- FYI: Loaded GT SMPL parameters from {len(frame_files)} frames in {gt_root}. Shape: {self.smpl_params.shape}")
        print(f"--- FYI: Loaded GT SMPL joints from {len(frame_files)} frames in {gt_root}. Shape: {self.smpl_joints.shape}")

    
    def load_smpl_params_for_frame(self, frame_id: int) -> torch.Tensor:
        """
        Load SMPL parameters for a given frame.

        Args:
            frame_id: Frame index to load SMPL parameters for.
        Returns:
            smpl_params: SMPL parameters tensor of shape (P, 86) for the specified frame.
        """
        if self.smpl_params is None:
            raise ValueError("SMPL parameters not loaded or available.")

        if frame_id < 0 or frame_id >= self.smpl_params.shape[0]:
            raise IndexError(f"Frame id {frame_id} out of bounds for SMPL parameters with {self.smpl_params.shape[0]} frames.")

        return self.smpl_params[frame_id]

    def load_smpl_joints_for_frame(self, frame_id: int) -> torch.Tensor:
        """
        Load SMPL joint positions for a given frame.

        Args:
            frame_id: Frame index to load SMPL joints for.
        Returns:
            smpl_joints: SMPL joint positions tensor of shape (P, 24, 3) for the specified frame.
        """
        if self.smpl_joints is None:
            raise ValueError("SMPL joints not loaded or available.")

        if frame_id < 0 or frame_id >= self.smpl_joints.shape[0]:
            raise IndexError(f"Frame id {frame_id} out of bounds for SMPL joints with {self.smpl_joints.shape[0]} frames.")

        return self.smpl_joints[frame_id] 
    
    def is_smpl_loading_available(self) -> bool:
        return self.smpl_root is not None and self.smpl_params is not None

    def _load_metric_alignment(self) -> Optional[dict]:
        alignment_path = self.preprocess_dir / "smpl_joints_alignment_transforms" / self.pred_method / "hi4d.npz"
        if not alignment_path.exists():
            raise FileNotFoundError(f"SMPL joints alignment transforms not found at {alignment_path}")
        data = np.load(alignment_path)
        self.rotations = torch.from_numpy(data["rotations"]).float()
        self.translations = torch.from_numpy(data["translations"]).float()
        self.scales = torch.from_numpy(data["scales"]).float()

    def align_input_smpl_joints(self, src_joints: torch.Tensor, frame_idx: int, person_idx: int) -> torch.Tensor:
        """
        Apply precomputed similarity transform to align SMPL joints to metric space.

        Args:
            src_joints: Source SMPL joints of shape (J, 3) in canonical space (arbitrary units).
            frame_idx: Frame index to select the transform.
            person_idx: Person index to select the transform.

        Returns:
            dst_joints: Transformed SMPL joints of shape (J, 3) in metric space. 
        """
        
        # parse the transforms
        rot = self.rotations[frame_idx, person_idx]
        trans_vec = self.translations[frame_idx, person_idx]
        scale_val = self.scales[frame_idx, person_idx]

        # apply the similarity transform to map 
        original_device = src_joints.device
        device = rot.device
        dst_joints = (rot @ src_joints.to(device).T).T * scale_val + trans_vec
        dst_joints = dst_joints.to(original_device)

        return dst_joints 


class FullSceneDataset(Dataset):
    """
    Exposes (frame_id, image, cam_intrinsics, smpl_param) for a chosen track id.

    Simplifying assumptions:
    1. All persons are on all frames
    """

    def __init__(self, preprocess_dir: Path, tids: List[int], mask_path: Path, cloud_downsample: int = 10, train_bg: bool = False):
        self.preprocess_dir = preprocess_dir
        self.tids = tids
        self.train_bg = train_bg
        self.mask_path = mask_path

        # Paths
        self.images_dir = self.preprocess_dir / "image"

        # Cache H,W from cam_dicts (assuming constant across frames; we still read per-frame K)
        first_image = self._load_image(self.images_dir / "0000.png")
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

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        im = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3]
        return im.contiguous() 

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

    def __len__(self):
        return len(self.pose_all)

    def __getitem__(self, idx: int):

        # Image
        img_path = self.images_dir / f"{idx:04d}.png"
        image = self._load_image(img_path)  # [H,W,3]
        assert image.shape[0] == self.H and image.shape[1] == self.W, f"Image shape mismatch: {image.shape} vs ({self.H},{self.W})"

        # Human masks
        mask_entry = ProgressiveSAMManager._load_entry_from_disk(self.mask_path, fid=idx, device='cpu')
        if mask_entry is not None:
            human_masks = mask_entry.refined
        else:
            human_masks = torch.zeros((0, self.H, self.W), dtype=torch.bool)  # No masks available

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
            "human_mask": human_masks,  # [P,H,W]
            "K": K,             # [3,3]
            "smpl_param": smpl_params,  # [P,86]
            "M_ext": M_ext,         # [4,4]
            "W": self.W,
            "H": self.H,
        }



def build_training_dataset(cfg, mask_path: Path) -> Dataset:
    dataset = FullSceneDataset(
        preprocess_dir=Path(cfg.preprocess_dir),
        tids=cfg.tids,
        mask_path=mask_path,
        cloud_downsample=cfg.cloud_downsample,
        train_bg=cfg.train_bg,
    )
    return dataset


def build_evaluation_dataset(cfg, pred_method) -> Dataset:
    ds_name = cfg.eval_ds_name
    if ds_name.lower() == "hi4d":
        dataset = Hi4DDataset(
            mask_root=cfg.gt_seg_masks_dir,
            smpl_root=cfg.gt_smpl_dir,
            preprocess_dir=cfg.preprocess_dir,
            pred_method=pred_method,
        )
    else:
        raise ValueError(f"Unsupported evaluation dataset name: {ds_name}")
    return dataset

def build_dataloader(cfg, dataset: Dataset, is_eval: bool = False) -> DataLoader:
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=not is_eval,
        num_workers=cfg.num_workers
    )
    print(f"--- FYI: DataLoader created with num_workers={cfg.num_workers}.")
    return dataloader
