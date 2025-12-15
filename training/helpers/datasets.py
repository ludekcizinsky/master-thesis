
from pathlib import Path
from typing import Optional, Tuple
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def extr_to_w2c_4x4(extr: torch.Tensor, device) -> torch.Tensor:
    w2c = torch.eye(4, device=device, dtype=torch.float32)
    w2c[:3, :4] = extr.to(device)
    return w2c

def intr_to_4x4(intr: torch.Tensor, device) -> torch.Tensor:
    intr4 = torch.eye(4, device=device, dtype=torch.float32)
    intr4[:3, :3] = intr.to(device)
    return intr4


class BaseDataset:
    def __init__(self, frames_dir: Path, masks_dir: Path, device: torch.device, sample_every: int = 1, depth_dir: Optional[Path] = None):
        self.device = device
        self._load_frames(frames_dir, sample_every)
        self._load_masks(masks_dir)

        first_image = self._load_img(self.frame_paths[0])
        self.trn_render_hw = (first_image.shape[0], first_image.shape[1])  # (H, W)

        if depth_dir is not None:
            self._load_depths(depth_dir)

    # --------- Path loaders
    def _load_frames(self, frames_dir: Path, sample_every: int = 1):
        frame_candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            frame_candidates.extend(frames_dir.glob(ext))
        self.frame_paths = sorted(set(frame_candidates))
        if sample_every > 1:
            self.frame_paths = self.frame_paths[::sample_every]
        if not self.frame_paths:
            raise RuntimeError(f"No frames found in {frames_dir}")

    def _load_masks(self, masks_dir: Path):
        self.mask_paths = []
        missing = []
        for p in self.frame_paths:
            base = p.stem
            candidates = [masks_dir / f"{base}{ext}" for ext in (".png", ".jpg", ".jpeg")]
            mask_path = next((c for c in candidates if c.exists()), None)
            if mask_path is None:
                missing.append(base)
            else:
                self.mask_paths.append(mask_path)
        if missing:
            raise RuntimeError(f"Missing masks for frames (by stem): {missing[:5]}")

    def _load_depths(self, depth_dir: Path):
        depth_files = sorted(p for p in depth_dir.glob("*.npy") if p.is_file())
        self.depth_paths = [depth_dir / p.name for p in depth_files]
        assert len(self.depth_paths) == len(self.frame_paths), "Number of depth files must match number of frames."

    # --------- Data loaders
    def _load_img(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        return arr.to(self.device)

    def _load_mask(self, path: Path, eps: float = 0.05) -> torch.Tensor:
        arr = torch.from_numpy(np.array(Image.open(path))).float()  # HxWxC or HxW
        if arr.dim() == 2:
            arr = arr.unsqueeze(-1) / 255.0  # HxWx1
            return arr.to(self.device) # already binary mask

        if arr.shape[-1] == 4:
            arr = arr[..., :3] # drop alpha
        # Foreground is any pixel whose max channel exceeds eps*255
        mask = (arr.max(dim=-1).values > eps * 255).float()  # HxW
        return mask.to(self.device).unsqueeze(-1)  # HxWx1, range [0,1]
    
    def _load_depth(self, path: Path) -> torch.Tensor:
        """
        Load and upsample depth map to training resolution.

        Args:
            path (Path): Path to the depth map file.
        Returns:
            torch.Tensor: Upsampled depth map tensor of shape HxW. Unit is the same as input depth map, so meters.
        """

        depth_np = torch.from_numpy(np.load(path)) # H_depthxW_depth

        height, width = self.trn_render_hw
        batched = depth_np.unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(batched, size=(height, width), mode="bilinear", align_corners=False)
        return upsampled.squeeze(0).squeeze(0).to(self.device).unsqueeze(-1)  # HxWx1


class Hi4dDataset(BaseDataset, Dataset):
    
    def __init__(self, 
                hi4d_scene_root_dir: Path,
                src_cam_id: int,
                depth_dir: Optional[Path],
                device: torch.device, 
                sample_every: int = 1):
        
        # Initialize base dataset
        self.root_dir = hi4d_scene_root_dir
        self.src_cam_id = src_cam_id
        self.device = device
        frames_dir = hi4d_scene_root_dir / "images" / f"{src_cam_id}"
        masks_dir = hi4d_scene_root_dir / "seg" / "img_seg_mask" / f"{src_cam_id}" / "all"
        super().__init__(frames_dir, masks_dir, device, sample_every, depth_dir)

        # Load camera parameters
        self.camera_params_path = self.root_dir/ "cameras" / "rgb_cameras.npz"
        self._load_cameras()

        # Load SMPLX parameters
        self.smplx_dir: Path = self.root_dir / "smplx"
        self._load_smplx()

    # -------- Data loaders for camera parameters and SMPLX
    def _load_camera_from_npz(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load intrinsics and extrinsics for a specific camera ID from a .npz file.

        Expects keys "ids", "intrinsics" [N,3,3], and "extrinsics" [N,3,4] in the file.
        Returns float tensors (intrinsics, extrinsics) optionally moved to `device`.
        """
        
        camera_npz_path = Path(self.camera_params_path)
        with np.load(camera_npz_path) as cams:
            missing = [k for k in ("ids", "intrinsics", "extrinsics") if k not in cams.files]
            if missing:
                raise KeyError(f"Missing keys {missing} in camera file {camera_npz_path}")

            ids = cams["ids"]
            matches = np.nonzero(ids == self.src_cam_id)[0]
            if len(matches) == 0:
                raise ValueError(f"Camera id {self.src_cam_id} not found in {camera_npz_path}")
            idx = int(matches[0])

            intrinsics = torch.from_numpy(cams["intrinsics"][idx]).float()
            extrinsics = torch.from_numpy(cams["extrinsics"][idx]).float()

            device = torch.device(self.device)
            intrinsics = intrinsics.to(device)
            extrinsics = extrinsics.to(device)

        return intrinsics, extrinsics


    def _load_cameras(self):
        intr, extr = self._load_camera_from_npz()
        w2c = extr_to_w2c_4x4(extr, self.device)
        self.c2w = torch.inverse(w2c) # shape is [4, 4]
        self.K = intr_to_4x4(intr, self.device) # shape is [4,4]

    def _load_smplx(self):
        frame_paths = sorted([p for p in os.listdir(self.smplx_dir) if p.endswith(".npz")])
        npzs = []
        for fp in frame_paths:
            npz = np.load(self.smplx_dir / f"{Path(fp).stem}.npz")
            npzs.append(npz)

        def stack_key(key):
            arrs = [torch.from_numpy(n[key]).float() for n in npzs]
            return torch.stack(arrs, dim=1).to(self.device)  # [P, F, ...]

        self.smplx = {
            "betas": stack_key("betas"),
            "root_pose": stack_key("root_pose"),   # [P,F,3] world axis-angle
            "body_pose": stack_key("body_pose"),
            "jaw_pose": stack_key("jaw_pose"),
            "leye_pose": stack_key("leye_pose"),
            "reye_pose": stack_key("reye_pose"),
            "lhand_pose": stack_key("lhand_pose"),
            "rhand_pose": stack_key("rhand_pose"),
            "trans": stack_key("trans"),           # [P,F,3] world translation
            "expr": stack_key("expression"),
        }

        self.smplx["expr"] = torch.zeros(self.smplx["expr"].shape[0], self.smplx["expr"].shape[1], 100, device=self.device)


    # -------- Dataset interface
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx: int):
        frame = self._load_img(self.frame_paths[idx])
        mask = self._load_mask(self.mask_paths[idx])
        K = self.K
        c2w = self.c2w
        smplx_params = {k: v[:, idx] for k, v in self.smplx.items()}

        # Prepare return values
        # - Mandatory
        to_return_values = {
            "frame_idx": torch.tensor(idx, device=self.device, dtype=torch.long),
            "frame_path": str(self.frame_paths[idx]),
            "image": frame,
            "mask": mask,
            "K": K,
            "c2w": c2w,
            "smplx_params": smplx_params,
        }

        # - (Optional) load depth map
        if hasattr(self, "depth_paths"):
            depth = self._load_depth(self.depth_paths[idx])
            to_return_values["depth"] = depth

        return to_return_values