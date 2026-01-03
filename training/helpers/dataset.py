import subprocess
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

def root_dir_to_image_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "images" / f"{cam_id}"

def root_dir_to_mask_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "seg" / "img_seg_mask" / f"{cam_id}" / "all"

def root_dir_to_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "smplx"

def root_dir_to_smpl_dir(root_dir: Path) -> Path:
    return root_dir / "smpl"

def root_dir_to_cameras_path(root_dir: Path) -> Path:
    return root_dir / "cameras" / "rgb_cameras.npz"

def root_dir_to_depth_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "depths" / f"{cam_id}"

def root_dir_to_skip_frames_path(root_dir: Path) -> Path:
    return root_dir / "skip_frames.csv"

class SceneDataset(Dataset):
    
    def __init__(self, 
                scene_root_dir: Path,
                src_cam_id: int,
                use_depth: Optional[bool] = False,
                use_smpl: Optional[bool] = False,
                device: Optional[torch.device] = "cuda", 
                sample_every: Optional[int] = 1,
                skip_frames: Optional[list] = []):
        
        # Initialize attributes
        self.root_dir = scene_root_dir
        self.src_cam_id = src_cam_id
        self.device = device
        self.frames_dir = root_dir_to_image_dir(scene_root_dir, src_cam_id)
        self.masks_dir = root_dir_to_mask_dir(scene_root_dir, src_cam_id)

        # Load frame paths (with optional subsampling)
        # Important: we use frame path names to match masks, SMPLX, depth, etc.
        # -> therefore also if we apply subsampling here, other modalities will be subsampled accordingly
        self._load_frame_paths(self.frames_dir, sample_every, skip_frames)

        # Load mask paths
        self._load_mask_paths(self.masks_dir)

        # Determine training resolution from first image
        first_image = self._load_img(self.frame_paths[0])
        self.trn_render_hw = (first_image.shape[0], first_image.shape[1])  # (H, W)

        # Load depth paths (if provided)
        if use_depth:
            self.depth_dir = root_dir_to_depth_dir(scene_root_dir, src_cam_id)
            self._load_depth_paths()

        # Load camera parameters
        self.camera_params_path = root_dir_to_cameras_path(scene_root_dir)
        self._load_cameras()

        # Load SMPLX parameters
        self.smplx_dir: Path = root_dir_to_smplx_dir(scene_root_dir)
        self._load_smplx_paths()

        # (Optional) Load SMPL parameters
        if use_smpl:
            self.smpl_dir: Path = root_dir_to_smpl_dir(scene_root_dir)
            self._load_smpl_paths()

    # --------- Path loaders
    def _load_frame_paths(self, frames_dir: Path, sample_every: int = 1, skip_frames: Optional[list] = []):

        # Collect all frame paths
        frame_candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            frame_candidates.extend(frames_dir.glob(ext))
        self.frame_paths = sorted(set(frame_candidates))

        # Apply skip frames
        if len(skip_frames) > 0:
            before_filter_count = len(self.frame_paths)
            filtered_frame_paths = []
            for p in self.frame_paths:
                frame_idx = int(p.stem)
                if frame_idx not in skip_frames:
                    filtered_frame_paths.append(p)
            self.frame_paths = filtered_frame_paths
            after_filter_count = len(self.frame_paths)
            print(f"-- Skipped {before_filter_count - after_filter_count} frames based on skip_frames list.")

        # Apply subsampling
        if sample_every > 1:
            self.frame_paths = self.frame_paths[::sample_every]

        # Check that we have frames
        if not self.frame_paths:
            raise RuntimeError(f"No frames found in {frames_dir}")

    def _load_mask_paths(self, masks_dir: Path):
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

    def _load_smplx_paths(self):
        self.smplx_paths = []
        missing = []
        for p in self.frame_paths:
            smplx_path = self.smplx_dir / f"{p.stem}.npz"
            if not smplx_path.exists():
                missing.append(p.stem)
            else:
                self.smplx_paths.append(smplx_path)
        if missing:
            raise RuntimeError(f"Missing SMPLX files for frames (by stem): {missing[:5]}")

    def _load_depth_paths(self):
        self.depth_paths = []
        missing = []
        for p in self.frame_paths:
            depth_path = self.depth_dir / f"{p.stem}.npy"
            if not depth_path.exists():
                missing.append(p.stem)
            else:
                self.depth_paths.append(depth_path)
        if missing:
            raise RuntimeError(f"Missing depth files for frames (by stem): {missing[:5]}")
        
    def _load_smpl_paths(self):
        self.smpl_paths = []
        missing = []
        for p in self.frame_paths:
            smpl_path = self.smpl_dir / f"{p.stem}.npz"
            if not smpl_path.exists():
                missing.append(p.stem)
            else:
                self.smpl_paths.append(smpl_path)
        if missing:
            raise RuntimeError(f"Missing SMPL files for frames (by stem): {missing[:5]}")


    # -------- Data loaders for camera parameters and SMPLX
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

    def _load_smplx(self, path: Path):

        npz = np.load(path)

        def add_key(key):
            arrs = torch.from_numpy(npz[key]).float()
            return arrs.to(self.device)  # [P, ...]

        smplx = {
            "betas": add_key("betas"),
            "root_pose": add_key("root_pose"),   # [P,3] world axis-angle
            "body_pose": add_key("body_pose"),
            "jaw_pose": add_key("jaw_pose"),
            "leye_pose": add_key("leye_pose"),
            "reye_pose": add_key("reye_pose"),
            "lhand_pose": add_key("lhand_pose"),
            "rhand_pose": add_key("rhand_pose"),
            "trans": add_key("trans"),           # [P,3] world translation
            "expr": add_key("expression"),
        }

        smplx["expr"] = torch.zeros(smplx["expr"].shape[0], smplx["expr"].shape[1], 100, device=self.device)

        return smplx

    def _load_smpl(self, path: Path):

        npz = np.load(path)

        def add_key(key):
            arrs = torch.from_numpy(npz[key]).float()
            return arrs.to(self.device)  # [P, ...]

        body_pose = add_key("body_pose")
        # SMPL often stores body pose as a flattened axis-angle vector [P, 69] (= 23 joints * 3).
        # Reshape to match the structured representation used elsewhere: [P, 23, 3].
        if body_pose.ndim == 2 and body_pose.shape[-1] == 69:
            body_pose = body_pose.reshape(body_pose.shape[0], 23, 3)

        smpl = {
            "betas": add_key("betas"),
            "body_pose": body_pose,
            "root_pose": add_key("global_orient"),   # [P,3] world axis-angle
            "trans": add_key("transl"),           # [P,3] world translation
            "contact": add_key("contact"),
        }

        return smpl

    # -------- Dataset interface
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx: int):
        frame = self._load_img(self.frame_paths[idx])
        frame_name = Path(self.frame_paths[idx]).stem
        mask = self._load_mask(self.mask_paths[idx])
        smplx_params = self._load_smplx(self.smplx_paths[idx])
        K = self.K
        c2w = self.c2w
        cam_id = self.src_cam_id

        # Prepare return values
        # - Mandatory
        to_return_values = {
            "frame_idx": torch.tensor(idx, device=self.device, dtype=torch.long),
            "frame_path": str(self.frame_paths[idx]),
            "frame_name": frame_name,
            "image": frame,
            "mask": mask,
            "K": K,
            "c2w": c2w,
            "smplx_params": smplx_params,
            "cam_id": cam_id,
        }

        # - (Optional) load depth map
        if hasattr(self, "depth_paths"):
            depth = self._load_depth(self.depth_paths[idx])
            to_return_values["depth"] = depth

        # - (Optional) load SMPL parameters
        if hasattr(self, "smpl_paths"):
            smpl_params = self._load_smpl(self.smpl_paths[idx])
            to_return_values["smpl_params"] = smpl_params

        return to_return_values


def fetch_data_if_available(tgt_scene_dir: Path, camera_id: int, frames_scene_dir: Path, masks_scene_dir: Path, cam_scene_dir: Optional[Path] = None,  
                                smplx_params_scene_dir: Optional[Path] = None, depths_scene_dir: Optional[Path] = None, smpl_params_scene_dir: Optional[Path] = None,
                                resolution_hw: Optional[Tuple[int, int]] = (1280, 940), frame_paths: Optional[list] = None):
    """
    Copy data from the specified scene directories to the tgt scene directory. If the given
    source directory is None, skip copying that data type. If the source directory does not exist,
    fill it with empty (dummy) data.
    """
    
    tgt_scene_dir.mkdir(parents=True, exist_ok=True)

    # Frames
    src_frames_dir = root_dir_to_image_dir(frames_scene_dir, camera_id)
    tgt_frames_dir = root_dir_to_image_dir(tgt_scene_dir, camera_id)
    tgt_frames_dir.parent.mkdir(parents=True, exist_ok=True)
    if src_frames_dir.exists():
        subprocess.run(["cp", "-r", str(src_frames_dir), str(tgt_frames_dir.parent)])
    else:
        assert frame_paths is not None, "Source frames directory does not exist; frame_paths must be provided to create dummy frames."
        frame_ext = ".jpg"
        frame_names = [Path(fp).stem for fp in frame_paths]
        for frame_name in frame_names:
            dummy_frame = Image.new("RGB", (resolution_hw[1], resolution_hw[0]), color=(0, 0, 0))
            dummy_frame_path = tgt_frames_dir / f"{frame_name}{frame_ext}"
            dummy_frame_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_frame.save(dummy_frame_path)

    # Masks
    src_masks_dir = root_dir_to_mask_dir(masks_scene_dir, camera_id)
    tgt_masks_dir = root_dir_to_mask_dir(tgt_scene_dir, camera_id)
    tgt_masks_dir.parent.mkdir(parents=True, exist_ok=True)
    if src_masks_dir.exists():
        subprocess.run(["cp", "-r", str(src_masks_dir), str(tgt_masks_dir.parent)])
    else:
        assert frame_paths is not None, "Source masks directory does not exist; frame_paths must be provided to create dummy masks."
        mask_ext = ".png"
        frame_names = [Path(fp).stem for fp in frame_paths]
        for frame_name in frame_names:
            dummy_mask = Image.new("L", (resolution_hw[1], resolution_hw[0]), color=0)  # Black mask
            dummy_mask_path = tgt_masks_dir / f"{frame_name}{mask_ext}"
            dummy_mask_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_mask.save(dummy_mask_path)

    # Camera parameters
    if cam_scene_dir is not None:
        src_cameras_path = root_dir_to_cameras_path(cam_scene_dir)
        tgt_cameras_path = root_dir_to_cameras_path(tgt_scene_dir)
        tgt_cameras_path.parent.mkdir(parents=True, exist_ok=True)
        if src_cameras_path.exists():
            subprocess.run(["cp", str(src_cameras_path), str(tgt_cameras_path)])

    # SMPLX parameters
    if smplx_params_scene_dir is not None:
        src_smplx_dir = root_dir_to_smplx_dir(smplx_params_scene_dir)
        tgt_smplx_dir = root_dir_to_smplx_dir(tgt_scene_dir)
        tgt_smplx_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_smplx_dir.exists():
            subprocess.run(["cp", "-r", str(src_smplx_dir), str(tgt_smplx_dir.parent)])
        else:
            raise ValueError(f"SMPLX parameters directory not found: {src_smplx_dir}")

    # Depths
    if depths_scene_dir is not None:
        src_depths_dir = root_dir_to_depth_dir(depths_scene_dir, camera_id)
        tgt_depths_dir = root_dir_to_depth_dir(tgt_scene_dir, camera_id)
        tgt_depths_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_depths_dir.exists():
            subprocess.run(["cp", "-r", str(src_depths_dir), str(tgt_depths_dir)])
        else:
            raise NotImplementedError(f"Depth directory not found: {src_depths_dir}")

    # SMPL parameters
    if smpl_params_scene_dir is not None:
        src_smpl_params_dir = root_dir_to_smpl_dir(smpl_params_scene_dir)
        tgt_smpl_params_dir = root_dir_to_smpl_dir(tgt_scene_dir)
        tgt_smpl_params_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_smpl_params_dir.exists():
            subprocess.run(["cp", "-r", str(src_smpl_params_dir), str(tgt_smpl_params_dir.parent)])
        else:
            raise ValueError(f"SMPL parameters directory not found: {src_smpl_params_dir}")

    # Skip frames
    src_skip_frames_path = root_dir_to_skip_frames_path(frames_scene_dir)
    tgt_skip_frames_path = root_dir_to_skip_frames_path(tgt_scene_dir)
    if src_skip_frames_path.exists():
        subprocess.run(["cp", str(src_skip_frames_path), str(tgt_skip_frames_path)])
