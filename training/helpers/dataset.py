import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
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

def root_dir_to_mesh_dir(root_dir: Path) -> Path:
    return root_dir / "seg" / "instance"

def root_dir_to_skip_frames_path(root_dir: Path) -> Path:
    return root_dir / "skip_frames.csv"

class SceneDataset(Dataset):
    
    def __init__(self, 
                scene_root_dir: Path,
                src_cam_id: int,
                use_depth: Optional[bool] = False,
                use_smpl: Optional[bool] = False,
                use_meshes: Optional[bool] = False,
                device: Optional[torch.device] = "cuda", 
                sample_every: Optional[int] = 1,
                skip_frames: Optional[list] = []):
        
        # Initialize attributes
        self.root_dir = scene_root_dir
        self.src_cam_id = src_cam_id
        self.device = device

        # Load frame paths (with optional subsampling)
        # Important: we use frame path names to match masks, SMPLX, depth, etc.
        # -> therefore also if we apply subsampling here, other modalities will be subsampled accordingly
        self.frames_dir = root_dir_to_image_dir(scene_root_dir, src_cam_id)
        self._load_frame_paths(sample_every, skip_frames)

        # Load mask paths
        self.masks_dir = root_dir_to_mask_dir(scene_root_dir, src_cam_id)
        self._load_mask_paths()

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

        # (Optional) Check for meshes directory
        if use_meshes:
            self.meshes_dir: Path = root_dir_to_mesh_dir(scene_root_dir)
            person_ids = [d.name for d in self.meshes_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            person_ids = sorted([int(pid) for pid in person_ids])
            self._load_mesh_paths(person_ids)

    # --------- Path loaders
    def _load_frame_paths(self, sample_every: int = 1, skip_frames: Optional[list] = []):

        # Collect all frame paths
        frame_candidates = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            frame_candidates.extend(self.frames_dir.glob(ext))
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
            raise RuntimeError(f"No frames found in {self.frames_dir}")

    def _load_mask_paths(self):
        self.mask_paths = []
        missing = []
        for p in self.frame_paths:
            base = p.stem
            candidates = [self.masks_dir / f"{base}{ext}" for ext in (".png", ".jpg", ".jpeg")]
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

    def _load_mesh_paths(self, person_ids: List[int]):
        """
        Build per-frame mesh paths for each person and record max sizes for padding.

        Design notes:
        - We store a list of mesh paths per frame and per person so __getitem__
          can load meshes on demand.
        - We precompute global max vertex/face counts across all persons/frames
          so padding is consistent for stacking and DataLoader collation.

        Args:
            person_ids: List of integer person IDs, e.g. [0, 1].
        """
        self.mesh_paths = []
        self.mesh_person_ids = person_ids
        self.mesh_max_verts = {pid: 0 for pid in person_ids}
        self.mesh_max_faces = {pid: 0 for pid in person_ids}
        self.mesh_max_verts_all = 0
        self.mesh_max_faces_all = 0
        missing = []
        for p in self.frame_paths:
            missing_per_frame = []
            paths_per_frame = []
            for person_id in person_ids:
                # example name: mesh-f00001.npz
                fnumber = int(p.stem)
                fname = f"mesh-f{fnumber:05d}.npz"
                person_mesh_path = self.meshes_dir / f"{person_id}" / fname
                if not person_mesh_path.exists():
                    missing_per_frame.append(fname)
                else:
                    paths_per_frame.append(person_mesh_path)
                    with np.load(person_mesh_path) as npz:
                        verts = npz["vertices"]
                        faces = npz["faces"]
                    self.mesh_max_verts[person_id] = max(self.mesh_max_verts[person_id], verts.shape[0])
                    self.mesh_max_faces[person_id] = max(self.mesh_max_faces[person_id], faces.shape[0])
                    self.mesh_max_verts_all = max(self.mesh_max_verts_all, verts.shape[0])
                    self.mesh_max_faces_all = max(self.mesh_max_faces_all, faces.shape[0])
            
            if missing_per_frame:
                missing.append(missing_per_frame)

            self.mesh_paths.append(paths_per_frame)
        if missing:
            raise RuntimeError(f"Missing mesh files for frames (by stem): {missing[:5]}")
        if self.mesh_max_verts_all == 0 or self.mesh_max_faces_all == 0:
            raise RuntimeError("Failed to determine mesh padding sizes; mesh files may be empty.")


    # -------- Data loaders for camera parameters and SMPLX
    def _load_img(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        return arr.to(self.device) # HxWx3, range [0,1]

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
            if not key in npz:
                arrs = torch.zeros((npz["betas"].shape[0], 10), device=self.device)
            else:
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
            "contact": add_key("contact"), # [P, N_contact] - may be dummy depending on source
        }

        return smpl
    
    def _load_meshes(self, per_person_path: List[Path]):
        """
        Load per-person meshes for a single frame and pad to fixed sizes.

        Design notes:
        - Meshes are returned as dense, padded tensors so the default PyTorch
          collate_fn can stack them across the batch.
        - True sizes are returned separately so metrics can ignore padding.

        Args:
            per_person_path: List of mesh paths (one per person) for this frame.

        Returns:
            Dict with:
              - "vertices": FloatTensor [P, Vmax, 3]
              - "faces": LongTensor [P, Fmax, 3]
              - "num_vertices": LongTensor [P]
              - "num_faces": LongTensor [P]
              - "person_ids": LongTensor [P]
        """
        num_persons = len(per_person_path)
        if num_persons != len(self.mesh_person_ids):
            raise RuntimeError(
                f"Expected {len(self.mesh_person_ids)} mesh paths, got {num_persons}."
            )

        verts_out = []
        faces_out = []
        num_verts = []
        num_faces = []

        for pid, mesh_path in zip(self.mesh_person_ids, per_person_path):
            npz = np.load(mesh_path)
            verts = npz["vertices"].astype(np.float32)
            faces = npz["faces"].astype(np.int64)

            max_v = self.mesh_max_verts_all
            max_f = self.mesh_max_faces_all

            verts_pad = np.zeros((max_v, 3), dtype=np.float32)
            faces_pad = np.zeros((max_f, 3), dtype=np.int64)
            verts_pad[: verts.shape[0]] = verts
            faces_pad[: faces.shape[0]] = faces

            verts_out.append(torch.from_numpy(verts_pad))
            faces_out.append(torch.from_numpy(faces_pad))
            num_verts.append(verts.shape[0])
            num_faces.append(faces.shape[0])

        meshes = {
            "vertices": torch.stack(verts_out, dim=0).to(self.device),
            "faces": torch.stack(faces_out, dim=0).to(self.device),
            "num_vertices": torch.tensor(num_verts, device=self.device, dtype=torch.long),
            "num_faces": torch.tensor(num_faces, device=self.device, dtype=torch.long),
            "person_ids": torch.tensor(self.mesh_person_ids, device=self.device, dtype=torch.long),
        }
        return meshes

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

        # - (Optional) load meshes
        if hasattr(self, "mesh_paths"):
            mesh_paths_per_person = self.mesh_paths[idx]
            meshes = self._load_meshes(mesh_paths_per_person)
            to_return_values["meshes"] = meshes

        return to_return_values


def fetch_masks_if_exist(masks_scene_dir: Path, tgt_scene_dir: Path, camera_id: int):
    src_masks_dir = root_dir_to_mask_dir(masks_scene_dir, camera_id)
    tgt_masks_dir = root_dir_to_mask_dir(tgt_scene_dir, camera_id)
    tgt_masks_dir.parent.mkdir(parents=True, exist_ok=True)
    if src_masks_dir.exists():
        subprocess.run(["cp", "-r", str(src_masks_dir), str(tgt_masks_dir.parent)])
        return True
    else:
        return False


def fetch_data_if_available(tgt_scene_dir: Path, camera_id: int, frames_scene_dir: Path, masks_scene_dir: Path, cam_scene_dir: Optional[Path] = None,  
                                smplx_params_scene_dir: Optional[Path] = None, depths_scene_dir: Optional[Path] = None, smpl_params_scene_dir: Optional[Path] = None, meshes_scene_dir: Optional[Path] = None,
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
    masks_were_fetched = fetch_masks_if_exist(masks_scene_dir, tgt_scene_dir, camera_id)
    if not masks_were_fetched:
        tgt_masks_dir = root_dir_to_mask_dir(tgt_scene_dir, camera_id)
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

    # Meshes
    if meshes_scene_dir is not None:
        src_meshes_dir = root_dir_to_mesh_dir(meshes_scene_dir)
        tgt_meshes_dir = root_dir_to_mesh_dir(tgt_scene_dir)
        tgt_meshes_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_meshes_dir.exists():
            subprocess.run(["cp", "-r", str(src_meshes_dir), str(tgt_meshes_dir.parent)])
        else:
            raise ValueError(f"Meshes directory not found: {src_meshes_dir}")

    # Skip frames
    src_skip_frames_path = root_dir_to_skip_frames_path(frames_scene_dir)
    tgt_skip_frames_path = root_dir_to_skip_frames_path(tgt_scene_dir)
    if src_skip_frames_path.exists():
        subprocess.run(["cp", str(src_skip_frames_path), str(tgt_skip_frames_path)])
