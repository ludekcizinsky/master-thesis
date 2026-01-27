import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data import Dataset
from tqdm import tqdm

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

def root_dir_to_mask_root_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "seg" / "img_seg_mask" / f"{cam_id}"

def root_dir_to_smplx_dir(root_dir: Path) -> Path:
    return root_dir / "smplx"

def root_dir_to_smpl_dir(root_dir: Path) -> Path:
    return root_dir / "smpl"

def root_dir_to_cameras_path(root_dir: Path) -> Path:
    return root_dir / "cameras" / "rgb_cameras.npz"

def root_dir_to_all_cameras_dir(root_dir: Path) -> Path:
    return root_dir / "all_cameras"

def root_dir_to_depth_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "depths" / f"{cam_id}"

def root_dir_to_mesh_dir(root_dir: Path) -> Path:
    return root_dir / "meshes"

def root_dir_to_skip_frames_path(root_dir: Path) -> Path:
    return root_dir / "skip_frames.csv"

def build_individual_mask_validity(num_valid: int, max_people: int, device: torch.device) -> torch.Tensor:
    valid = torch.zeros(max_people, device=device, dtype=torch.bool)
    if num_valid > 0:
        valid[: min(num_valid, max_people)] = True
    return valid

def filter_individual_masks_by_validity(
    individual_masks: torch.Tensor, valid: torch.Tensor
) -> List[torch.Tensor]:
    """
    Filter padded individual masks based on a validity boolean mask.

    Args:
        individual_masks: [B, P, H, W, 1] or [P, H, W, 1]
        valid: [B, P] or [P]
    Returns:
        List of tensors, one per batch item, each of shape [P_valid, H, W, 1].
    """
    if individual_masks.dim() == 4:
        return [individual_masks[valid.bool()]]
    if individual_masks.dim() != 5:
        raise ValueError(f"Unexpected individual_masks shape: {individual_masks.shape}")
    if valid.dim() != 2:
        raise ValueError(f"Unexpected valid shape: {valid.shape}")
    filtered = []
    for i in range(individual_masks.shape[0]):
        filtered.append(individual_masks[i][valid[i].bool()])
    return filtered

class SceneDataset(Dataset):
    
    def __init__(self, 
                scene_root_dir: Path,
                src_cam_id: int,
                use_masks: Optional[bool] = True,
                use_cameras: Optional[bool] = True,
                use_smplx: Optional[bool] = True,
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
        self.use_masks = bool(use_masks)
        self.use_cameras = bool(use_cameras)
        self.use_smplx = bool(use_smplx)
        self.max_people = 5

        # Load frame paths (with optional subsampling)
        # Important: we use frame path names to match masks, SMPLX, depth, etc.
        # -> therefore also if we apply subsampling here, other modalities will be subsampled accordingly
        self.frames_dir = root_dir_to_image_dir(scene_root_dir, src_cam_id)
        self._load_frame_paths(sample_every, skip_frames)

        # Determine training resolution from first image
        first_image = self._load_img(self.frame_paths[0])
        self.trn_render_hw = (first_image.shape[0], first_image.shape[1])  # (H, W)

        # Load mask paths
        if self.use_masks:
            self.masks_dir = root_dir_to_mask_dir(scene_root_dir, src_cam_id)
            self._load_mask_paths()
            self.individual_mask_dirs = self._find_individual_mask_dirs()
            self.num_individual_masks = len(self.individual_mask_dirs)
            if self.num_individual_masks > 0:
                if self.num_individual_masks > self.max_people:
                    raise ValueError(
                        f"Found {self.num_individual_masks} individual mask dirs, "
                        f"but max_people is set to {self.max_people}."
                    )
                self._load_individual_mask_paths()

        # Load depth paths (if provided)
        if use_depth:
            self.depth_dir = root_dir_to_depth_dir(scene_root_dir, src_cam_id)
            self._load_depth_paths()

        # Load camera parameters
        if self.use_cameras:
            self.cameras_dir = root_dir_to_all_cameras_dir(scene_root_dir) / f"{src_cam_id}"
            self._load_camera_paths()

        # Load SMPLX parameters
        if self.use_smplx:
            self.smplx_dir: Path = root_dir_to_smplx_dir(scene_root_dir)
            self._load_smplx_paths()

        # (Optional) Load SMPL parameters
        if use_smpl:
            self.smpl_dir: Path = root_dir_to_smpl_dir(scene_root_dir)
            self._load_smpl_paths()

        # (Optional) Check for meshes directory
        if use_meshes:
            self.meshes_dir: Path = root_dir_to_mesh_dir(scene_root_dir)
            self._load_mesh_paths()

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

    def _find_individual_mask_dirs(self) -> List[Path]:
        masks_root_dir = root_dir_to_mask_root_dir(self.root_dir, self.src_cam_id)
        if not masks_root_dir.exists():
            return []
        candidate_dirs = [
            d for d in masks_root_dir.iterdir()
            if d.is_dir() and d.name != "all"
        ]

        def _sort_key(path: Path):
            name = path.name
            if name.isdigit():
                return (0, int(name))
            return (1, name)

        return sorted(candidate_dirs, key=_sort_key)

    def _load_individual_mask_paths(self):
        self.individual_mask_paths = []
        for person_dir in self.individual_mask_dirs:
            person_paths = []
            missing = []
            for p in self.frame_paths:
                base = p.stem
                candidates = [person_dir / f"{base}{ext}" for ext in (".png", ".jpg", ".jpeg")]
                mask_path = next((c for c in candidates if c.exists()), None)
                if mask_path is None:
                    missing.append(base)
                else:
                    person_paths.append(mask_path)
            if missing:
                raise RuntimeError(
                    f"Missing individual masks in '{person_dir.name}' for frames (by stem): {missing[:5]}"
                )
            self.individual_mask_paths.append(person_paths)

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

    def _load_camera_paths(self):
        self.camera_paths = []
        missing = []
        for p in self.frame_paths:
            cam_path = self.cameras_dir / f"{p.stem}.npz"
            if not cam_path.exists():
                missing.append(p.stem)
            else:
                self.camera_paths.append(cam_path)
        if missing:
            raise RuntimeError(f"Missing camera files for frames (by stem): {missing[:5]}")
        
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

    def _load_mesh_paths(self):
        """
        Build per-frame mesh paths and record max sizes for padding.

        Design notes:
        - We store a list of mesh paths per frame so __getitem__
          can load meshes on demand.
        - We precompute global max vertex/face counts across all frames
          so padding is consistent for stacking and DataLoader collation.
        """
        self.mesh_paths = []
        self.mesh_max_verts = 100000
        self.mesh_max_faces = 200000
        missing = []
        for p in tqdm(self.frame_paths, desc="Loading mesh paths and computing max sizes"):
            mesh_path = self.meshes_dir / f"{p.stem}.obj"
            if not mesh_path.exists():
                missing.append(p.stem)
                continue
#            mesh = trimesh.load_mesh(mesh_path, process=False)
            #if isinstance(mesh, trimesh.Scene):
                #mesh = trimesh.util.concatenate(tuple(mesh.dump()))
            #if not isinstance(mesh, trimesh.Trimesh):
                #missing.append(p.stem)
                #continue
            #verts = np.asarray(mesh.vertices)
            #faces = np.asarray(mesh.faces)
            #self.mesh_max_verts = max(self.mesh_max_verts, verts.shape[0])
            #self.mesh_max_faces = max(self.mesh_max_faces, faces.shape[0])
            self.mesh_paths.append(mesh_path)
        if missing:
            raise RuntimeError(f"Missing mesh files for frames (by stem): {missing[:5]}")
        if self.mesh_max_verts == 0 or self.mesh_max_faces == 0:
            raise RuntimeError("Failed to determine mesh padding sizes; mesh files may be empty.")
        
        print(f"-- Max mesh vertices: {self.mesh_max_verts}, faces: {self.mesh_max_faces}")


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


    def _load_camera_from_frame(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load intrinsics and extrinsics from a per-frame camera file.

        Expects keys "intrinsics" [1,3,3] or [3,3] and "extrinsics" [1,3,4] or [3,4].
        Returns float tensors (intrinsics, extrinsics) moved to `device`.
        """

        with np.load(path) as cams:
            missing = [k for k in ("intrinsics", "extrinsics") if k not in cams.files]
            if missing:
                raise KeyError(f"Missing keys {missing} in camera file {path}")

            intrinsics = cams["intrinsics"]
            extrinsics = cams["extrinsics"]

        if intrinsics.ndim == 3:
            intrinsics = intrinsics[0]
        if extrinsics.ndim == 3:
            extrinsics = extrinsics[0]

        if intrinsics.shape != (3, 3) or extrinsics.shape != (3, 4):
            raise ValueError(
                f"Unexpected camera shapes in {path}: "
                f"intrinsics={intrinsics.shape}, extrinsics={extrinsics.shape}"
            )

        device = torch.device(self.device)
        intrinsics = torch.from_numpy(intrinsics).float().to(device)
        extrinsics = torch.from_numpy(extrinsics).float().to(device)

        return intrinsics, extrinsics

    @staticmethod
    def _load_smplx(path: Path, device: torch.device = torch.device("cuda")):

        npz = np.load(path)

        def add_key(key):
            arrs = torch.from_numpy(npz[key]).float()
            return arrs.to(device)  # [P, ...]

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

        smplx["expr"] = torch.zeros(smplx["expr"].shape[0], smplx["expr"].shape[1], 100, device=device)
        return smplx

    @staticmethod
    def _load_smpl(path: Path, device: torch.device = torch.device("cuda")):

        npz = np.load(path)

        def add_key(key):
            if not key in npz:
                arrs = torch.zeros((npz["betas"].shape[0], 10), device=device)
            else:
                arrs = torch.from_numpy(npz[key]).float()
            return arrs.to(device)  # [P, ...]

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
        }
        if "contact" in npz:
            smpl["contact"] = add_key("contact")  # [P, N_contact]

        return smpl
    
    def _load_mesh(self, mesh_path: Path):
        """
        Load a single mesh for a frame and pad to fixed sizes.

        Design notes:
        - Meshes are returned as dense, padded tensors so the default PyTorch
          collate_fn can stack them across the batch.
        - True sizes are returned separately so metrics can ignore padding.

        Returns:
            Dict with:
              - "vertices": FloatTensor [Vmax, 3]
              - "faces": LongTensor [Fmax, 3]
              - "num_vertices": LongTensor []
              - "num_faces": LongTensor []
        """
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.dump()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise RuntimeError(f"Failed to load mesh at {mesh_path}")
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        max_v = self.mesh_max_verts
        max_f = self.mesh_max_faces

        verts_pad = np.zeros((max_v, 3), dtype=np.float32)
        faces_pad = np.zeros((max_f, 3), dtype=np.int64)
        verts_pad[: verts.shape[0]] = verts
        faces_pad[: faces.shape[0]] = faces

        meshes = {
            "vertices": torch.from_numpy(verts_pad).to(self.device),
            "faces": torch.from_numpy(faces_pad).to(self.device),
            "num_vertices": torch.tensor(verts.shape[0], device=self.device, dtype=torch.long),
            "num_faces": torch.tensor(faces.shape[0], device=self.device, dtype=torch.long),
        }
        return meshes

    # -------- Dataset interface
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx: int):
        frame = self._load_img(self.frame_paths[idx])
        frame_name = Path(self.frame_paths[idx]).stem
        # Prepare return values
        # - Mandatory
        to_return_values = {
            "frame_idx": torch.tensor(idx, device=self.device, dtype=torch.long),
            "frame_path": str(self.frame_paths[idx]),
            "frame_name": frame_name,
            "image": frame,
        }
        if self.use_masks:
            to_return_values["mask"] = self._load_mask(self.mask_paths[idx])
            if self.max_people is not None:
                per_person_masks = []
                if hasattr(self, "individual_mask_paths") and self.num_individual_masks > 0:
                    per_person_masks = [
                        self._load_mask(person_paths[idx])
                        for person_paths in self.individual_mask_paths
                    ]
                if not per_person_masks:
                    zero_mask = torch.zeros_like(to_return_values["mask"])
                    per_person_masks = [zero_mask for _ in range(self.max_people)]
                elif len(per_person_masks) < self.max_people:
                    zero_mask = torch.zeros_like(per_person_masks[0])
                    per_person_masks.extend(
                        [zero_mask for _ in range(self.max_people - len(per_person_masks))]
                    )
                to_return_values["individual_mask"] = torch.stack(per_person_masks, dim=0)
                num_valid = self.num_individual_masks if hasattr(self, "num_individual_masks") else 0
                to_return_values["individual_mask_valid"] = build_individual_mask_validity(
                    num_valid, self.max_people, self.device
                )
        if self.use_cameras:
            intr, extr = self._load_camera_from_frame(self.camera_paths[idx])
            w2c = extr_to_w2c_4x4(extr, self.device)
            to_return_values["c2w"] = torch.inverse(w2c)
            to_return_values["K"] = intr_to_4x4(intr, self.device)
            to_return_values["cam_id"] = self.src_cam_id
        if self.use_smplx:
            to_return_values["smplx_params"] = self._load_smplx(self.smplx_paths[idx])

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
            mesh_path = self.mesh_paths[idx]
            meshes = self._load_mesh(mesh_path)
            to_return_values["meshes"] = meshes

        return to_return_values


def fetch_masks_if_exist(masks_scene_dir: Path, tgt_scene_dir: Path, camera_id: int):
    if masks_scene_dir is None:
        return False
    src_masks_root = root_dir_to_mask_root_dir(masks_scene_dir, camera_id)
    tgt_masks_root = root_dir_to_mask_root_dir(tgt_scene_dir, camera_id)
    tgt_masks_root.parent.mkdir(parents=True, exist_ok=True)
    if src_masks_root.exists():
        subprocess.run(["cp", "-r", str(src_masks_root), str(tgt_masks_root.parent)])
        return True
    else:
        return False

def fetch_cameras_if_exist(cam_scene_dir: Optional[Path], tgt_scene_dir: Path, camera_id: int, was_it_fetched: dict):
    if cam_scene_dir is not None:
        src_cameras_dir = root_dir_to_all_cameras_dir(cam_scene_dir) / f"{camera_id}"
        tgt_cameras_dir = root_dir_to_all_cameras_dir(tgt_scene_dir) / f"{camera_id}"
        tgt_cameras_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_cameras_dir.exists():
            subprocess.run(["cp", "-r", str(src_cameras_dir), str(tgt_cameras_dir.parent)])
        was_it_fetched["cameras"] = True
    else:
        was_it_fetched["cameras"] = False 


def fetch_data_if_available(tgt_scene_dir: Path, camera_id: int, frames_scene_dir: Path, masks_scene_dir: Path, cam_scene_dir: Optional[Path] = None,  
                                smplx_params_scene_dir: Optional[Path] = None, depths_scene_dir: Optional[Path] = None, smpl_params_scene_dir: Optional[Path] = None, meshes_scene_dir: Optional[Path] = None,
                                resolution_hw: Optional[Tuple[int, int]] = (1280, 940), frame_paths: Optional[list] = None):
    """
    Copy data from the specified scene directories to the tgt scene directory. If the given
    source directory is None, skip copying that data type. If the source directory does not exist,
    fill it with empty (dummy) data.
    """

    was_it_fetched = dict()
    
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
    was_it_fetched["frames"] = True

    # Masks
    masks_were_fetched = fetch_masks_if_exist(masks_scene_dir, tgt_scene_dir, camera_id)
    if not masks_were_fetched and frame_paths is not None:
        tgt_masks_dir = root_dir_to_mask_dir(tgt_scene_dir, camera_id)
        mask_ext = ".png"
        frame_names = [Path(fp).stem for fp in frame_paths]
        for frame_name in frame_names:
            dummy_mask = Image.new("L", (resolution_hw[1], resolution_hw[0]), color=0)  # Black mask
            dummy_mask_path = tgt_masks_dir / f"{frame_name}{mask_ext}"
            dummy_mask_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_mask.save(dummy_mask_path)
    was_it_fetched["masks"] = masks_were_fetched

    # Camera parameters
    fetch_cameras_if_exist(cam_scene_dir, tgt_scene_dir, camera_id, was_it_fetched)


    # SMPLX parameters
    if smplx_params_scene_dir is not None:
        src_smplx_dir = root_dir_to_smplx_dir(smplx_params_scene_dir)
        tgt_smplx_dir = root_dir_to_smplx_dir(tgt_scene_dir)
        tgt_smplx_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_smplx_dir.exists():
            subprocess.run(["cp", "-r", str(src_smplx_dir), str(tgt_smplx_dir.parent)])
        else:
            raise ValueError(f"SMPLX parameters directory not found: {src_smplx_dir}")
        
        was_it_fetched["smplx"] = True
    else:
        was_it_fetched["smplx"] = False

    # Depths
    if depths_scene_dir is not None:
        src_depths_dir = root_dir_to_depth_dir(depths_scene_dir, camera_id)
        tgt_depths_dir = root_dir_to_depth_dir(tgt_scene_dir, camera_id)
        tgt_depths_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_depths_dir.exists():
            subprocess.run(["cp", "-r", str(src_depths_dir), str(tgt_depths_dir)])
        else:
            raise NotImplementedError(f"Depth directory not found: {src_depths_dir}")
        was_it_fetched["depths"] = True
    else:
        was_it_fetched["depths"] = False

    # SMPL parameters
    if smpl_params_scene_dir is not None:
        src_smpl_params_dir = root_dir_to_smpl_dir(smpl_params_scene_dir)
        tgt_smpl_params_dir = root_dir_to_smpl_dir(tgt_scene_dir)
        tgt_smpl_params_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_smpl_params_dir.exists():
            subprocess.run(["cp", "-r", str(src_smpl_params_dir), str(tgt_smpl_params_dir.parent)])
        else:
            raise ValueError(f"SMPL parameters directory not found: {src_smpl_params_dir}")
        was_it_fetched["smpl"] = True
    else:
        was_it_fetched["smpl"] = False

    # Meshes
    if meshes_scene_dir is not None:
        src_meshes_dir = root_dir_to_mesh_dir(meshes_scene_dir)
        tgt_meshes_dir = root_dir_to_mesh_dir(tgt_scene_dir)
        tgt_meshes_dir.parent.mkdir(parents=True, exist_ok=True)
        if src_meshes_dir.exists():
            subprocess.run(["cp", "-r", str(src_meshes_dir), str(tgt_meshes_dir.parent)])
        else:
            raise ValueError(f"Meshes directory not found: {src_meshes_dir}")
        was_it_fetched["meshes"] = True
    else:
        was_it_fetched["meshes"] = False

    # Skip frames
    src_skip_frames_path = root_dir_to_skip_frames_path(frames_scene_dir)
    tgt_skip_frames_path = root_dir_to_skip_frames_path(tgt_scene_dir)
    if src_skip_frames_path.exists():
        subprocess.run(["cp", str(src_skip_frames_path), str(tgt_skip_frames_path)])
        was_it_fetched["skip_frames"] = True
    else:
        was_it_fetched["skip_frames"] = False

    # (Optional) meta.npz
    meta_file_path = frames_scene_dir / "meta.npz"
    if meta_file_path.exists():
        tgt_meta_file_path = tgt_scene_dir / "meta.npz"
        subprocess.run(["cp", str(meta_file_path), str(tgt_meta_file_path)])
        was_it_fetched["meta"] = True
    else:
        was_it_fetched["meta"] = False
    
    return was_it_fetched
