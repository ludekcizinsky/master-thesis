import os
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Dict, Any, Iterable
from dataclasses import dataclass
from scipy.spatial import cKDTree

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


class FullSceneDataset(Dataset):
    """
    Exposes (frame_id, image, mask, cam_intrinsics, smpl_param) for a chosen track id.

    Simplifying assumptions:
    1. All persons are on all frames
    """

    def __init__(self, preprocess_dir: Path, tids: List[int], cloud_downsample: int = 10):
        self.preprocess_dir = preprocess_dir
        self.tids = tids

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

    def __len__(self):
        return len(self.pose_all)

    def __getitem__(self, idx: int):

        # Image
        img_path = self.images_dir / f"{idx:04d}.png"
        image = load_image(img_path)  # [H,W,3]
        assert image.shape[0] == self.H and image.shape[1] == self.W, f"Image shape mismatch: {image.shape} vs ({self.H},{self.W})"

        # Masks for each tid
        masks = []
        for tid in self.tids:
            mask_path = self.mask_dir / str(tid) / f"{idx:04d}.png"
            if mask_path.exists():
                mask = load_mask(mask_path)  # [H,W]
            else:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            masks.append(mask)
        mask = torch.stack(masks, dim=0)  # [P,H,W] combined mask
        assert mask.shape[0] == len(self.tids), f"Mask shape {mask.shape} does not match number of tids {len(self.tids)}"


        # Camera 
        # - Extrinsics
        M_ext = self.pose_all[idx]  # [4,4]
        assert M_ext.shape == (4, 4), f"Extrinsics shape mismatch: {M_ext.shape} vs {(4, 4)}"

        # - Intrinsics
        intrinsics = self.intrinsics_all[idx][:3, :3]  # [3,3]
        K = intrinsics.clone()
        assert K.shape == (3, 3), f"K shape mismatch: {K.shape} vs {(3, 3)}"

        # SMPL params (86D)
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
    

@dataclass
class MegaSAMFrame:
    """Convenient container for a single frame."""
    image: torch.Tensor        # [H, W, 3], uint8 or float32 in [0,1] (see normalize)
    depth: Optional[torch.Tensor]   # [H, W], float32 (meters or MegaSaM units); may be None
    K: torch.Tensor            # [3, 3], float32
    w2c: torch.Tensor          # [4, 4], float32
    c2w: torch.Tensor 
    H: int
    W: int
    static_mask: Optional[torch.Tensor]  # [H, W] bool, True=STATIC (kept), False=DYNAMIC (ignored)


class MegaSAMDataset(Dataset):
    """
    Torch dataset for MegaSaM NPZ outputs:
        required keys:  images[N,H,W,3], depths[N,H,W], intrinsic[3,3], cam_c2w[N,4,4]
        optional keys:  static_mask[N,H,W] (bool or uint8), motion_prob[N,H,W] (float32)

    It also provides a helper to aggregate a static 3D point cloud across frames for 3DGS init.
    """
    def __init__(
        self,
        npz_path_or_data: str | Dict[str, np.ndarray],
        *,
        normalize_images: bool = True,
        center_pixels: bool = True,
        device: torch.device | str = "cpu",
    ):
        if isinstance(npz_path_or_data, str):
            data = np.load(npz_path_or_data, allow_pickle=True)
            print(f"--- FYI: loaded MegaSaM NPZ and the following keys found: {list(data.keys())}")
        else:
            data = npz_path_or_data

        # Required arrays
        images = data["images"]                  # (N,H,W,3) uint8
        depths = data.get("depths", None)        # (N,H,W) float32 (optional for pure color training)
        K = data["intrinsic"]                    # (3,3)
        c2w = data["cam_c2w"]                    # (N,4,4)

        assert images.ndim == 4 and images.shape[-1] in (3, 4)
        N, H, W, _ = images.shape
        print(f"--- FYI: MegaSaM dataset with {N} frames of size {W}x{H} found in {npz_path_or_data if isinstance(npz_path_or_data, str) else 'dict'}")
        if images.shape[-1] == 4:
            images = images[..., :3]  # drop alpha if present

        # Broadcast K to all frames if needed
        if K.ndim == 2:
            K = np.broadcast_to(K[None, ...], (N, 3, 3))
            print(f"--- FYI: broadcasting single intrinsic matrix to all {N} frames")
        elif K.ndim == 3:
            assert K.shape[0] == N
        else:
            raise ValueError("intrinsic must be shape (3,3) or (N,3,3)")

        # default: everything static
        static_mask = np.ones((N, H, W), dtype=bool)

        # Pose alignment 
        T0 = c2w[len(c2w) // 2]  
        T0_inv = np.linalg.inv(T0)  # Inverse of the first camera pose

        # Apply T0_inv to all camera poses
        self.w2c = np.matmul(T0_inv[np.newaxis, :, :], c2w)
        self.c2w = np.linalg.inv(self.w2c)

        # Cache tensors
        self.N = N
        self.H = H
        self.W = W
        self.center_pixels = center_pixels
        self.device = torch.device(device)
        self.images = torch.from_numpy(images.copy())  # uint8
        if normalize_images:
            self.images = self.images.float() / 255.0  # [0,1]
        self.depths = None if depths is None else torch.from_numpy(depths.astype(np.float32))
        self.K = torch.from_numpy(K.astype(np.float32))
        self.static_mask = torch.from_numpy(static_mask.astype(bool))
        self.w2c = torch.from_numpy(self.w2c.astype(np.float32))
        self.c2w = torch.from_numpy(self.c2w.astype(np.float32))

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> MegaSAMFrame:
        img = self.images[idx]                        # [H,W,3]
        dep = None if self.depths is None else self.depths[idx]  # [H,W]
        K = self.K[idx]
        w2c = self.w2c[idx]
        c2w = self.c2w[idx]
        sm = self.static_mask[idx]                    # [H,W] bool

        return MegaSAMFrame(
            image=img,
            depth=dep,
            K=K,
            w2c=w2c,
            c2w=c2w,
            H=self.H,
            W=self.W,
            static_mask=sm
        )

    # --------- Helpers for 3DGS ---------

    @torch.no_grad()
    def sample_pixels(
        self,
        idx: int,
        num: int,
        static_only: bool = False,
        return_uv: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Randomly sample pixels from a frame (optionally only static).
        Returns colors (and optional depths) for photometric / depth supervision.
        """
        frame = self[idx]
        H, W = frame.H, frame.W
        if static_only and frame.static_mask is not None:
            valid = frame.static_mask.view(-1)
        else:
            valid = torch.ones(H*W, dtype=torch.bool, device=self.images.device)

        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            raise RuntimeError("No valid pixels to sample (check your masks).")

        choice = valid_idx[torch.randint(0, valid_idx.numel(), (num,), device=valid_idx.device)]
        v = choice // W
        u = choice % W

        colors = frame.image[v, u, :]  # [num, 3], uint8 or float
        depths = None
        if frame.depth is not None:
            depths = frame.depth[v, u]  # [num]

        out = {
            "rgb": colors,             # [num,3]
            "K": frame.K,              # [3,3]
            "c2w": frame.c2w,          # [4,4]
            "w2c": frame.w2c,          # [4,4]
            "H": torch.tensor(H),
            "W": torch.tensor(W),
        }
        if depths is not None:
            out["depth"] = depths
        if return_uv:
            out["uv"] = torch.stack([u, v], dim=-1)  # [num,2]
        return out

    @torch.no_grad()
    def build_static_point_cloud(
        self,
        *,
        every_k: int = 1,                  # use every k-th frame
        downsample: int = 2,               # pixel stride
        min_depth: float = 1e-6,
        max_depth: float = float("inf"),
        return_colors: bool = True,
        use_depth_conf_mask: Optional[torch.Tensor] = None,  # [N,H,W] bool; ANDed with static mask
        device: str | torch.device = "cpu",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Back-project static pixels across frames into a single 3D point cloud in world coords.
        Returns:
          points: [M,3] float32
          colors: [M,3] float32 in [0,1] if dataset images normalized, else uint8 scaled later
        """
        assert self.depths is not None, "Depths are required to build a point cloud."

        pts_all = []
        cols_all = [] if return_colors else None

        K_np = self.K.cpu().numpy()
        for i in range(0, self.N, every_k):
            frame = self[i]
            K = K_np[i]
            K_inv = np.linalg.inv(K)

            dep = frame.depth.cpu().numpy()  # [H,W]
            sm = frame.static_mask.cpu().numpy() if frame.static_mask is not None else np.ones_like(dep, dtype=bool)
            if use_depth_conf_mask is not None:
                sm = sm & use_depth_conf_mask[i].cpu().numpy()

            # downsample grid
            yy = np.arange(0, frame.H, downsample)
            xx = np.arange(0, frame.W, downsample)
            u, v = np.meshgrid(xx, yy)  # u:cols (x), v:rows (y)
            if self.center_pixels:
                u = u + 0.5
                v = v + 0.5
            
            v, u = v.astype(np.int32), u.astype(np.int32)  # [H',W']
            d = dep[v, u]  # [H',W']
            keep = (sm[v, u]) & np.isfinite(d) & (d > min_depth) & (d < max_depth)

            if keep.sum() == 0:
                continue

            u_keep = u[keep].reshape(-1)
            v_keep = v[keep].reshape(-1)
            d_keep = d[keep].reshape(-1)

            # backproject to camera coords: x_cam = d * K^{-1} [u, v, 1]^T
            homo = np.stack([u_keep, v_keep, np.ones_like(u_keep)], axis=-1)  # [M,3]
            local_dirs = (K_inv @ homo.T).T                                     # [M,3]
            x_cam = local_dirs * d_keep[:, None]                                # [M,3]

            # to world: X_w = R * x_cam + t
            R = frame.c2w[:3, :3].cpu().numpy()
            t = frame.c2w[:3, 3].cpu().numpy()
            Xw = (R @ x_cam.T).T + t[None, :]                                   # [M,3]
            pts_all.append(torch.from_numpy(Xw).to(device=device, dtype=torch.float32))

            if return_colors:
                col = frame.image.cpu().numpy()[v[keep], u[keep], :]  # [M,3]
                cols_all.append(torch.from_numpy(col).to(device=device, dtype=torch.float32))

        points = torch.cat(pts_all, dim=0) if pts_all else torch.empty(0, 3, device=device)
        colors = torch.cat(cols_all, dim=0) if (return_colors and cols_all) else None
        return points, colors

    @staticmethod
    @torch.no_grad()
    def estimate_scales_from_knn(pts: torch.Tensor, k: int = 16, percentile: float = 50.0) -> torch.Tensor:
        """
        Heuristic to set initial Gaussian scales from local point spacing.
        Returns per-point isotropic scale sigma: [N,1] (float32).
        """
        if pts.numel() == 0:
            return torch.empty(0, 1, dtype=torch.float32, device=pts.device)

        pts_np = pts.detach().cpu().numpy()
        tree = cKDTree(pts_np)
        dists, _ = tree.query(pts_np, k=min(k+1, max(2, pts_np.shape[0])))  # first neighbor is itself (0)
        # ignore the zero self-distance
        local = dists[:, 1:]  # [N,k]
        sigma = np.percentile(local, percentile, axis=1)  # [N]
        return torch.from_numpy(sigma.astype(np.float32)).unsqueeze(1).to(pts.device)