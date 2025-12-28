import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from omegaconf import DictConfig

# Silence noisy deprecation notices from upstream libs to keep logs readable.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import hydra
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset

import torch.nn.functional as F

from fused_ssim import fused_ssim
import wandb

from collections import deque

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "submodules", "lhm")
    ),
)
from training.helpers.gs_renderer import GS3DRenderer
from training.helpers.dataset import SceneDataset, fetch_data_if_available, root_dir_to_image_dir, root_dir_to_mask_dir, root_dir_to_skip_frames_path, root_dir_to_depth_dir
from training.helpers.debug import overlay_smplx_mesh_pyrender, save_depth_comparison
from training.helpers.eval_metrics import ssim, psnr, lpips, _ensure_nchw, segmentation_mask_metrics
from training.helpers.difix import difix_refine

from submodules.difix3d.src.pipeline_difix import DifixPipeline


# ---------------------------------------------------------------------------
# Mask estimation functions
# ---------------------------------------------------------------------------
def estimate_masks_from_smplx_batch(batch: Dict[str, Any], smplx_model) -> torch.Tensor:
    """
    Estimate binary foreground masks by rasterizing posed SMPL-X meshes into each camera view.

    Expected `batch` format (as produced by `training.helpers.dataset.SceneDataset` + PyTorch collation):
      - `batch["image"]`: `torch.Tensor` of shape `[B, H, W, 3]` in `[0, 1]` (used only for H/W and device).
      - `batch["K"]`: `torch.Tensor` of shape `[B, 4, 4]` (camera intrinsics; fx/fy/cx/cy read from it).
      - `batch["c2w"]`: `torch.Tensor` of shape `[B, 4, 4]` (camera-to-world transform).
      - `batch["smplx_params"]`: `dict[str, torch.Tensor]` where each tensor has shape `[B, P, ...]` and
        contains the keys:
          - `"betas"`: `[B, P, D]`
          - `"root_pose"`: `[B, P, 3]`
          - `"body_pose"`: `[B, P, 21, 3]`
          - `"jaw_pose"`, `"leye_pose"`, `"reye_pose"`: `[B, P, 3]`
          - `"lhand_pose"`, `"rhand_pose"`: `[B, P, 15, 3]`
          - `"trans"`: `[B, P, 3]`

    Returns:
      - `masks`: `torch.Tensor` of shape `[B, H, W, 1]` on the same device as `batch["image"]`,
        with values in `{0.0, 1.0}`.
    """

    # Parse batch data
    images: torch.Tensor = batch["image"]
    Ks: torch.Tensor = batch["K"]
    c2ws: torch.Tensor = batch["c2w"]
    smplx_params: Dict[str, torch.Tensor] = batch["smplx_params"]

    device = images.device
    bsize, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])

    # SMPL-X mesh topology.
    smplx_layer = getattr(smplx_model, "smplx_layer", None).to(device)
    faces = np.asarray(getattr(smplx_layer, "faces"), dtype=np.int64)

    # Convert CV camera convention to OpenGL for pyrender.
    cv_to_gl = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=c2ws.dtype,
    )

    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    out_masks: List[torch.Tensor] = []

    # Expression is not needed for silhouette quality; keep it zero.
    expr_dim = int(getattr(getattr(smplx_model, "smpl_x", None), "expr_param_dim", 0))

    for bi in range(bsize):
        K = Ks[bi].detach().cpu()
        fx, fy, cx, cy = (
            float(K[0, 0]),
            float(K[1, 1]),
            float(K[0, 2]),
            float(K[1, 2]),
        )
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e4)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.5, 0.5, 0.5))

        # Add all persons as meshes to the scene (pyrender will handle occlusions via depth).
        num_people = int(smplx_params["betas"][bi].shape[0])
        for pid in range(num_people):
            betas = smplx_params["betas"][bi, pid : pid + 1]
            root_pose = smplx_params["root_pose"][bi, pid : pid + 1]
            body_pose = smplx_params["body_pose"][bi, pid : pid + 1]
            jaw_pose = smplx_params["jaw_pose"][bi, pid : pid + 1]
            leye_pose = smplx_params["leye_pose"][bi, pid : pid + 1]
            reye_pose = smplx_params["reye_pose"][bi, pid : pid + 1]
            lhand_pose = smplx_params["lhand_pose"][bi, pid : pid + 1]
            rhand_pose = smplx_params["rhand_pose"][bi, pid : pid + 1]
            trans = smplx_params["trans"][bi, pid : pid + 1]

            if expr_dim > 0:
                expression = torch.zeros((1, expr_dim), device=device, dtype=betas.dtype)
            else:
                expression = None

            out = smplx_layer(
                global_orient=root_pose,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                left_hand_pose=lhand_pose,
                right_hand_pose=rhand_pose,
                betas=betas,
                transl=trans,
                expression=expression,
            )
            verts = out.vertices[0].detach().cpu().numpy()
            mesh_tm = trimesh.Trimesh(verts, faces, process=False)
            scene.add(pyrender.Mesh.from_trimesh(mesh_tm, smooth=False))

        # Camera pose for pyrender.
        c2w = c2ws[bi]
        w2c_cv = torch.inverse(c2w)
        w2c_gl = torch.matmul(cv_to_gl, w2c_cv)
        c2w_gl = torch.inverse(w2c_gl)
        pose = c2w_gl.clone()
        pose[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=pose.dtype)
        scene.add(camera, pose=pose.detach().cpu().numpy())

        # Finally, render. Depending on pyrender version, DEPTH_ONLY may return either
        # the depth array directly or a (color, depth) tuple.
        depth_out = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        depth = depth_out[1] if isinstance(depth_out, tuple) else depth_out
        mask_np = (depth > 0).astype(np.float32)  # [H, W]
        out_masks.append(torch.from_numpy(mask_np).to(device=device, dtype=images.dtype).unsqueeze(-1))

    renderer.delete()
    return torch.stack(out_masks, dim=0)

def overlay_mask_on_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    color_rgb: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Alpha-blend a (binary) mask over an RGB image for visualization.

    Args:
        image: `[H, W, 3]` float tensor in `[0, 1]`.
        mask: `[H, W, 1]` float tensor in `{0, 1}` (or `[0, 1]` for soft masks).
        color_rgb: Foreground tint color.
        alpha: Tint opacity on foreground pixels.

    Returns:
        `[H, W, 3]` float tensor in `[0, 1]`.
    """
    mask01 = mask.clamp(0.0, 1.0)
    color = torch.tensor(color_rgb, device=image.device, dtype=image.dtype).view(1, 1, 3)
    alpha_t = torch.as_tensor(alpha, device=image.device, dtype=image.dtype)
    return image * (1.0 - alpha_t * mask01) + color * (alpha_t * mask01)


def get_masked_images(images: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Apply binary masks to images.

    Args:
        images: `[B, H, W, 3]` float tensor in `[0, 1]`.
        masks: `[B, H, W, 1]` float tensor in `{0, 1}`.

    Returns:
        `[B, H, W, 3]` float tensor in `[0, 1]`.
    """
    return images * masks


def get_gt_est_masks_overlay(gt_masks: torch.Tensor, est_masks: torch.Tensor) -> torch.Tensor:
    """
    Create an overlay image visualizing ground-truth and estimated masks.

    Args:
        gt_masks: `[B, H, W, 1]` float tensor in `{0, 1}`.
        est_masks: `[B, H, W, 1]` float tensor in `{0, 1}`.
    Returns:
        `[B, H, W, 3]` float tensor in `[0, 1]` where:
          - GT-only pixels are green,
          - Est-only pixels are red,
          - Overlapping pixels are yellow,
          - Background pixels are black.
    """

    bsize, H, W = gt_masks.shape[0], gt_masks.shape[1], gt_masks.shape[2]
    overlays: List[torch.Tensor] = []
    for bi in range(bsize):
        gt_mask = gt_masks[bi, :, :, 0] > 0.5
        est_mask = est_masks[bi, :, :, 0] > 0.5
        gt_only = gt_mask & (~est_mask)
        est_only = (~gt_mask) & est_mask
        overlap = gt_mask & est_mask
        overlay = torch.zeros((H, W, 3), device=gt_masks.device, dtype=gt_masks.dtype)
        # GT-only: green
        overlay[:, :, 1][gt_only] = 1.0
        # Est-only: red
        overlay[:, :, 0][est_only] = 1.0
        # Overlap: yellow
        overlay[:, :, 0][overlap] = 1.0
        overlay[:, :, 1][overlap] = 1.0
        overlays.append(overlay)
    return torch.stack(overlays, dim=0)


def save_segmentation_debug_figures(
    gt_masked_frames: torch.Tensor,
    est_masked_frames: torch.Tensor,
    mask_overlays: torch.Tensor,
    frame_names: List[str],
    iou: torch.Tensor,
    recall: torch.Tensor,
    f1: torch.Tensor,
    save_dir: Path,
):
    """
    Save per-sample debug figures showing masked images and mask overlay.

    Args:
        gt_masked_frames: `[B, H, W, 3]` masked with GT masks.
        est_masked_frames: `[B, H, W, 3]` masked with estimated masks.
        mask_overlays: `[B, H, W, 3]` overlay of estimated (red) on GT (green).
        frame_names: list of length `B`, used for filenames.
        iou/recall/f1: `[B]` tensors with per-frame metrics.
        save_dir: directory to save PNGs into.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    titles = [
        "Masked Frames using GT Mask",
        "Masked Frames using Est. Mask",
        "Estimated Mask (red) overlayed\non Ground Truth Mask (green)",
    ]

    for idx in range(gt_masked_frames.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        imgs = [
            gt_masked_frames[idx].detach().cpu().numpy(),
            est_masked_frames[idx].detach().cpu().numpy(),
            mask_overlays[idx].detach().cpu().numpy(),
        ]
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(f"IoU: {float(iou[idx]):.3f} | Recall: {float(recall[idx]):.3f} | F1: {float(f1[idx]):.3f}")
        out_path = save_dir / f"{frame_names[idx]}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def root_dir_to_difix_debug_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "est_images_debug" / f"{cam_id}"

def root_dir_to_est_masks_debug_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "est_masks_debug" / f"{cam_id}"

def save_image(tensor: torch.Tensor, filename: str):
    """
    Accepts HWC, CHW, or BCHW; if batch > 1, saves the first item.
    Assumes values in [0,1]. If channels are a multiple of 3, tiles them along width.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] % 3 == 0:
        c = tensor.shape[0]
        if c not in (1, 3):
            n = c // 3
            tensor = tensor.view(n, 3, tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor[i] for i in range(n)], dim=2)
        image = tensor.permute(1, 2, 0).detach().cpu().numpy()
    elif tensor.dim() == 3 and tensor.shape[-1] % 3 == 0:
        c = tensor.shape[-1]
        if c not in (1, 3):
            n = c // 3
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], n, 3)
            tensor = torch.cat([tensor[:, :, i] for i in range(n)], axis=1)
        image = tensor.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported tensor shape for save_image: {tensor.shape}")
    image = (image * 255).clip(0, 255).astype("uint8")
    Image.fromarray(image).save(filename)

def save_binary_mask(tensor: torch.Tensor, filename: str):
    """
    Accepts HWC, CHW, or BCHW where C=1; if batch > 1, saves the first item.
    Assumes values in [0,1]. Saves as a binary (0/255) PNG image.
    """

    # First item only
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # HWC or CHW with C=1 convert to HxWx1
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() == 3 and tensor.shape[-1] == 1:
        pass
    else:
        raise ValueError(f"Unsupported tensor shape for save_binary_mask: {tensor.shape}")
    
    # Binarize and save
    mask = (tensor.detach().cpu().numpy() > 0.5).astype("uint8") * 255
    Image.fromarray(mask.squeeze(-1)).save(filename)


def build_renderer():
    renderer = GS3DRenderer(
        human_model_path="/scratch/izar/cizinsky/pretrained/pretrained_models/human_model_files",
        subdivide_num=1,
        smpl_type="smplx_2",
        feat_dim=1024,
        query_dim=1024,
        use_rgb=True,
        sh_degree=3,
        mlp_network_config={'activation': 'silu', 'n_hidden_layers': 2, 'n_neurons': 512},
        xyz_offset_max_step=1.0,
        clip_scaling=[100, 0.01, 0.05, 3000],
        shape_param_dim=10,
        expr_param_dim=100,
        cano_pose_type=1,
        fix_opacity=False,
        fix_rotation=False,
        decoder_mlp=False,
        skip_decoder=True,
        decode_with_extra_info=None,
        gradient_checkpointing=True,
        apply_pose_blendshape=False,
        dense_sample_pts=40000,
    )

    return renderer


def get_masks_based_bbox(masks: torch.Tensor, pad: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    Get bounding boxes around masks with optional padding.

    Args:
        masks: Tensor of shape [B, H, W, 1] with mask values in [0, 1].
        pad: Number of pixels to pad the bounding box on each side.

    Returns:
        List of bounding boxes [(y_min, y_max, x_min, x_max)] for each mask in the batch.
    """

    per_item_bboxes = []
    global_bbox = None

    for b in range(masks.shape[0]):
        ys, xs = torch.where(masks[b, :, :, 0] > 0.5)
        # No mask, use full image
        if ys.numel() == 0 or xs.numel() == 0:
            y_min, y_max = 0, masks.shape[1] - 1
            x_min, x_max = 0, masks.shape[2] - 1
        else:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            # Add padding
            y_center = (y_min + y_max) / 2
            x_center = (x_min + x_max) / 2
            h_half = (y_max - y_min) / 2 + pad
            w_half = (x_max - x_min) / 2 + pad
            y_min = max(int(y_center - h_half), 0)
            y_max = min(int(y_center + h_half), masks.shape[1] - 1)
            x_min = max(int(x_center - w_half), 0)
            x_max = min(int(x_center + w_half), masks.shape[2] - 1)

        bbox = (y_min, y_max, x_min, x_max)
        per_item_bboxes.append(bbox)

        if global_bbox is None:
            global_bbox = bbox
        else:
            global_bbox = (
                min(global_bbox[0], y_min),
                max(global_bbox[1], y_max),
                min(global_bbox[2], x_min),
                max(global_bbox[3], x_max),
            )

    if global_bbox is None:
        return []

    return [global_bbox for _ in per_item_bboxes]

def bbox_crop(bboxes: List[Tuple[int, int, int, int]], to_crop: torch.Tensor) -> torch.Tensor:
    """
    Crop a batch of tensors according to the provided bounding boxes.

    Args:
        bboxes: List of bounding boxes [(y_min, y_max, x_min, x_max)] for each tensor in the batch.
        to_crop: Tensor of shape [B, H, W, C] to be cropped where C can be any number of channels.

    Returns:
        Cropped tensor of shape [B, H_crop, W_crop, C].
    """

    # Crop tensors
    cropped = []
    for b in range(to_crop.shape[0]):
        y_min, y_max, x_min, x_max = bboxes[b]
        cropped.append(to_crop[b, y_min : y_max + 1, x_min : x_max + 1, :])

    return torch.stack(cropped, dim=0)


def load_skip_frames(scene_dir: Path) -> List[int]:
    """
    Load skip frames from skip_frames.csv in the scene directory.

    Note: the frame indicies are actual frame indexes and the frames dir may
    not always start with frame 0. Therefore, the returnd indices correspond to
    the actual frame indices to skip. e.g. if we have frames 10, 11, 12, 13 and skip_frames.csv contains "11,13",
    we will skip frames 11 and 13.

    Args:
        scene_dir: Path to the scene directory.
    Returns:
        List of frame indices to skip.
    """

    skip_frames_file = root_dir_to_skip_frames_path(scene_dir)
    if not skip_frames_file.exists():
        return []
    with open(skip_frames_file, "r") as f:
        line = f.readline().strip()
        skip_frames = [int(idx_str) for idx_str in line.split(",") if idx_str.isdigit()]
    return skip_frames

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class MultiHumanTrainer:
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tuner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess_dir = Path(cfg.preprocessing_dir)
        self.output_dir = Path(cfg.output_dir)
        self.trn_data_dir = Path(cfg.input_data_dir) / "train" / f"{cfg.scene_name}" / f"{self.cfg.exp_name}"
        self.trn_data_dir.mkdir(parents=True, exist_ok=True)
        self.test_data_dir = Path(cfg.input_data_dir) / "test" / f"{cfg.scene_name}" / f"{self.cfg.exp_name}"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_params = tuple(cfg.train_params)
        self.sample_every = cfg.sample_every
        self.renderer : GS3DRenderer = build_renderer().to(self.tuner_device)
        self.wandb_run = None

        # Initialize wandb 
        self._init_wandb()

        # Prepare datasets for training and testing
        self._init_train_dataset()
        self._init_test_scene_dir()

        # Preare model
        self._load_model()

        # Initialize difix
        self._init_nv_generation()

        # Generate novel views 
        is_nv_gen_done = False
        n_its_nv_gen = 0
        while not is_nv_gen_done:
            is_nv_gen_done = self.novel_view_generation_step()
            n_its_nv_gen += 1
            print(" Novel view generation step ", n_its_nv_gen, " done: ", is_nv_gen_done)

        # Prepare training dataset from all the new cameras
        for cam_id in self.difix_cam_ids_all:
            scene_ds = SceneDataset(
                self.trn_data_dir,
                int(cam_id),
                use_depth=self.cfg.use_depth,
                device=self.tuner_device,
                sample_every=1,
            )
            self.trn_datasets.append(scene_ds)
        self.trn_dataset = ConcatDataset(self.trn_datasets)

    # ---------------- Datasets ----------------------------
    def _init_train_dataset(self):

        src_cam_id: int = self.cfg.nvs_eval.source_camera_id

        # Fetch training dataset from the specified directories 
        # to the training data dir if not already present
        fetch_data_if_available(
            self.trn_data_dir,
            src_cam_id,
            Path(self.cfg.frames_scene_dir),
            Path(self.cfg.masks_scene_dir),
            Path(self.cfg.cameras_scene_dir),
            Path(self.cfg.smplx_params_scene_dir),
            Path(self.cfg.depths_scene_dir) if self.cfg.use_depth else None,
        )

        # Load skip frames if needed
        skip_frames = load_skip_frames(self.trn_data_dir)

        # Create training dataset
        self.trn_datasets = list() 
        trn_ds = SceneDataset(
            self.trn_data_dir, 
            src_cam_id, 
            use_depth=self.cfg.use_depth, 
            device=self.tuner_device, 
            sample_every=self.sample_every,
            skip_frames=skip_frames,
        )
        self.curr_trn_frame_paths = trn_ds.frame_paths
        self.trn_datasets.append(trn_ds)
        self.trn_dataset = trn_ds
        self.trn_render_hw = self.trn_dataset.trn_render_hw
        print(f"Training dataset initialised at {self.trn_data_dir} with {len(self.trn_dataset)} images.")

    def _init_test_scene_dir(self):

        # Parse settings
        src_cam_id: int = self.cfg.nvs_eval.source_camera_id
        tgt_cam_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        all_cam_ids = [src_cam_id] + tgt_cam_ids
        frames_dir = Path(self.cfg.test_frames_scene_dir)
        masks_dir = Path(self.cfg.test_masks_scene_dir)
        cameras_dir = Path(self.cfg.test_cameras_scene_dir)
        smplx_params_dir = Path(self.cfg.test_smplx_params_scene_dir)
        depths_dir = Path(self.cfg.test_depths_scene_dir) if self.cfg.test_depths_scene_dir is not None else None


        # Fetch testing dataset from the specified directories 
        for cam_id in all_cam_ids:
            fetch_data_if_available(
                self.test_data_dir,
                cam_id,
                frames_dir,
                masks_dir,
                cameras_dir,
                smplx_params_dir,
                depths_dir,
            )

    # ---------------- Model  ------------------------------
    def _load_model(self):

        # Get query points + tranform from neural to zero pose
        # - in theory, to infer query points, we can use betas, face and and joint offset
        # - in practice, I currently use betas only
        smplx_params = self.trn_dataset[0]["smplx_params"]
        self.query_points, self.tranform_mat_neutral_pose = self.renderer.get_query_points(smplx_params, self.tuner_device)

        # Load Canonical 3DGS
        root_gs_model_dir = self.preprocess_dir / "canon_3dgs_lhm"
        self.gs_model_list = torch.load(root_gs_model_dir / "union" / "hi4d_gs.pt", map_location=self.tuner_device, weights_only=False)
        

    # ---------------- Training utilities ----------------
    def _trainable_tensors(self) -> List[torch.Tensor]:
        params = []
        for pidx, gauss in enumerate(self.gs_model_list):
            print(f"For the {pidx}-th gauss model, adding trainable tensors:")
            for name in self.train_params:
                t = getattr(gauss, name, None)
                if torch.is_tensor(t):
                    t.requires_grad_(True)
                    print(f"    - {name}")
                    params.append(t)

        return params


    def forward(self, batch):

        # Parse batch data
        Ks = batch["K"] # [B, 4, 4],
        c2ws = batch["c2w"] # [B, 4, 4]
        smplx_params = batch["smplx_params"] # each key has value of shape [B, P, ...] where P is num persons

        # Render target views
        num_views = Ks.shape[0]
        render_h, render_w = self.trn_render_hw
        render_res_list = []
        for view_idx in range(num_views):
            smplx_single_view = {k: v[view_idx] for k, v in smplx_params.items()}
            render_res = self.renderer.animate_and_render(
                self.gs_model_list,
                self.query_points,
                smplx_single_view,
                c2ws[view_idx],
                Ks[view_idx],
                render_h,
                render_w,
            )
            render_res_list.append(render_res)

        # Parse outputs
        pred_rgb = torch.cat([res["comp_rgb"] for res in render_res_list], dim=0)  # [B, H, W, 3], 0-1
        pred_mask = torch.cat([res["comp_mask"][..., :1] for res in render_res_list], dim=0)  # [B, H, W, 1]
        pred_depth = torch.cat([res["comp_depth"][..., :1] for res in render_res_list], dim=0)  # [B, H, W, 1]

        return pred_rgb, pred_mask, pred_depth

    def _canonical_regularization(self):
        """Return combined canonical regularization and its components."""
        asap_terms = []
        acap_terms = []
        for gauss in self.gs_model_list:
            # Gaussian Shape Regularization (ASAP): encourage isotropic scales.
            # If scaling is stored in log-space, exp() keeps positivity; otherwise it’s a smooth surrogate.
            scales = gauss.scaling
            scales_pos = torch.exp(scales)
            asap = ((scales_pos - scales_pos.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1).mean()
            asap_terms.append(asap)

            # Positional anchoring (ACAP): hinge on offset magnitude beyond margin.
            offsets = gauss.offset_xyz
            acap = torch.clamp(offsets.norm(dim=-1) - self.cfg.regularization['acap_margin'], min=0.0).mean()
            acap_terms.append(acap)

            # Compute the percentage of Gaussians that are outside the margin
            with torch.no_grad():
                num_outside = (offsets.norm(dim=-1) > self.cfg.regularization['acap_margin']).sum().item()
                total = offsets.shape[0]
                percent_outside = (num_outside / total) * 100.0
                to_log = {"debug/acap_percent_outside": percent_outside}
                if self.wandb_run is not None:
                    wandb.log(to_log)

        asap_loss = torch.stack(asap_terms).mean() * self.cfg.regularization["asap_w"]
        acap_loss = torch.stack(acap_terms).mean() * self.cfg.regularization["acap_w"]

        return asap_loss, acap_loss

    @staticmethod
    def _infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    def _build_loader(self, dataset: SceneDataset) -> DataLoader:
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=0, drop_last=False
        )
        return loader

    # ---------------- Logging utilities ----------------
    def _init_wandb(self):
        if not self.cfg.wandb.enable or wandb is None:
            return
        if self.wandb_run is None:
            self.wandb_run = wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config={
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "lr": self.cfg.lr,
                    "weight_decay": self.cfg.weight_decay,
                    "grad_clip": self.cfg.grad_clip,
                    "train_params": self.train_params,
                    "exp_name": self.cfg.exp_name,
                    "scene_name": self.cfg.scene_name,
                    "output_dir": str(self.output_dir),
                    "loss_weights": self.cfg.loss_weights,
                    "sample_every": self.cfg.sample_every,
                },
                name=self.cfg.exp_name,
                tags=list(self.cfg.wandb.tags) if "tags" in self.cfg.wandb else None,
            )

    # ---------------- Training loop ----------------
    def train_loop(self):

        # Prepare dataset and dataloader
        trn_loader = self._build_loader(self.trn_dataset)
        trn_iter = self._infinite_loader(trn_loader)

        # Initialize optimizer
        params = self._trainable_tensors()
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Pre-optimization visualization (epoch 0).
        if self.cfg.eval_pretrain:
            self.eval_loop(0)

        # Training loop
        for epoch in range(self.cfg.epochs):
            trn_ds_size = len(self.trn_dataset)
            total_ds_size = trn_ds_size 
            n_batches_per_current_epoch = total_ds_size // self.cfg.batch_size
            running_loss = 0.0
            pbar = tqdm(
                total=n_batches_per_current_epoch,
                desc=f"Epoch {epoch + 1}/{self.cfg.epochs}",
                leave=False,
            )

            batch_idx = 0

            while batch_idx < n_batches_per_current_epoch:
                
                # Sample batch
                batch = next(trn_iter)

                # Parse batch data
                fnames = batch["frame_name"] # list of length B
                cam_ids = batch["cam_id"] # [B]
                frames = batch["image"] # [B, H, W, 3],
                masks = batch["mask"] # [B, H, W, 1],
                depths = batch.get("depth", None) # [B, H, W, 1],
                bsize = int(frames.shape[0])
                batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
                batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]

                # create a mask for samples in the batch that come from src camera
                src_cam_id = self.cfg.nvs_eval.source_camera_id
                cam_id_tensor = batch["cam_id"]  # [B]
                is_src_cam = (cam_id_tensor == src_cam_id)  # [B]

                # Reset gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                pred_rgb, pred_mask, pred_depth = self.forward(batch)

                # Compute masked ground truth
                mask3 = masks
                if mask3.shape[-1] == 1:
                    mask3 = mask3.repeat(1, 1, 1, 3)
                gt_masked = frames * mask3
                if depths is not None:
                    gt_depth_masked = depths * masks

                # Apply mask also to render if it novel view and this mechanism is enabled
                if self.cfg.apply_mask_to_render_for_nvs:
                    # For src view, use all-ones mask to not affect the render
                    effective_masks = masks.clone().to(pred_rgb.dtype)
                    effective_masks[is_src_cam] = 1.0
                    effective_mask3 = effective_masks
                    if effective_mask3.shape[-1] == 1:
                        effective_mask3 = effective_mask3.repeat(1, 1, 1, 3)
                    pred_rgb = pred_rgb * effective_mask3
                    pred_mask = pred_mask * effective_masks
                    if pred_depth is not None:
                        pred_depth = pred_depth * effective_masks

                # BBOX crop around the union person mask
                if self.cfg.use_bbox_crop:
                    bboxes = get_masks_based_bbox(masks, pad=self.cfg.bbox_pad)
                    loss_pred_rgb = bbox_crop(bboxes, pred_rgb)
                    loss_pred_mask = bbox_crop(bboxes, pred_mask)
                    loss_gt_masked = bbox_crop(bboxes, gt_masked)
                    if depths is not None:
                        loss_pred_depth = bbox_crop(bboxes, pred_depth)
                        loss_gt_depth_masked = bbox_crop(bboxes, gt_depth_masked)
                    loss_masks = bbox_crop(bboxes, masks)
                else:
                    loss_pred_rgb = pred_rgb
                    loss_pred_mask = pred_mask
                    loss_gt_masked = gt_masked
                    if depths is not None:
                        loss_pred_depth = pred_depth
                        loss_gt_depth_masked = gt_depth_masked
                    loss_masks = masks

                # Compute loss
                rgb_loss = self.cfg.loss_weights["rgb"] * F.mse_loss(loss_pred_rgb, loss_gt_masked)
                # - only compute silhouette loss on the source camera.
                #   (Novel-view masks may be bootstrapped and noisy; using them for silhouette supervision
                #   can inject strong, conflicting gradients.)
                if is_src_cam.any():
                    loss_pred_mask_src = loss_pred_mask[is_src_cam]
                    loss_masks_src = loss_masks[is_src_cam]
                    sil_loss = self.cfg.loss_weights["sil"] * F.mse_loss(loss_pred_mask_src, loss_masks_src)
                else:
                    sil_loss = torch.zeros((), device=loss_pred_mask.device, dtype=loss_pred_mask.dtype)
                if depths is not None:
                    depth_loss = self.cfg.loss_weights["depth"] * F.mse_loss(loss_pred_depth, loss_gt_depth_masked)
                else:
                    depth_loss = torch.zeros((), device=loss_pred_rgb.device, dtype=loss_pred_rgb.dtype)
                ssim_val = fused_ssim(_ensure_nchw(loss_pred_rgb), _ensure_nchw(loss_gt_masked), padding="valid")
                ssim_loss = self.cfg.loss_weights["ssim"] * (1.0 - ssim_val)
                asap_loss, acap_loss = self._canonical_regularization()
                reg_loss = asap_loss + acap_loss

                loss = rgb_loss + sil_loss + ssim_loss + reg_loss + depth_loss
                
                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip)
                optimizer.step()

                # Log the results
                running_loss += loss.item()
                pbar.set_postfix(
                   loss=f"{loss.item():.4f}",
                    asap=f"{asap_loss.item():.4f}",
                    acap=f"{acap_loss.item():.4f}",
                    depth=f"{depth_loss.item():.4f}",
                )

                if self.wandb_run is not None:
                    wandb.log(
                        {
                            "loss/combined": loss.item(),
                            "loss/rgb": rgb_loss.item(),
                            "loss/sil": sil_loss.item(),
                            "loss/ssim": ssim_loss.item(),
                            "loss/depth": depth_loss.item(),
                            "loss/reg": reg_loss.item(),
                            "loss/asap": asap_loss.item(),
                            "loss/acap": acap_loss.item(),
                            "epoch": epoch + 1,
                        }
                    )

                # space for debug stuff is here
                if batch_idx in [0] and (epoch + 1) % 2 == 0:
                    # - create a joined image from pred_masked and gt_masked for debugging
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name / f"epoch_{epoch+1:04d}" / "rgb"
                    debug_save_dir.mkdir(parents=True, exist_ok=True)
                    overlay = pred_rgb*0.5 + gt_masked*0.5
                    joined_image = torch.cat([pred_rgb, gt_masked, overlay], dim=3)  # Concatenate along width
                    for i in range(joined_image.shape[0]):
                        image = joined_image[i:i+1]
                        frame_name = fnames[i]
                        cam_id = cam_ids[i].item()
                        debug_image_path = debug_save_dir / f"rgb_loss_input_cam{cam_id}_{frame_name}.png"
                        save_image(image.permute(0, 3, 1, 2), str(debug_image_path))

                    # - save depth comparison images
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name / f"epoch_{epoch+1:04d}" / "depth"
                    debug_save_dir.mkdir(parents=True, exist_ok=True)
                    for i in range(pred_depth.shape[0]):
                        frame_name = fnames[i] 
                        cam_id = cam_ids[i].item()
                        save_path = debug_save_dir / f"depth_comparison_cam{cam_id}_frame_{frame_name}.png"
                        save_depth_comparison(pred_depth[i].squeeze(-1), gt_depth_masked[i].squeeze(-1), str(save_path))

                batch_idx += 1
                pbar.update(1)

            # End of epoch
            # - Progress bar stuff
            pbar.close()
            # - Log epoch level metrics
            avg_loss = running_loss / max(1, n_batches_per_current_epoch)
            if self.wandb_run is not None:
                to_log = {
                    "loss/combined_epoch": avg_loss, 
                    "epoch": epoch + 1, 
                    "loss/n_batches_per_epoch": n_batches_per_current_epoch,
                    "loss/total_ds_size": total_ds_size,
                    "loss/trn_ds_size": trn_ds_size,
                }
                wandb.log(to_log)

            # - Run eval loop if neccesary
            if getattr(self.cfg, "eval_every_epoch", 0) > 0 and (epoch + 1) % self.cfg.eval_every_epoch == 0:
                self.eval_loop(epoch + 1)

        # End of training    
        if self.wandb_run is not None:
            self.wandb_run.finish()

    # ---------------- Novel view generation -------------------
    def _init_nv_generation(self):
        src_cam_id = self.cfg.nvs_eval.source_camera_id
        self.left_cam_id, self.right_cam_id = src_cam_id, src_cam_id
        self.camera_ids = [4, 16, 28, 40, 52, 64, 76, 88]
        src_cam_idx = self.camera_ids.index(src_cam_id)
        half_n = (len(self.camera_ids) - 1) // 2
        other_half_n = len(self.camera_ids) - 1 - half_n

        self.from_src_left_traversed_cams = deque((self.camera_ids[src_cam_idx-1::-1] + self.camera_ids[:src_cam_idx:-1])[:half_n])
        self.from_src_right_traversed_cams = deque((self.camera_ids[src_cam_idx+1:] + self.camera_ids[:src_cam_idx])[:other_half_n])
        self.difix_cam_ids_all = list(self.from_src_left_traversed_cams) + list(self.from_src_right_traversed_cams)
        print("NV camera traversal initialized:")
        print(f"    Left traversed cams: {self.from_src_left_traversed_cams}")
        print(f"    Right traversed cams: {self.from_src_right_traversed_cams}")

    @torch.no_grad()
    def prepare_nv_rgb_frames(self, cam_id: int, prev_cam_id: int, difix_pipe: Optional[DifixPipeline] = None):
        """
        Prepare new dataset for the given camera by rendering novel views and refining them using Difix.

        Args:
            difix_pipe: DifixPipeline instance for refinement.
            cam_id: Target camera ID for which to create the new dataset.
            prev_cam_id: Previous camera ID to use as reference for Difix.

        Notes:
        1. render the new camera views (out: render_frames)
        2. refine the rendered frames using Difix with reference to previous camera views
        3. create new dataset from the refined frames copy other data from gt dataset
        """

        # Load dataset for the previous camera (will use only images + masks as reference for Difix)
        is_prev_cam_always_source = self.cfg.difix.is_prev_cam_always_source
        # - if previous cam is always source cam, use that
        if is_prev_cam_always_source:
            sample_every = self.sample_every
            prev_cam_id = self.cfg.nvs_eval.source_camera_id
            skip_frames = load_skip_frames(self.trn_data_dir)
        # - we just started, so previous cam dataset is the original GT dataset for both left/right
        elif self.left_cam_id == self.right_cam_id:
            sample_every = self.sample_every
            skip_frames = load_skip_frames(self.trn_data_dir)
        # - otherwise, previous cam dataset is the refined dataset from last Difix step
        else:
            sample_every = 1  # use all frames from the refined dataset (they have been already subsampled)
            skip_frames = []  # no skip frames for refined datasetk
        # - finally, load the previous cam dataset
        prev_cam_dataset = SceneDataset(self.trn_data_dir, prev_cam_id, device=self.tuner_device, 
                                        sample_every=sample_every, skip_frames=skip_frames)


        # Load new camera dataset (with dummy images and masks which I will replace with refined later)
        # - generate initial contents of the new cam dataset
        # (if dummy dirs are used, images/masks will be overwritten later)
        refined_frames_save_dir = root_dir_to_image_dir(self.trn_data_dir, cam_id)
        if os.path.exists(refined_frames_save_dir):
            return # already done
        masks_scene_dir = Path(self.cfg.masks_scene_dir) if not self.cfg.use_estimated_masks else Path("/dummy/masks/dir")
        fetch_data_if_available(
            self.trn_data_dir,
            cam_id,
            Path("/dummy/frames/dir"), 
            masks_scene_dir,   
            frame_paths=self.curr_trn_frame_paths,
        )

        # - finally, load the new cam dataset
        new_cam_sample_every = 1 # because curr trn frame paths are already subsampled
        new_cam_dataset = SceneDataset(self.trn_data_dir, cam_id, device=self.tuner_device, sample_every=new_cam_sample_every)
        new_cam_loader = DataLoader(
            new_cam_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # Render in batches novel views from target camera
        for batch in tqdm(new_cam_loader, desc=f"Preparing NV RGB frames for cam {cam_id}"):

            # - Update smplx params with neutral pose transform
            frame_indices = batch["frame_idx"] # [B]
            frame_names = batch["frame_name"] # [B]
            bsize = int(batch["image"].shape[0])
            batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
            batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]

            # - Forward pass to render novel views using GT smplx params and cameras
            pred_rgb, _, _ = self.forward(batch)

            # - Refine rendered images using Difix
            # -- Load reference images from previous camera view
            ref_images = []
            for i in range(bsize):
                sample_dict = prev_cam_dataset[frame_indices[i].item()]
                ref_img, ref_mask = sample_dict["image"], sample_dict["mask"]  # [H, W, 3], [H, W, 1], 0-1
                ref_frame = ref_img * ref_mask # [H, W, 3], 0-1
                ref_images.append(ref_frame.to(self.tuner_device))
            ref_images = torch.stack(ref_images, dim=0)  # [B, H, W, 3]
            # -- Run difix refinement
            refined_rgb = difix_refine(self.cfg.difix, pred_rgb, ref_images, difix_pipe) # [B, H, W, 3]
            # -- Save refined frames
            for i in range(refined_rgb.shape[0]):
                save_path = refined_frames_save_dir / f"{frame_names[i]}.jpg"
                save_image(refined_rgb[i].permute(2, 0, 1), str(save_path))

            # - Debug: save side-by-side comparison of reference, rendered, refined
            difix_debug_dir = root_dir_to_difix_debug_dir(self.trn_data_dir, cam_id)
            difix_debug_dir.mkdir(parents=True, exist_ok=True)
            joined = torch.cat([ref_images, pred_rgb, refined_rgb], dim=2)  # side-by-side along width
            for i in range(joined.shape[0]):
                save_path = difix_debug_dir / f"{frame_names[i]}.png"
                save_image(joined[i].permute(2, 0, 1), str(save_path))

    @torch.no_grad()
    def prepare_nv_depth_maps(self, cam_id: int, allow_overwrite: bool = False):
        
        # Check whether depth maps for this camera already exist
        depths_save_dir = root_dir_to_depth_dir(self.trn_data_dir, cam_id)
        if depths_save_dir.exists() and not allow_overwrite:
            return # already done

        # Call external script to prepare depth maps for the new camera
        print(f"Preparing NV depth maps for cam {cam_id}")
        script_path = Path(__file__).resolve().parents[1] / "submodules" / "da3" / "nv_inference.py"
        cmd = [
            "conda", "run", "-n", "da3",
            "python", str(script_path),
            "--scene-dir", str(self.trn_data_dir),
            "--camera-id", str(cam_id),
        ]
        subprocess.run(cmd, check=True)


    @torch.no_grad()
    def prepare_nv_masks(self, cam_id: int):

        # Load dataset
        new_cam_sample_every = 1 # because nv frames are already subsampled / filtered
        new_cam_dataset = SceneDataset(self.trn_data_dir, cam_id, device=self.tuner_device, sample_every=new_cam_sample_every)
        new_cam_loader = DataLoader(
            new_cam_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )


        # Compute masks for the nv dataset
        for batch in tqdm(new_cam_loader, desc=f"Estimating masks for cam {cam_id}"):
            # - Parse batch data
            frame_names = batch["frame_name"] # [B]
            rgb_frames = batch["image"] # [B, H, W, 3], 0-1

            # - Estimate binary masks
            mask_estimation_method = self.cfg.mask_estimator_kind
            if mask_estimation_method == "rgb_render_based":
                eps = 10.0 / 255.0
                binary_masks = (rgb_frames > eps).any(dim=-1, keepdim=True).float() # [B, H, W, 1]
            elif mask_estimation_method == "smplx_mesh_based":
                binary_masks = estimate_masks_from_smplx_batch(batch, self.renderer.smplx_model)  # [B, H, W, 1]

            # - Save binary masks
            masks_save_dir = root_dir_to_mask_dir(self.trn_data_dir, cam_id)
            masks_save_dir.mkdir(parents=True, exist_ok=True)
            for i in range(binary_masks.shape[0]):
                save_path = masks_save_dir / f"{frame_names[i]}.png"
                save_binary_mask(binary_masks[i].permute(2, 0, 1), str(save_path))

            # - Overlay the estimated masks on refined images and save for debugging
            debug_masks_overlay_dir = root_dir_to_est_masks_debug_dir(self.trn_data_dir, cam_id)
            debug_masks_overlay_dir.mkdir(parents=True, exist_ok=True)
            for i in range(binary_masks.shape[0]):
                overlay = overlay_mask_on_image(
                    rgb_frames[i],
                    binary_masks[i],
                    color_rgb=(1.0, 0.0, 0.0),
                    alpha=0.5,
                )
                save_path = debug_masks_overlay_dir / f"{frame_names[i]}.png"
                save_image(overlay.permute(2, 0, 1), str(save_path))

    @torch.no_grad()
    def novel_view_generation_step(self) -> bool:

        # Select next left/right cameras to process (if any left)
        previous_cams = []
        new_cams = []
        # - Left (only if any left)
        left_changed = False
        if len(self.from_src_left_traversed_cams) > 0: 
            prev_left_cam_id = self.left_cam_id
            new_left_cam_id = self.from_src_left_traversed_cams.popleft() 
            new_cams.append(new_left_cam_id)
            previous_cams.append(prev_left_cam_id)
            left_changed = True
        else:
            new_left_cam_id = self.left_cam_id

        # - Right (only if any left)
        right_changed = False
        if len(self.from_src_right_traversed_cams) > 0:
            prev_right_cam_id = self.right_cam_id
            new_right_cam_id = self.from_src_right_traversed_cams.popleft()
            new_cams.append(new_right_cam_id)
            previous_cams.append(prev_right_cam_id)
            right_changed = True
        else:
            new_right_cam_id = self.right_cam_id

        # If no new cameras to process, return
        if not left_changed and not right_changed:
            return True # returns true to signal that difix is done

        # Generation pipeline
        # - New rgb frames
        # -- Initialize Difix pipeline
        is_enabled = self.cfg.difix.trn_enable
        if not is_enabled:
            difix_pipe = None
        else:
            difix_pipe = DifixPipeline.from_pretrained(
                self.cfg.difix.model_id, 
                trust_remote_code=True, 
                requires_safety_checker=False,
                num_views=2,
            )
            difix_pipe.to(self.tuner_device)
            difix_pipe.set_progress_bar_config(disable=True)

        # -- Generate new datasets for the new cameras
        for cam_id, prev_cam_id in zip(new_cams, previous_cams):
            self.prepare_nv_rgb_frames(
                cam_id,
                prev_cam_id,
                difix_pipe,
            )
        # -- Delete difix pipe to free memory
        if difix_pipe is not None:
            del difix_pipe
            torch.cuda.empty_cache()

        # - New depth maps for the new cameras
        for cam_id in new_cams:
            self.prepare_nv_depth_maps(cam_id)

        # - New mask maps for the new cameras (if needed)
        if self.cfg.use_estimated_masks:
            for cam_id in new_cams:
                self.prepare_nv_masks(cam_id)

        # Update the left and right previous cam ids
        self.left_cam_id = new_left_cam_id
        if left_changed:
            print(f"Updated NV left cam: {prev_left_cam_id} -> {self.left_cam_id}")
        self.right_cam_id = new_right_cam_id
        if right_changed:
            print(f"Updated NV right cam: {prev_right_cam_id} -> {self.right_cam_id}")

        return False


    # ---------------- Evaluation -------------------
    @torch.no_grad()
    def eval_loop_nvs(self, epoch):

        # Parse the evaluation setup
        target_camera_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        root_save_dir: Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        root_save_dir.mkdir(parents=True, exist_ok=True)

        # Load skip frames
        skip_frames = load_skip_frames(self.test_data_dir)

        # Init source camera dataset for Difix reference images
        src_cam_id: int = self.cfg.nvs_eval.source_camera_id
        src_cam_dataset = SceneDataset(self.test_data_dir, src_cam_id, 
                                       device=self.tuner_device, skip_frames=skip_frames)

        # Init Difix if enabled for the evaluation
        if self.cfg.difix.eval_enable:
            difix_pipe = DifixPipeline.from_pretrained(
                "nvidia/difix",
                trust_remote_code=True,
                requires_safety_checker=False,
                num_views=1,
            )
            difix_pipe.to(self.tuner_device)
            difix_pipe.set_progress_bar_config(disable=True)
        else:
            difix_pipe = None

        # Collector for metrics
        metrics_all_cams_per_frame = list()

        for tgt_cam_id in target_camera_ids:
            # Prepare dataset and dataloader for target camera
            val_dataset = SceneDataset(self.test_data_dir, tgt_cam_id, 
                                       device=self.tuner_device, skip_frames=skip_frames)
            loader = DataLoader(
                val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
            )

            # Prepare paths where to save results
            save_dir = root_save_dir / f"{tgt_cam_id}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Render in batches novel views from target camera
            for batch in tqdm(loader, desc=f"Evaluation NVS cam {tgt_cam_id}", leave=False):

                # Parse batch data
                frames = batch["image"]             # [B, H, W, 3],
                masks = batch["mask"]               # [B, H, W, 1],
                bsize = int(frames.shape[0])
                batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
                batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]
                gt_h, gt_w = frames.shape[1], frames.shape[2]
                frame_indices = batch["frame_idx"]      # [B]
                frame_names = batch["frame_name"]      # [B]
                frame_paths = batch["frame_path"]      # List[B]

                # Forward pass to render novel views
                renders, _, _ = self.forward(batch)

                # Apply masks to the gt 
                masks3 = masks.repeat(1, 1, 1, 3)
                masked_gt = frames * masks3

                # debug - for difix goes here just in case
                # [Optional] Apply Difix to refine the renders
                if difix_pipe is not None:
                    # - Refine rendered images using Difix
                    # -- Load reference images from previous camera view
                    ref_images = []
                    for i in range(bsize):
                        sample_dict = src_cam_dataset[frame_indices[i].item()]
                        ref_img, ref_mask = sample_dict["image"], sample_dict["mask"]  # [H, W, 3], [H, W, 1], 0-1
                        ref_frame = ref_img * ref_mask # [H, W, 3], 0-1
                        ref_images.append(ref_frame.to(self.tuner_device))
                    ref_images = torch.stack(ref_images, dim=0)  # [B, H, W, 3]
                    # -- Run difix refinement
                    refined_rgb = difix_refine(self.cfg.difix, renders, ref_images, difix_pipe, is_eval=True) # [B, H, W, 3]

                    # - Debug: save side-by-side comparison of reference, rendered, refined
                    difix_debug_dir = save_dir / "difix_debug_comparisons" / f"{tgt_cam_id}"
                    difix_debug_dir.mkdir(parents=True, exist_ok=True)
                    joined = torch.cat([ref_images, renders, refined_rgb, masked_gt], dim=2)  # side-by-side along width
                    for i in range(joined.shape[0]):
                        save_path = difix_debug_dir / f"{frame_names[i]}.png"
                        save_image(joined[i].permute(2, 0, 1), str(save_path))

                    # - Use refined renders for evaluation
                    renders = refined_rgb

                # Combine masked render and masked GT with overlay
                render_vs_gt_dir = save_dir / "render_vs_gt"
                render_vs_gt_dir.mkdir(parents=True, exist_ok=True)
                columns = [renders, masked_gt]
                joined = torch.cat(columns, dim=2)  # side-by-side along width
                for i in range(joined.shape[0]):
                    save_path = render_vs_gt_dir / Path(frame_paths[i]).name
                    save_image(joined[i].permute(2, 0, 1), str(save_path))

                # Before evaluation, apply downsampling if needed
                if self.cfg.nvs_eval.downscale_factor > 1:
                    ds_factor = self.cfg.nvs_eval.downscale_factor
                    renders = F.interpolate(
                        renders.permute(0, 3, 1, 2),
                        size=(gt_h // ds_factor, gt_w // ds_factor),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    frames = F.interpolate(
                        frames.permute(0, 3, 1, 2),
                        size=(gt_h // ds_factor, gt_w // ds_factor),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                    masks = F.interpolate(
                        masks.permute(0, 3, 1, 2),
                        size=(gt_h // ds_factor, gt_w // ds_factor),
                        mode="nearest",
                    ).permute(0, 2, 3, 1)
                    masks3 = masks.repeat(1, 1, 1, 3)
                    frames = frames * masks3
                    
                    masks = torch.ones_like(masks)
                    masks3 = masks.repeat(1, 1, 1, 3)
                
                # Save what goes into metrics computation for debugging
                metrics_debug_dir = save_dir / "metrics_debug_inputs"
                metrics_debug_dir.mkdir(parents=True, exist_ok=True)
                masked_renders = renders * masks3
                masked_gt = frames * masks3
                overlayed = 0.5 * masked_renders + 0.5 * masked_gt
                columns = [masked_renders, masked_gt, overlayed]
                joined = torch.cat(columns, dim=2)  # side-by-side along width
                for i in range(joined.shape[0]):
                    save_path = metrics_debug_dir / Path(frame_paths[i]).name
                    save_image(joined[i].permute(2, 0, 1), str(save_path))
                    
                # Compute metrics
                psnr_vals = psnr(
                    images=frames,
                    masks=masks.squeeze(-1),
                    renders=renders,
                )
                ssim_vals = ssim(
                    images=frames,
                    masks=masks.squeeze(-1),
                    renders=renders,
                )
                lpips_vals = lpips(
                    images=frames,
                    masks=masks.squeeze(-1),
                    renders=renders,
                )
                for fname, psnr_val, ssim_val, lpips_val in zip(frame_names, psnr_vals, ssim_vals, lpips_vals):
                    fid = int(fname)
                    metrics_all_cams_per_frame.append(
                        (tgt_cam_id, fid, psnr_val.item(), ssim_val.item(), lpips_val.item())
                    )
            
            # Create a video from render vs gt images using call to ffmpeg
            render_vs_gt_dir = root_save_dir / f"{tgt_cam_id}" / "render_vs_gt"
            video_path = root_save_dir / f"{tgt_cam_id}" / f"cam_{tgt_cam_id}_nvs_epoch_{epoch:04d}.mp4"
            jpg_frames = sorted(p for p in render_vs_gt_dir.glob("*.jpg") if p.stem.isdigit())
            if not jpg_frames:
                raise FileNotFoundError(f"No render_vs_gt frames found in {render_vs_gt_dir}")
            start_number = int(jpg_frames[0].stem)
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-start_number", str(start_number),
                "-framerate", "20",
                "-i", str(render_vs_gt_dir / "%06d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
 
        # Save metrics to CSV
        # - save per-frame metrics
        df = pd.DataFrame(metrics_all_cams_per_frame, columns=["camera_id", "frame_id", "psnr", "ssim", "lpips"])
        csv_path = root_save_dir / "metrics_all_cams_per_frame.csv"
        df.to_csv(csv_path, index=False)

        # - save average metrics per camera
        df_avg_cam = df.groupby("camera_id").agg({"psnr": "mean", "ssim": "mean", "lpips": "mean"}).reset_index()
        csv_path_avg_cam = root_save_dir / "metrics_avg_per_camera.csv"
        df_avg_cam.to_csv(csv_path_avg_cam, index=False)

        # - log to wandb the per cam metrics
        if self.cfg.wandb.enable:
            for _, row in df_avg_cam.iterrows():
                to_log = {f"eval_nv/cam_{int(row['camera_id'])}/{metric_name}": row[metric_name] for metric_name in ["psnr", "ssim", "lpips"]}
                to_log["epoch"] = epoch
                wandb.log(to_log)

        # - save overall average metrics excluding self.cfg.nvs_eval.source_camera_id
        df_excluding_source = df[df["camera_id"] != self.cfg.nvs_eval.source_camera_id]
        overall_avg = {
            "psnr": df_excluding_source["psnr"].mean(),
            "ssim": df_excluding_source["ssim"].mean(),
            "lpips": df_excluding_source["lpips"].mean(),
        }
        overall_avg_path = root_save_dir / "novel_view_results.txt"
        with open(overall_avg_path, "w") as f:
            for k, v in overall_avg.items():
                f.write(f"{k}: {v:.4f}\n")

        # - log the overall average metrics to wandb
        if self.cfg.wandb.enable:
            to_log = {f"eval_nv/all_cam/{metric_name}": v for metric_name, v in overall_avg.items()}
            to_log["epoch"] = epoch
            wandb.log(to_log)

        # - log the source training view metrics to wandb
        if self.cfg.wandb.enable:
            df_source = df[df["camera_id"] == self.cfg.nvs_eval.source_camera_id]
            if not df_source.empty:
                source_avg = {
                    "psnr": df_source["psnr"].mean(),
                    "ssim": df_source["ssim"].mean(),
                    "lpips": df_source["lpips"].mean(),
                }
                to_log = {f"eval_nv/trn_cam/{metric_name}": v for metric_name, v in source_avg.items()}
                to_log["epoch"] = epoch
                wandb.log(to_log)

        # Delete difix pipe to free memory
        if difix_pipe is not None:
            del difix_pipe
            torch.cuda.empty_cache()

    @torch.no_grad()
    def eval_loop_segmentation(self, epoch): 

        # Init save directory        
        save_dir : Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        tgt_cam_ids = self.cfg.nvs_eval.target_camera_ids
        src_cam_id = self.cfg.nvs_eval.source_camera_id
        metrics_per_frame = []
        for tgt_cam_id in tgt_cam_ids:

            # Init save directory for this cam
            cam_save_dir : Path = save_dir / f"{tgt_cam_id}"
            cam_save_dir.mkdir(parents=True, exist_ok=True)

            # From cam save dir, init segm_metrics_debug dir
            segm_metrics_debug_dir = cam_save_dir / "segm_metrics_debug_inputs"
            segm_metrics_debug_dir.mkdir(parents=True, exist_ok=True)

            # Prepare gt datasets for given camera
            skip_frames = load_skip_frames(self.trn_data_dir)
            # - For the estimated dataset, use same sample_every and skip frames if evaluating on source camera
            if tgt_cam_id == src_cam_id:
                est_sample_every = self.sample_every
                est_skip_frames = skip_frames
            else:
                est_sample_every = 1
                est_skip_frames = []  # no skip frames for test dataset

            est_dataset = SceneDataset(self.trn_data_dir, tgt_cam_id, 
                                       device=self.tuner_device,sample_every=est_sample_every, skip_frames=est_skip_frames)
            # - For the gt dataset, always use the test dataset with original sample_every and skip frames
            gt_dataset = SceneDataset(self.test_data_dir, tgt_cam_id, 
                                       device=self.tuner_device, sample_every=self.sample_every, skip_frames=skip_frames)

            # Prepare dataloader
            loader = DataLoader(
                est_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
            )


            # Evaluate in batches
            for batch in tqdm(loader, desc=f"Segmentation Eval cam {tgt_cam_id}", leave=False):

                # Parse estimated masks
                est_masks = batch["mask"]               # [B, H, W, 1]
                est_frame_indices = batch["frame_idx"]      # [B]
                fnames = batch["frame_name"]      # [B]
                frames = batch["image"]             # [B, H, W, 3]

                # Parse gt masks
                gt_masks = []
                gt_frame_names = []
                for i in range(est_masks.shape[0]):
                    sample_dict = gt_dataset[est_frame_indices[i].item()]
                    gt_mask = sample_dict["mask"]  # [H, W, 1], 0-1
                    gt_frame_names.append(sample_dict["frame_name"])
                    gt_masks.append(gt_mask.to(self.tuner_device))
                gt_masks = torch.stack(gt_masks, dim=0)  # [B, H, W, 1]
                assert list(gt_frame_names) == list(fnames), "Estimated and GT frame names do not match!"

                # Compute metrics
                segm_metrics = segmentation_mask_metrics(
                    gt_masks=gt_masks.squeeze(-1),
                    pred_masks=est_masks.squeeze(-1),
                )
                segm_iou, segm_recall, segm_f1 = segm_metrics["segm_iou"], segm_metrics["segm_recall"], segm_metrics["segm_f1"]
                for fname, iou, recall, f1 in zip(fnames, segm_iou, segm_recall, segm_f1):
                    fid = int(fname)
                    metrics_per_frame.append(
                        (tgt_cam_id, fid, iou.item(), recall.item(), f1.item())
                    )

                # Save debug inputs for this batch
                gt_masked_frames = get_masked_images(frames, gt_masks)
                est_masked_frames = get_masked_images(frames, est_masks)
                gt_est_masks_overlay = get_gt_est_masks_overlay(gt_masks, est_masks) 
                save_segmentation_debug_figures(
                    gt_masked_frames,
                    est_masked_frames,
                    gt_est_masks_overlay,
                    frame_names=fnames,
                    iou=segm_iou,
                    recall=segm_recall,
                    f1=segm_f1,
                    save_dir=segm_metrics_debug_dir,
                )
                

        # Save metrics to CSV
        # - save per-frame metrics
        df = pd.DataFrame(metrics_per_frame, columns=["camera_id", "frame_id", "iou", "recall", "f1"])
        csv_path = save_dir / "segmentation_metrics_per_frame.csv"
        df.to_csv(csv_path, index=False)

        # - save average metrics per camera
        df_avg_cam = df.groupby("camera_id").agg({"iou": "mean", "recall": "mean", "f1": "mean"}).reset_index()
        csv_path_avg_cam = save_dir / "segmentation_metrics_avg_per_camera.csv"
        df_avg_cam.to_csv(csv_path_avg_cam, index=False)

        # - log to wandb the per cam metrics
        if self.cfg.wandb.enable:
            for _, row in df_avg_cam.iterrows():
                to_log = {f"eval_segm/cam_{int(row['camera_id'])}/{metric_name}": row[metric_name] for metric_name in ["iou", "recall", "f1"]}
                to_log["epoch"] = epoch
                wandb.log(to_log)

        # - save average across all cameras except from source camera
        df_excluding_source = df[df["camera_id"] != self.cfg.nvs_eval.source_camera_id]
        overall_avg = {
            "iou": df_excluding_source["iou"].mean(),
            "recall": df_excluding_source["recall"].mean(),
            "f1": df_excluding_source["f1"].mean(),
        }
        overall_avg_path = save_dir / "segmentation_overall_results.txt"
        with open(overall_avg_path, "w") as f:
            for k, v in overall_avg.items():
                f.write(f"{k}: {v:.4f}\n")
        
        # - log the overall average metrics to wandb
        if self.cfg.wandb.enable:
            to_log = {f"eval_segm/all_cam/{metric_name}": v for metric_name, v in overall_avg.items()}
            to_log["epoch"] = epoch
            wandb.log(to_log)

        # - log the source training view metrics to wandb
        if self.cfg.wandb.enable:
            df_source = df[df["camera_id"] == self.cfg.nvs_eval.source_camera_id]
            if not df_source.empty:
                source_avg = {
                    "iou": df_source["iou"].mean(),
                    "recall": df_source["recall"].mean(),
                    "f1": df_source["f1"].mean(),
                }
                to_log = {f"eval_segm/trn_cam/{metric_name}": v for metric_name, v in source_avg.items()}
                to_log["epoch"] = epoch
                wandb.log(to_log)
 
    @torch.no_grad()
    def eval_loop(self, epoch):
        self.eval_loop_nvs(epoch)
        self.eval_loop_segmentation(epoch)


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    os.environ["TORCH_HOME"] = str(cfg.torch_home)
    os.environ["HF_HOME"] = str(cfg.hf_home)
    tuner = MultiHumanTrainer(cfg)
    tuner.train_loop()


if __name__ == "__main__":
    main()
