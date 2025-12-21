import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from omegaconf import DictConfig

# Silence noisy deprecation notices from upstream libs to keep logs readable.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import hydra
import numpy as np
from PIL import Image
import kornia

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, ConcatDataset

import torch.nn.functional as F

from fused_ssim import fused_ssim
import pyiqa
import wandb

from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "submodules", "lhm")
    ),
)
from training.helpers.gs_renderer import GS3DRenderer
from training.helpers.dataset import SceneDataset, fetch_data_if_available, root_dir_to_image_dir, root_dir_to_mask_dir
from training.helpers.debug import overlay_smplx_mesh_pyrender, save_depth_comparison

from submodules.difix3d.src.pipeline_difix import DifixPipeline

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

# Cached LPIPS metric instance; built lazily on first use.
_LPIPS_METRIC: Optional[torch.nn.Module] = None


def _get_lpips_net(device: torch.device) -> torch.nn.Module:
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None:
        # Spatial LPIPS gives a per-pixel distance map (pyiqa handles input normalisation to [-1,1])
        _LPIPS_METRIC = pyiqa.create_metric(
            "lpips", device=device, net="vgg", spatial=True, as_loss=False
        ).eval()
    return _LPIPS_METRIC.to(device)


def _ensure_nchw(t: torch.Tensor) -> torch.Tensor:
    """Convert tensors from NHWC (renderer output) to NCHW for library calls."""
    return t.permute(0, 3, 1, 2).contiguous()


def _mask_sums(mask: torch.Tensor) -> torch.Tensor:
    """Sum mask activations per sample (expects shape [B,1,H,W])."""
    return mask.sum(dim=(2, 3))


def ssim(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked SSIM per sample.
    
    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]
    Returns: 
        ssim_vals: [B] masked SSIM values 
    """
    target = _ensure_nchw(images.float())
    preds = _ensure_nchw(renders.float())
    mask = masks.unsqueeze(1).float()

    # Kornia returns per-channel SSIM; average across channels before masking
    ssim_map = kornia.metrics.ssim(preds, target, window_size=11, max_val=1.0)
    ssim_map = ssim_map.mean(1, keepdim=True)

    # Reduce over the masked region for each batch element
    numerator = (ssim_map * mask).sum(dim=(2, 3))
    mask_sum = _mask_sums(mask)
    safe_mask_sum = mask_sum.clamp_min(1e-6)
    result = numerator / safe_mask_sum
    result = torch.where(mask_sum < 1e-5, torch.zeros_like(result), result)
    return result.squeeze(1)


def psnr(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Masked PSNR per sample.

    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]

    Returns:
        psnr_vals: [B] masked PSNR values 
    """
    target = images.float()
    preds = renders.float()
    mask = masks.unsqueeze(-1).float()

    diff2 = (preds - target) ** 2
    masked_diff2 = diff2 * mask
    # Compute masked MSE then convert to PSNR
    numerator = masked_diff2.sum(dim=(1, 2, 3))
    denom = mask.sum(dim=(1, 2, 3))
    safe_denom = denom.clamp_min(1e-6)
    mse = numerator / safe_denom
    mse = mse.clamp_min(1e-12)
    psnr_vals = 10.0 * torch.log10((max_val ** 2) / mse)
    psnr_vals = torch.where(denom < 1e-5, torch.zeros_like(psnr_vals), psnr_vals)
    return psnr_vals


def lpips(images: torch.Tensor, masks: torch.Tensor, renders: torch.Tensor) -> torch.Tensor:
    """Masked spatial LPIPS per sample using pyiqa.


    Args:
        images: [B,H,W,C] ground truth images in [0,1] 
        masks: [B,H,W] binary masks in [0,1]
        renders: [B,H,W,C] rendered images in [0,1]
    Returns:
        lpips_vals: [B] masked LPIPS values 
    """
    target = _ensure_nchw(images.float()).clamp(0.0, 1.0)
    preds = _ensure_nchw(renders.float()).clamp(0.0, 1.0)
    mask = masks.unsqueeze(1).float()

    net = _get_lpips_net(preds.device)
    with torch.no_grad():
        dmap = net(preds, target)

    # Match mask resolution to the LPIPS map and average within the mask
    if dmap.shape[-2:] != mask.shape[-2:]:
        mask_resized = F.interpolate(mask, size=dmap.shape[-2:], mode="nearest")
    else:
        mask_resized = mask

    numerator = (dmap * mask_resized).sum(dim=(1, 2, 3))
    denom = mask_resized.sum(dim=(1, 2, 3))
    safe_denom = denom.clamp_min(1e-6)
    lpips_vals = numerator / safe_denom
    lpips_vals = torch.where(denom < 1e-5, torch.zeros_like(lpips_vals), lpips_vals)
    return lpips_vals



# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def root_dir_to_difix_debug_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "difix_debug_comparisons" / f"{cam_id}"

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
        self._init_difix()
        is_difix_done = False
        n_its_difix = 0
        while not is_difix_done:
            is_difix_done = self.difix_step()
            n_its_difix += 1
            print(" Difix step ", n_its_difix, " done: ", is_difix_done)


        for cam_id in self.difix_cam_ids_all:
            scene_ds = SceneDataset(
                self.trn_data_dir,
                int(cam_id),
                depth_dir=None,
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
            Path(self.cfg.depths_scene_dir) if self.cfg.depths_scene_dir is not None else None,
        )

        # Create training dataset
        self.trn_datasets = list() 
        trn_ds = SceneDataset(
            self.trn_data_dir, 
            src_cam_id, 
            depth_dir=None, 
            device=self.tuner_device, 
            sample_every=self.sample_every,
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

    @staticmethod
    def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
        """Convert a [H,W,C] tensor in [0,1] to uint8 numpy."""
        return (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")

    @staticmethod
    def _pad_image_to_multiple(image: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        pad_h = (multiple - (image.shape[0] % multiple)) % multiple
        pad_w = (multiple - (image.shape[1] % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return image, (0, 0)
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        return padded, (pad_h, pad_w)

    def _prepare_image_for_difix(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        multiple = int(getattr(self.cfg.difix, "resolution_multiple", 8))
        multiple = max(1, multiple)
        metadata: Dict[str, Any] = {
            "orig_hw": image.shape[:2],
        }
        processed, pad_hw = self._pad_image_to_multiple(image, multiple)
        metadata["pad_hw"] = pad_hw
        metadata["processed_hw"] = processed.shape[:2]
        return processed, metadata

    def _restore_image_from_difix(self, image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        orig_h, orig_w = metadata.get("orig_hw", image.shape[:2])
        return image[:orig_h, :orig_w, :]

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
                frames = batch["image"] # [B, H, W, 3],
                masks = batch["mask"] # [B, H, W, 1],
                # depths = batch["depth"] # [B, H, W, 1],
                bsize = int(frames.shape[0])
                batched_neutral_pose_transform = self.tranform_mat_neutral_pose.unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
                batch["smplx_params"]["transform_mat_neutral_pose"] = batched_neutral_pose_transform # [B, P, 55, 4, 4]

                # Reset gradients
                optimizer.zero_grad(set_to_none=True)

                # Forward pass
                pred_rgb, pred_mask, pred_depth = self.forward(batch)

                # Compute masked ground truth
                mask3 = masks
                if mask3.shape[-1] == 1:
                    mask3 = mask3.repeat(1, 1, 1, 3)
                gt_masked = frames * mask3
                # gt_depth_masked = depths * masks

                # BBOX crop around the union person mask
                if self.cfg.use_bbox_crop:
                    bboxes = get_masks_based_bbox(masks, pad=self.cfg.bbox_pad)
                    loss_pred_rgb = bbox_crop(bboxes, pred_rgb)
                    loss_pred_mask = bbox_crop(bboxes, pred_mask)
                    loss_gt_masked = bbox_crop(bboxes, gt_masked)
                    # loss_pred_depth = bbox_crop(bboxes, pred_depth)
                    # loss_gt_depth_masked = bbox_crop(bboxes, gt_depth_masked)
                    loss_masks = bbox_crop(bboxes, masks)
                else:
                    loss_pred_rgb = pred_rgb
                    loss_pred_mask = pred_mask
                    loss_gt_masked = gt_masked
                    # loss_pred_depth = pred_depth
                    # loss_gt_depth_masked = gt_depth_masked
                    loss_masks = masks

                # Compute loss
                rgb_loss = self.cfg.loss_weights["rgb"] * F.mse_loss(loss_pred_rgb, loss_gt_masked)
                sil_loss = self.cfg.loss_weights["sil"] * F.mse_loss(loss_pred_mask, loss_masks)
                # depth_loss = self.cfg.loss_weights["depth"] * F.mse_loss(loss_pred_depth, loss_gt_depth_masked)
                ssim_val = fused_ssim(_ensure_nchw(loss_pred_rgb), _ensure_nchw(loss_gt_masked), padding="valid")
                ssim_loss = self.cfg.loss_weights["ssim"] * (1.0 - ssim_val)
                asap_loss, acap_loss = self._canonical_regularization()
                reg_loss = asap_loss + acap_loss

                loss = rgb_loss + sil_loss + ssim_loss + reg_loss # + depth_loss
                
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
                )

                if self.wandb_run is not None:
                    wandb.log(
                        {
                            "loss/combined": loss.item(),
                            "loss/rgb": rgb_loss.item(),
                            "loss/sil": sil_loss.item(),
                            "loss/ssim": ssim_loss.item(),
                            # "loss/depth": depth_loss.item(),
                            "loss/reg": reg_loss.item(),
                            "loss/asap": asap_loss.item(),
                            "loss/acap": acap_loss.item(),
                            "epoch": epoch + 1,
                        }
                    )

                # space for debug stuff is here
                if batch_idx in [5] and (epoch + 1) % 2 == 0:
                    # - create a joined image from pred_masked and gt_masked for debugging
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name / f"epoch_{epoch+1:04d}" / "rgb"
                    debug_save_dir.mkdir(parents=True, exist_ok=True)
                    overlay = pred_rgb*0.5 + gt_masked*0.5
                    joined_image = torch.cat([pred_rgb, gt_masked, overlay], dim=3)  # Concatenate along width
                    for i in range(joined_image.shape[0]):
                        image = joined_image[i:i+1]
                        global_idx = batch_idx * self.cfg.batch_size + i
                        debug_image_path = debug_save_dir / f"rgb_loss_input_{global_idx}.png"
                        save_image(image.permute(0, 3, 1, 2), str(debug_image_path))

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

    # ---------------- Difix -------------------
    def _init_difix(self):
        src_cam_id = self.cfg.nvs_eval.source_camera_id
        self.left_cam_id, self.right_cam_id = src_cam_id, src_cam_id
        self.camera_ids = [4, 16, 28, 40, 52, 64, 76, 88]
        src_cam_idx = self.camera_ids.index(src_cam_id)
        half_n = (len(self.camera_ids) - 1) // 2
        other_half_n = len(self.camera_ids) - 1 - half_n

        self.from_src_left_traversed_cams = deque((self.camera_ids[src_cam_idx-1::-1] + self.camera_ids[:src_cam_idx:-1])[:half_n])
        self.from_src_right_traversed_cams = deque((self.camera_ids[src_cam_idx+1:] + self.camera_ids[:src_cam_idx])[:other_half_n])
        self.difix_cam_ids_all = list(self.from_src_left_traversed_cams) + list(self.from_src_right_traversed_cams)
        print("Difix NVS camera traversal initialized:")
        print(f"    Left traversed cams: {self.from_src_left_traversed_cams}")
        print(f"    Right traversed cams: {self.from_src_right_traversed_cams}")

    @torch.no_grad()
    def _difix_prepare_new_view_ds(self, difix_pipe: DifixPipeline, cam_id: int, prev_cam_id: int):
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
        # - we just started, so previous cam dataset is the original GT dataset for both left/right
        elif self.left_cam_id == self.right_cam_id:
            sample_every = self.sample_every
        # - otherwise, previous cam dataset is the refined dataset from last Difix step
        else:
            sample_every = 1  # use all frames from the refined dataset (they have been already subsampled)
        # - finally, load the previous cam dataset
        prev_cam_dataset = SceneDataset(self.trn_data_dir, prev_cam_id, device=self.tuner_device, depth_dir=None, sample_every=sample_every)


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
        new_cam_dataset = SceneDataset(self.trn_data_dir, cam_id, device=self.tuner_device, depth_dir=None, sample_every=new_cam_sample_every)
        new_cam_loader = DataLoader(
            new_cam_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # Render in batches novel views from target camera
        for batch in tqdm(new_cam_loader, desc=f"Difix Refinement for cam {cam_id}", leave=False):

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
            is_enabled = self.cfg.difix.trn_enable
            refined_rgb = self.difix_refine(pred_rgb, ref_images, difix_pipe, is_enabled) # [B, H, W, 3]
            # -- Save refined frames
            for i in range(refined_rgb.shape[0]):
                save_path = refined_frames_save_dir / f"{frame_names[i]}.jpg"
                save_image(refined_rgb[i].permute(2, 0, 1), str(save_path))

            # - Compute masks from refined rgb (no alpha)
            if self.cfg.use_estimated_masks:
                # refined_rgb is assumed in [0, 1]
                eps = 10.0 / 255.0   
                binary_masks = (refined_rgb > eps).any(dim=-1, keepdim=True).float()
                masks_save_dir = root_dir_to_mask_dir(self.trn_data_dir, cam_id)
                # -- Save binary masks
                for i in range(binary_masks.shape[0]):
                    save_path = masks_save_dir / f"{frame_names[i]}.png"
                    save_binary_mask(binary_masks[i].permute(2, 0, 1), str(save_path))

            # - Debug: save side-by-side comparison of reference, rendered, refined
            difix_debug_dir = root_dir_to_difix_debug_dir(self.trn_data_dir, cam_id)
            difix_debug_dir.mkdir(parents=True, exist_ok=True)
            joined = torch.cat([ref_images, pred_rgb, refined_rgb], dim=2)  # side-by-side along width
            for i in range(joined.shape[0]):
                save_path = difix_debug_dir / f"{frame_names[i]}.png"
                save_image(joined[i].permute(2, 0, 1), str(save_path))

        # TODO: Compute depth maps for the refined dataset and save them

    @torch.no_grad()
    def difix_step(self) -> bool:

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

        # Initialize Difix pipeline
        difix_pipe = DifixPipeline.from_pretrained(
            self.cfg.difix.model_id, 
            trust_remote_code=True, 
            requires_safety_checker=False,
            num_views=2,
        )
        difix_pipe.to(self.tuner_device)
        difix_pipe.set_progress_bar_config(disable=True)

        # Process each new camera
        for cam_id, prev_cam_id in zip(new_cams, previous_cams):
            self._difix_prepare_new_view_ds(
                difix_pipe,
                cam_id,
                prev_cam_id,
            )

        # Update the left and right previous cam ids
        self.left_cam_id = new_left_cam_id
        if left_changed:
            print(f"Difix updated NV left cam: {prev_left_cam_id} -> {self.left_cam_id}")
        self.right_cam_id = new_right_cam_id
        if right_changed:
            print(f"Difix updated NV right cam: {prev_right_cam_id} -> {self.right_cam_id}")

        # Delete difix pipe to free memory
        del difix_pipe
        torch.cuda.empty_cache()

        return False

    @torch.no_grad()
    def difix_refine(self, renders: torch.Tensor, reference_images: torch.Tensor, difix_pipe: DifixPipeline, enabled: bool = True, is_eval=False) -> torch.Tensor:
        """Refine rendered images using Difix with reference images.
        Args:
            renders: [B, H, W, 3] rendered images to refine
            reference_images: [B, H, W, 3] reference images for refinement
            difix_pipe: DifixPipeline instance
        Returns:
            refined_renders: [B, H, W, 3] refined rendered images
        """
        if not enabled:
            return renders

        # - Run the refinement
        refined_renders = []
        for i in tqdm(range(renders.shape[0]), desc="Difix refinement", total=renders.shape[0], leave=False):
            # -- Img to refine
            img_to_refine = self._tensor_to_uint8(renders[i])
            # -- Reference image
            reference_image = self._tensor_to_uint8(reference_images[i])

            img_prepared, transform_meta = self._prepare_image_for_difix(img_to_refine)
            ref_prepared, _ = self._prepare_image_for_difix(reference_image)

            # -- Run Difix
            if not is_eval:
                refined_image = difix_pipe(
                    self.cfg.difix.prompt,
                    image=Image.fromarray(img_prepared),
                    ref_image=Image.fromarray(ref_prepared),
                    height=img_prepared.shape[0],
                    width=img_prepared.shape[1],
                    num_inference_steps=self.cfg.difix.num_inference_steps,
                    timesteps=self.cfg.difix.timesteps,
                    guidance_scale=self.cfg.difix.guidance_scale,
                    negative_prompt=self.cfg.difix.negative_prompt,
                ).images[0]
            # (the only difference in eval is that we do not provide ref image)
            else:
                refined_image = difix_pipe(
                    self.cfg.difix.prompt,
                    image=Image.fromarray(img_prepared),
                    height=img_prepared.shape[0],
                    width=img_prepared.shape[1],
                    num_inference_steps=self.cfg.difix.num_inference_steps,
                    timesteps=self.cfg.difix.timesteps,
                    guidance_scale=self.cfg.difix.guidance_scale,
                    negative_prompt=self.cfg.difix.negative_prompt,
                ).images[0]

            # -- Collect results
            refined_np = np.array(refined_image)
            refined_np = self._restore_image_from_difix(refined_np, transform_meta)
            refined_tensor = torch.from_numpy(refined_np).float() / 255.0
            assert (
                refined_tensor.shape == renders[i].shape
            ), "Refined image has different shape than the original render after restoration."
            refined_renders.append(refined_tensor.to(self.tuner_device))

        # - Stack refined renders
        refined_renders = torch.stack(refined_renders, dim=0) # [B, H, W, 3]

        return refined_renders
 

    # ---------------- Evaluation -------------------
    @torch.no_grad()
    def eval_loop(self, epoch):

        # Parse the evaluation setup
        target_camera_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        root_save_dir: Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        root_save_dir.mkdir(parents=True, exist_ok=True)

        # Init source camera dataset for Difix reference images
        src_cam_id: int = self.cfg.nvs_eval.source_camera_id
        src_cam_dataset = SceneDataset(self.test_data_dir, src_cam_id, device=self.tuner_device, depth_dir=None)

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
            val_dataset = SceneDataset(self.test_data_dir, tgt_cam_id, device=self.tuner_device, depth_dir=None)
            loader = DataLoader(
                val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
            )

            # Prepare paths where to save results
            save_dir = root_save_dir / f"{tgt_cam_id}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Render in batches novel views from target camera
            for batch in tqdm(loader, desc=f"NVS cam {tgt_cam_id}", leave=False):

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
                    refined_rgb = self.difix_refine(renders, ref_images, difix_pipe, is_eval=True) # [B, H, W, 3]

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


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    os.environ["TORCH_HOME"] = str(cfg.torch_home)
    os.environ["HF_HOME"] = str(cfg.hf_home)
    tuner = MultiHumanTrainer(cfg)
    tuner.train_loop()


if __name__ == "__main__":
    # DataLoader workers need the spawn context to safely use CUDA tensors.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
