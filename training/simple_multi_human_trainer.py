import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
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
from torch.utils.data import DataLoader
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
from training.helpers.datasets import Hi4dDataset
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
        self.preprocess_dir = Path(cfg.preprocessing_dir).expanduser()
        self.output_dir = Path(cfg.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_params = tuple(cfg.train_params)
        self.renderer : GS3DRenderer = build_renderer().to(self.tuner_device)
        self.wandb_run = None

        # Prepare dataset
        #self.frames_dir = self.output_dir / "frames"
        #self.masks_dir = self.output_dir / "masks" / "union"
        self.depth_dir = self.preprocess_dir / "depth_maps" / "raw"
        root_gt_dir_path: Path = Path(self.cfg.nvs_eval.root_gt_dir_path)
        src_cam_id: int = self.cfg.nvs_eval.source_camera_id
        self.trn_dataset = Hi4dDataset(root_gt_dir_path, src_cam_id, self.depth_dir, self.tuner_device)

        # Preare model
        self._load_model()

        # Initialize difix
        self._init_difix()

    # ---------------- Model  ------------------------------
    def _load_model(self):

        # Get query points + tranform from neural to zero pose
        self.query_points, self.tranform_mat_neutral_pose = self.renderer.get_query_points(self.trn_dataset.smplx, self.tuner_device)

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

        # Initialize wandb if enabled and not already initialized
        if self.wandb_run is None:
            self._init_wandb()

        # Prepare dataset and dataloader
        self.trn_render_hw = self.trn_dataset.trn_render_hw
        loader = DataLoader(
            self.trn_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=0, drop_last=False
        )

        # Initialize optimizer
        params = self._trainable_tensors()
        optimizer = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Pre-optimization visualization (epoch 0).
        if self.cfg.eval_pretrain:
            self.eval_loop(0)

        # Training loop
        for epoch in range(self.cfg.epochs):
            running_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.cfg.epochs}", leave=False)
            batch_idx = 0
            for batch in pbar:

                # Parse batch data
                frames = batch["image"] # [B, H, W, 3],
                masks = batch["mask"] # [B, H, W, 1],
                depths = batch["depth"] # [B, H, W, 1],
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
                gt_depth_masked = depths * masks

                # BBOX crop around the union person mask
                if self.cfg.use_bbox_crop:
                    bboxes = get_masks_based_bbox(masks, pad=self.cfg.bbox_pad)
                    loss_pred_rgb = bbox_crop(bboxes, pred_rgb)
                    loss_pred_mask = bbox_crop(bboxes, pred_mask)
                    loss_gt_masked = bbox_crop(bboxes, gt_masked)
                    loss_pred_depth = bbox_crop(bboxes, pred_depth)
                    loss_gt_depth_masked = bbox_crop(bboxes, gt_depth_masked)
                    loss_masks = bbox_crop(bboxes, masks)
                else:
                    loss_pred_rgb = pred_rgb
                    loss_pred_mask = pred_mask
                    loss_gt_masked = gt_masked
                    loss_pred_depth = pred_depth
                    loss_gt_depth_masked = gt_depth_masked
                    loss_masks = masks

                # Compute loss
                rgb_loss = self.cfg.loss_weights["rgb"] * F.mse_loss(loss_pred_rgb, loss_gt_masked)
                sil_loss = self.cfg.loss_weights["sil"] * F.mse_loss(loss_pred_mask, loss_masks)
                depth_loss = self.cfg.loss_weights["depth"] * F.mse_loss(loss_pred_depth, loss_gt_depth_masked)
                ssim_val = fused_ssim(_ensure_nchw(loss_pred_rgb), _ensure_nchw(loss_gt_masked), padding="valid")
                ssim_loss = self.cfg.loss_weights["ssim"] * (1.0 - ssim_val)
                asap_loss, acap_loss = self._canonical_regularization()
                reg_loss = asap_loss + acap_loss

                loss = rgb_loss + sil_loss + ssim_loss + depth_loss + reg_loss
                
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
                            "loss/depth": depth_loss.item(),
                            "loss/reg": reg_loss.item(),
                            "loss/asap": asap_loss.item(),
                            "loss/acap": acap_loss.item(),
                        }
                    )

                # space for debug stuff is here
                if batch_idx == 5 and (epoch + 1) % 2 == 0:
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

                    # - save depth comparison images
                    debug_save_dir = self.output_dir / "debug" / self.cfg.exp_name / f"epoch_{epoch+1:04d}" / "depth"
                    debug_save_dir.mkdir(parents=True, exist_ok=True)
                    for i in range(pred_depth.shape[0]):
                        global_idx = self.cfg.sample_every * (batch_idx * self.cfg.batch_size + i)
                        save_path = debug_save_dir / f"depth_comparison_frame_{global_idx:06d}.png"
                        save_depth_comparison(pred_depth[i].squeeze(-1), gt_depth_masked[i].squeeze(-1), str(save_path))

                batch_idx += 1

            # End of epoch
            # - Report average loss
            avg_loss = running_loss / max(1, len(loader))
            if self.wandb_run is not None:
                wandb.log({"loss/combined_epoch": avg_loss, "epoch": epoch + 1})

            # - Run eval loop if neccesary
            if getattr(self.cfg, "eval_every_epoch", 0) > 0 and (epoch + 1) % self.cfg.eval_every_epoch == 0:
                self.eval_loop(epoch + 1)
            
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
        print("Difix NVS camera traversal initialized:")
        print(f"    Left traversed cams: {self.from_src_left_traversed_cams}")
        print(f"    Right traversed cams: {self.from_src_right_traversed_cams}")


    @torch.no_grad()
    def difix_refine(self):

        # Select next left/right cameras to process (if any left)
        previous_cams = []
        new_cams = []
        # - Left (only if any left)
        if len(self.from_src_left_traversed_cams) > 0: 
            prev_left_cam_id = self.left_cam_id
            self.left_cam_id = self.from_src_left_traversed_cams.popleft() 
            new_cams.append(self.left_cam_id)
            previous_cams.append(prev_left_cam_id)

        # - Right (only if any left)
        if len(self.from_src_right_traversed_cams) > 0:
            prev_right_cam_id = self.right_cam_id
            self.right_cam_id = self.from_src_right_traversed_cams.popleft()
            new_cams.append(self.right_cam_id)
            previous_cams.append(prev_right_cam_id)

        # Prepare rendering inputs
        root_gt_dir_path: Path = Path(self.cfg.nvs_eval.root_gt_dir_path)


        # Process each new camera:
        # 1. render the new camera views (out: render_frames + alpha_masks)
        # 2. load the previous refined frames (or source frames for the first cameras) (out: ref_frames)
        for cam_id, prev_cam_id in zip(new_cams, previous_cams):
            pass
    
    def difix_eval_refine(self, input_data, difix_pipe: DifixPipeline, save_dir: Path, ):

        # Parse input data
        renders, masked_gt, frame_indices = input_data  # renders: [B, H, W, 3], masked_gt: [B, H, W, 3], frame_indices: [B]

        # - Run the refinement
        before_refinement_renders = renders.clone()
        refined_renders = []
        ref_images = []
        for i in tqdm(range(renders.shape[0]), desc="Difix refinement", total=renders.shape[0], leave=False):
            # -- Img to refine = rendered image
            img_to_refine = (renders[i].cpu().numpy() * 255).astype("uint8")
            # -- Reference image = src view gt image + mask
            sample_dict = self.trn_dataset[frame_indices[i].item()]
            src_frame = sample_dict["image"]  # [H, W, 3], 0-1
            src_mask = sample_dict["mask"]    # [H, W, 1], 0-1 
            src_frame_np = (src_frame.cpu().numpy() * 255).astype("uint8")
            src_mask_np = src_mask.cpu().numpy().astype(np.float32)  # [H, W, 1]
            reference_image = (src_frame_np.astype(np.float32) * src_mask_np).clip(0, 255).astype("uint8")
            # -- Run Difix
            refined_image = difix_pipe(
                self.cfg.difix.prompt,
                image=Image.fromarray(img_to_refine),
                ref_image=Image.fromarray(reference_image),
                num_inference_steps=self.cfg.difix.num_inference_steps,
                timesteps=self.cfg.difix.timesteps,
                guidance_scale=self.cfg.difix.guidance_scale,
            ).images[0]
            # -- Collect results
            refined_image = torch.from_numpy(np.array(refined_image)).float() / 255.
            refined_renders.append(refined_image.to(self.tuner_device))
            reference_image = torch.from_numpy(reference_image).float() / 255.
            ref_images.append(reference_image.to(self.tuner_device))

        # - Stack refined renders and reference images
        renders = torch.stack(refined_renders, dim=0)
        ref_images = torch.stack(ref_images, dim=0)

        # - Save before vs after refinement images for debugging
        difix_debug_dir = save_dir / "difix_debug"
        difix_debug_dir.mkdir(parents=True, exist_ok=True)
        joined = torch.cat([ref_images, before_refinement_renders, renders, masked_gt], dim=2)  # side-by-side along width
        for i in range(joined.shape[0]):
            save_path = difix_debug_dir / f"{frame_indices[i].item():06d}.png"
            save_image(joined[i].permute(2, 0, 1), str(save_path))

        return renders

    # ---------------- Evaluation -------------------
    @torch.no_grad()
    def eval_loop(self, epoch):

        # Parse the evaluation setup
        target_camera_ids: List[int] = self.cfg.nvs_eval.target_camera_ids
        root_gt_dir_path: Path = Path(self.cfg.nvs_eval.root_gt_dir_path)
        root_save_dir: Path = self.output_dir / "evaluation" / self.cfg.exp_name / f"epoch_{epoch:04d}"
        root_save_dir.mkdir(parents=True, exist_ok=True)

        # Init Difix if enabled for the evaluation
        if self.cfg.difix.eval_enable:
            difix_pipe = DifixPipeline.from_pretrained(self.cfg.difix.model_id, trust_remote_code=True, requires_safety_checker=False)
            difix_pipe.to(self.tuner_device)
            difix_pipe.set_progress_bar_config(disable=True)
        else:
            difix_pipe = None

        # Collector for metrics
        metrics_all_cams_per_frame = list()

        for tgt_cam_id in target_camera_ids:


            val_dataset = Hi4dDataset(root_gt_dir_path, tgt_cam_id, device=self.tuner_device, depth_dir=None)
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
                frame_paths = batch["frame_path"]      # List[B]

                # Forward pass to render novel views
                renders, _, _ = self.forward(batch)

                # Apply masks to the gt 
                masks3 = masks.repeat(1, 1, 1, 3)
                masked_gt = frames * masks3

                # debug - for difix goes here just in case
                # [Optional] Apply Difix to refine the renders
                if difix_pipe is not None:
                    renders = self.difix_eval_refine((renders, masked_gt, frame_indices), difix_pipe, save_dir)

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
                for fid, psnr_val, ssim_val, lpips_val in zip(frame_indices, psnr_vals, ssim_vals, lpips_vals):
                    metrics_all_cams_per_frame.append(
                        (tgt_cam_id, fid.item(), psnr_val.item(), ssim_val.item(), lpips_val.item())
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
    main()
