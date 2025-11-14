import hydra
import os
import sys
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

from pathlib import Path
import shutil
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from tqdm import tqdm

from PIL import Image

import math
from training.helpers.trainer_init import init_logging
from training.helpers.smpl_utils import update_skinning_weights, filter_dynamic_splats, get_joints_from_pose_params
from training.helpers.dataset import Hi4DDataset, build_training_dataset, build_dataloader, build_evaluation_dataset
from training.helpers.model_init import (
    SceneSplats,
    create_splats_with_optimizers,
    init_trainable_smpl_params,
)
from training.helpers.render import render_splats
from training.helpers.losses import prepare_input_for_loss
from training.helpers.checkpointing import ModelCheckpointManager
from training.helpers.progressive_sam import ProgressiveSAMManager
from training.helpers.visualisation_utils import colourise_depth, save_orbit_visualization, save_epoch_smpl_overlays, save_smpl_overlay_image 
from training.helpers.evaluation_metrics import (
    compute_all_metrics, 
    aggregate_batch_tid_metric_dicts, 
    aggregate_global_tids_metric_dicts
)

from fused_ssim import fused_ssim


class Trainer:
    def __init__(self, cfg: DictConfig, internal_run_id: str = None):
        # Initial param setup
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.current_epoch = 0
        self.is_smpl_optim_enabled = self.current_epoch < cfg.pose_opt_end_epoch
        # Cache pose optimisation schedule so we can decide which optimiser to run each step
        self.debug_vis_freq = int(cfg.get("debug_vis_freq", 0))
        self.pose_correction_epoch = int(cfg.pose_correction_epoch)
        self.pose_opt_start_epoch = int(cfg.pose_opt_start_epoch)
        self.pose_opt_end_epoch = int(cfg.pose_opt_end_epoch)
        self.pose_opt_interval = int(cfg.pose_opt_interval)
        self.pose_opt_duration = int(cfg.pose_opt_duration)
        self.pose_lr_scale = float(cfg.pose_lr_scale)
        depth_cfg = cfg.depth_loss
        self.depth_loss_start_epoch = int(depth_cfg.start_epoch)
        self.depth_loss_end_epoch = int(depth_cfg.end_epoch)
        self.depth_loss_interval = int(depth_cfg.interval)
        self.depth_loss_duration = int(depth_cfg.duration)
        self.depth_order_weight = float(depth_cfg.depth_order_weight)
        self.interpenetration_weight = float(depth_cfg.interpenetration_weight)
        self.depth_normalise_by_pixels = bool(depth_cfg.normalise_by_pixels)
        self.depth_decay_milestone = max(1, int(depth_cfg.decay_milestone))
        self.depth_iterations = 0
        self._previous_smpl_params: Dict[int, torch.Tensor] = {}
        print(f"--- FYI: using device {self.device}")

        # Checkpoint manager
        self.ckpt_manager = ModelCheckpointManager(
            Path(cfg.output_dir),
            cfg.group_name,
            cfg.tids,
        )
        self.checkpoint_dir = self.ckpt_manager.root
        self.baseline_vis_dir = Path(self.cfg.preprocess_dir) / "baseline_smpl_vis"

        if not cfg.resume:
            reset_static = bool(cfg.train_bg)
            reset_tids = list(cfg.tids) if len(cfg.tids) > 0 else []
            self.ckpt_manager.reset(reset_static=reset_static, reset_tids=reset_tids)

        # Setup experiment dirs
        if internal_run_id is not None:
            run_name, run_id = internal_run_id.split("_")
            self.experiment_dir = Path(cfg.train_dir) /  f"{run_name}_{run_id}"
        else:
            self.experiment_dir = Path(cfg.train_dir) / f"{wandb.run.name}_{wandb.run.id}"
        print(f"--- FYI: experiment output dir: {self.experiment_dir}")

        # Load dataset and create dataloader
        mask_path = self.checkpoint_dir / "progressive_sam"
        self.dataset = build_training_dataset(cfg, mask_path=mask_path)
        self._orbit_debug_frame = len(self.dataset) // 2 if len(self.dataset) > 0 else None

        if len(self.dataset) > 0:
            self.orbit_reference_w2c = self.dataset.pose_all[0].clone()
        else:
            self.orbit_reference_w2c = torch.eye(4)

        print(f"--- FYI: dataset has {len(self.dataset)} samples and using batch size 1")

        # Define 3dgs model of the scene and optimizers
        self.all_gs, self.all_optimisers, self.all_strategies = create_splats_with_optimizers(
            self.device, cfg, self.dataset, checkpoint_manager=self.ckpt_manager
        )
        if len(cfg.tids) > 0:
            with torch.no_grad():
                updated_weights = update_skinning_weights(
                    self.all_gs,
                    k=self.cfg.lbs_knn,
                    eps=1e-6,
                    device=self.device,
                )
            self.lbs_weights = [w.detach() for w in updated_weights]
            print(f"--- FYI: in total have {len(self.lbs_weights)} sets of skinning weights (one per dynamic splat set)")
        else:
            self.lbs_weights = None

        # Initialize Progressive SAM manager
        self.progressive_sam = ProgressiveSAMManager(
            mask_cfg=cfg.mask_refinement,
            tids=list(cfg.tids),
            device=self.device,
            default_lbs_knn=int(cfg.lbs_knn),
            checkpoint_dir=self.checkpoint_dir,
            preprocessing_dir=Path(cfg.preprocess_dir),
            is_preprocessing=cfg.is_preprocessing, 
            training_dir=self.experiment_dir,
        )
        self.progressive_sam.init_state(
            cfg.resume, self.dataset, 
            self.all_gs, self.lbs_weights, self.current_epoch
        )

        # Create dataloader
        self.loader = build_dataloader(cfg, self.dataset)

        # Define trainable SMPL parameters and optimizers
        self.smpl_params, smpl_param_list = init_trainable_smpl_params(
            self.dataset, cfg, self.device, checkpoint_manager=self.ckpt_manager
        )
        if smpl_param_list:
            # One optimiser continues updating all parameters, the other is restricted to SMPL-only passes
            self.smpl_joint_optimizer = optim.Adam(smpl_param_list, lr=self.cfg.smpl_lr)
            self.smpl_pose_optimizer = optim.Adam(
                smpl_param_list, lr=self.cfg.smpl_lr * self.pose_lr_scale
            )
            self.smpl_joint_optimizer.zero_grad(set_to_none=True)
            self.smpl_pose_optimizer.zero_grad(set_to_none=True)
        else:
            self.smpl_joint_optimizer = None
            self.smpl_pose_optimizer = None

        if self.cfg.save_pose_overlays_every_epoch > 0 and not self.cfg.is_preprocessing:
            self._ensure_baseline_visualisations()
            self._copy_baseline_to_experiment()

    def _parse_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)  # [B,H,W,3]
        human_masks = batch["human_mask"].to(self.device)  # [B,P,H,W]
        K = batch["K"].to(self.device)           # [B,3,3]
        w2c = batch["M_ext"].to(self.device)     # [B,4,4]
        H, W = batch["image"].shape[1:3]
        assert images.shape[0] == 1, "Mask refinement currently expects batch size 1."
        fid_tensor = batch["fid"]
        fid = int(fid_tensor.item()) if torch.is_tensor(fid_tensor) else int(fid_tensor)

        return images, human_masks, K, w2c, H, W, fid

    def _in_pose_only_window(self) -> bool:
        """Return True when the schedule requests SMPL-only optimisation."""
        if self.pose_opt_end_epoch <= self.pose_opt_start_epoch:
            return False
        if self.current_epoch < self.pose_opt_start_epoch or self.current_epoch >= self.pose_opt_end_epoch:
            return False
        interval = self.pose_opt_interval if self.pose_opt_interval > 0 else 1
        duration = max(1, self.pose_opt_duration)
        position = (self.current_epoch - self.pose_opt_start_epoch) % interval
        return position < duration

    def _in_depth_loss_window(self) -> bool:
        """Return True during epochs where we should run the depth-only step."""
        if self.depth_loss_end_epoch <= self.depth_loss_start_epoch:
            return False
        if self.current_epoch < self.depth_loss_start_epoch or self.current_epoch >= self.depth_loss_end_epoch:
            return False
        interval = self.depth_loss_interval if self.depth_loss_interval > 0 else 1
        duration = max(1, self.depth_loss_duration)
        position = (self.current_epoch - self.depth_loss_start_epoch) % interval
        return position < duration

    def _get_frame_smpl_params(self, fid: int, allow_grad: bool) -> Optional[torch.Tensor]:
        frame_smpl_params = self.smpl_params.get(fid)
        if frame_smpl_params is None:
            return None
        return frame_smpl_params if allow_grad else frame_smpl_params.detach()

    def _baseline_has_content(self, directory: Path) -> bool:
        if not directory.exists():
            return False
        return any(directory.rglob("*.png"))

    def _ensure_baseline_visualisations(self):
        if self._baseline_has_content(self.baseline_vis_dir):
            print(f"--- FYI: Found cached baseline SMPL visualisations at {self.baseline_vis_dir}")
            return

        print("--- FYI: Baseline SMPL visualisations not found. Generating now...")
        target_dir = self.experiment_dir / "visualizations" / "smpl"
        if target_dir.exists():
            shutil.rmtree(target_dir)

        save_epoch_smpl_overlays(
            dataset=self.dataset,
            smpl_params_per_frame=self.smpl_params,
            experiment_dir=self.experiment_dir,
            epoch=0,
            device=self.device,
            gender=getattr(self.cfg, "smpl_gender", "neutral"),
            alpha=getattr(self.cfg, "pose_overlay_alpha", 0.6),
        )
        if self.baseline_vis_dir.exists():
            shutil.rmtree(self.baseline_vis_dir)
        shutil.copytree(target_dir, self.baseline_vis_dir)
        print(f"--- FYI: Cached baseline SMPL visualisations at {self.baseline_vis_dir}")

    def _copy_baseline_to_experiment(self):
        if not self._baseline_has_content(self.baseline_vis_dir):
            return
        target_dir = self.experiment_dir / "visualizations" / "smpl"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(self.baseline_vis_dir, target_dir)

    @torch.no_grad()
    def evaluate(self, batch: Dict[str, Any], tids: List[int], render_bg: bool):
        # Parse batch
        _, _, K, w2c, H, W, fid = self._parse_batch(batch)

        # Track which dynamic splat sets correspond to the requested tids
        smpl_param_forward = self._get_frame_smpl_params(fid, allow_grad=False)
        selected_indices, smpl_param_forward, lbs_weights = filter_dynamic_splats(
            all_gs=self.all_gs,
            all_lbs_weights = self.lbs_weights,
            smpl_params=smpl_param_forward,
            sel_tids=tids,
            cfg_tids=self.cfg.tids,
        )

        # Prepare the splats to render based on requested tids/background
        static_component = self.all_gs.static if render_bg and self.all_gs.static is not None else None
        # Only include dynamic splats if we actually selected a matching SMPL slice
        dynamic_components = (
            [self.all_gs.dynamic[idx] for idx in selected_indices] if smpl_param_forward is not None else []
        )

        if static_component is None and len(dynamic_components) == 0:
            raise RuntimeError("No splats selected for rendering. Enable background rendering or provide valid tids.")

        gs_to_render = SceneSplats(
            static=static_component,
            dynamic=dynamic_components,
            smpl_c_info=self.all_gs.smpl_c_info,
        )

        renders, _, _ = render_splats(
            gs_to_render, smpl_param_forward, 
            lbs_weights, w2c, K, H, W, 
            sh_degree=self.cfg.sh_degree,
            render_mode="RGB+ED"
        )
        colors, depths = renders[..., 0:3], renders[..., 3:4]

        return colors, depths, smpl_param_forward

    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        # Parse batch
        images, human_masks, K, w2c, H, W, fid = self._parse_batch(batch)

        # Confidence guided SMPL parameter optimization (optimise only for reliable frames)
        frame_reliable = self.progressive_sam.is_frame_reliable(fid)
        pose_only_window = self._in_pose_only_window()
        # Before pose correction we still tweak unreliable frames, but only through SMPL updates
        delayed_pose_window = (not frame_reliable) and (self.current_epoch < self.pose_correction_epoch)
        allow_smpl_grad = (
            self.is_smpl_optim_enabled and (frame_reliable or pose_only_window or delayed_pose_window)
        )
        smpl_param_forward = self._get_frame_smpl_params(fid, allow_grad=allow_smpl_grad)
        # Pause Gaussian updates while we are running SMPL-only refinement
        should_update_gaussians = not (pose_only_window or delayed_pose_window)

        # Forward pass: render
        colors, alphas, info = render_splats(
            self.all_gs, smpl_param_forward, 
            self.lbs_weights, w2c, K, H, W, 
            sh_degree=self.cfg.sh_degree
        )

        # Neccessary step for later adaptive densification
        self.all_strategies.pre_backward_step(
            scene_splats=self.all_gs,
            optimizers=self.all_optimisers,
            it_number=it_number,
            info=info
        )

        # Losses
        gt_render, pred_render, pred_original = prepare_input_for_loss(
            gt_imgs=images,
            renders=colors,
            human_masks=human_masks,
            cfg=self.cfg
        )  # [B,h,w,3], [B,h,w,3] (cropped if needed)

        render_for_loss = pred_original # if frame_reliable else pred_render

        if len(self.cfg.tids) > 0 and self.cfg.train_bg:
            gt_render_dyn, pred_render_dyn, pred_original_dyn = prepare_input_for_loss(
                gt_imgs=images,
                renders=colors,
                human_masks=human_masks,
                cfg=self.cfg,
                dynamic_only=True
            )
        else:
            gt_render_dyn = pred_render_dyn = pred_original_dyn = None

        # - L1
        l1_loss = F.l1_loss(render_for_loss, gt_render)
        if gt_render_dyn is not None and pred_original_dyn is not None:
            l1_loss += F.l1_loss(pred_original_dyn, gt_render_dyn)

        # - SSIM
        ssim_loss = 1.0 - fused_ssim(
            render_for_loss.permute(0, 3, 1, 2), gt_render.permute(0, 3, 1, 2), padding="valid"
        )
        if gt_render_dyn is not None and pred_original_dyn is not None:
            ssim_loss += 1.0 - fused_ssim(
                pred_original_dyn.permute(0, 3, 1, 2), gt_render_dyn.permute(0, 3, 1, 2), padding="valid"
            )

        # - Alpha
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.alpha_lambda > 0.0:
            human_masks_combined = human_masks.sum(dim=1).clamp(0.0, 1.0)  # [B,H,W]
            alpha_loss = F.mse_loss(
                alphas.squeeze(-1),
                human_masks_combined
            )

        # - Combined
        loss = l1_loss*self.cfg.l1_lambda + ssim_loss * self.cfg.ssim_lambda + alpha_loss * self.cfg.alpha_lambda

        # - Add Regularizers
        static_splats = [self.all_gs.static] if self.all_gs.static is not None else []
        all_splats = static_splats + self.all_gs.dynamic
        op_reg_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.opacity_reg > 0.0:
            for splats in all_splats:
                if len(splats["opacities"]) == 0:
                    continue
                op_reg_loss += self.cfg.opacity_reg * torch.sigmoid(splats["opacities"]).mean()
            loss += op_reg_loss

        scale_reg_loss = torch.tensor(0.0, device=self.device)
        if self.cfg.scale_reg > 0.0:
            for splats in all_splats:
                if len(splats["scales"]) == 0:
                    continue
                scale_reg_loss += self.cfg.scale_reg * torch.exp(splats["scales"]).mean()
            loss += scale_reg_loss
        
        # Backprop
        loss.backward()

        def _assign_subset_grads(entry: Any) -> None:
            if isinstance(entry, torch.Tensor):
                parent = getattr(entry, "_parent_tensor", None)
                slc = getattr(entry, "_parent_slice", None)
                if parent is not None and slc is not None and parent.grad is not None:
                    start, end = slc
                    grad_view = parent.grad[(slice(None), slice(start, end))]
                    entry.grad = grad_view
            elif isinstance(entry, dict):
                for value in entry.values():
                    _assign_subset_grads(value)
            elif isinstance(entry, (list, tuple)):
                for item in entry:
                    _assign_subset_grads(item)

        _assign_subset_grads(info)

        smpl_grad_norm = 0.0
        if allow_smpl_grad:
            total_sq = 0.0
            for param in self.smpl_params.values():
                if param.grad is not None:
                    total_sq += param.grad.detach().pow(2).sum().item()
            if total_sq > 0.0:
                smpl_grad_norm = math.sqrt(total_sq)

        # Update weights step
        static_optims = [self.all_optimisers.static] if self.all_gs.static is not None else []
        all_optims = static_optims + self.all_optimisers.dynamic
        for optimizers in all_optims:
            for opt in optimizers.values():
                if should_update_gaussians:
                    opt.step()
                opt.zero_grad(set_to_none=True)

        smpl_optim_performed = False
        if allow_smpl_grad:
            if pose_only_window and self.smpl_pose_optimizer is not None:
                # Dedicated SMPL-only pass (shape frozen)
                self.smpl_pose_optimizer.step()
                smpl_optim_performed = True
            elif (frame_reliable or delayed_pose_window) and self.smpl_joint_optimizer is not None:
                # Joint updates keep SMPL coupled to the Gaussian parameters
                self.smpl_joint_optimizer.step()
                smpl_optim_performed = True

        for opt in (self.smpl_joint_optimizer, self.smpl_pose_optimizer):
            if opt is not None:
                opt.zero_grad(set_to_none=True)


        # Adaptive densification step
        self.all_strategies.post_backward_step(
            scene_splats=self.all_gs,
            optimizers=self.all_optimisers,
            it_number=it_number,
            info=info,
            packed=self.cfg.packed
        )

        # Clear any gradients stored on auxiliary tensors returned in info to avoid
        # keeping cloned grad buffers alive across iterations.
        def _clear_aux_grads(entry: Any) -> None:
            if isinstance(entry, torch.Tensor):
                entry.grad = None
            elif isinstance(entry, dict):
                for value in entry.values():
                    _clear_aux_grads(value)
            elif isinstance(entry, (list, tuple)):
                for item in entry:
                    _clear_aux_grads(item)

        if isinstance(info, tuple) and len(info) == 2:
            static_info, dynamic_infos = info
            _clear_aux_grads(static_info)
            _clear_aux_grads(dynamic_infos)
        else:
            _clear_aux_grads(info)
        info = None
        if smpl_param_forward is not None:
            # Keep a detached snapshot for potential temporal losses
            self._previous_smpl_params[fid] = smpl_param_forward.detach().clone()


        # Update skinning weights
        if len(self.cfg.tids) > 0:
            with torch.no_grad():
                new_lbs_weights_list = update_skinning_weights(self.all_gs, k=self.cfg.lbs_knn, eps=1e-6)
                self.lbs_weights = [new_lbs_weights.detach() for new_lbs_weights in new_lbs_weights_list]

        if (
            self.debug_vis_freq > 0
            and self.current_epoch % self.debug_vis_freq == 0
            and self._orbit_debug_frame is not None
            and int(fid) == self._orbit_debug_frame
            and len(self.all_gs.dynamic) > 0
        ):
            smpl_frame_params = self.smpl_params.get(int(fid))
            if smpl_frame_params is not None:
                orbit_out_path = (
                    self.experiment_dir / "visualizations" / "orbit" / f"epoch_{self.current_epoch:04d}.mp4"
                )
                save_orbit_visualization(
                    scene_splats=self.all_gs,
                    smpl_params=smpl_frame_params.detach(),
                    lbs_weights=self.lbs_weights,
                    base_w2c=w2c[0],
                    K=K[0],
                    image_size=(H, W),
                    device=self.device,
                    sh_degree=self.cfg.sh_degree,
                    out_path=orbit_out_path,
                )

        # Log values
        log_values = {
            "loss/combined": float(loss.item()),
            "loss/masked_l1": float(l1_loss.item()),
            "loss/masked_ssim": float(ssim_loss.item()),
            "loss/opacity_reg": float(op_reg_loss.item()),
            "loss/scale_reg": float(scale_reg_loss.item()),
            "smpl_optim/step": 1.0 if smpl_optim_performed else 0.0,
            "smpl_optim/grad_norm": smpl_grad_norm,
            "epoch": self.current_epoch,
        }

        total_n_gs = 0
        for i, gs in enumerate(self.all_gs.dynamic):
            log_values[f"splats/num_gs_dynamic_{i}"] = gs["means"].shape[0]
            total_n_gs += gs["means"].shape[0]

        if self.all_gs.static is not None:
            log_values[f"splats/num_gs_static"] = self.all_gs.static["means"].shape[0]
            total_n_gs += self.all_gs.static["means"].shape[0]
        log_values["splats/num_gs_total"] = total_n_gs

        return log_values

    def _render_depth_for_tid(
        self,
        tid: int,
        smpl_params: Optional[torch.Tensor],
        w2c: torch.Tensor,
        K: torch.Tensor,
        H: int,
        W: int,
    ) -> Optional[torch.Tensor]:
        """
        Render RGB+Depth for a single TID and return the depth map tensor.
        """
        if smpl_params is None:
            return None

        selected_indices, smpl_param_forward, lbs_weights = filter_dynamic_splats(
            all_gs=self.all_gs,
            all_lbs_weights=self.lbs_weights,
            smpl_params=smpl_params,
            sel_tids=[tid],
            cfg_tids=self.cfg.tids,
        )

        if len(selected_indices) == 0 or smpl_param_forward is None or lbs_weights is None:
            return None

        dynamic_components = [self.all_gs.dynamic[idx] for idx in selected_indices]
        if len(dynamic_components) == 0:
            return None

        gs_to_render = SceneSplats(
            static=None,
            dynamic=dynamic_components,
            smpl_c_info=self.all_gs.smpl_c_info,
        )

        renders, _, _ = render_splats(
            gs_to_render,
            smpl_param_forward,
            lbs_weights,
            w2c,
            K,
            H,
            W,
            sh_degree=self.cfg.sh_degree,
            render_mode="RGB+ED",
        )
        depths = renders[..., 3:4]
        return depths

    def step_depth(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute auxiliary losses that require explicit per-person depth reasoning.
        Returns both a depth-order loss and a coarse interpenetration penalty.
        """
        if len(self.cfg.tids) == 0:
            return {
                "loss/depth_order": 0.0,
                "loss/interpenetration": 0.0,
                "loss/combined": 0.0,
            }

        images, human_masks, K, w2c, H, W, fid = self._parse_batch(batch)
        if human_masks.shape[0] != 1:
            raise ValueError("Depth order loss currently expects batch size 1.")

        allow_smpl_grad = self.is_smpl_optim_enabled
        smpl_param_forward = self._get_frame_smpl_params(fid, allow_grad=allow_smpl_grad)
        if smpl_param_forward is None:
            return {
                "loss/depth_order": 0.0,
                "loss/interpenetration": 0.0,
                "loss/combined": 0.0,
            }

        # Per-TID collections used to build the front-depth map.
        depth_maps: List[torch.Tensor] = []
        depth_valid_masks: List[torch.Tensor] = []
        mask_tensors: List[torch.Tensor] = []

        for tid_idx, tid in enumerate(self.cfg.tids):
            depths = self._render_depth_for_tid(tid, smpl_param_forward, w2c, K, H, W)
            if depths is None:
                continue

            depth_map = depths[0, ..., 0]  # [H,W]
            depth_maps.append(depth_map)
            depth_valid_masks.append(depth_map > 0.0)
            mask_tensor = human_masks[0, tid_idx, ...].to(depth_map.device)
            mask_tensors.append(mask_tensor)

        if not depth_maps:
            return {
                "loss/depth_order": 0.0,
                "loss/interpenetration": 0.0,
                "loss/combined": 0.0,
            }

        stacked_depths = torch.stack(depth_maps, dim=0)  # [T,H,W]
        stacked_valid = torch.stack(depth_valid_masks, dim=0)

        # Replace invalid depths with +inf and take the min to approximate the frontmost actor.
        large_val = torch.tensor(torch.finfo(stacked_depths.dtype).max, device=stacked_depths.device)
        masked_depths = torch.where(stacked_valid, stacked_depths, large_val)
        front_depth, _ = masked_depths.min(dim=0)
        front_valid = stacked_valid.any(dim=0)

        total_loss = torch.tensor(0.0, device=self.device)
        total_pixels = 0
        for depth_map, mask_tensor, valid_depth in zip(depth_maps, mask_tensors, depth_valid_masks):
            # Only supervise pixels where SAM says this TID is visible and we rendered valid depths.
            pixel_mask = (mask_tensor > 0.5) & valid_depth & front_valid
            if pixel_mask.any():
                diff = depth_map - front_depth
                total_loss = total_loss + (diff[pixel_mask] ** 2).sum()
                total_pixels += int(pixel_mask.sum().item())

        # Recreate Multiply's mask filtering: only trust pixels belonging to exactly one actor.
        mask_sum = human_masks[0].sum(dim=0)
        single_actor = mask_sum <= (1.0 + 1e-2)
        non_empty = mask_sum >= 0.7

        sam_mask_idx = human_masks[0].argmax(dim=0, keepdim=True)  # [1,H,W]
        gathered_depths = torch.gather(stacked_depths, dim=0, index=sam_mask_idx).squeeze(0)
        gathered_valid = torch.gather(stacked_valid.float(), dim=0, index=sam_mask_idx).squeeze(0).bool()

        valid_mask = front_valid & single_actor & non_empty & gathered_valid

        front_flat = front_depth[valid_mask]
        gt_flat = gathered_depths[valid_mask]

        if front_flat.numel() == 0 or gt_flat.numel() == 0:
            depth_order_loss_raw = torch.zeros(1, device=self.device, requires_grad=True)
            valid_count = torch.tensor(0, device=self.device)
        else:
            diff = gt_flat - front_flat
            exclude_mask = ~(diff.abs() < 1e-8)
            if exclude_mask.any():
                selected_diff = diff[exclude_mask]
                depth_order_loss_raw = F.softplus(selected_diff).sum()
                valid_count = torch.tensor(int(exclude_mask.sum().item()), device=self.device)
            else:
                depth_order_loss_raw = torch.zeros(1, device=self.device, requires_grad=True)
                valid_count = torch.tensor(0, device=self.device)

        if self.depth_normalise_by_pixels and valid_count.item() > 0:
            depth_order_loss_raw = depth_order_loss_raw / valid_count

        decay_factor = 1.0 - min(self.current_epoch, self.depth_decay_milestone) / float(self.depth_decay_milestone)
        depth_order_loss = depth_order_loss_raw * self.depth_order_weight * decay_factor

        if self.interpenetration_weight > 0.0:
            interpenetration_loss = (
                self._compute_interpenetration_loss() * self.interpenetration_weight * decay_factor
            )
        else:
            interpenetration_loss = torch.tensor(0.0, device=self.device)

        total_loss = depth_order_loss + interpenetration_loss

        smpl_optim_performed = False
        if total_loss.requires_grad:
            total_loss.backward()

            static_optims = [self.all_optimisers.static] if self.all_gs.static is not None else []
            all_optims = static_optims + self.all_optimisers.dynamic
            for optimizers in all_optims:
                for opt in optimizers.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            if allow_smpl_grad and self.smpl_joint_optimizer is not None:
                self.smpl_joint_optimizer.step()
                smpl_optim_performed = True
            if self.smpl_joint_optimizer is not None:
                self.smpl_joint_optimizer.zero_grad(set_to_none=True)
            if self.smpl_pose_optimizer is not None:
                self.smpl_pose_optimizer.zero_grad(set_to_none=True)

        depth_value = float(depth_order_loss.detach().item())
        interpenetration_value = float(interpenetration_loss.detach().item())
        combined_value = depth_value * self.depth_order_weight + interpenetration_value * self.interpenetration_weight

        logs = {
            "loss/depth_order": depth_value,
            "loss/interpenetration": interpenetration_value,
            "loss/combined": combined_value,
            "smpl_optim/step": 1.0 if smpl_optim_performed else 0.0,
        }
        return logs

    def _compute_interpenetration_loss(self) -> torch.Tensor:
        """
        Approximate an interpenetration penalty by treating each TID's Gaussians as
        isotropic spheres and penalising overlaps between different TIDs.
        """
        if len(self.cfg.tids) < 2 or len(self.all_gs.dynamic) == 0:
            return torch.tensor(0.0, device=self.device)

        margin = float(self.cfg.interpenetration_margin)
        total_loss = torch.tensor(0.0, device=self.device)
        pair_count = 0

        for idx_a in range(len(self.cfg.tids)):
            for idx_b in range(idx_a + 1, len(self.cfg.tids)):
                if idx_a >= len(self.all_gs.dynamic) or idx_b >= len(self.all_gs.dynamic):
                    continue

                gs_a = self.all_gs.dynamic[idx_a]
                gs_b = self.all_gs.dynamic[idx_b]
                if gs_a["means"].shape[0] == 0 or gs_b["means"].shape[0] == 0:
                    continue

                means_a = gs_a["means"]
                means_b = gs_b["means"]
                # Estimate an isotropic radius from anisotropic Gaussian scales.
                scales_a = torch.exp(gs_a["scales"]).mean(dim=-1)
                scales_b = torch.exp(gs_b["scales"]).mean(dim=-1)

                # Compute all pairwise center distances and penalise any overlap between
                # the two point sets, with a small safety margin.
                diff = means_a[:, None, :] - means_b[None, :, :]
                dists = torch.linalg.norm(diff, dim=-1)  # [Na, Nb]
                radii_sum = scales_a[:, None] + scales_b[None, :] + margin

                penetration = torch.clamp(radii_sum - dists, min=0.0)
                if penetration.numel() == 0:
                    continue

                total_loss = total_loss + penetration.mean()
                pair_count += 1

        if pair_count == 0:
            return torch.tensor(0.0, device=self.device)
        return total_loss / pair_count

    def evaluation_loop(self, selected_tids: List[int], render_bg: bool, epoch: int):
        mask_path = Path(self.cfg.preprocess_dir) / "sam2_masks"
        eval_dataset = build_training_dataset(self.cfg, mask_path=mask_path)
        eval_dataloader = build_dataloader(self.cfg, eval_dataset, is_eval=True)

        gt_provided = self.cfg.gt_seg_masks_dir is not None or self.cfg.gt_smpl_dir is not None
        if gt_provided:
            gt_dataset: Hi4DDataset = build_evaluation_dataset(self.cfg)
        else:
            gt_dataset = None

        # --- Individual human evaluations
        if len(selected_tids) > 0:
            tid_video_metrics = dict()
            for tid in selected_tids:
                save_qual_dir = self.experiment_dir / "visualizations" / "fg_render" / f"tid_{tid}" / f"epoch_{epoch:04d}"
                save_qual_rgb_dir = save_qual_dir / "rgb"
                save_qual_mask_dir = save_qual_dir / "gt_mask"
                os.makedirs(save_qual_rgb_dir, exist_ok=True)
                os.makedirs(save_qual_mask_dir, exist_ok=True)
                if gt_dataset is not None and gt_dataset.is_seg_mask_loading_available():
                    save_qual_pred_mask_dir = save_qual_dir / "pred_mask"
                    os.makedirs(save_qual_pred_mask_dir, exist_ok=True)

                tid_idx = self.cfg.tids.index(tid)
                tid_metrics = list()
                for batch in eval_dataloader:
                    images, human_masks, _, _, _, _, fid = self._parse_batch(batch)
                    pred_masks = human_masks[:, tid_idx, ...]  # [B,H,W]
                    colors, _, pred_smpl = self.evaluate(batch, [tid], render_bg=False)
                    pred_smpl = pred_smpl.unsqueeze(0)  # [1,1,86]

                    # Load predicted masks if available to compute segmentation metrics
                    if gt_dataset is not None and gt_dataset.is_seg_mask_loading_available():
                        gt_masks_all = gt_dataset.load_segmentation_masks(fid).to(self.device)
                        gt_masks = gt_masks_all[tid_idx, ...].unsqueeze(0)  # [B,H,W]
                    else:
                        gt_masks = pred_masks
                        pred_masks = None

                    # Load ground-truth SMPL if available
                    if gt_dataset is not None and gt_dataset.is_smpl_loading_available():
                        gt_smpl_joints_all = gt_dataset.load_smpl_joints_for_frame(fid).to(self.device) # [P,24,3]
                        gt_smpl_joints = gt_smpl_joints_all[tid_idx].unsqueeze(0)  # [B,24,3]

                        # pred smpl pose params -> joints
                        pred_smpl_joints_raw = get_joints_from_pose_params(pred_smpl[0]) # [1,86] -> [1,24,3]

                        # align to the gt dataset
                        pred_smpl_joints = gt_dataset.align_input_smpl_joints(pred_smpl_joints_raw[0], fid, tid_idx) # [24,3]
                        pred_smpl_joints = pred_smpl_joints.unsqueeze(0)  # [1,24,3]
                        
                    else:
                        gt_smpl_joints = None
                        pred_smpl_joints = None

                    # Compute quantitative rendering metrics
                    metrics = compute_all_metrics(
                        images=images,
                        masks=gt_masks,
                        renders=colors,
                        pred_masks=pred_masks,
                        gt_smpl_joints=gt_smpl_joints,
                        pred_smpl_joints=pred_smpl_joints,
                    )
                    tid_metrics.append(metrics)

                    # save the rendered image
                    img_np = (colors[0].cpu().numpy() * 255).astype("uint8")
                    img_pil = Image.fromarray(img_np)
                    img_pil.save(save_qual_rgb_dir / f"{fid:04d}.png")

                    # save the mask image
                    mask_np = (gt_masks[0].cpu().numpy() * 255).astype("uint8")
                    mask_pil = Image.fromarray(mask_np)
                    mask_pil.save(save_qual_mask_dir / f"{fid:04d}.png")

                    # if pred masks provided also save them
                    if pred_masks is not None:
                        pred_mask_np = (pred_masks[0].cpu().numpy() * 255).astype("uint8")
                        pred_mask_pil = Image.fromarray(pred_mask_np)
                        pred_mask_pil.save(save_qual_pred_mask_dir / f"{fid:04d}.png")
                
                # Aggregate metrics
                tid_metrics_across_frames = aggregate_batch_tid_metric_dicts(tid_metrics)
                tid_video_metrics[tid] = tid_metrics_across_frames
            
            # aggregate across tids
            individual_tid_metrics_across_frames = aggregate_global_tids_metric_dicts(tid_video_metrics)
        else:
            individual_tid_metrics_across_frames = dict()

        # --- Joined human evaluation (all selected tids together)
        if len(selected_tids) > 0:
            save_qual_dir = self.experiment_dir / "visualizations" / "fg_render" / "all" / f"epoch_{epoch:04d}"
            save_qual_rgb_dir = save_qual_dir / "rgb"
            save_qual_mask_dir = save_qual_dir / "gt_mask"
            os.makedirs(save_qual_rgb_dir, exist_ok=True)
            os.makedirs(save_qual_mask_dir, exist_ok=True)
            if gt_dataset is not None and gt_dataset.is_seg_mask_loading_available():
                save_qual_pred_mask_dir = save_qual_dir / "pred_mask"
                os.makedirs(save_qual_pred_mask_dir, exist_ok=True)

            joined_tid_metrics = list()
            for batch in eval_dataloader:
                images, human_masks, _, _, _, _, fid = self._parse_batch(batch)
                pred_masks_joined = (human_masks.sum(dim=1).clamp(0.0, 1.0))  # [B,H,W] 
                colors, _, pred_smpl = self.evaluate(batch, selected_tids, render_bg=False)
                pred_smpl = pred_smpl.unsqueeze(0)  # [B,len(tids),86]

                # Load predicted masks if available to compute segmentation metrics
                if gt_dataset is not None and gt_dataset.is_seg_mask_loading_available():
                    gt_masks_all = gt_dataset.load_segmentation_masks(fid).to(self.device)
                    gt_masks_joined = (gt_masks_all.sum(dim=0).clamp(0.0, 1.0)).unsqueeze(0)  # [B,H,W] 
                else:
                    gt_masks_joined = pred_masks_joined
                    pred_masks_joined = None

                # Load ground-truth SMPL if available
                gt_smpl_joints = None
                pred_smpl_joints = None

                # Compute quantitative metrics
                metrics = compute_all_metrics(
                    images=images,
                    masks=gt_masks_joined,
                    renders=colors,
                    pred_masks=pred_masks_joined,
                    gt_smpl_joints=gt_smpl_joints,
                    pred_smpl_joints=pred_smpl_joints,
                )
                joined_tid_metrics.append(metrics)

                # save the rendered image
                img_np = (colors[0].cpu().numpy() * 255).astype("uint8")
                img_pil = Image.fromarray(img_np)
                img_pil.save(save_qual_rgb_dir / f"{fid:04d}.png")

                # save the mask image
                mask_np = (gt_masks_joined[0].cpu().numpy() * 255).astype("uint8")
                mask_pil = Image.fromarray(mask_np)
                mask_pil.save(save_qual_mask_dir / f"{fid:04d}.png")

                # if pred masks provided also save them
                if pred_masks_joined is not None:
                    pred_mask_np = (pred_masks_joined[0].cpu().numpy() * 255).astype("uint8")
                    pred_mask_pil = Image.fromarray(pred_mask_np)
                    pred_mask_pil.save(save_qual_pred_mask_dir / f"{fid:04d}.png")
            
            # aggregate metrics 
            joined_tid_metrics_across_frames = aggregate_batch_tid_metric_dicts(joined_tid_metrics)
        else:
            joined_tid_metrics_across_frames = dict()

        # --- Full render evaluation (background + all humans)
        if render_bg:
            save_qual_dir = self.experiment_dir / "visualizations" / "full_render" / f"epoch_{epoch:04d}"
            save_qual_rgb_dir = save_qual_dir / "rgb"
            os.makedirs(save_qual_rgb_dir, exist_ok=True)

            full_render_metrics = list()
            for batch in eval_dataloader:
                images, human_masks, _, _, _, _, fid = self._parse_batch(batch)
                colors, _, pred_smpl = self.evaluate(batch, selected_tids, render_bg=True)

                # Compute quantitative metrics
                masks_include_all = torch.ones_like(human_masks[:, 0, ...])  # [B,H,W]
                metrics = compute_all_metrics(
                    images=images,
                    masks=masks_include_all,
                    renders=colors,
                )
                full_render_metrics.append(metrics)

                # save the rendered image
                img_np = (colors[0].cpu().numpy() * 255).astype("uint8")
                img_pil = Image.fromarray(img_np)
                img_pil.save(save_qual_rgb_dir / f"{fid:04d}.png")
        
            # aggregate metrics
            full_render_metrics_across_frames = aggregate_batch_tid_metric_dicts(full_render_metrics)
        else:
            full_render_metrics_across_frames = dict()

        # Log to wandb
        dict_to_log = {}
        for metric_name, value in individual_tid_metrics_across_frames.items():
            dict_to_log[f"eval/individual_tids/{metric_name}"] = value
        for metric_name, value in joined_tid_metrics_across_frames.items():
            dict_to_log[f"eval/joined_tids/{metric_name}"] = value
        for metric_name, value in full_render_metrics_across_frames.items():
            dict_to_log[f"eval/full_render/{metric_name}"] = value
        dict_to_log["epoch"] = epoch

        print(f"--- Joined TIDs Evaluation results at epoch {epoch} for TIDs {selected_tids}:")
        for key, value in dict_to_log.items():
            print(f"    {key}: {value:.4f}")

        wandb.log(dict_to_log)


    def train_loop(self, max_epochs: int = 1):
        iteration = 0

        with tqdm(total=max_epochs, desc="Training Progress", dynamic_ncols=True) as pbar:
            for epoch in range(max_epochs):
                self.current_epoch = epoch
                if self.progressive_sam.should_update(epoch):
                    self.progressive_sam.update_masks(self.dataset, self.all_gs, self.lbs_weights, epoch=epoch)
                    self.loader = build_dataloader(self.cfg, self.dataset)
                    psam_logs = {
                        "progressive_sam/num_reliable_frames": len(self.progressive_sam.reliable_frames),
                        "progressive_sam/num_unreliable_frames": len(self.progressive_sam.unreliable_frames),
                        "progressive_sam/reliability_threshold": self.progressive_sam.iou_threshold,
                    }
                else:
                    psam_logs = {}

                depth_window = self._in_depth_loss_window()
                smpl_epoch_updated = False
                for batch in self.loader:
                    iteration += 1
                    if depth_window:
                        logs = self.step_depth(batch)
                        self.depth_iterations += 1
                        if logs.get("smpl_optim/step", 0.0) > 0:
                            smpl_epoch_updated = True
                    else:
                        effective_iteration = iteration - self.depth_iterations
                        logs = self.step(batch, effective_iteration)
                        if logs.get("smpl_optim/step", 0.0) > 0:
                            smpl_epoch_updated = True

                    # Update progress bar display
                    loss_display = logs.get("loss/combined", 0.0)
                    pbar.set_postfix({
                        "loss": f"{loss_display:.4f}",
                        "epoch": epoch,
                        "n_gs": logs.get("splats/num_gs_total", 0),
                    })

                    # Log to wandb
                    logs["iteration"] = iteration
                    logs["epoch"] = epoch
                    if psam_logs:
                        logs.update(psam_logs)
                    wandb.log(logs)

                self.is_smpl_optim_enabled = self.current_epoch < self.cfg.pose_opt_end_epoch
                if not self.is_smpl_optim_enabled and (self.current_epoch == self.cfg.pose_opt_end_epoch):
                    print(f"--- FYI: SMPL parameter optimization disabled from epoch {self.current_epoch} onwards.")

                completed_epochs = epoch + 1
                if (
                    smpl_epoch_updated
                    and self.cfg.save_pose_overlays_every_epoch > 0
                    and completed_epochs % self.cfg.save_pose_overlays_every_epoch == 0
                ):
                    save_epoch_smpl_overlays(
                        dataset=self.dataset,
                        smpl_params_per_frame=self.smpl_params,
                        experiment_dir=self.experiment_dir,
                        epoch=completed_epochs,
                        device=self.device,
                        gender=getattr(self.cfg, "smpl_gender", "neutral"),
                        alpha=getattr(self.cfg, "pose_overlay_alpha", 0.6),
                    )

                if (
                    self.cfg.eval_every_epochs > 0
                    and completed_epochs % self.cfg.eval_every_epochs == 0
                    and completed_epochs > 0
                ):
                    self.evaluation_loop(
                        selected_tids=list(self.cfg.tids),
                        render_bg=self.cfg.train_bg,
                        epoch=completed_epochs,
                    )

                if self.cfg.save_freq > 0 and completed_epochs % self.cfg.save_freq == 0:
                    self.ckpt_manager.save(self.all_gs, completed_epochs, smpl_params=self.smpl_params)
                pbar.update(1)

        if self.cfg.save_freq > 0 and max_epochs > 0 and (max_epochs % self.cfg.save_freq != 0):
            self.ckpt_manager.save(self.all_gs, max_epochs, smpl_params=self.smpl_params)
        

@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):

    print("ℹ️ Initializing Trainer")
    init_logging(cfg)
    trainer = Trainer(cfg)
    print("✅ Trainer initialized.\n")
    if cfg.is_preprocessing:
        return

    print("ℹ️ Starting training")
    trainer.train_loop(max_epochs=cfg.max_epochs)
    wandb.finish()
    print("✅ Training completed.\n")

if __name__ == "__main__":
    main()
