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
from training.helpers.smpl_utils import update_skinning_weights, filter_dynamic_splats
from training.helpers.dataset import build_dataset, build_dataloader
from training.helpers.model_init import (
    SceneSplats,
    create_splats_with_optimizers,
    init_trainable_smpl_params,
)
from training.helpers.render import render_splats
from training.helpers.losses import prepare_input_for_loss
from training.helpers.checkpointing import ModelCheckpointManager
from training.helpers.progressive_sam import ProgressiveSAMManager
from training.helpers.visualisation_utils import colourise_depth, save_orbit_visualization, save_epoch_smpl_overlays
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
        self._previous_smpl_params: Dict[int, torch.Tensor] = {}
        print(f"--- FYI: using device {self.device}")

        # Checkpoint manager
        self.ckpt_manager = ModelCheckpointManager(
            Path(cfg.output_dir),
            cfg.group_name,
            cfg.tids,
        )
        self.checkpoint_dir = self.ckpt_manager.root

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

        if self.cfg.save_pose_overlays_every_epoch > 0:
            baseline_dir = Path(self.cfg.preprocess_dir) / "baseline_smpl_vis"
            if not baseline_dir.exists():
                raise FileNotFoundError(f"Baseline SMPL visualisations not found: {baseline_dir}")
            target_dir = self.experiment_dir / "visualizations" / "smpl"
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(baseline_dir, target_dir)

        # Load dataset and create dataloader
        mask_path = self.checkpoint_dir / "progressive_sam"
        self.dataset = build_dataset(cfg, mask_path=mask_path)
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
        mask_cfg = cfg.mask_refinement
        mask_container = OmegaConf.to_container(mask_cfg, resolve=True) if mask_cfg is not None else {}
        self.progressive_sam = ProgressiveSAMManager(
            mask_cfg=mask_container,
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

    def _get_frame_smpl_params(self, fid: int, allow_grad: bool) -> Optional[torch.Tensor]:
        frame_smpl_params = self.smpl_params.get(fid)
        if frame_smpl_params is None:
            return None
        return frame_smpl_params if allow_grad else frame_smpl_params.detach()

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

        return colors, depths

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
            and it_number % self.debug_vis_freq == 0
            and self._orbit_debug_frame is not None
            and int(fid) == self._orbit_debug_frame
            and len(self.all_gs.dynamic) > 0
        ):
            smpl_frame_params = self.smpl_params.get(int(fid))
            if smpl_frame_params is not None:
                orbit_out_path = (
                    self.experiment_dir / "visualizations" / "orbit" / f"iter_{it_number:06d}.mp4"
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

    def evaluation_loop(self, selected_tids: List[int], render_bg: bool, epoch: int):
        mask_path = Path(self.cfg.preprocess_dir) / "sam2_masks"
        self.dataset = build_dataset(self.cfg, mask_path=mask_path)
        eval_dataloader = build_dataloader(self.cfg, self.dataset, is_eval=True)

        # --- Individual human evaluations
        if len(selected_tids) > 0:
            tid_video_metrics = dict()
            for tid in selected_tids:
                save_qual_dir = self.experiment_dir / "visualizations" / f"tid_{tid}" / f"epoch_{epoch:04d}"
                save_qual_rgb_dir = save_qual_dir / "rgb"
                save_qual_depth_dir = save_qual_dir / "depth"
                os.makedirs(save_qual_rgb_dir, exist_ok=True)
                os.makedirs(save_qual_depth_dir, exist_ok=True)

                tid_idx = self.cfg.tids.index(tid)
                tid_metrics = list()
                for batch in eval_dataloader:
                    images, human_masks, _, _, _, _, fid = self._parse_batch(batch)
                    tid_masks = human_masks[:, tid_idx, ...]  # [B,H,W]
                    colors, depths = self.evaluate(batch, [tid], render_bg=False)

                    # Compute quantitative metrics
                    metrics = compute_all_metrics(
                        images=images,
                        masks=tid_masks,
                        renders=colors,
                    )
                    tid_metrics.append(metrics)

                    # save the rendered image
                    img_np = (colors[0].cpu().numpy() * 255).astype("uint8")
                    img_pil = Image.fromarray(img_np)
                    img_pil.save(save_qual_rgb_dir / f"{fid:04d}.png")

                    # save the depth map
                    depth_viz = colourise_depth(depths[0], self.cfg)
                    depth_pil = Image.fromarray(depth_viz)
                    depth_pil.save(save_qual_depth_dir / f"{fid:04d}.png")
                
                # Aggregate metrics
                tid_metrics_across_frames = aggregate_batch_tid_metric_dicts(tid_metrics)
                tid_video_metrics[tid] = tid_metrics_across_frames
            
            # aggregate across tids
            individual_tid_metrics_across_frames = aggregate_global_tids_metric_dicts(tid_video_metrics)
        else:
            individual_tid_metrics_across_frames = dict()

        # --- Joined human evaluation (all selected tids together)
        if len(selected_tids) > 0:
            save_qual_dir = self.experiment_dir / "visualizations" / "all_humans" / f"epoch_{epoch:04d}"
            save_qual_rgb_dir = save_qual_dir / "rgb"
            save_qual_depth_dir = save_qual_dir / "depth"
            os.makedirs(save_qual_rgb_dir, exist_ok=True)
            os.makedirs(save_qual_depth_dir, exist_ok=True)

            joined_tid_metrics = list()
            for batch in eval_dataloader:
                images, human_masks, _, _, _, _, fid = self._parse_batch(batch)
                human_masks_joined = (human_masks.sum(dim=1).clamp(0.0, 1.0))  # [B,H,W] 
                colors, depths = self.evaluate(batch, selected_tids, render_bg=False)

                # Compute quantitative metrics
                metrics = compute_all_metrics(
                    images=images,
                    masks=human_masks_joined,
                    renders=colors,
                )
                joined_tid_metrics.append(metrics)

                # save the rendered image
                img_np = (colors[0].cpu().numpy() * 255).astype("uint8")
                img_pil = Image.fromarray(img_np)
                img_pil.save(save_qual_rgb_dir / f"{fid:04d}.png")

                # save the depth map
                depth_viz = colourise_depth(depths[0], self.cfg)
                depth_pil = Image.fromarray(depth_viz)
                depth_pil.save(save_qual_depth_dir / f"{fid:04d}.png")
            
            # aggregate metrics 
            joined_tid_metrics_across_frames = aggregate_batch_tid_metric_dicts(joined_tid_metrics)
        else:
            joined_tid_metrics_across_frames = dict()

        # --- Full render evaluation (background + all humans)
        if render_bg:
            save_qual_dir = self.experiment_dir / "visualizations" / "full_render" / f"epoch_{epoch:04d}"
            save_qual_rgb_dir = save_qual_dir / "rgb"
            save_qual_depth_dir = save_qual_dir / "depth"
            os.makedirs(save_qual_rgb_dir, exist_ok=True)
            os.makedirs(save_qual_depth_dir, exist_ok=True)

            full_render_metrics = list()
            for batch in eval_dataloader:
                images, human_masks, _, _, _, _, fid = self._parse_batch(batch)
                colors, depths = self.evaluate(batch, selected_tids, render_bg=True)

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

                # save the depth map
                depth_viz = colourise_depth(depths[0], self.cfg)
                depth_pil = Image.fromarray(depth_viz)
                depth_pil.save(save_qual_depth_dir / f"{fid:04d}.png")
        
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


    def train_loop(self, iters: int = 2000):
        iteration = 0
        epoch = 0


        with tqdm(total=iters, desc="Training Progress", dynamic_ncols=True) as pbar:
            while iteration < iters:
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

                smpl_epoch_updated = False
                for batch in self.loader:
                    iteration += 1
                    logs = self.step(batch, iteration)
                    if logs.get("smpl_optim/step", 0.0) > 0:
                        smpl_epoch_updated = True

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{logs['loss/combined']:.4f}",
                        "epoch": epoch,
                        "n_gs": logs.get("splats/num_gs_total", 0),
                    })

                    # Log to wandb
                    logs["iteration"] = iteration
                    logs["epoch"] = epoch
                    if psam_logs:
                        logs.update(psam_logs)
                    wandb.log(logs)

                    if self.cfg.save_freq > 0 and (iteration % self.cfg.save_freq == 0):
                        self.ckpt_manager.save(self.all_gs, iteration, smpl_params=self.smpl_params)

                    if iteration >= iters:
                        break

                epoch += 1
                self.is_smpl_optim_enabled = self.current_epoch < self.cfg.pose_opt_end_epoch
                if not self.is_smpl_optim_enabled and (self.current_epoch == self.cfg.pose_opt_end_epoch):
                    print(f"--- FYI: SMPL parameter optimization disabled from epoch {self.current_epoch} onwards.")

                if (
                    smpl_epoch_updated
                    and self.cfg.save_pose_overlays_every_epoch > 0
                    and epoch % self.cfg.save_pose_overlays_every_epoch == 0
                ):
                    save_epoch_smpl_overlays(
                        dataset=self.dataset,
                        smpl_params_per_frame=self.smpl_params,
                        experiment_dir=self.experiment_dir,
                        epoch=epoch,
                        device=self.device,
                        gender=getattr(self.cfg, "smpl_gender", "neutral"),
                        alpha=getattr(self.cfg, "pose_overlay_alpha", 0.6),
                    )

                if self.cfg.eval_every_epochs > 0 and (epoch % self.cfg.eval_every_epochs == 0) and epoch > 0:
                    self.evaluation_loop(
                        selected_tids=list(self.cfg.tids),
                        render_bg=self.cfg.train_bg,
                        epoch=epoch,
                    )

        if self.cfg.save_freq > 0 and iteration % self.cfg.save_freq != 0:
            self.ckpt_manager.save(self.all_gs, iteration, smpl_params=self.smpl_params)
        

@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):

    print("ℹ️ Initializing Trainer")
    init_logging(cfg)
    trainer = Trainer(cfg)
    print("✅ Trainer initialized.\n")
    if cfg.is_preprocessing:
        return

    print("ℹ️ Starting training")
    trainer.train_loop(iters=cfg.iters)
    wandb.finish()
    print("✅ Training completed.\n")

if __name__ == "__main__":
    main()
