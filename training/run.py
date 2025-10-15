import hydra
import os
import sys
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm

from training.helpers.trainer_init import init_logging
from training.helpers.smpl_utils import update_skinning_weights
from training.helpers.dataset import FullSceneDataset
from training.helpers.model_init import create_splats_with_optimizers, init_trainable_smpl_params
from training.helpers.render import render_splats
from training.helpers.losses import prepare_input_for_loss
from training.helpers.checkpointing import ModelCheckpointManager
from training.helpers.progressive_sam import ProgressiveSAMManager
from training.helpers.visualisation_utils import VisualisationManager

from fused_ssim import fused_ssim


class Trainer:
    def __init__(self, cfg: DictConfig, internal_run_id: str = None):
        # Initial param setup
        self.cfg = cfg
        self.visualise_cam_preds = cfg.visualise_cam_preds
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.current_epoch = 0
        print(f"--- FYI: using device {self.device}")

        # Checkpoint manager
        self.ckpt_manager = ModelCheckpointManager(
            Path(cfg.output_dir),
            cfg.group_name,
            cfg.tids,
        )
        self.checkpoint_dir = self.ckpt_manager.root

        if not getattr(cfg, "resume", False):
            reset_static = bool(cfg.train_bg)
            reset_tids = list(cfg.tids) if len(cfg.tids) > 0 else []
            self.ckpt_manager.reset(reset_static=reset_static, reset_tids=reset_tids)

        # Setup experiment dirs
        preprocess_path = Path(cfg.preprocess_dir)
        if internal_run_id is not None:
            run_name, run_id = internal_run_id.split("_")
            self.experiment_dir = Path(cfg.train_dir) /  f"{run_name}_{run_id}"
        else:
            self.experiment_dir = Path(cfg.train_dir) / f"{wandb.run.name}_{wandb.run.id}"
        self.trn_viz_debug_dir = self.experiment_dir / "visualizations" / "debug"
        os.makedirs(self.trn_viz_debug_dir, exist_ok=True)
        print(f"--- FYI: experiment output dir: {self.experiment_dir}")

        # Initialize Progressive SAM manager
        mask_cfg = getattr(cfg, "mask_refinement", None)
        mask_container = OmegaConf.to_container(mask_cfg, resolve=True) if mask_cfg is not None else {}
        self.progressive_sam = ProgressiveSAMManager(
            mask_cfg=mask_container,
            tids=list(cfg.tids),
            device=self.device,
            default_lbs_knn=int(cfg.lbs_knn),
            preprocess_dir=preprocess_path,
            checkpoint_dir=self.checkpoint_dir,
        )
        self.mask_loss_weight = self.progressive_sam.loss_weight
        self.mask_enabled = self.progressive_sam.enabled
        if cfg.resume:
            self.progressive_sam.init_from_disk()
        else:
            self.progressive_sam.clear_cache()
            self.progressive_sam.clear_ckpt_dir()

        # Load dataset and create dataloader
        self.dataset = FullSceneDataset(
            preprocess_path,
            cfg.tids,  # list of tids to train on
            cloud_downsample=cfg.cloud_downsample,
            train_bg=cfg.train_bg,
        )
        if len(self.dataset) > 0:
            self.orbit_reference_w2c = self.dataset.pose_all[0].clone()
        else:
            self.orbit_reference_w2c = torch.eye(4)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
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

        # Define trainable SMPL parameters and optimizer
        self.smpl_params, self.smpl_param_optimizer = init_trainable_smpl_params(
            self.dataset, cfg, self.device, checkpoint_manager=self.ckpt_manager
        )
        self.smpl_snapshot_frame: Optional[int] = None
        self.smpl_snapshot_params: Optional[List[torch.Tensor]] = None

        # Init Visualisation manager responsible for periodic visualizations
        self.pose_overlay_period = 5
        self.last_pose_overlay_epoch: int = -1
        self.visualisation_manager = VisualisationManager(
            cfg=self.cfg,
            mask_enabled=self.mask_enabled,
            progressive_sam=self.progressive_sam,
            trn_viz_dir=self.trn_viz_debug_dir,
            scene_splats=self.all_gs,
            lbs_weights=self.lbs_weights,
            device=self.device,
            sh_degree=self.cfg.sh_degree,
            dataset=self.dataset,
            pose_overlay_period=self.pose_overlay_period,
        )

    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        # Parse batch
        images = batch["image"].to(self.device)  # [B,H,W,3]
        K = batch["K"].to(self.device)           # [B,3,3]
        w2c = batch["M_ext"].to(self.device)     # [B,4,4]
        H, W = batch["image"].shape[1:3]
        assert images.shape[0] == 1, "Mask refinement currently expects batch size 1."
        fid_tensor = batch["fid"]
        fid = int(fid_tensor.item()) if torch.is_tensor(fid_tensor) else int(fid_tensor)

        # Confidence guided SMPL parameter optimization (optimise only for reliable frames)
        frame_smpl_params = self.smpl_params.get(fid)
        frame_reliable = self.progressive_sam.is_frame_reliable(fid)
        if frame_smpl_params is None:
            smpl_param_forward = None
        else:
            smpl_param_forward = frame_smpl_params if frame_reliable else frame_smpl_params.detach()

        # Forward pass: mask refinement
        mask_output = self.progressive_sam.process_batch(
            fid=fid,
            image=images[0],
            smpl_params=smpl_param_forward,
            w2c=w2c[0],
            K=K[0],
            scene_splats=self.all_gs,
            lbs_weights=self.lbs_weights,
            device=self.device,
            image_size=(H, W),
            lbs_knn=int(self.cfg.lbs_knn),
        )
        human_masks = mask_output.human_masks
        mask_loss = mask_output.mask_loss
        alpha_stack = mask_output.alpha_stack
        viz_entries = mask_output.viz_entries

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

        render_for_loss = pred_original if frame_reliable else pred_render

        # - L1
        l1_loss = F.l1_loss(render_for_loss, gt_render)

        # - SSIM
        ssim_loss = 1.0 - fused_ssim(
            render_for_loss.permute(0, 3, 1, 2), gt_render.permute(0, 3, 1, 2), padding="valid"
        )

        # - Combined 
        loss = l1_loss*self.cfg.l1_lambda + ssim_loss * self.cfg.ssim_lambda 

        # - Mask loss (only when mask refinement is enabled and on the epoch of rebuilding the SAM cache)
        apply_mask_loss = (
            self.mask_enabled
            and alpha_stack is not None
            and self.mask_loss_weight > 0.0
            and self.progressive_sam.last_rebuild_epoch == self.current_epoch
        )
        if apply_mask_loss:
            loss = loss + self.mask_loss_weight * mask_loss

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

        # Update weights step
        static_optims = [self.all_optimisers.static] if self.all_gs.static is not None else []
        all_optims = static_optims + self.all_optimisers.dynamic
        for optimizers in all_optims:
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

        if self.smpl_param_optimizer is not None:
            if frame_reliable:
                self.smpl_param_optimizer.step()
            self.smpl_param_optimizer.zero_grad(set_to_none=True)


        # Adaptive densification step
        self.all_strategies.post_backward_step(
            scene_splats=self.all_gs,
            optimizers=self.all_optimisers,
            it_number=it_number,
            info=info,
            packed=self.cfg.packed
        )


        # Update skinning weights
        if len(self.cfg.tids) > 0:
            with torch.no_grad():
                new_lbs_weights_list = update_skinning_weights(self.all_gs, k=self.cfg.lbs_knn, eps=1e-6)
                self.lbs_weights = [new_lbs_weights.detach() for new_lbs_weights in new_lbs_weights_list]
                self.visualisation_manager.lbs_weights = self.lbs_weights

        # Visualizations
        self.last_pose_overlay_epoch = self.visualisation_manager.run_visualisation_step(
            gt_render=gt_render,
            pred_render=pred_render,
            pred_original=pred_original,
            viz_entries=viz_entries,
            fid=fid,
            current_epoch=self.current_epoch,
            smpl_param_forward=smpl_param_forward,
            w2c=w2c[0],
            K=K[0],
            H=H,
            W=W,
            smpl_snapshot_frame=self.smpl_snapshot_frame,
            smpl_snapshot_params=self.smpl_snapshot_params,
            smpl_params_per_frame=self.smpl_params,
            last_pose_overlay_epoch=self.last_pose_overlay_epoch,
        )

        # Log values
        mask_loss_scalar = float(mask_loss.item()) if apply_mask_loss else 0.0
        reliability_pct = 0.0
        if self.mask_enabled:
            reliability_pct = 100.0 * self.progressive_sam.get_reliability_ratio()

        log_values = {
            "loss/combined": float(loss.item()),
            "loss/masked_l1": float(l1_loss.item()),
            "loss/masked_ssim": float(ssim_loss.item()),
            "loss/opacity_reg": float(op_reg_loss.item()),
            "loss/scale_reg": float(scale_reg_loss.item()),
            "loss/mask_refinement": mask_loss_scalar,
            "mask/reliable_pct": reliability_pct,
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

    def train_loop(self, iters: int = 2000):
        iteration = 0
        epoch = 0

        with tqdm(total=iters, desc="Training Progress", dynamic_ncols=True) as pbar:
            while iteration < iters:
                self.current_epoch = epoch
                if self.progressive_sam.should_rebuild(epoch):
                    self.progressive_sam.rebuild_cache(
                        self.dataset,
                        self.all_gs,
                        self.lbs_weights,
                        epoch=epoch,
                        lbs_knn=int(self.cfg.lbs_knn),
                        device=self.device,
                    )
                    snapshot_fid = self.progressive_sam.get_lowest_iou_reliable_frame()
                    if snapshot_fid is not None and snapshot_fid in self.smpl_params:
                        self.smpl_snapshot_frame = snapshot_fid
                        param_tensor = self.smpl_params[snapshot_fid].detach().clone()
                        self.smpl_snapshot_params = [param_tensor[i].detach().cpu().clone() for i in range(param_tensor.shape[0])]
                        self.last_pose_overlay_epoch = -1

                for batch in self.loader:
                    iteration += 1
                    logs = self.step(batch, iteration)

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
                    wandb.log(logs)

                    if self.cfg.save_freq > 0 and (iteration % self.cfg.save_freq == 0):
                        self.ckpt_manager.save(self.all_gs, iteration, smpl_params=self.smpl_params)
                        self.progressive_sam.save(iteration)

                    if iteration >= iters:
                        break

                epoch += 1

        if self.cfg.save_freq > 0 and iteration % self.cfg.save_freq != 0:
            self.ckpt_manager.save(self.all_gs, iteration, smpl_params=self.smpl_params)
            self.progressive_sam.save(iteration)
        

@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):

    print("ℹ️ Initializing Trainer")
    init_logging(cfg)
    trainer = Trainer(cfg)
    print("✅ Trainer initialized.\n")

    print("ℹ️ Starting training")
    trainer.train_loop(iters=cfg.iters)
    wandb.finish()
    print("✅ Training completed.\n")

if __name__ == "__main__":
    main()
