import hydra
import os
import sys
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../submodules/humans4d")))
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from tqdm import tqdm

from training.helpers.utils import init_logging, save_loss_visualization
from training.helpers.smpl_utils import update_skinning_weights
from training.helpers.dataset import FullSceneDataset
from training.helpers.model_init import create_splats_with_optimizers
from training.helpers.render import render_splats
from training.helpers.losses import prepare_input_for_loss
from training.helpers.checkpointing import GaussianCheckpointManager

from fused_ssim import fused_ssim


class Trainer:
    def __init__(self, cfg: DictConfig, internal_run_id: str = None):
        self.cfg = cfg
        self.visualise_cam_preds = cfg.visualise_cam_preds
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"--- FYI: using device {self.device}")

        self.dataset = FullSceneDataset(
            Path(cfg.preprocess_dir),
            cfg.tids,  # list of tids to train on
            cloud_downsample=cfg.cloud_downsample,
            train_bg=cfg.train_bg,
        )
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        print(f"--- FYI: dataset has {len(self.dataset)} samples and using batch size 1")

        if internal_run_id is not None:
            run_name, run_id = internal_run_id.split("_")
            self.experiment_dir = Path(cfg.train_dir) /  f"{run_name}_{run_id}"
        else:
            self.experiment_dir = Path(cfg.train_dir) / f"{wandb.run.name}_{wandb.run.id}"
        self.trn_viz_debug_dir = self.experiment_dir / "visualizations" / "debug"
        self.ckpt_manager = GaussianCheckpointManager(Path(cfg.output_dir), cfg.group_name, cfg.tids)
        self.checkpoint_dir = self.ckpt_manager.root
        os.makedirs(self.trn_viz_debug_dir, exist_ok=True)
        print(f"--- FYI: experiment output dir: {self.experiment_dir}")

        # Define model and optimizers
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
    
    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        images = batch["image"].to(self.device)  # [B,H,W,3]
        K  = batch["K"].to(self.device)      # [B,3,3]
        w2c = batch["M_ext"].to(self.device)  # [B,4,4]
        masks  = batch["mask"].to(self.device)   # [B,P,H,W]
        H, W = batch["image"].shape[1:3] 
        smpl_param = batch["smpl_param"].to(self.device)  # [B, P, 86]

        # Forward pass: render
        colors, alphas, info = render_splats(
            self.all_gs, smpl_param, 
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
        gt_render, pred_render = prepare_input_for_loss(
            gt_imgs=images, 
            renders=colors, 
            human_masks=masks, 
            cfg=self.cfg
        )  # [B,h,w,3], [B,h,w,3] (cropped if needed)

        # - L1
        l1_loss = F.l1_loss(pred_render, gt_render)

        # - SSIM
        ssim_loss = 1.0 - fused_ssim(
            pred_render.permute(0, 3, 1, 2), gt_render.permute(0, 3, 1, 2), padding="valid"
        )

        # - Combined 
        loss = l1_loss*self.cfg.l1_lambda + ssim_loss * self.cfg.ssim_lambda 

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
                self.lbs_weights = [new_lbs_weights.detach() for new_lbs_weights in new_lbs_weights_list]  # redundant but explicit

        # Periodic debug visualization
        if (it_number % 500 == 0 and self.cfg.visualise_cam_preds) or (it_number == 1):
            save_loss_visualization(
                image_input=images,
                gt=gt_render,
                prediction=colors,
                out_path=self.trn_viz_debug_dir / f"lossviz_it{it_number:05d}.png"
            )

        # Log values
        log_values = {
            "loss/combined": float(loss.item()),
            "loss/masked_l1": float(l1_loss.item()),
            "loss/masked_ssim": float(ssim_loss.item()),
            "loss/opacity_reg": float(op_reg_loss.item()),
            "loss/scale_reg": float(scale_reg_loss.item()),
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
        it = 0
        with tqdm(total=iters, desc="Training Progress", dynamic_ncols=True) as pbar:
            while it < iters:
                for batch in self.loader:
                    it += 1
                    logs = self.step(batch, it)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{logs['loss/combined']:.4f}",
                    })

                    # Log to wandb
                    logs["iteration"] = it
                    wandb.log(logs)

                    if self.cfg.save_freq > 0 and (it % self.cfg.save_freq == 0):
                        self.ckpt_manager.save(self.all_gs, it)

                    if it >= iters:
                        break

        if self.cfg.save_freq > 0 and it % self.cfg.save_freq != 0:
            self.ckpt_manager.save(self.all_gs, it)
        

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
