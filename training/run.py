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

from fused_ssim import fused_ssim

from gsplat.strategy import DefaultStrategy



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
        )
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        print(f"--- FYI: dataset has {len(self.dataset)} samples and using batch size 1")

        if internal_run_id is not None:
            run_name, run_id = internal_run_id.split("_")
            self.experiment_dir = Path(cfg.train_dir) /  f"{run_name}_{run_id}"
        else:
            self.experiment_dir = Path(cfg.train_dir) / f"{wandb.run.name}_{wandb.run.id}"
        self.trn_viz_debug_dir = self.experiment_dir / "visualizations" / "debug"
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.trn_viz_debug_dir, exist_ok=True)
        print(f"--- FYI: experiment output dir: {self.experiment_dir}")

        # Define model and optimizers
        self.all_gs, self.all_optimisers, self.smpl_c_info = create_splats_with_optimizers(self.device, cfg, self.dataset)
        self.lbs_weights = [self.smpl_c_info["weights_c"].clone() for _ in self.all_gs[1:]]  # [M,24]

        # Adaptive densification strategy
        self.all_strategies = list()
        for splats, optimizers in zip(self.all_gs, self.all_optimisers): 
            strategy = DefaultStrategy(verbose=True)
            strategy.refine_stop_iter = cfg.gs_refine_stop_iter
            strategy.refine_every = cfg.gs_refine_every
            strategy.refine_start_iter = cfg.gs_refine_start_iter
            strategy.check_sanity(splats, optimizers)
            strategy_state = strategy.initialize_state(scene_scale=1.0)
            self.all_strategies.append((strategy, strategy_state))

    
    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        images = batch["image"].to(self.device)  # [B,H,W,3]
        masks  = batch["mask"].to(self.device)   # [B,H,W]
        K  = batch["K"].to(self.device)      # [B,3,3]
        w2c = batch["M_ext"].to(self.device)  # [B,4,4]
        smpl_param = batch["smpl_param"].to(self.device)  # [B, P, 86]
        H, W = batch["image"].shape[1:3] 

        # Forward pass: render
        colors, alphas, info_per_gs = render_splats(
            self.all_gs, self.smpl_c_info, smpl_param, self.lbs_weights, w2c, K, H, W, sh_degree=self.cfg.sh_degree, kind="all"
        )

        with torch.no_grad():
            for i in range(len(self.all_gs)):
                splats = self.all_gs[i]
                optimizers = self.all_optimisers[i]
                strategy, strategy_state = self.all_strategies[i]
                info = info_per_gs[i]
                strategy.step_pre_backward(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=it_number,
                    info=info,
                )

        # Losses
        gt_render, pred_render = images, colors  # full image supervision

        # - L1
        l1_loss = F.l1_loss(pred_render, gt_render)

        # - SSIM
        ssim_loss = 1.0 - fused_ssim(
            pred_render.permute(0, 3, 1, 2), gt_render.permute(0, 3, 1, 2), padding="valid"
        )

        # - Combined 
        loss = l1_loss*self.cfg.l1_lambda + ssim_loss * self.cfg.ssim_lambda 

        # - Add Regularizers
        if self.cfg.opacity_reg > 0.0:
            op_reg_loss = 0.0
            for splats in self.all_gs:
                if len(splats["opacities"]) == 0:
                    continue
                op_reg_loss += self.cfg.opacity_reg * torch.sigmoid(splats["opacities"]).mean()
            loss += op_reg_loss
        if self.cfg.scale_reg > 0.0:
            scale_reg_loss = 0.0
            for splats in self.all_gs:
                if len(splats["scales"]) == 0:
                    continue
                scale_reg_loss += self.cfg.scale_reg * torch.exp(splats["scales"]).mean()
            loss += scale_reg_loss
        
        # Backprop
        loss.backward()

        # Update weights step
        for optimizers in self.all_optimisers:
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)


        # Adaptive densification step
        with torch.no_grad():
            for i in range(len(self.all_gs)):
                splats = self.all_gs[i]
                optimizers = self.all_optimisers[i]
                strategy, strategy_state = self.all_strategies[i]
                info = info_per_gs[i]
                strategy.step_post_backward(
                    params=splats,
                    optimizers=optimizers,
                    state=strategy_state,
                    step=it_number,
                    info=info,
                    packed=self.cfg.packed,
                )

        # Update skinning weights
        with torch.no_grad():
            dynamic_gs = self.all_gs[1:]
            new_lbs_weights_list = update_skinning_weights(dynamic_gs, self.smpl_c_info, k=self.cfg.lbs_knn, eps=1e-6)
            self.lbs_weights = [new_lbs_weights.detach() for new_lbs_weights in new_lbs_weights_list]  # redundant but explicit

        # Periodic debug visualization
        if it_number % 500 == 0 and self.cfg.visualise_cam_preds:
            save_loss_visualization(
                image_input=images,
                gt=gt_render,
                prediction=colors,
                out_path=self.trn_viz_debug_dir / f"lossviz_it{it_number:05d}.png"
            )

        log_values = {
            "loss/combined": float(loss.item()),
            "loss/masked_l1": float(l1_loss.item()),
            "loss/masked_ssim": float(ssim_loss.item()),
            "loss/opacity_reg": float(op_reg_loss.item()),
            "loss/scale_reg": float(scale_reg_loss.item()),
            "splats/num_gs_total": sum([g["means"].shape[0] for g in self.all_gs])
        }
        for i, gs in enumerate(self.all_gs):
            name = "static" if i == 0 else f"dynamic_{i}"
            log_values[f"splats/num_gs_{name}"] = gs["means"].shape[0]

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

                    if it >= iters:
                        break
        

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
