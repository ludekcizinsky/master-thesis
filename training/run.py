import hydra
import os
import sys
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

from pathlib import Path
from typing import Dict, Any, List

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
from training.helpers.progressive_sam import compute_refined_masks, suppress_sam_logging
from training.helpers.visualisation_utils import save_mask_refinement_figure

from fused_ssim import fused_ssim


class Trainer:
    def __init__(self, cfg: DictConfig, internal_run_id: str = None):
        self.cfg = cfg
        self.visualise_cam_preds = cfg.visualise_cam_preds
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"--- FYI: using device {self.device}")

        mask_cfg = getattr(cfg, "mask_refinement", None)
        mask_container = OmegaConf.to_container(mask_cfg, resolve=True) if mask_cfg is not None else {}
        self.mask_loss_weight = float(mask_container.get("loss_weight", 0.0))
        self.mask_alpha_threshold = float(mask_container.get("alpha_threshold", 0.3))
        self.mask_rebuild_every = max(int(mask_container.get("rebuild_every_epochs", 10)), 1)
        self.mask_predictor_cfg = mask_container.get("sam2", {})
        self.mask_enabled = bool(mask_container.get("enabled", False)) and len(cfg.tids) > 0
        self.sam_predictor = None
        self.sam_device = None
        self.mask_cache: dict[int, dict[str, Any]] = {}
        self.current_epoch = 0
        self.last_mask_rebuild_epoch = -1

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
        self.ckpt_manager = GaussianCheckpointManager(
            Path(cfg.output_dir),
            cfg.group_name,
            cfg.tids,
        )
        self.checkpoint_dir = self.ckpt_manager.root
        os.makedirs(self.trn_viz_debug_dir, exist_ok=True)

        if not getattr(cfg, "resume", False):
            reset_static = bool(cfg.train_bg)
            reset_tids = list(cfg.tids) if len(cfg.tids) > 0 else []
            self.ckpt_manager.reset(reset_static=reset_static, reset_tids=reset_tids)
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

        if self.mask_enabled:
            self.sam_predictor = self._init_sam_predictor()

    def _init_sam_predictor(self):

        gh = GlobalHydra.instance()
        if gh.is_initialized():
            gh.clear()

        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam_cfg = self.mask_predictor_cfg
        model_id = sam_cfg.get("model_id", "facebook/sam2.1-hiera-large")
        device_str = sam_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        sam_model = build_sam2_hf(model_id, device=device_str)
        predictor = SAM2ImagePredictor(
            sam_model,
            mask_threshold=float(sam_cfg.get("mask_threshold", 0.0)),
        )
        import logging
        logging.getLogger("sam2").setLevel(logging.ERROR)
        logging.getLogger("sam2").propagate = False
        self.sam_device = device_str
        logging.getLogger("sam2").setLevel(logging.ERROR)
        logging.getLogger("sam2").propagate = False
        return predictor

    def _rebuild_mask_cache(self) -> None:
        if not self.mask_enabled or len(self.cfg.tids) == 0:
            return

        if self.sam_predictor is None:
            self.sam_predictor = self._init_sam_predictor()

        print(f"--- FYI: rebuilding SAM mask cache for epoch {self.current_epoch}")
        self.mask_cache.clear()

        with torch.no_grad():
            for sample in self.dataset:
                fid = int(sample["fid"])
                image_np = np.clip(sample["image"].cpu().numpy(), 0.0, 1.0)
                image_uint8 = (image_np * 255.0).round().astype(np.uint8)
                with suppress_sam_logging():
                    self.sam_predictor.set_image(image_uint8)
                    refined_results = compute_refined_masks(
                        scene_splats=self.all_gs,
                        smpl_params=sample["smpl_param"].to(self.device),
                        w2c=sample["M_ext"].to(self.device),
                        K=sample["K"].to(self.device),
                        image_size=(int(sample["H"]), int(sample["W"])),
                        alpha_threshold=self.mask_alpha_threshold,
                        predictor=self.sam_predictor,
                        predictor_cfg=self.mask_predictor_cfg,
                        lbs_weights=self.lbs_weights,
                        lbs_knn=int(self.cfg.lbs_knn),
                        device=self.device,
                    )

                if not refined_results:
                    continue

                refined_tensor = torch.stack([res.refined_mask for res in refined_results], dim=0).to(self.device)
                alpha_tensor = torch.stack([res.alpha for res in refined_results], dim=0).to(self.device)
                initial_tensor = torch.stack([res.initial_mask for res in refined_results], dim=0).to(self.device)
                vis_pos = [res.vis_positive_points for res in refined_results]
                vis_neg = [res.vis_negative_points for res in refined_results]

                self.mask_cache[fid] = {
                    "refined": refined_tensor,
                    "alpha": alpha_tensor,
                    "initial": initial_tensor,
                    "vis_pos": vis_pos,
                    "vis_neg": vis_neg,
                }

        self.last_mask_rebuild_epoch = self.current_epoch
    
    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        images = batch["image"].to(self.device)  # [B,H,W,3]
        K = batch["K"].to(self.device)           # [B,3,3]
        w2c = batch["M_ext"].to(self.device)     # [B,4,4]
        H, W = batch["image"].shape[1:3]
        smpl_param = batch["smpl_param"].to(self.device)  # [B, P, 86]

        assert images.shape[0] == 1, "Mask refinement currently expects batch size 1."
        image_np = np.clip(images[0].detach().cpu().numpy(), 0.0, 1.0)
        image_uint8 = (image_np * 255.0).round().astype(np.uint8)
        fid_tensor = batch["fid"]
        fid = int(fid_tensor.item()) if torch.is_tensor(fid_tensor) else int(fid_tensor)

        if len(self.cfg.tids) > 0:
            human_masks = torch.ones((1, len(self.cfg.tids), H, W), device=self.device, dtype=images.dtype)
        else:
            human_masks = torch.zeros((1, 1, H, W), device=self.device, dtype=images.dtype)

        mask_loss = torch.tensor(0.0, device=self.device)
        alpha_stack = None
        viz_entries: List[dict[str, Any]] = []

        if self.mask_enabled and len(self.cfg.tids) > 0:
            cache_entry = self.mask_cache.get(fid)

            if cache_entry is None:
                with torch.no_grad():
                    with suppress_sam_logging():
                        self.sam_predictor.set_image(image_uint8)
                        refined_results = compute_refined_masks(
                            scene_splats=self.all_gs,
                            smpl_params=smpl_param[0],
                            w2c=w2c[0],
                            K=K[0],
                            image_size=(H, W),
                            alpha_threshold=self.mask_alpha_threshold,
                            predictor=self.sam_predictor,
                            predictor_cfg=self.mask_predictor_cfg,
                            lbs_weights=self.lbs_weights,
                            lbs_knn=int(self.cfg.lbs_knn),
                            device=self.device,
                        )

                if refined_results:
                    refined_tensor = torch.stack([res.refined_mask for res in refined_results], dim=0).to(self.device)
                    alpha_tensor = torch.stack([res.alpha for res in refined_results], dim=0).to(self.device)
                    initial_tensor = torch.stack([res.initial_mask for res in refined_results], dim=0).to(self.device)
                    vis_pos = [res.vis_positive_points for res in refined_results]
                    vis_neg = [res.vis_negative_points for res in refined_results]
                    self.mask_cache[fid] = {
                        "refined": refined_tensor,
                        "alpha": alpha_tensor,
                        "initial": initial_tensor,
                        "vis_pos": vis_pos,
                        "vis_neg": vis_neg,
                    }
                    cache_entry = self.mask_cache[fid]

            if cache_entry is not None:
                refined_tensor = cache_entry["refined"]
                alpha_tensor = cache_entry["alpha"]
                initial_tensor = cache_entry["initial"]
                human_masks = refined_tensor.unsqueeze(0)
                alpha_stack = alpha_tensor
                mask_loss = F.mse_loss(alpha_tensor, refined_tensor)
                for idx_h in range(refined_tensor.shape[0]):
                    viz_entries.append({
                        "initial": initial_tensor[idx_h].detach().cpu().numpy(),
                        "refined": refined_tensor[idx_h].detach().cpu().numpy(),
                        "pos": cache_entry["vis_pos"][idx_h],
                        "neg": cache_entry["vis_neg"][idx_h],
                    })

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
            human_masks=human_masks,
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
        if self.mask_enabled and alpha_stack is not None and self.mask_loss_weight > 0.0:
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

        if (
            self.mask_enabled
            and viz_entries
            and ((it_number % 500 == 0 and self.cfg.visualise_cam_preds) or (it_number == 1))
        ):
            for idx, entry in enumerate(viz_entries):
                out_path = self.trn_viz_debug_dir / f"maskref_it{it_number:05d}_human{idx:02d}.png"
                save_mask_refinement_figure(
                    image_np,
                    entry["initial"],
                    entry["refined"],
                    entry["pos"],
                    entry["neg"],
                    out_path,
                )

        # Log values
        mask_loss_scalar = float(mask_loss.item()) if self.mask_enabled else 0.0

        log_values = {
            "loss/combined": float(loss.item()),
            "loss/masked_l1": float(l1_loss.item()),
            "loss/masked_ssim": float(ssim_loss.item()),
            "loss/opacity_reg": float(op_reg_loss.item()),
            "loss/scale_reg": float(scale_reg_loss.item()),
            "loss/mask_refinement": mask_loss_scalar,
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
                if (
                    self.mask_enabled
                    and len(self.cfg.tids) > 0
                    and (
                        epoch == 0
                        or self.last_mask_rebuild_epoch < 0
                        or (epoch - self.last_mask_rebuild_epoch) >= self.mask_rebuild_every
                    )
                ):
                    self._rebuild_mask_cache()

                for batch in self.loader:
                    iteration += 1
                    logs = self.step(batch, iteration)

                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{logs['loss/combined']:.4f}",
                        "epoch": epoch,
                    })

                    # Log to wandb
                    logs["iteration"] = iteration
                    logs["epoch"] = epoch
                    wandb.log(logs)

                    if self.cfg.save_freq > 0 and (iteration % self.cfg.save_freq == 0):
                        self.ckpt_manager.save(self.all_gs, iteration)

                    if iteration >= iters:
                        break

                epoch += 1

        if self.cfg.save_freq > 0 and iteration % self.cfg.save_freq != 0:
            self.ckpt_manager.save(self.all_gs, iteration)
        

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
