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
import numpy as np

from tqdm import tqdm

from training.helpers.utils import init_logging, save_loss_visualization
from training.helpers.losses import anchor_to_smpl_surface, opacity_distance_penalty
from training.helpers.dataset import FullSceneDataset
from training.helpers.model_init import create_splats_with_optimizers

from fused_ssim import fused_ssim

from gsplat.strategy import DefaultStrategy
from gsplat import rasterization


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
        self.splats, self.optimizers, smpl_c_info = create_splats_with_optimizers(self.device, cfg, self.dataset)
        self.smpl_verts_c = smpl_c_info["verts_c"]
        self.weights_c = smpl_c_info["weights_c"]  # [M,24]
        self.smpl_server = smpl_c_info["smpl_server"]

        # Keep a frozen copy of the initial canonical means for anchor-loss in "init" mode
        self.init_means_c = self.smpl_verts_c.clone()
        self.M0 = self.init_means_c.shape[0]

        # Adaptive densification strategy
        self.strategy = DefaultStrategy(verbose=True)
        self.strategy.refine_stop_iter = cfg.gs_refine_stop_iter
        self.strategy.refine_every = cfg.gs_refine_every
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    def _update_skinning_weights(self, current_weights_c, k: int = 4, eps: float = 1e-6, p: float = 1.0):
        """
        Recompute ALL per-Gaussian LBS weights using inverse-distance weighting over k-NN SMPL verts.

        For each Gaussian center g, find its k nearest SMPL vertices v_j with distances d_j.
        If any d_j == 0 (within eps), assign equal weight among those zero-distance neighbors.
        Otherwise, use weights proportional to 1 / d_j^p (row-normalized).

        Args:
            current_weights_c: torch.Tensor [M_old, 24] or None
                Only used for device/dtype fallback; values are ignored.
            k: int
                Number of nearest SMPL vertices.
            eps: float
                Numerical stability & zero-distance threshold.
            p: float
                Power for inverse-distance weighting (p=1 => 1/d, p=2 => 1/d^2, ...).

        Requires (on self):
            self.splats["means"]         -> [M_total, 3]
            self.smpl_server.verts_c     -> [1, N_smpl, 3]
            self.smpl_server.weights_c   -> [1, N_smpl, 24]

        Returns:
            new_W: torch.Tensor [M_total, 24]
        """
        # Device / dtype
        if current_weights_c is not None:
            device = current_weights_c.device
            dtype  = current_weights_c.dtype
        else:
            device = self.splats["means"].device
            dtype  = self.splats["means"].dtype

        means_all = self.splats["means"].to(device=device, dtype=dtype)            # [M, 3]
        smpl_V    = self.smpl_server.verts_c[0].to(device=device, dtype=dtype)     # [N, 3]
        smpl_W    = self.smpl_server.weights_c[0].to(device=device, dtype=dtype)   # [N, 24]

        M = means_all.shape[0]
        N = smpl_V.shape[0]
        k_eff = min(max(int(k), 1), N)

        # Distances: [M, N]
        dists = torch.cdist(means_all, smpl_V)  # O(M*N)

        # k-NN: [M, k]
        topk_d, topk_idx = torch.topk(dists, k=k_eff, dim=1, largest=False)

        # Gather neighbor weights: [M, k, 24]
        smpl_W_knn = smpl_W[topk_idx]

        # Handle zero-distance rows robustly
        zero_mask = topk_d <= eps                         # [M, k]
        any_zero  = zero_mask.any(dim=1, keepdim=True)    # [M, 1]

        # Inverse-distance weights for non-zero case: w_j ∝ 1 / d_j^p
        inv = 1.0 / torch.clamp(topk_d, min=eps).pow(p)   # [M, k]
        w_geo = inv / (inv.sum(dim=1, keepdim=True) + eps)

        # If any zero distances: distribute uniformly across zero-distance neighbors
        zcount = zero_mask.sum(dim=1, keepdim=True).clamp(min=1)   # [M,1]
        w_zero = zero_mask.float() / zcount                        # [M, k]

        # Select per-row: use zero-based uniform weights if any zero exists, else inverse-distance
        w_final = torch.where(any_zero, w_zero, w_geo)             # [M, k]

        # Blend neighbor SMPL weights: [M, 24]
        new_W = (w_final.unsqueeze(-1) * smpl_W_knn).sum(dim=1)

        # Safety renorm
        new_W = torch.clamp(new_W, min=0)
        new_W = new_W / (new_W.sum(dim=1, keepdim=True) + eps)

        return new_W

    def _rasterize_splats(self, smpl_param, w2c, K, H, W):

        device, dtype = self.device, torch.float32

        # Prepare gaussians
        # - Means
        means = self.means_can_to_cam(smpl_param)  # [M,3]
        # - Quats
        quats = self.rotations()  # [M,4]
        # - Scales
        scales = self.scales() # [M,3]
        # - Colours
        colors = self.get_colors()
        # - Opacity
        opacity = self.opacity()

        # Define cameras
        viewmats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)   # [1,4,4]
        Ks = K.to(device, dtype).contiguous()                  # [1,3,3]

        # Define cameras
        dtype = torch.float32
        viewmats = w2c.to(device, dtype).contiguous()  # [1,4,4]
        Ks = K.to(device, dtype).contiguous()                  # [1,3,3]

        # Render
        colors, alphas, info = rasterization(
            means, quats, scales, opacity, colors, viewmats, Ks, W, H, 
            sh_degree=self.cfg.sh_degree, packed=self.cfg.packed
        )

        return colors, alphas, info

    
    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        images = batch["image"].to(self.device)  # [B,H,W,3]
        masks  = batch["mask"].to(self.device)   # [B,H,W]
        K  = batch["K"].to(self.device)      # [B, 3,3]
        w2c = batch["M_ext"].to(self.device)  # [B,4,4]
        smpl_param = batch["smpl_param"].to(self.device)  # [B,86]
        H, W = images.shape[1:3]

        # Forward pass: render
        renders, alphas, info = self._rasterize_splats(smpl_param, w2c, K, H, W)

        with torch.no_grad():
            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=it_number,
                info=info,
            )

        # Losses
        gt_render, pred_render = prepare_input_for_loss(images, renders, masks, kind=self.cfg.mask_type)

        # - L1
        l1_loss = F.l1_loss(pred_render, gt_render)

        # - SSIM
        ssim_loss = 1.0 - fused_ssim(
            pred_render.permute(0, 3, 1, 2), gt_render.permute(0, 3, 1, 2), padding="valid"
        )

        # - Anchor loss (to keep splats close to SMPL surface)
        if self.cfg.ma_lambda > 0.0:
            ma_loss = anchor_to_smpl_surface(
                means_c=self.splats["means"],
                smpl_verts_c=self.smpl_verts_c,
                free_radius=self.cfg.ma_free_radius,
            )
        else:
            ma_loss = torch.tensor(0.0, device=self.device)


        # - Opacity distance penalty (to keep splats near the surface)
        if self.cfg.opa_lambda > 0.0:
            opa_loss = opacity_distance_penalty(
                opa_logits=self.splats["opacities"],
                means_c=self.splats["means"],
                smpl_verts_c=self.smpl_verts_c,
                free_radius=self.cfg.opa_free_radius,
            )
        else:
            opa_loss = torch.tensor(0.0, device=self.device)

        # - Combined 
        loss = l1_loss*self.cfg.l1_lambda + ssim_loss * self.cfg.ssim_lambda + ma_loss*self.cfg.ma_lambda + opa_loss*self.cfg.opa_lambda

        # - Add Regularizers
        if self.cfg.opacity_reg > 0.0:
            loss += self.cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
        if self.cfg.scale_reg > 0.0:
            loss += self.cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

        # --- Backprop
        loss.backward()

        # Update weights step
        for opt in self.optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=it_number,
                info=info,
                packed=self.cfg.packed,
            )

        # Update skinning weights
        with torch.no_grad():
            self.weights_c = self._update_skinning_weights(self.weights_c, k=self.cfg.lbs_knn, eps=1e-6)
            self.weights_c = self.weights_c.detach()  # redundant but explicit

        # Periodic debug visualization
        if it_number % 500 == 0 and self.cfg.visualise_cam_preds:
            save_loss_visualization(
                image_input=images,
                gt=gt_render,
                prediction=renders,
                out_path=self.trn_viz_debug_dir / f"lossviz_it{it_number:05d}.png"
            )

        return {
            "loss/combined": float(loss.item()),
            "loss/masked_l1": float(l1_loss.item()),
            "loss/masked_ssim": float(ssim_loss.item()),
            "loss/anchor_means": float(ma_loss.item()),
            "loss/opacity_penalty": float(opa_loss.item()),
            "loss/opacity_reg": float(torch.sigmoid(self.splats["opacities"]).mean().item()),
            "loss/scale_reg": float(torch.exp(self.splats["scales"]).mean().item()),
            "splats/num": len(self.splats["means"]),
        }

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

                    if it % self.cfg.save_freq == 0:
                        self.export_canonical_npz(self.checkpoint_dir / f"model_canonical_it{it:05d}.npz")

                    if it >= iters:
                        break
        
        # End of training: save model and canonical viz
        self.export_canonical_npz(self.checkpoint_dir / "model_canonical_final.npz")

    @torch.no_grad()
    def export_canonical_npz(self, path: Path):
        data = {
            "means_c": self.splats["means"].detach().cpu().numpy(),          # [M,3]
            "log_scales": self.splats["scales"].detach().cpu().numpy(),    # [M,3]
            "quats": self.rotations().detach().cpu().numpy(),        # [M,4] unit quats [w,x,y,z]
            "colors": self.get_colors().detach().cpu().numpy(),      # [M,3], [0,1]
            "opacity": self.opacity().detach().cpu().numpy(),        # [M]
            "weights_c": self.weights_c.detach().cpu().numpy(),  # [M,24]
        }
        np.savez(path, **data)

    @torch.no_grad()
    def load_canonical_npz(self, path: Path):
        data = np.load(path)
        self.splats["means"].data = torch.from_numpy(data["means_c"]).to(self.device)
        self.splats["scales"].data = torch.from_numpy(data["log_scales"]).to(self.device)
        self.splats["quats"].data = torch.from_numpy(data["quats"]).to(self.device)
        self.splats["opacities"].data = torch.logit(torch.from_numpy(data["opacity"]).to(self.device))
        colors = torch.from_numpy(data["colors"]).to(self.device)
        self.splats["sh0"].data = colors[:, :1, :]
        self.splats["shN"].data = colors[:, 1:, :]
        self.weights_c = torch.from_numpy(data["weights_c"]).to(self.device)
        print(f"--- FYI: loaded canonical model from {path}, #GS={len(self.splats['means'])}")

@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):

    print("ℹ️ Initializing Trainer")
    init_logging(cfg)
    trainer = Trainer(cfg)
    print("✅ Trainer initialized.\n")
    quit()

    print("ℹ️ Starting training")
    trainer.train_loop(iters=cfg.iters)
    internal_run_id = f"{wandb.run.name}_{wandb.run.id}"
    wandb.finish()
    print("✅ Training completed.\n")

    # print("ℹ️ Starting evaluation")
    # os.system(f"python training/evaluate.py internal_run_id={internal_run_id} split=val")
    # print("✅ Evaluation completed.")

if __name__ == "__main__":
    main()
