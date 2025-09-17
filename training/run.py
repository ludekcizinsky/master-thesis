import math
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

from training.helpers.utils import init_logging, save_loss_visualization, lbs_apply
from training.helpers.losses import anchor_to_smpl_surface, opacity_distance_penalty
from training.helpers.render import rasterize_splats
from gsplat.strategy import DefaultStrategy
from training.helpers.dataset import HumanOnlyDataset

from utils.smpl_deformer.smpl_server import SMPLServer
from fused_ssim import fused_ssim


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def create_splats_with_optimizers(device, cfg):

    smpl_server = SMPLServer().to(device).eval()
    with torch.no_grad():
        verts_c = smpl_server.verts_c[0].to(device)      # [6890,3]
        weights_c = smpl_server.weights_c[0].to(device)  # [6890,24]

    M = verts_c.shape[0]

    # --- Splats (the model)
    # means 
    smpl_vertices = verts_c.clone()                     # [M,3]

    # anisotropic per-axis log-scales
    init_sigma = 0.02
    log_scales = torch.full((M, 3), math.log(init_sigma), device=device) # [M,3]

    # rotations as quaternions [w,x,y,z], init to identity
    quats_init = torch.zeros(M, 4, device=device)
    quats_init[:, 0] = 1.0

    # opacities
    colors = torch.rand(M, 3, device=device)      # [M,3]
    init_opacity = 0.1
    opacity_logit = torch.full((M,), torch.logit(torch.tensor(init_opacity, device=device)))

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(smpl_vertices), cfg.means_lr),
        ("scales", torch.nn.Parameter(log_scales), cfg.scales_lr),
        ("quats", torch.nn.Parameter(quats_init), cfg.quats_lr),
        ("opacities", torch.nn.Parameter(opacity_logit), cfg.opacities_lr),
    ]


    # color is SH coefficients
    colors = torch.zeros((M, (cfg.sh_degree + 1) ** 2, 3))  # [M, K, 3]
    colors[:, 0, :] = rgb_to_sh(torch.rand(M, 3))  # [M,3]
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), cfg.sh0_lr))
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), cfg.shN_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # --- Optimizers
    BS = 1
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return splats, optimizers, weights_c, smpl_server, verts_c


def prepare_input_for_loss(gt_imgs: torch.Tensor, renders: torch.Tensor, masks: torch.Tensor, kind="bbox_crop"):
    """
    Args:
        gt_imgs: [B,H,W,3] in [0,1]
        renders: [B,H,W,3] in [0,1]
        masks:   [B,H,W] in {0,1}

    Returns:
        gt_crop: [B,h,w,3] in [0,1]
        pr_crop: [B,h,w,3] in [0,1]
    """

    gt_imgs = gt_imgs.clamp(0, 1).float()          # [B,H,W,3]
    renders = renders.clamp(0, 1).float()

    if kind == "bbox_crop":
        # Apply mask to the ground truth (keep only target person pixels)
        gt_imgs *= masks.unsqueeze(-1)

        # bbox crop around the mask to avoid huge easy background
        ys, xs = torch.where(masks[0] > 0.5)
        if ys.numel() > 0:
            pad = 8
            y0 = max(int(ys.min().item()) - pad, 0)
            y1 = min(int(ys.max().item()) + pad + 1, gt_imgs.shape[1])
            x0 = max(int(xs.min().item()) - pad, 0)
            x1 = min(int(xs.max().item()) + pad + 1, gt_imgs.shape[2])

            gt_crop  = gt_imgs[:, y0:y1, x0:x1, :]
            pr_crop  = renders[:, y0:y1, x0:x1, :]
        else:
            # fallback if mask empty
            gt_crop, pr_crop = gt_imgs, renders

        return gt_crop, pr_crop
    elif kind == "bbox":
        # Apply mask to the ground truth (keep only target person pixels)
        gt_imgs *= masks.unsqueeze(-1)

        # Mask out the render outside the bbox of the mask (keep full GT)
        pad = 8
        ys, xs = torch.where(masks[0] > 0.5)
        y0 = max(int(ys.min().item()) - pad, 0)
        y1 = min(int(ys.max().item()) + pad + 1, renders.shape[1])
        x0 = max(int(xs.min().item()) - pad, 0)
        x1 = min(int(xs.max().item()) + pad + 1, renders.shape[2])

        pr_bbox = torch.zeros_like(renders)
        pr_bbox[:, y0:y1, x0:x1, :] = renders[:, y0:y1, x0:x1, :]

        return gt_imgs, pr_bbox

    elif kind == "tight_crop":
        # Apply tight mask (keep only target person pixels)
        gt_imgs *= masks.unsqueeze(-1)
        renders *= masks.unsqueeze(-1)

        # bbox crop around the mask to avoid huge easy background
        ys, xs = torch.where(masks[0] > 0.5)
        if ys.numel() > 0:
            y0 = int(ys.min().item())
            y1 = int(ys.max().item()) + 1
            x0 = int(xs.min().item())
            x1 = int(xs.max().item()) + 1

            gt_crop  = gt_imgs[:, y0:y1, x0:x1, :]
            pr_crop  = renders[:, y0:y1, x0:x1, :]
        else:
            # fallback if mask empty
            gt_crop, pr_crop = gt_imgs, renders
        
        return gt_crop, pr_crop

    elif kind == "tight":
        # Apply tight mask (keep only target person pixels)
        gt_imgs *= masks.unsqueeze(-1)
        renders *= masks.unsqueeze(-1)

        return gt_imgs, renders

    else:
        raise ValueError(f"Unknown prepare_input_for_loss kind: {kind}")

class Trainer:
    def __init__(self, cfg: DictConfig, internal_run_id: str = None):
        self.cfg = cfg
        self.visualise_cam_preds = cfg.visualise_cam_preds
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"--- FYI: using device {self.device}")

        self.dataset = HumanOnlyDataset(Path(cfg.preprocess_dir), cfg.tid, split=cfg.split, downscale=cfg.downscale, val_fids=cfg.val_fids)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        print(f"--- FYI: dataset has {len(self.dataset)} samples and using batch size 1")

        if internal_run_id is not None:
            run_name, run_id = internal_run_id.split("_")
            self.experiment_dir = Path(cfg.train_dir) / f"tid_{cfg.tid}" / f"{run_name}_{run_id}"
        else:
            self.experiment_dir = Path(cfg.train_dir) / f"tid_{cfg.tid}" / f"{wandb.run.name}_{wandb.run.id}"
        self.trn_viz_debug_dir = self.experiment_dir / "visualizations" / "debug"
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.trn_viz_debug_dir, exist_ok=True)
        print(f"--- FYI: experiment output dir: {self.experiment_dir}")

        # Define model and optimizers
        out = create_splats_with_optimizers(self.device, cfg)
        self.splats, self.optimizers, self.weights_c, self.smpl_server, self.smpl_verts_c = out
        print("--- FYI: Model initialized. Number of GS:", len(self.splats["means"]))

        # Keep a frozen copy of the initial canonical means for anchor-loss in "init" mode
        self.init_means_c = self.splats["means"].detach().clone()  # [M0,3]
        self.M0 = self.init_means_c.shape[0]

        # Adaptive densification strategy
        self.strategy = DefaultStrategy(verbose=True)
        self.strategy.refine_stop_iter = cfg.gs_refine_stop_iter
        self.strategy.refine_every = cfg.gs_refine_every
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    def _update_skinning_weights(self, current_weights_c, k: int = 4, eps: float = 1e-6):
        """
        Update / extend skinning weights when #splats (Gaussians) changes.

        Args:
            current_weights_c: torch.Tensor [M_old, 24]
                Existing per-Gaussian LBS weights.
            k: int
                # of nearest SMPL vertices to use for interpolation.
            eps: float
                Small epsilon for numerical stability.

        Requires (on self):
            self.splats["means"]          -> [M_new_total, 3] canonical positions of Gaussians
            self.smpl_verts_c             -> [N_smpl, 3] canonical SMPL vertex positions
            self.smpl_weights_c           -> [N_smpl, 24] SMPL skinning weight matrix (rows sum ~1)

        Returns:
            updated_weights_c: torch.Tensor [M_new_total, 24]
        """
        device = current_weights_c.device
        dtype  = current_weights_c.dtype

        means_all = self.splats["means"].to(device=device, dtype=dtype)                # [M_total, 3]
        smpl_V    = self.smpl_server.verts_c[0].to(device=device, dtype=dtype)                   # [N_smpl, 3]
        smpl_W    = self.smpl_server.weights_c[0].to(device=device, dtype=dtype)                 # [N_smpl, 24]

        M_old = current_weights_c.shape[0]
        M_tot = means_all.shape[0]

        # Case 1: fewer/equal splats than weights -> truncate (common when pruning)
        if M_tot <= M_old:
            return current_weights_c[:M_tot].clone()

        # Case 2: we added new splats -> extend
        M_new = M_tot - M_old
        means_new = means_all[M_old:]                                                  # [M_new, 3]
        print(f"--- FYI: extending skinning weights: {M_old} -> {M_tot} splats")

        # k-NN in canonical space to SMPL vertices
        k_eff = min(k, smpl_V.shape[0])
        # distances: [M_new, N_smpl]
        dists = torch.cdist(means_new, smpl_V)                                         
        # take k nearest
        topk_d, topk_idx = torch.topk(dists, k=k_eff, dim=1, largest=False)            # [M_new, k], [M_new, k]

        # inverse-distance weights (safer than 1/d with eps)
        inv = 1.0 / (topk_d + eps)                                                     # [M_new, k]
        w_geo = inv / (inv.sum(dim=1, keepdim=True) + eps)                             # row-normalize

        # gather SMPL skinning weights for those neighbors
        # smpl_W[topk_idx]: [M_new, k, 24]
        smpl_W_knn = smpl_W[topk_idx]

        # distance-weighted mix -> [M_new, 24]
        new_W = (w_geo.unsqueeze(-1) * smpl_W_knn).sum(dim=1)

        # final safety renorm (keep convex comb)
        new_W = torch.clamp(new_W, min=0)
        new_W = new_W / (new_W.sum(dim=1, keepdim=True) + eps)

        # stitch: keep old rows, append new rows
        updated = torch.empty((M_tot, smpl_W.shape[1]), device=device, dtype=dtype)
        updated[:M_old] = current_weights_c
        updated[M_old:] = new_W
        return updated

    def means_can_to_cam(self, smpl_param: torch.Tensor) -> torch.Tensor:
        out = self.smpl_server(smpl_param, absolute=False)
        T_rel = out["smpl_tfs"][0].to(self.device)  # [24,4,4]

        with torch.no_grad():
            self.weights_c = self._update_skinning_weights(self.weights_c, k=4, eps=1e-6)
            self.weights_c = self.weights_c.detach()  # redundant but explicit

        means_cam = lbs_apply(self.splats["means"], self.weights_c, T_rel)
        return means_cam

    def opacity(self) -> torch.Tensor:
        """[M] in (0,1)"""
        return torch.sigmoid(self.splats["opacities"])

    def get_colors(self) -> torch.Tensor:
        """[M,3] in [0,1]"""
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        return colors

    def scales(self) -> torch.Tensor:
        """Per-axis std devs [M,3], clamped."""
        s = torch.exp(self.splats["scales"])
        min_sigma = 1e-4
        max_sigma = 1.0
        return s.clamp(min_sigma, max_sigma)

    def rotations(self) -> torch.Tensor:
        """
        Return unit quaternions [w, x, y, z] with shape [M,4]
        """
        q = self.splats["quats"]
        return q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    
    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        images = batch["image"].to(self.device)  # [B,H,W,3]
        masks  = batch["mask"].to(self.device)   # [B,H,W]
        K  = batch["K"].to(self.device)      # [B, 3,3]
        smpl_param = batch["smpl_param"].to(self.device)  # [B,86]
        H, W = images.shape[1:3]

        # Forward pass: render
        renders, alphas, info = rasterize_splats(
            trainer=self,
            smpl_param=smpl_param,
            K=K,
            img_wh=(W, H),
            sh_degree=self.cfg.sh_degree,
            packed=self.cfg.packed,
        ) # renders of shape [1,H,W,3]


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

            )
        else:
            ma_loss = torch.tensor(0.0, device=self.device)


        # - Opacity distance penalty (to keep splats near the surface)
        if self.cfg.opa_lambda > 0.0:
            opa_loss = opacity_distance_penalty(
                opa_logits=self.splats["opacities"],
                means_c=self.splats["means"],
                smpl_verts_c=self.smpl_verts_c,
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

        # Periodic debug visualization
        if it_number % 1000 == 0 and self.cfg.visualise_cam_preds:
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

    print("ℹ️ Starting training")
    trainer.train_loop(iters=cfg.iters)
    internal_run_id = f"{wandb.run.name}_{wandb.run.id}"
    wandb.finish()
    print("✅ Training completed.\n")

    print("ℹ️ Starting evaluation")
    os.system(f"python training/evaluate.py internal_run_id={internal_run_id} split=val")
    print("✅ Evaluation completed.")

if __name__ == "__main__":
    main()
