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
from training.helpers.render import gsplat_render
from gsplat.strategy import DefaultStrategy
from training.helpers.dataset import HumanOnlyDataset


from utils.smpl_deformer.smpl_server import SMPLServer



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
        ("means", torch.nn.Parameter(smpl_vertices), cfg.lrs.mean),
        ("scales", torch.nn.Parameter(log_scales), cfg.lrs.scale),
        ("quats", torch.nn.Parameter(quats_init), cfg.lrs.quats),
        ("opacities", torch.nn.Parameter(opacity_logit), cfg.lrs.opacity),
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

    return splats, optimizers, weights_c, smpl_server


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.sh_degree = cfg.sh_degree
        self.visualise_cam_preds = cfg.visualise_cam_preds
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        print(f"--- FYI: using device {self.device}")

        self.dataset = HumanOnlyDataset(Path(cfg.preprocess_dir), cfg.tid, downscale=cfg.downscale)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)
        print(f"--- FYI: dataset has {len(self.dataset)} samples and using batch size 1")

        self.experiment_dir = Path(cfg.train_dir) / f"{wandb.run.name}_{wandb.run.id}"
        self.trn_viz_canon_dir = self.experiment_dir / "visualizations" / "canonical"
        self.trn_viz_debug_dir = self.experiment_dir / "visualizations" / "debug"
        os.makedirs(self.trn_viz_canon_dir, exist_ok=True)
        os.makedirs(self.trn_viz_debug_dir, exist_ok=True)
        print(f"--- FYI: experiment output dir: {self.experiment_dir}")

        # Define model and optimizers
        self.grad_clip = cfg.grad_clip
        out = create_splats_with_optimizers(self.device, cfg)
        self.splats, self.optimizers, self.weights_c, self.smpl_server = out
        print("--- FYI: Model initialized. Number of GS:", len(self.splats["means"]))

        # Adaptive densification strategy
        self.strategy = DefaultStrategy(verbose=True)
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
        image = batch["image"].squeeze(0).to(self.device)  # [3,H,W]
        mask  = batch["mask"].squeeze(0).to(self.device)   # [H,W]
        K  = batch["K"].squeeze(0).to(self.device)      # [3,3]
        smpl_param = batch["smpl_param"].to(self.device)  # [1,86]
        H, W = image.shape[-2:]

        # Forward pass: render
        rgb_pred, _, info = gsplat_render(
            trainer=self,
            smpl_param=smpl_param,
            K=K,
            img_wh=(W, H),
            sh_degree=self.sh_degree
        )  # [3,H,W]

        with torch.no_grad():
            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=it_number,
                info=info,
            )

        # Losses
        # - RGB L1 (masked)
        mask3 = mask.expand_as(rgb_pred)
        l_rgb = (mask3 * (rgb_pred - image).abs()).mean()

        # - Regularizers
        l_scale_reg = self.scales().mean() * 1e-3
        l_opacity_reg = (self.opacity() ** 2).mean() * 1e-4

        # Total loss
        loss = l_rgb + l_scale_reg + l_opacity_reg
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
                packed=True,
            )

        # Periodic debug visualization
        if it_number % 1000 == 0 and self.visualise_cam_preds:
            save_loss_visualization(
                image=image,
                mask=mask,
                rgb_pred=rgb_pred,
                out_path=self.trn_viz_debug_dir / f"lossviz_it{it_number:05d}.png"
            )

        return {
            "loss": float(loss.item()),
            "l_rgb": float(l_rgb.item()),
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
                        "loss": f"{logs['loss']:.4f}",
                        "rgb": f"{logs['l_rgb']:.4f}",
                    })

                    # Log to wandb
                    wandb.log({
                        "loss": logs["loss"],
                        "l_rgb": logs["l_rgb"],
                        "iteration": it,
                        "num_splats": len(self.splats["means"]),
                    })

                    if it >= iters:
                        break
        
        # End of training: save model and canonical viz
        self.export_canonical_npz(self.experiment_dir / "model_canonical.npz")

    @torch.no_grad()
    def export_canonical_npz(self, path: Path):
        data = {
            "means_c": self.splats["means"].detach().cpu().numpy(),          # [M,3]
            "log_scales": self.splats["scales"].detach().cpu().numpy(),    # [M,3]
            "quats": self.rotations().detach().cpu().numpy(),        # [M,4] unit quats [w,x,y,z]
            "colors": self.get_colors().detach().cpu().numpy(),      # [M,3], [0,1]
            "opacity": self.opacity().detach().cpu().numpy(),        # [M]
        }
        np.savez(path, **data)

@hydra.main(config_path="../configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig):

    print("ℹ️ Initializing Trainer")
    init_logging(cfg)
    trainer = Trainer(cfg)
    print("✅ Trainer initialized.\n")

    print("ℹ️ Starting training")
    trainer.train_loop(iters=cfg.iters)
    print("✅ Training completed.")

if __name__ == "__main__":
    main()
