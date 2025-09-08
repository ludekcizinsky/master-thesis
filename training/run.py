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

from training.helpers.utils import init_logging, project_points, save_loss_visualization, lbs_apply
from training.helpers.render import gsplat_render
from gsplat.strategy import DefaultStrategy
from training.helpers.models import CanonicalGaussians
from training.helpers.dataset import HumanOnlyDataset

from utils.smpl_deformer.smpl_server import SMPLServer


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

    # colors and opacities
    colors = torch.rand(M, 3, device=device)      # [M,3]
    init_opacity = 0.1
    opacity_logit = torch.full((M,), torch.logit(torch.tensor(init_opacity, device=device)))


    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(smpl_vertices), cfg.lrs.mean),
        ("scales", torch.nn.Parameter(log_scales), cfg.lrs.scale),
        ("quats", torch.nn.Parameter(quats_init), cfg.lrs.quats),
        ("opacities", torch.nn.Parameter(opacity_logit), cfg.lrs.opacity),
        ("colors", torch.nn.Parameter(colors), cfg.lrs.color), # TODO: understand better how they represent color in gsplat
    ]

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

        # Warm-start colors from first available frame (project once)
        self._init_colors()

    def means_can_to_cam(self, smpl_param: torch.Tensor) -> torch.Tensor:

        out = self.smpl_server(smpl_param, absolute=False)
        T_rel = out["smpl_tfs"][0].to(self.device)       # [24,4,4]
        means_cam = lbs_apply(self.splats["means"], self.weights_c, T_rel)  # [M,3]
        return means_cam

    def opacity(self) -> torch.Tensor:
        """[M] in (0,1)"""
        return torch.sigmoid(self.splats["opacities"])

    def get_colors(self) -> torch.Tensor:
        """[M,3] in [0,1]"""
        return self.splats["colors"].clamp(0, 1)

    def scales(self) -> torch.Tensor:
        """Per-axis std devs [M,3], clamped."""
        s = torch.exp(self.splats["scales"])
        min_sigma = 1e-4
        max_sigma = 1.0
        return s.clamp(min_sigma, max_sigma)

    def rotations(self) -> torch.Tensor:
        """
        Return unit quaternions [w, x, y, z] with shape [M,4],
        which matches your rasteriser's expected format.
        """
        q = self.splats["quats"]
        return q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    

    @torch.no_grad()
    def _init_colors(self):
        # Use first frame in dataset to initialize colors by sampling image at projections
        sample = self.dataset[0]
        image = sample["image"].to(self.device)  # [3,H,W]
        K = sample["K"].to(self.device)
        smpl_param = sample["smpl_param"].unsqueeze(0).to(self.device)  # [1,86]

        means_cam = self.means_can_to_cam(smpl_param)  # [M,3]
        uv, _ = project_points(means_cam, K)  # [M,2]

        # Bilinear sample colors (ignore out-of-bounds)
        H, W = image.shape[-2:]
        u = uv[:, 0].clamp(0, W - 1 - 1e-3)
        v = uv[:, 1].clamp(0, H - 1 - 1e-3)
        grid = torch.stack([(u / (W - 1)) * 2 - 1, (v / (H - 1)) * 2 - 1], dim=-1).view(1, -1, 1, 2)  # [1,M,1,2]
        # grid_sample expects [N,C,H,W] and grid [N,H_out,W_out,2]
        sampled = F.grid_sample(image.unsqueeze(0), grid, align_corners=True, mode="bilinear")  # [1,3,M,1]
        colors = sampled.squeeze(0).squeeze(-1).permute(1, 0)  # [M,3]
        self.splats["colors"].data.copy_(colors.clamp(0, 1))


    def step(self, batch: Dict[str, Any], it_number: int) -> Dict[str, float]:
        image = batch["image"].squeeze(0).to(self.device)  # [3,H,W]
        mask  = batch["mask"].squeeze(0).to(self.device)   # [H,W]
        K  = batch["K"].squeeze(0).to(self.device)      # [3,3]
        smpl_param = batch["smpl_param"].to(self.device)  # [1,86]
        H, W = image.shape[-2:]

        rgb_pred = gsplat_render(
            trainer=self,
            smpl_param=smpl_param,
            K=K,
            img_wh=(W, H)
        )  # [3,H,W]

        # Losses
        # - RGB L1 (masked)
        mask3 = mask.expand_as(rgb_pred)
        l_rgb = (mask3 * (rgb_pred - image).abs()).mean()

        # - Regularizers
        l_scale_reg = self.scales().mean() * 1e-3
        l_opacity_reg = (self.opacity() ** 2).mean() * 1e-4

        # Total loss
        loss = l_rgb + l_scale_reg + l_opacity_reg

        # Update weights step
        # self.opt.zero_grad(set_to_none=True)
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            parameters = [p for p in self.splats.values() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        # self.opt.step()
        for opt in self.optimizers.values():
            opt.step()

        # Periodic debug visualization
        if it_number % 50 == 0 and self.visualise_cam_preds:
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
