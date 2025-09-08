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

from training.helpers.utils import init_logging, project_points, save_loss_visualization
from training.helpers.render import gsplat_render
from training.helpers.models import CanonicalGaussians
from training.helpers.dataset import HumanOnlyDataset

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

        # Gaussians in canonical space
        self.gaus = CanonicalGaussians(self.device).to(self.device)
        param_groups = [
            {"params": [self.gaus.means_c],       "lr": cfg.lrs.mean},
            {"params": [self.gaus.log_scales],    "lr": cfg.lrs.scale},
            {"params": [self.gaus.opacity_logit], "lr": cfg.lrs.opacity},
            {"params": [self.gaus.colors],        "lr": cfg.lrs.colour},
        ]

        # Optimizer: Gaussians
        self.opt = torch.optim.Adam(param_groups)
        self.grad_clip = cfg.grad_clip
        params = list(self.gaus.parameters())
        print(f"--- FYI: training {sum(p.numel() for p in params)} parameters")

        # Warm-start colors from first available frame (project once)
        self._init_colors()

    @torch.no_grad()
    def _init_colors(self):
        # Use first frame in dataset to initialize colors by sampling image at projections
        sample = self.dataset[0]
        image = sample["image"].to(self.device)  # [3,H,W]
        K = sample["K"].to(self.device)
        smpl_param = sample["smpl_param"].unsqueeze(0).to(self.device)  # [1,86]

        means_cam = self.gaus.means_cam(smpl_param)  # [M,3]
        uv, _ = project_points(means_cam, K)  # [M,2]

        # Bilinear sample colors (ignore out-of-bounds)
        H, W = image.shape[-2:]
        u = uv[:, 0].clamp(0, W - 1 - 1e-3)
        v = uv[:, 1].clamp(0, H - 1 - 1e-3)
        grid = torch.stack([(u / (W - 1)) * 2 - 1, (v / (H - 1)) * 2 - 1], dim=-1).view(1, -1, 1, 2)  # [1,M,1,2]
        # grid_sample expects [N,C,H,W] and grid [N,H_out,W_out,2]
        sampled = F.grid_sample(image.unsqueeze(0), grid, align_corners=True, mode="bilinear")  # [1,3,M,1]
        colors = sampled.squeeze(0).squeeze(-1).permute(1, 0)  # [M,3]
        self.gaus.colors.data.copy_(colors.clamp(0, 1))


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
        l_scale_reg = self.gaus.scales().mean() * 1e-3
        l_opacity_reg = (self.gaus.opacity() ** 2).mean() * 1e-4

        # Total loss
        loss = l_rgb + l_scale_reg + l_opacity_reg

        # Update weights step
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.gaus.parameters(), self.grad_clip)
        self.opt.step()

        # Periodic debug visualization
        if it_number % 20 == 0 and self.visualise_cam_preds:
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
        self.gaus.export_canonical_npz(self.experiment_dir / "model_canonical.npz")


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
