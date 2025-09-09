import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import torch

from preprocess.helpers.video_utils import frames_to_video
from training.helpers.render import rasterize_splats
from training.run import Trainer

import wandb
import numpy as np
import hydra

from omegaconf import OmegaConf


@hydra.main(config_path="../configs", config_name="show2d.yaml", version_base=None)
def main(cfg):

    print("ℹ️ Loading run config from wandb")
    api = wandb.Api()
    wandb_run_id = cfg.internal_run_id.split("_")[-1]
    run = api.run(f"{cfg.entity}/{cfg.project}/{wandb_run_id}")
    run_cfg = OmegaConf.create(dict(run.config))
    print("✅ Run config loaded.\n")

    print("ℹ️ initializing trainer")
    trainer = Trainer(run_cfg, internal_run_id=cfg.internal_run_id)
    print("✅ Trainer initialized.\n")

    print("ℹ️ Loading model parameters from npz")
    model_dir = trainer.experiment_dir / "model"
    if "model_canonical_final.npz" not in os.listdir(model_dir):
        print(f"--- FYI: 'model_canonical_final.npz' not found in {model_dir}, loading most recent model instead")
        npz_path = sorted(list(model_dir.glob("model_canonical_it*.npz")))[-1]
    else:
        npz_path = model_dir / "model_canonical_final.npz"
    print(f"--- FYI: loading model from {npz_path}")
    trainer.load_canonical_npz(npz_path)
    print("✅ Model parameters loaded.\n")

    print("ℹ️ Rendering and saving visualizations")
    device = trainer.device
    output_dir = Path(trainer.experiment_dir) / "renders_2d"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(trainer.dataset)), desc="Rendering"):
        batch = trainer.dataset[i]  # batch of size 1
        images = batch["image"].to(device).unsqueeze(0)  # [B,H,W,3]
        masks  = batch["mask"].to(device).unsqueeze(0)  # [B,H,W]
        K  = batch["K"].to(device).unsqueeze(0)      # [B, 3,3]
        smpl_param = batch["smpl_param"].to(device).unsqueeze(0)  # [B,86]
        H, W = images.shape[1:3]

        # Forward pass: render
        with torch.no_grad():
            renders, alphas, info = rasterize_splats(
                trainer=trainer,
                smpl_param=smpl_param,
                K=K,
                img_wh=(W, H),
                sh_degree=run_cfg.sh_degree,
                packed=run_cfg.packed,
                masks=masks
            ) # renders of shape [1,H,W,3]

        # Save the render only
        render_np = (renders[0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(render_np).save(frames_dir / f"frame_{i:05d}.png")

    print(f"--- FYI: Save the frames to {frames_dir}")
    frames_to_video(frames_dir, output_dir / "render.mp4", framerate=12)
    print(f"✅ Done. Video saved to {output_dir / 'render.mp4'}\n")

if __name__ == "__main__":
    main()