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

import subprocess

def stack_videos_with_titles(top_path: str, bottom_path: str, 
                             top_title: str, bottom_title: str, 
                             output_path: str):
    """
    Stack two videos vertically with titles overlaid.

    Args:
        top_path (str): Path to the top video.
        bottom_path (str): Path to the bottom video.
        top_title (str): Title text for the top video.
        bottom_title (str): Title text for the bottom video.
        output_path (str): Path where the output video will be saved.
    """
    filter_complex = (
        f"[0:v]drawtext=text='{top_title}':fontcolor=white:fontsize=36:x=10:y=10[v0]; "
        f"[1:v]drawtext=text='{bottom_title}':fontcolor=white:fontsize=36:x=10:y=10[v1]; "
        f"[v0][v1]vstack=inputs=2[out]"
    )

    cmd = [
        "ffmpeg", "-y",  # overwrite output if exists
        "-i", top_path,
        "-i", bottom_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        output_path
    ]

    subprocess.run(cmd, check=True)


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
    model_dir = trainer.experiment_dir / "checkpoints"
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
    masked_images_dir = output_dir / "masked_images"
    masked_images_dir.mkdir(parents=True, exist_ok=True)

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

        # Save the render
        render_np = (renders[0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(render_np).save(frames_dir / f"frame_{i:05d}.png")

        # Save the masked image
        # - first get masked image
        masked_image = images * masks.unsqueeze(-1)  # [1,H,W,3]
        # - then convert to PIL and save
        masked_image_np = (masked_image[0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(masked_image_np).save(masked_images_dir / f"frame_{i:05d}.png")

    frames_to_video(frames_dir, output_dir / "predictions.mp4", framerate=12)
    frames_to_video(masked_images_dir, output_dir / "masked_images.mp4", framerate=12)
    stack_videos_with_titles(
        top_path=str(output_dir / "predictions.mp4"),
        bottom_path=str(output_dir / "masked_images.mp4"),
        top_title="Prediction",
        bottom_title="GT",
        output_path=str(output_dir / "comparison.mp4")
    )
    print(f"✅ Done. Comparison video saved to {output_dir / 'comparison.mp4'}\n")

if __name__ == "__main__":
    main()