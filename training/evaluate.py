import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning
)

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

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

@hydra.main(config_path="../configs", config_name="evaluate.yaml", version_base=None)
def main(cfg):


    print("ℹ️ Loading run config from wandb")
    api = wandb.Api()
    wandb_run_id = cfg.internal_run_id.split("_")[-1]
    run = api.run(f"{cfg.entity}/{cfg.project}/{wandb_run_id}")
    run_cfg = OmegaConf.create(dict(run.config))
    run_cfg.split = cfg.split  # override split if specified in cfg
    print("✅ Run config loaded.\n")

    print("ℹ️ initializing trainer")
    trainer = Trainer(run_cfg, internal_run_id=cfg.internal_run_id)
    device = trainer.device
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



    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    if cfg.lpips_net == "alex":
        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
    elif cfg.lpips_net == "vgg":
        # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=False
        ).to(device)
    else:
        raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

    output_dir = Path(trainer.experiment_dir) / "evaluation" / cfg.split
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    for i in tqdm(range(len(trainer.dataset)), desc="Evaluating"):

        # Forward pass: render
        batch = trainer.dataset[i]  # batch of size 1
        images = batch["image"].to(device).unsqueeze(0)  # [B,H,W,3]
        masks  = batch["mask"].to(device).unsqueeze(0)  # [B,H,W]
        K  = batch["K"].to(device).unsqueeze(0)      # [B, 3,3]
        smpl_param = batch["smpl_param"].to(device).unsqueeze(0)  # [B,86]
        H, W = images.shape[1:3]

        with torch.no_grad():
            rendered_img, _, _ = rasterize_splats(
                trainer=trainer,
                smpl_param=smpl_param,
                K=K,
                img_wh=(W, H),
                sh_degree=run_cfg.sh_degree,
                packed=run_cfg.packed,
                masks=masks
            ) # renders of shape [1,H,W,3]

        # Compute metrics
        masked_image = images * masks.unsqueeze(-1)  # [1,H,W,3]
        masked_image_p = masked_image.permute(0, 3, 1, 2)  # [1, 3, H, W]
        rendered_img_p = rendered_img.permute(0, 3, 1, 2)  # [1, 3, H, W]
        metrics["psnr"].append(psnr(rendered_img_p, masked_image_p))
        metrics["ssim"].append(ssim(rendered_img_p, masked_image_p))
        metrics["lpips"].append(lpips(rendered_img_p, masked_image_p))


    print("\n--- Evaluation results ---")
    for k in metrics.keys():
        metrics[k] = torch.stack(metrics[k]).mean().item()
        print(f"{k}: {metrics[k]:.4f}")
    print("--------------------------\n")


if __name__ == "__main__":
    main()