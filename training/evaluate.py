import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path
import json

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
from matplotlib import pyplot as plt
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
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

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
        images = images.clamp(0, 1).float()          # [1,H,W,3]
        rendered_img = rendered_img.clamp(0, 1).float()

        # Apply the SAME mask to both pred & gt (if you want masked evaluation)
        masks = (masks > 0.5).float()                # [1,H,W]
        rendered_img_masked = rendered_img * masks.unsqueeze(-1)
        masked_image       = images      * masks.unsqueeze(-1)

        # Optional but recommended: bbox crop around the mask to avoid huge easy background
        ys, xs = torch.where(masks[0] > 0.5)
        if ys.numel() > 0:
            pad = 8
            y0 = max(int(ys.min().item()) - pad, 0)
            y1 = min(int(ys.max().item()) + pad + 1, images.shape[1])
            x0 = max(int(xs.min().item()) - pad, 0)
            x1 = min(int(xs.max().item()) + pad + 1, images.shape[2])

            gt_crop  = masked_image[:, y0:y1, x0:x1, :]
            pr_crop  = rendered_img_masked[:, y0:y1, x0:x1, :]
        else:
            # fallback if mask empty
            gt_crop, pr_crop = masked_image, rendered_img_masked

        # NCHW for metrics
        gt_p = gt_crop.permute(0, 3, 1, 2).contiguous()
        pr_p = pr_crop.permute(0, 3, 1, 2).contiguous()

        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

        lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=("alex" if cfg.lpips_net=="alex" else "vgg"),
            normalize=True
        ).to(device)

        metrics["psnr"].append(psnr(pr_p, gt_p))
        metrics["ssim"].append(ssim(pr_p, gt_p))
        metrics["lpips"].append(lpips(pr_p, gt_p))

        # Save image
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow((masked_image[0].cpu().numpy() * 255).astype(np.uint8))
        axs[0].set_title("Ground truth")
        axs[0].axis("off")
        axs[1].imshow((rendered_img[0].cpu().numpy() * 255).astype(np.uint8))
        axs[1].set_title(f"Rendered (PSNR ↑: {metrics['psnr'][-1]:.2f}, SSIM ↑: {metrics['ssim'][-1]:.3f}, LPIPS ↓: {metrics['lpips'][-1]:.3f})")
        axs[1].axis("off")
        plt.tight_layout()
        plt.savefig(image_dir / f"{i:05d}.png")
        plt.close(fig)


    # aggregate metrics
    # - mean
    agg_metrics = {f"eval/{k}_{cfg.split}_mean": float(torch.stack(v).mean().item()) for k, v in metrics.items()}
    # - std
    agg_metrics.update({f"eval/{k}_{cfg.split}_std": float(torch.stack(v).std().item()) for k, v in metrics.items()})
    # - num samples
    agg_metrics["eval/val_samples"] = len(trainer.dataset)

    # save metrics as json
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(agg_metrics, f, indent=4)
    print(f"✅ Saved evaluation metrics to {output_dir / 'metrics.json'}: {agg_metrics}")

    # upload the aggregate metrics to wandb
    if cfg.log_to_wandb:
        # resume the wandb run
        wandb_path = Path("/scratch/izar/cizinsky/thesis/")
        wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            id=wandb_run_id,
            resume="must",
            dir=wandb_path,
        )

        # log as summary metrics
        wandb.run.summary.update(agg_metrics)
        print(f"✅ Uploaded evaluation metrics to wandb run {wandb_run_id}: {agg_metrics}")
        wandb.finish()


if __name__ == "__main__":
    main()