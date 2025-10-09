import os
import sys
from pathlib import Path

import hydra
import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hydra.core.global_hydra import GlobalHydra

os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"


from training.helpers.checkpointing import GaussianCheckpointManager
from training.helpers.dataset import FullSceneDataset
from training.helpers.model_init import create_splats_with_optimizers
from training.helpers.progressive_sam import get_sam_masks


def _save_visualization(
    image: np.ndarray,
    mask: np.ndarray,
    pos_pts: torch.Tensor,
    neg_pts: torch.Tensor,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.imshow(mask.astype(float), cmap="Greens", alpha=0.35)

    if pos_pts.numel() > 0:
        ax.scatter(
            pos_pts[:, 0],
            pos_pts[:, 1],
            s=30,
            c="lime",
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label="positive",
        )

    if neg_pts.numel() > 0:
        ax.scatter(
            neg_pts[:, 0],
            neg_pts[:, 1],
            s=30,
            c="red",
            marker="x",
            linewidths=1.0,
            label="negative",
        )

    if pos_pts.numel() > 0 or neg_pts.numel() > 0:
        ax.legend(loc="upper right")

    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@hydra.main(config_path="configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.merge(
        OmegaConf.create(
            {
                "alpha_threshold": 0.3,
                "frame_index": 35,
            }
        ),
        cfg,
    )

    alpha_threshold = float(cfg.get("alpha_threshold", 0.3))
    frame_index = int(cfg.get("frame_index", 0))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"--- FYI: using device {device}")

    dataset = FullSceneDataset(
        Path(cfg.preprocess_dir),
        cfg.tids,
        cloud_downsample=int(cfg.cloud_downsample),
        train_bg=bool(cfg.train_bg),
    )
    print(f"--- FYI: dataset has {len(dataset)} frames.")
    if frame_index < 0 or frame_index >= len(dataset):
        raise IndexError(f"Frame index {frame_index} is out of dataset bounds (0-{len(dataset) - 1}).")

    ckpt_manager = GaussianCheckpointManager(
        Path(cfg.output_dir),
        cfg.group_name,
        cfg.tids,
    )

    scene_splats, _, _ = create_splats_with_optimizers(
        device,
        cfg,
        dataset,
        checkpoint_manager=ckpt_manager,
    )

    sample = dataset[frame_index]
    smpl_params = sample["smpl_param"].to(device)
    if smpl_params.ndim == 3:
        smpl_params = smpl_params.squeeze(0)

    w2c = sample["M_ext"].to(device)
    K = sample["K"].to(device)
    H = int(sample["H"])
    W = int(sample["W"])

    results = get_sam_masks(
        scene_splats,
        smpl_params,
        w2c,
        K,
        (H, W),
        alpha_threshold=alpha_threshold,
        device=device,
    )

    image_np = np.clip(sample["image"].cpu().numpy(), 0.0, 1.0)
    output_root = Path("playground/outputs")

    # build sam predictor
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()              # drop the existing global state
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

    for idx, res in enumerate(results):
        print(f"--- Human {idx}: mask {tuple(res.mask.shape)}, alpha {tuple(res.alpha.shape)}")
        print(
            f"    positives: {res.positive_points.shape[0]}, negatives: {res.negative_points.shape[0]}"
        )
        out_path = output_root / f"frame_{frame_index:04d}_human_{idx:02d}.png"
        _save_visualization(
            image_np,
            res.mask.numpy(),
            res.positive_points,
            res.negative_points,
            out_path,
        )
        print(f"--- FYI: saved visualization to {out_path}")


if __name__ == "__main__":
    main()
