import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("TORCH_HOME", "/scratch/izar/cizinsky/.cache")
os.environ.setdefault("HF_HOME", "/scratch/izar/cizinsky/.cache")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "training")))

from training.helpers.checkpointing import GaussianCheckpointManager
from training.helpers.dataset import FullSceneDataset
from training.helpers.model_init import create_splats_with_optimizers
from training.helpers.progressive_sam import compute_refined_masks
from training.helpers.visualisation_utils import save_mask_refinement_figure


def _merge_with_defaults(cfg: DictConfig) -> DictConfig:
    defaults = OmegaConf.create(
        {
            "alpha_threshold": 0.3,
            "frame_index": 0,
            "sam2": {
                "model_id": "facebook/sam2.1-hiera-large",
                "mask_threshold": 0.0,
                "multimask_output": False,
                "max_points": 10,
                "use_initial_mask": True,
            },
        }
    )
    merged = OmegaConf.merge(defaults, cfg)
    return merged


def _build_predictor(cfg: DictConfig, device: torch.device):
    gh = GlobalHydra.instance()
    if gh.is_initialized():
        gh.clear()

    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam_device = "cuda" if device.type == "cuda" else "cpu"
    sam_model = build_sam2_hf(cfg.sam2.model_id, device=sam_device)
    predictor = SAM2ImagePredictor(
        sam_model,
        mask_threshold=float(cfg.sam2.get("mask_threshold", 0.0)),
    )
    return predictor


@hydra.main(config_path="configs", config_name="train.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = _merge_with_defaults(cfg)

    alpha_threshold = float(cfg.alpha_threshold)
    frame_index = int(cfg.frame_index)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"--- FYI: using device {device}")

    dataset = FullSceneDataset(
        Path(cfg.preprocess_dir),
        cfg.tids,
        cloud_downsample=int(cfg.cloud_downsample),
        train_bg=bool(cfg.train_bg),
    )
    print(f"--- FYI: dataset has {len(dataset)} frames.")
    if not (0 <= frame_index < len(dataset)):
        raise IndexError(f"Frame index {frame_index} is out of bounds (0..{len(dataset) - 1}).")

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

    image_np = np.clip(sample["image"].cpu().numpy(), 0.0, 1.0)
    image_uint8 = (image_np * 255.0).round().astype(np.uint8)

    predictor = _build_predictor(cfg, device)
    predictor.set_image(image_uint8)

    predictor_cfg = OmegaConf.to_container(cfg.sam2, resolve=True)

    refined_results = compute_refined_masks(
        scene_splats=scene_splats,
        smpl_params=smpl_params,
        w2c=w2c,
        K=K,
        image_size=(H, W),
        alpha_threshold=alpha_threshold,
        predictor=predictor,
        predictor_cfg=predictor_cfg,
        device=device,
    )

    output_root = Path("playground/outputs")
    for idx, result in enumerate(refined_results):
        print(
            f"--- Human {idx}: initial mask {tuple(result.initial_mask.shape)}, refined mask {tuple(result.refined_mask.shape)}"
        )
        print(
            f"    positives: {result.positive_points.shape[0]}, negatives: {result.negative_points.shape[0]}"
        )
        out_path = output_root / f"frame_{frame_index:04d}_human_{idx:02d}.png"
        save_mask_refinement_figure(
            image_np,
            result.initial_mask.numpy(),
            result.refined_mask.numpy(),
            result.vis_positive_points,
            result.vis_negative_points,
            out_path,
        )
        print(f"--- FYI: saved visualization to {out_path}")


if __name__ == "__main__":
    main()
