import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import hydra
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("TORCH_HOME", "/scratch/izar/cizinsky/.cache")
os.environ.setdefault("HF_HOME", "/scratch/izar/cizinsky/.cache")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "training")))

from training.helpers.checkpointing import GaussianCheckpointManager
from training.helpers.dataset import FullSceneDataset
from training.helpers.model_init import create_splats_with_optimizers
from training.helpers.progressive_sam import SamMaskResult, get_sam_masks




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


def _downsample_points(points: Optional[np.ndarray], max_points: int) -> Optional[np.ndarray]:
    if points is None:
        return None
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=int)
    return points[idx]


def _prepare_point_prompts(
    result: SamMaskResult, max_points: int = 10
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    pos = result.positive_points.cpu().numpy() if result.positive_points.numel() > 0 else None
    neg = result.negative_points.cpu().numpy() if result.negative_points.numel() > 0 else None

    pos = _downsample_points(pos, max_points)
    neg = _downsample_points(neg, max_points)

    if pos is None and neg is None:
        return None, None, pos, neg

    coords: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    if pos is not None:
        coords.append(pos)
        labels.append(np.ones(pos.shape[0], dtype=np.int32))
    if neg is not None:
        coords.append(neg)
        labels.append(np.zeros(neg.shape[0], dtype=np.int32))

    point_coords = np.concatenate(coords, axis=0).astype(np.float32)
    point_labels = np.concatenate(labels, axis=0)
    return point_coords, point_labels, pos, neg


def _save_comparison_figure(
    image: np.ndarray,
    original_mask: np.ndarray,
    refined_mask: np.ndarray,
    pos_pts: Optional[np.ndarray],
    neg_pts: Optional[np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].imshow(original_mask.astype(float), cmap="Greens", alpha=0.35)
    legend_added = False
    if pos_pts is not None and pos_pts.size > 0:
        axes[0].scatter(
            pos_pts[:, 0],
            pos_pts[:, 1],
            s=45,
            c="lime",
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            label="positive",
        )
        legend_added = True
    if neg_pts is not None and neg_pts.size > 0:
        axes[0].scatter(
            neg_pts[:, 0],
            neg_pts[:, 1],
            s=45,
            c="red",
            marker="x",
            linewidths=1.2,
            label="negative",
        )
        legend_added = True
    if legend_added:
        axes[0].legend(loc="upper right")
    axes[0].set_title("SAM2 Input Prompts")
    axes[0].set_axis_off()

    axes[1].imshow(image)
    axes[1].imshow(original_mask.astype(float), cmap="Reds", alpha=0.4)
    axes[1].imshow(refined_mask.astype(float), cmap="Blues", alpha=0.6)
    axes[1].set_title("Refined Mask Overlay")
    axes[1].set_axis_off()

    # add labels for the masks in the second subplot
    red_patch = matplotlib.patches.Patch(color="red", label="Original Mask")
    blue_patch = matplotlib.patches.Patch(color="blue", label="Refined Mask")
    axes[1].legend(handles=[red_patch, blue_patch], loc="upper right")

    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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

    mask_results = get_sam_masks(
        scene_splats,
        smpl_params,
        w2c,
        K,
        (H, W),
        alpha_threshold=alpha_threshold,
        device=device,
    )

    image_np = np.clip(sample["image"].cpu().numpy(), 0.0, 1.0)
    image_uint8 = (image_np * 255.0).round().astype(np.uint8)

    predictor = _build_predictor(cfg, device)
    predictor.set_image(image_uint8)
    multimask_output = bool(cfg.sam2.get("multimask_output", False))

    output_root = Path("playground/outputs")

    for idx, result in enumerate(mask_results):
        print(f"--- Human {idx}: mask {tuple(result.mask.shape)}, alpha {tuple(result.alpha.shape)}")
        print(
            f"    positives: {result.positive_points.shape[0]}, negatives: {result.negative_points.shape[0]}"
        )

        point_coords, point_labels, pos_vis, neg_vis = _prepare_point_prompts(
            result, max_points=int(cfg.sam2.get("max_points", 50))
        )

        mask_input = None
        if cfg.sam2.get("use_initial_mask", True):
            mask_np = result.mask.numpy().astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np)[None, None, :, :]
            target_size = predictor.model.sam_prompt_encoder.mask_input_size
            if mask_tensor.shape[-2:] != target_size:
                mask_tensor = F.interpolate(
                    mask_tensor,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            mask_input = mask_tensor.cpu().numpy()

        refined_masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )

        if refined_masks.ndim == 3:
            selected_idx = 0
            if refined_masks.shape[0] > 1:
                selected_idx = int(np.argmax(scores))
            refined_mask = refined_masks[selected_idx]
        else:
            refined_mask = refined_masks

        refined_mask_binary = np.asarray(refined_mask > 0.5, dtype=bool)

        out_path = output_root / f"frame_{frame_index:04d}_human_{idx:02d}.png"
        _save_comparison_figure(
            image_np,
            result.mask.numpy(),
            refined_mask_binary,
            pos_vis,
            neg_vis,
            out_path,
        )
        print(f"--- FYI: saved visualization to {out_path}")


if __name__ == "__main__":
    main()
