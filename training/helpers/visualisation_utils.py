from pathlib import Path
from typing import Optional

import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_mask_refinement_figure(
    image: np.ndarray,
    initial_mask: np.ndarray,
    refined_mask: np.ndarray,
    positive_pts: Optional[np.ndarray],
    negative_pts: Optional[np.ndarray],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image)
    axes[0].imshow(initial_mask.astype(float), cmap="Greens", alpha=0.35)
    legend_items = []
    if positive_pts is not None and positive_pts.size > 0:
        axes[0].scatter(
            positive_pts[:, 0],
            positive_pts[:, 1],
            s=45,
            c="lime",
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            label="positive",
        )
        legend_items.append("positive")
    if negative_pts is not None and negative_pts.size > 0:
        axes[0].scatter(
            negative_pts[:, 0],
            negative_pts[:, 1],
            s=45,
            c="red",
            marker="x",
            linewidths=1.2,
            label="negative",
        )
        legend_items.append("negative")
    if legend_items:
        axes[0].legend(loc="upper right")
    axes[0].set_title("SAM2 Input Prompts")
    axes[0].set_axis_off()

    axes[1].imshow(image)
    axes[1].imshow(initial_mask.astype(float), cmap="Reds", alpha=0.4)
    axes[1].imshow(refined_mask.astype(float), cmap="Blues", alpha=0.6)
    axes[1].set_title("Refined Mask Overlay")
    axes[1].set_axis_off()

    # add legend for red and blue overlays
    red_patch = matplotlib.patches.Patch(color="red", alpha=0.4, label="Initial Mask")
    blue_patch = matplotlib.patches.Patch(color="blue", alpha=0.6, label="Refined Mask")
    axes[1].legend(handles=[red_patch, blue_patch], loc="upper right")

    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_loss_visualization(
    image_input: torch.Tensor,       # [B, H,W, 3], GT image in [0,1]
    gt: torch.Tensor,        # [B, H,W, 3], 0–1
    prediction: torch.Tensor,    # [B, H,W, 3], predicted image in [0,1]
    out_path: str,
):
    """
    Saves a side-by-side visualization of:
    - original image
    - masked image (image * mask)
    - predicted image
    """

    # it may happen that gt and prediction have different sizes due to cropping
    # in which case, resize them to the input image size, use black background
    if image_input.shape[1:3] != gt.shape[1:3]:
        B, H, W = image_input.shape[0:3]
        gt_resized = torch.zeros((B, H, W, 3), device=gt.device)
        # pr_resized = torch.zeros((B, H, W, 3), device=prediction.device)

        h0 = (H - gt.shape[1]) // 2
        w0 = (W - gt.shape[2]) // 2
        h1 = h0 + gt.shape[1]
        w1 = w0 + gt.shape[2]

        gt_resized[:, h0:h1, w0:w1, :] = gt
        # pr_resized[:, h0:h1, w0:w1, :] = prediction

        gt = gt_resized
        # prediction = pr_resized

    comparison = torch.cat([image_input, gt, prediction], dim=2)  # [B,H,3W,3]
    comparison = comparison[0]  # Take first in batch

    # Convert to uint8 for saving
    img = (comparison.cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(img).save(out_path)

    return out_path