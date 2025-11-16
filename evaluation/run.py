#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.helpers.evaluation_metrics import (  # noqa: E402
    compute_all_metrics,
    aggregate_batch_tid_metric_dicts,
)

from evaluation.helpers.rendering import (  # noqa: E402
    common_frame_names,
    load_image,
    load_mask,
    save_masked_renders,
)

from evaluation.helpers.misc import chunked 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-path",
        type=Path,
        help="Directory with reference RGB frames",
    )
    parser.add_argument(
        "--gt-masks-path",
        type=Path,
        help="Directory with ground-truth masks",
    )
    parser.add_argument(
        "--renders-path",
        type=Path,
        help="Directory with rendered RGB frames to evaluate",
    )
    parser.add_argument(
        "--metrics-output-path",
        type=Path,
        help="Directory to save computed metrics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many frames to process per batch",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default: auto-detect)",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    device = args.device

    # Directories
    images_dir = args.images_path
    masks_dir = args.gt_masks_path
    renders_dir = args.renders_path
    masked_output_dir = renders_dir / "masked_renders"
    masked_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_dir = args.metrics_output_path
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Rendering metrics
    frame_names = common_frame_names(images_dir, masks_dir, renders_dir)
    print(f"Found {len(frame_names)} common PNG frames. Processing on device: {device}.")
    print(f"Masked renders will be saved to: {masked_output_dir}")

    metric_batches: List[Dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for chunk in chunked(frame_names, args.batch_size):
            images = torch.stack([load_image(images_dir / name) for name in chunk], dim=0).to(device)
            masks = torch.stack([load_mask(masks_dir / name) for name in chunk], dim=0).to(device)
            renders = torch.stack([load_image(renders_dir / name) for name in chunk], dim=0).to(device)

            metrics = compute_all_metrics(images, masks, renders)
            metric_batches.append({k: v.detach().cpu() for k, v in metrics.items()})
            save_masked_renders(renders, masks, chunk, masked_output_dir)

    averages = aggregate_batch_tid_metric_dicts(metric_batches)
    print("\nAverage metrics across frames:")
    for name in ("ssim", "psnr", "lpips"):
        value = averages.get(name, None)
        if value is None:
            continue
        print(f"  {name.upper():>5}: {value:.4f}")
    
    # save the metrics to cvs file
    metrics_csv_path = metrics_output_dir / "rendering_metrics.csv"
    with open(metrics_csv_path, "w") as f:
        f.write("metric,value\n")
        for name, value in averages.items():
            f.write(f"{name},{value:.6f}\n")
    print(f"\nMetrics saved to: {metrics_csv_path}")

if __name__ == "__main__":
    main()
