#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import torch
from PIL import Image
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.helpers.evaluation_metrics import (  # noqa: E402
    compute_all_metrics,
    aggregate_batch_tid_metric_dicts,
)

from evaluation.helpers.misc import (
    load_image,
    save_masked_renders,
)

from evaluation.helpers.segmentation import (
    load_masks_for_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-path",
        type=Path,
        help="Directory with reference RGB frames",
    )
    parser.add_argument(
        "--renders-path",
        type=Path,
        help="Directory with rendered RGB frames to evaluate",
    )
    parser.add_argument(
        "--gt-masks-path",
        type=Path,
        help="Directory with ground-truth masks",
    )
    parser.add_argument(
        "--gt-masks-ds-type",
        type=str,
        help="Dataset type for ground-truth masks (e.g., 'multiply', 'progressive_sam')",
    )
    parser.add_argument(
        "--pred-masks-path",
        type=Path,
        help="Directory with predicted masks",
        default=None,
    )
    parser.add_argument(
        "--pred-masks-ds-type",
        type=str,
        help="Dataset type for predicted masks (e.g., 'multiply', 'progressive_sam')",
        default=None,
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
    renders_dir = args.renders_path
    gt_masks_dir = args.gt_masks_path
    pred_masks_dir = args.pred_masks_path
    masked_output_dir = renders_dir / "masked_renders"
    masked_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_dir = args.metrics_output_path
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    # Load data for evaluation
    frame_names = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() == ".png")
    images = torch.stack([load_image(images_dir / name) for name in frame_names], dim=0)
    renders = torch.stack([load_image(renders_dir / name) for name in frame_names], dim=0)
    gt_masks, pred_masks = load_masks_for_evaluation(
        gt_masks_dir_path=gt_masks_dir,
        gt_ds=args.gt_masks_ds_type,
        pred_masks_dir_path=pred_masks_dir,
        pred_ds=args.pred_masks_ds_type,
        device="cpu",
    )
    gt_smpl_joints = None
    pred_smpl_joints = None
    print(f"Found {len(frame_names)} PNG frames. Processing on device: {device}.")

    # Compute Metrics
    metric_batches: List[Dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for batch_start in range(0, len(frame_names), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(frame_names))
            print(f"Processing frames {batch_start} to {batch_end-1}...")

            images_batch = images[batch_start:batch_end].to(device)
            renders_batch = renders[batch_start:batch_end].to(device)
            gt_masks_batch = gt_masks[batch_start:batch_end].to(device)
            if pred_masks is not None:
                pred_masks_batch = pred_masks[batch_start:batch_end].to(device)
            else:
                pred_masks_batch = None

            frame_names_batch = frame_names[batch_start:batch_end]
            metrics = compute_all_metrics(images_batch, gt_masks_batch, renders_batch, pred_masks_batch, gt_smpl_joints, pred_smpl_joints)
            metric_batches.append({k: v.detach().cpu() for k, v in metrics.items()})
            save_masked_renders(renders_batch, gt_masks_batch, frame_names_batch, masked_output_dir)

    averages = aggregate_batch_tid_metric_dicts(metric_batches)
    print("\nAverage metrics across frames:")
    for name in sorted(averages.keys()):
        value = averages.get(name, None)
        if value is None:
            continue
        print(f"  {name.upper():>5}: {value:.4f}")
    
    # save the metrics to csv file
    metrics_csv_path = metrics_output_dir / "metrics.csv"
    with open(metrics_csv_path, "w") as f:
        f.write("metric,value\n")
        for name, value in averages.items():
            f.write(f"{name},{value:.6f}\n")
    print(f"\nMetrics saved to: {metrics_csv_path}")


if __name__ == "__main__":
    main()
