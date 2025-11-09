#!/usr/bin/env python3
"""Compute masked SSIM/PSNR/LPIPS for frame triplets and report averages."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.helpers.evaluation_metrics import (  # noqa: E402
    compute_all_metrics,
    aggregate_batch_tid_metric_dicts,
)


def _list_png_names(directory: Path) -> List[str]:
    return sorted(p.name for p in directory.iterdir() if p.suffix.lower() == ".png")


def _common_frame_names(image_dir: Path, mask_dir: Path, render_dir: Path) -> List[str]:
    image_names = set(_list_png_names(image_dir))
    mask_names = set(_list_png_names(mask_dir))
    render_names = set(_list_png_names(render_dir))
    common = sorted(image_names & mask_names & render_names)
    if not common:
        raise RuntimeError("No overlapping frame names found across the provided directories.")
    missing = {
        "images": sorted(render_names - image_names),
        "masks": sorted(render_names - mask_names),
    }
    for label, files in missing.items():
        if files:
            print(f"Warning: {len(files)} {label} missing; first few: {files[:5]}")
    return common


def _load_image(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_mask(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr)


def _chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _resolve_mask_output_dir(renders_dir: Path) -> Path:
    output_dir = renders_dir / "masked_renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_masked_renders(renders: torch.Tensor, masks: torch.Tensor, frame_names: List[str], output_dir: Path) -> None:
    masked = (renders * masks.unsqueeze(-1)).clamp(0.0, 1.0)
    masked_np = (masked.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    for idx, name in enumerate(frame_names):
        Image.fromarray(masked_np[idx]).save(output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-path",
        type=Path,
        default=Path("/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/image"),
        help="Directory with reference RGB frames (default: %(default)s)",
    )
    parser.add_argument(
        "--gt-masks-path",
        type=Path,
        default=Path("/scratch/izar/cizinsky/multiply-output/training/original_taichi/pseudo_gt_masks"),
        help="Directory with ground-truth masks (default: %(default)s)",
    )
    parser.add_argument(
        "--renders-path",
        type=Path,
        default=Path("/scratch/izar/cizinsky/multiply-output/training/original_taichi/joined_fg"),
        help="Directory with rendered RGB frames to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="How many frames to process per batch (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (default: auto-detect)",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    images_dir = args.images_path
    masks_dir = args.gt_masks_path
    renders_dir = args.renders_path
    masked_output_dir = _resolve_mask_output_dir(renders_dir)

    frame_names = _common_frame_names(images_dir, masks_dir, renders_dir)
    print(f"Found {len(frame_names)} common PNG frames. Processing on device: {device}.")
    print(f"Masked renders will be saved to: {masked_output_dir}")

    metric_batches: List[Dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for chunk in _chunked(frame_names, args.batch_size):
            images = torch.stack([_load_image(images_dir / name) for name in chunk], dim=0).to(device)
            masks = torch.stack([_load_mask(masks_dir / name) for name in chunk], dim=0).to(device)
            renders = torch.stack([_load_image(renders_dir / name) for name in chunk], dim=0).to(device)

            metrics = compute_all_metrics(images, masks, renders)
            metric_batches.append({k: v.detach().cpu() for k, v in metrics.items()})
            _save_masked_renders(renders, masks, chunk, masked_output_dir)

    averages = aggregate_batch_tid_metric_dicts(metric_batches)
    print("\nAverage metrics across frames:")
    for name in ("ssim", "psnr", "lpips"):
        value = averages.get(name, None)
        if value is None:
            continue
        print(f"  {name.upper():>5}: {value:.4f}")


if __name__ == "__main__":
    main()
