#!/usr/bin/env python3
"""Generate boolean masks for non-white pixels in PNG frame sequences."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


def iter_png_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() == ".png":
            yield path


def build_mask(image_path: Path, tolerance: int) -> Image.Image:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        data = np.asarray(rgb, dtype=np.uint8)
    diff_from_white = 255 - data
    non_white = np.any(diff_from_white > tolerance, axis=-1)
    mask = (non_white.astype(np.uint8) * 255)
    return Image.fromarray(mask)


def apply_mask(image_path: Path, mask: Image.Image) -> Image.Image:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb_data = np.asarray(rgb, dtype=np.uint8)
    mask_data = np.asarray(mask, dtype=np.uint8)
    if mask_data.shape != rgb_data.shape[:2]:
        raise ValueError(f"Mask and image sizes differ for {image_path}")
    keep = mask_data > 0
    result = np.zeros_like(rgb_data)
    result[keep] = rgb_data[keep]
    return Image.fromarray(result)


def save_masks(
    src: Path,
    dst: Path,
    tolerance: int,
    visualize_src: Optional[Path] = None,
    visualize_dst: Optional[Path] = None,
) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    if (visualize_src is None) != (visualize_dst is None):
        raise ValueError("Both visualize_src and visualize_dst must be provided together")
    if visualize_dst is not None:
        visualize_dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for png_file in iter_png_files(src):
        mask = build_mask(png_file, tolerance)
        mask.save(dst / png_file.name)
        if visualize_src and visualize_dst:
            source_image = visualize_src / png_file.name
            if not source_image.exists():
                raise FileNotFoundError(f"Missing source frame: {source_image}")
            visual = apply_mask(source_image, mask)
            visual.save(visualize_dst / png_file.name)
        count += 1
    return count


def bounded_int(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 255:
        raise argparse.ArgumentTypeError("value must be between 0 and 255")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="Directory containing source PNG frames")
    parser.add_argument("dst", type=Path, help="Output directory for boolean masks")
    parser.add_argument(
        "--tolerance",
        type=bounded_int,
        default=20,
        help="How far from pure white (0-255) a pixel must be to count as foreground (default: 20)",
    )
    parser.add_argument(
        "--visualize-src",
        type=Path,
        help="Directory with original frames to apply masks (optional)",
    )
    parser.add_argument(
        "--visualize-dst",
        type=Path,
        help="Directory where visualizations should be saved (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = save_masks(
        args.src,
        args.dst,
        args.tolerance,
        visualize_src=args.visualize_src,
        visualize_dst=args.visualize_dst,
    )
    print(f"Saved {count} masks to {args.dst}")


if __name__ == "__main__":
    main()
