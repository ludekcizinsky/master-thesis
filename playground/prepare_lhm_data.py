#!/usr/bin/env python3
"""
Export refined masks from .pt files to binary PNG files.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import torch
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert refined SAM masks saved as .pt files into binary PNG masks."
    )
    parser.add_argument(
        "--input_mask_dir",
        type=Path,
        help="Directory containing *.pt mask files (e.g., mask_0000.pt).",
    )
    parser.add_argument(
        "--input_img_dir",
        type=Path,
        help="Directory containing corresponding input images (e.g., frame_0000.png).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Destination directory where binary masks will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to the refined mask (>= becomes 1). Default: 0.5",
    )
    return parser.parse_args()


def frame_id_from_path(path: Path, index: int, fallback_width: int) -> str:
    """Extract a frame id from the filename or fall back to a zero-padded index."""
    match = re.search(r"(\d+)(?=\.pt$)", path.name)
    if match:
        return match.group(1)
    return f"{index:0{fallback_width}d}"


def save_binary_png(mask: torch.Tensor, output_path: Path) -> None:
    """Persist a (H, W) uint8 tensor to disk as a PNG, using available backend."""
    array = mask.cpu().numpy().astype("uint8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image

        Image.fromarray(array, mode="L").save(output_path)
        return
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio

        imageio.imwrite(output_path, array)
    except ImportError as exc:
        raise SystemExit(
            f"Install Pillow or imageio to save PNG files (missing: {exc.name})."
        ) from exc

def save_png(tensor: torch.Tensor, output_path: Path) -> None:
    """Persist a (C, H, W) uint8 tensor to disk as a PNG, using available backend."""
    array = tensor.cpu().numpy().astype("uint8").transpose(1, 2, 0)  # H, W, C
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image

        Image.fromarray(array, mode="RGB").save(output_path)
        return
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio

        imageio.imwrite(output_path, array)
    except ImportError as exc:
        raise SystemExit(
            f"Install Pillow or imageio to save PNG files (missing: {exc.name})."
        ) from exc

def load_refined(pt_path: Path) -> torch.Tensor:
    """Load a .pt file and return the refined mask tensor."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "refined" not in data:
        raise ValueError(f"File {pt_path} does not contain a 'refined' entry.")

    refined = data["refined"]
    if not isinstance(refined, torch.Tensor):
        raise ValueError(f"'refined' in {pt_path} is not a torch.Tensor.")
    if refined.ndim == 2:
        refined = refined.unsqueeze(0)
    if refined.ndim != 3:
        raise ValueError(f"'refined' tensor in {pt_path} must be 3D (P, H, W).")
    return refined

def load_image(img_path: Path) -> torch.Tensor:
    """Load an image file and return it as a tensor."""
    try:
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # C, H, W
        return img_tensor
    except ImportError:
        raise SystemExit("Install Pillow to load images.")

def process_masks(pt_files: Iterable[Path], img_files: Iterable[Path], output_dir: Path, threshold: float) -> None:
    pt_files = list(sorted(pt_files))
    img_files = list(sorted(img_files))

    fallback_width = max(4, len(str(max(len(pt_files) - 1, 0))))
    expected_people: Optional[int] = None
    frame_counter = 0

    for pt_path in pt_files:
        try:
            refined = load_refined(pt_path)
        except ValueError as exc:
            print(f"Skipping {pt_path.name}: {exc}", file=sys.stderr)
            continue

        img_path = img_files[frame_counter]
        img_tensor = load_image(img_path)

        if expected_people is None:
            expected_people = refined.shape[0]
        elif refined.shape[0] != expected_people:
            raise ValueError(
                f"Inconsistent number of people: {pt_path} has {refined.shape[0]}, "
                f"expected {expected_people}."
            )

        frame_id = frame_id_from_path(pt_path, frame_counter, fallback_width)
        binary = (refined >= threshold).to(torch.uint8) * 255

        for person_idx in range(binary.shape[0]):
            # prepare output directory
            masks_dir = output_dir / "masks" / f"{person_idx:02d}"
            masks_dir.mkdir(parents=True, exist_ok=True)
            masked_images_dir = output_dir / "masked_images" / f"{person_idx:02d}"
            masked_images_dir.mkdir(parents=True, exist_ok=True)

            # save binary mask
            out_path = masks_dir / f"{frame_id}.png"
            mask_to_save = binary[person_idx]
            save_binary_png(mask_to_save, out_path)

            # save masked image
            masked_img = img_tensor.clone()
            masked_img[:, mask_to_save == 0] = 0  
            out_path = masked_images_dir / f"{frame_id}.png"
            save_png(masked_img, out_path)

        print(
            f"Processed {pt_path.name}: {binary.shape[0]} masks -> frame {frame_id}",
            file=sys.stderr,
        )
        frame_counter += 1


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    process_masks(args.input_mask_dir.glob("*.pt"), args.input_img_dir.glob("*.png"), args.output_dir, args.threshold)

    print(f"All done! Binary masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
