from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MASK_EXT = ".png"


@dataclass
class Config:
    scene_dir: Annotated[Path, tyro.conf.arg(aliases=["--scenes-dir"])]
    overlay_alpha: float = 0.5
    max_tracks: int = 8


def _sorted_frames(frames_dir: Path) -> List[Path]:
    frames = [p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

    def sort_key(p: Path) -> Tuple[int, str]:
        return (int(p.stem), p.name) if p.stem.isdigit() else (1_000_000_000, p.name)

    return sorted(frames, key=sort_key)


def _sorted_masks(mask_dir: Path) -> List[Path]:
    masks = [p for p in mask_dir.iterdir() if p.suffix.lower() == MASK_EXT]

    def sort_key(p: Path) -> Tuple[int, str]:
        return (int(p.stem), p.name) if p.stem.isdigit() else (1_000_000_000, p.name)

    return sorted(masks, key=sort_key)


def _track_color_palette() -> List[Tuple[int, int, int]]:
    return [
        (0, 102, 255),    # track 0 - blue
        (255, 165, 0),    # track 1 - orange
        (0, 200, 0),      # track 2 - green
        (220, 0, 0),      # track 3 - red
        (170, 0, 255),    # track 4 - purple
        (0, 200, 200),    # track 5 - cyan
        (255, 215, 0),    # track 6 - gold
        (255, 105, 180),  # track 7 - pink
    ]


def _assert_mask_counts(frames: List[Path], mask_dirs: List[Path]) -> None:
    frame_count = len(frames)
    for mask_dir in mask_dirs:
        masks = _sorted_masks(mask_dir)
        if len(masks) != frame_count:
            msg = (
                f"Mask count mismatch in {mask_dir}: "
                f"{len(masks)} masks vs {frame_count} frames."
            )
            raise AssertionError(msg)


def _numeric_track_dirs(mask_dirs: List[Path]) -> List[Path]:
    numeric_dirs = [d for d in mask_dirs if d.name.isdigit()]
    return sorted(numeric_dirs, key=lambda p: int(p.name))


def _apply_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be single channel.")
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return image
    overlay = image.copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color_arr * alpha
    return overlay


def main() -> None:
    cfg = tyro.cli(Config)

    frames_dir = cfg.scene_dir / "frames"
    masks_dir = cfg.scene_dir / "masks"
    if not frames_dir.exists():
        print(f"Frames dir not found: {frames_dir}")
        sys.exit(1)
    if not masks_dir.exists():
        print(f"Masks dir not found: {masks_dir}")
        sys.exit(1)

    frames = _sorted_frames(frames_dir)
    if not frames:
        print(f"No frames found in {frames_dir}")
        sys.exit(1)

    mask_dirs = [d for d in masks_dir.iterdir() if d.is_dir()]
    if not mask_dirs:
        print(f"No mask subfolders found in {masks_dir}")
        sys.exit(1)

    _assert_mask_counts(frames, mask_dirs)

    track_dirs = _numeric_track_dirs(mask_dirs)
    if len(track_dirs) > cfg.max_tracks:
        raise ValueError(
            f"Found {len(track_dirs)} tracks but max_tracks={cfg.max_tracks}."
        )

    out_dir = cfg.scene_dir / "quality_checks" / "mask_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = _track_color_palette()
    track_masks = [ _sorted_masks(track_dir) for track_dir in track_dirs ]

    for frame_idx, frame_path in enumerate(tqdm(frames, desc="Overlay masks")):
        with Image.open(frame_path) as img:
            image = np.array(img.convert("RGB"), dtype=np.float32)

        for track_idx, mask_paths in enumerate(track_masks):
            mask_path = mask_paths[frame_idx]
            with Image.open(mask_path) as m:
                mask = np.array(m.convert("L"))
            if mask.shape[0] != image.shape[0] or mask.shape[1] != image.shape[1]:
                raise ValueError(
                    f"Mask size mismatch for {mask_path}: "
                    f"{mask.shape} vs {image.shape}"
                )
            color = palette[track_idx % len(palette)]
            image = _apply_overlay(image, mask, color, cfg.overlay_alpha)

        out_path = out_dir / f"{frame_path.stem}.jpg"
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    main()
