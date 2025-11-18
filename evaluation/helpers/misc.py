from typing import Iterable, List
import numpy as np
import torch
from pathlib import Path
from typing import List

from PIL import Image


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]


def _list_png_names(directory: Path) -> List[str]:
    return sorted(p.name for p in directory.iterdir() if p.suffix.lower() == ".png")


def common_frame_names(image_dir: Path, mask_dir: Path, render_dir: Path) -> List[str]:
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


def load_image(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)

def save_masked_renders(renders: torch.Tensor, masks: torch.Tensor, frame_names: List[str], output_dir: Path) -> None:
    masked = (renders * masks.unsqueeze(-1)).clamp(0.0, 1.0)
    masked_np = (masked.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    for idx, name in enumerate(frame_names):
        Image.fromarray(masked_np[idx]).save(output_dir / name)

def save_renders(renders: torch.Tensor, frame_names: List[str], output_dir: Path) -> None:
    renders_np = (renders.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
    for idx, name in enumerate(frame_names):
        Image.fromarray(renders_np[idx]).save(output_dir / name)