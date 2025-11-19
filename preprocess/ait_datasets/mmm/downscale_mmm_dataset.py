#!/usr/bin/env python
"""Downscale MMM dataset assets and adjust camera parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downscale MMM dataset images and camera intrinsics. "
            "Creates new *_down{factor}x outputs next to the originals."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to a sequence directory (contains org_image/, stage_img/, cameras/, opt_cam/).",
    )
    parser.add_argument(
        "--org-factor",
        type=int,
        default=2,
        help="Downscale factor for org_image/ and opt_cam/ assets (default: 2).",
    )
    parser.add_argument(
        "--stage-factor",
        type=int,
        default=4,
        help="Downscale factor for stage_img/ and cameras/ assets (default: 4).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Temporal downsample factor (use every Nth frame; default: 1).",
    )
    return parser.parse_args()


def valid_images(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def downscale_image(src: Path, dst: Path, factor: int) -> None:
    with Image.open(src) as img:
        new_width = max(1, img.width // factor)
        new_height = max(1, img.height // factor)
        resized = img.resize((new_width, new_height), Image.LANCZOS)
        ensure_dir(dst.parent)
        resized.save(dst)


def downscale_flat_folder(
    src: Path, dst: Path, factor: int, downsample: int
) -> None:
    if not src.exists():
        print(f"[WARN] Source folder {src} missing, skipping.")
        return
    ensure_dir(dst)
    files = list(valid_images(sorted(src.iterdir())))
    if not files:
        print(f"[INFO] No images found in {src}, nothing to do.")
        return
    selected = files[::downsample]
    print(
        f"[INFO] Downscaling {len(selected)} / {len(files)} images "
        f"from {src} -> {dst} (every {downsample})"
    )
    for image_path in selected:
        target = dst / image_path.name
        downscale_image(image_path, target, factor)


def downscale_stage_images(
    src_root: Path, dst_root: Path, factor: int, downsample: int
) -> None:
    if not src_root.exists():
        print(f"[WARN] Stage image folder {src_root} missing, skipping.")
        return
    for camera_dir in sorted(path for path in src_root.iterdir() if path.is_dir()):
        target_dir = dst_root / camera_dir.name
        downscale_flat_folder(camera_dir, target_dir, factor, downsample)


def scale_intrinsics_matrix(matrix: np.ndarray, scale: float) -> np.ndarray:
    scaled = np.array(matrix, dtype=np.float64, copy=True)
    scaled[..., :2, :] *= scale
    return scaled


def update_rgb_cameras(
    cameras_dir: Path, scale_factor: int, org_factor_for_name: int, downsample: int
) -> None:
    scale = 1.0 / scale_factor
    rgb_path = cameras_dir / "rgb_cameras.npz"
    if not rgb_path.exists():
        print(f"[WARN] No rgb_cameras.npz under {cameras_dir}, skipping.")
        return
    data = np.load(rgb_path)
    save_path = cameras_dir / (
        f"rgb_cameras_down{org_factor_for_name}x_subsample{downsample}x.npz"
    )
    updated = {}
    for key in data.files:
        arr = data[key]
        if key in {"intrinsics", "intrinsics_ori"}:
            updated[key] = scale_intrinsics_matrix(arr, scale)
        elif key == "shape":
            new_shape = np.maximum(arr // scale_factor, 1)
            updated[key] = new_shape
        else:
            updated[key] = arr
    np.savez(save_path, **updated)
    print(f"[INFO] Wrote {save_path}")


def update_opt_cam(opt_cam_dir: Path, factor: int, downsample: int) -> None:
    if not opt_cam_dir.exists():
        print(f"[WARN] opt_cam directory {opt_cam_dir} missing, skipping.")
        return
    scale = 1.0 / factor
    target_dir = (
        opt_cam_dir.parent
        / f"{opt_cam_dir.name}_down{factor}x_subsample{downsample}x"
    )
    ensure_dir(target_dir)
    files = sorted(opt_cam_dir.glob("*.npz"))
    if not files:
        print(f"[INFO] No .npz files in {opt_cam_dir}, nothing to do.")
        return
    selected = files[::downsample]
    for src in selected:
        data = np.load(src)
        updated = {key: data[key] for key in data.files}
        if "K" in updated:
            updated["K"] = scale_intrinsics_matrix(updated["K"], scale)
        dst = target_dir / src.name
        np.savez(dst, **updated)
    print(f"[INFO] Wrote {len(files)} opt_cam files to {target_dir}")


def main() -> None:
    args = parse_args()
    if args.org_factor < 1 or args.stage_factor < 1:
        raise ValueError("Downscale factors must be >= 1")
    if args.downsample < 1:
        raise ValueError("Downsample factor must be >= 1")
    root = args.dataset_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root {root} does not exist")

    print(
        f"[INFO] Downscaling dataset at {root} "
        f"(org_image x{args.org_factor}, stage_img x{args.stage_factor}, "
        f"downsample every {args.downsample})"
    )
    org_src = None
    for candidate in ("org_image", "org_img"):
        path = root / candidate
        if path.exists():
            org_src = path
            break
    if org_src is None:
        raise FileNotFoundError(
            f"Neither org_image nor org_img exists in {root}, nothing to process."
        )
    org_dst = (
        root
        / f"{org_src.name}_down{args.org_factor}x_subsample{args.downsample}x"
    )
    downscale_flat_folder(
        org_src,
        org_dst,
        args.org_factor,
        args.downsample,
    )
    stage_src = root / "stage_img"
    if stage_src.exists():
        downscale_stage_images(
            stage_src,
            root
            / f"stage_img_down{args.stage_factor}x_subsample{args.downsample}x",
            args.stage_factor,
            args.downsample,
        )
    else:
        print("[INFO] stage_img directory missing; skipping stage outputs.")

    cameras_dir = root / "cameras"
    if cameras_dir.exists():
        update_rgb_cameras(
            cameras_dir, args.stage_factor, args.org_factor, args.downsample
        )
    else:
        print("[INFO] cameras directory missing; skipping rgb_cameras output.")

    opt_cam_dir = root / "opt_cam"
    if opt_cam_dir.exists():
        update_opt_cam(opt_cam_dir, args.org_factor, args.downsample)
    else:
        print("[INFO] opt_cam directory missing; skipping opt_cam outputs.")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
