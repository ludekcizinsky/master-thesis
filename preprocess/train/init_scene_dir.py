from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import tyro
from PIL import Image


@dataclass
class Args:
    video_path: str
    seq_name: str
    cam_id: int
    output_dir: str


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}


def _list_frame_paths(frame_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(frame_dir.glob(ext))
    if not paths:
        return []
    try:
        paths.sort(key=lambda p: int(p.stem))
    except Exception:
        paths.sort()
    return paths


def _ensure_scene_dirs(scene_dir: Path) -> None:
    for name in (
        "all_cameras",
        "images",
        "seg",
        "smpl",
        "smplx",
        "depths",
        "meshes",
        "canon_3dgs_lhm",
    ):
        (scene_dir / name).mkdir(parents=True, exist_ok=True)


def _save_frame_rgb(frame_rgb, out_path: Path) -> None:
    img = Image.fromarray(frame_rgb)
    img.save(out_path, format="JPEG", quality=95)


def _extract_frames_from_video(video_path: Path, out_dir: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    idx = 1
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out_path = out_dir / f"{idx:06d}.jpg"
        _save_frame_rgb(frame_rgb, out_path)
        idx += 1
    cap.release()

    if idx == 1:
        raise RuntimeError(f"No frames extracted from {video_path}")


def _copy_frames_from_dir(frame_dir: Path, out_dir: Path) -> None:
    frame_paths = _list_frame_paths(frame_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frame_dir}")

    for idx, path in enumerate(frame_paths, start=1):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            out_path = out_dir / f"{idx:06d}.jpg"
            rgb.save(out_path, format="JPEG", quality=95)


def main() -> None:
    args = tyro.cli(Args)
    video_path = Path(args.video_path)
    scene_dir = Path(args.output_dir) / args.seq_name
    cam_dir = scene_dir / "images" / str(args.cam_id)

    _ensure_scene_dirs(scene_dir)
    cam_dir.mkdir(parents=True, exist_ok=True)

    if _is_video_file(video_path):
        _extract_frames_from_video(video_path, cam_dir)
    elif video_path.is_dir():
        _copy_frames_from_dir(video_path, cam_dir)
    else:
        raise RuntimeError(f"Unsupported video_path: {video_path}")


if __name__ == "__main__":
    main()
