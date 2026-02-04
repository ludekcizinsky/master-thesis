from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import sys

import numpy as np
import tyro

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from preprocess.custom.helpers.cameras_and_smplx_reformat import (
    generate_static_cameras,
    load_skip_frames,
)


@dataclass
class Args:
    scene_dir: Path
    num_of_cameras: int


def _collect_frame_numbers(smplx_dir: Path) -> List[int]:
    frames = [p.stem for p in smplx_dir.glob("*.npz") if p.is_file()]
    frames = [f for f in frames if f.isdigit()]
    return sorted([int(f) for f in frames])


def _infer_fname_num_digits(frame_numbers: List[int]) -> int:
    if not frame_numbers:
        return 6
    return max(6, len(str(frame_numbers[0])))


def _load_centers_world(smplx_dir: Path, frame_numbers: List[int], fname_num_digits: int):
    centers_world = []
    centers_frame_numbers = []
    for frame_number in frame_numbers:
        smplx_path = smplx_dir / f"{frame_number:0{fname_num_digits}d}.npz"
        if not smplx_path.exists():
            continue
        with np.load(smplx_path) as data:
            if "trans" not in data:
                continue
            trans = data["trans"]
            if trans.size == 0:
                continue
            centers_world.append(np.mean(trans, axis=0))
            centers_frame_numbers.append(frame_number)
    return centers_world, centers_frame_numbers


def _collect_camera_ids(camera_dir: Path) -> List[int]:
    cam_ids = []
    for p in camera_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            cam_ids.append(int(p.name))
    return sorted(cam_ids)


def main() -> None:
    args = tyro.cli(Args)
    scene_dir = args.scene_dir

    smplx_dir = scene_dir / "smplx"
    cameras_dir = scene_dir / "all_cameras"
    if not smplx_dir.exists():
        raise FileNotFoundError(f"Missing smplx directory: {smplx_dir}")
    if not cameras_dir.exists():
        raise FileNotFoundError(f"Missing all_cameras directory: {cameras_dir}")

    frame_numbers = _collect_frame_numbers(smplx_dir)
    if not frame_numbers:
        raise RuntimeError(f"No SMPL-X frames found in {smplx_dir}")

    skip_frames = set(load_skip_frames(scene_dir))
    if skip_frames:
        frame_numbers = [f for f in frame_numbers if f not in skip_frames]
    if not frame_numbers:
        raise RuntimeError("No frames left after applying skip_frames.csv.")

    fname_num_digits = _infer_fname_num_digits(frame_numbers)

    centers_world, centers_frame_numbers = _load_centers_world(
        smplx_dir, frame_numbers, fname_num_digits
    )
    if not centers_world:
        raise RuntimeError("No valid SMPL-X translations found to estimate camera centers.")

    cam_ids = _collect_camera_ids(cameras_dir)
    if not cam_ids:
        raise RuntimeError(f"No camera ids found under {cameras_dir}")
    if len(cam_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one source camera directory under {cameras_dir}, found {len(cam_ids)}"
        )
    src_cam_id = cam_ids[0]
    start_cam_id = 100

    generate_static_cameras(
        scene_root_dir=scene_dir,
        frame_numbers=frame_numbers,
        centers_world=np.stack(centers_world, axis=0),
        centers_frame_numbers=centers_frame_numbers,
        body_radius_m=1.3,
        src_cam_id=src_cam_id,
        start_cam_id=start_cam_id,
        num_cams=args.num_of_cameras,
        fname_num_digits=fname_num_digits,
        enforce_coverage=False,
        up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )


if __name__ == "__main__":
    main()
