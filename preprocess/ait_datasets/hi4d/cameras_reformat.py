from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def root_dir_to_source_cameras_file(root_dir: Path) -> Path:
    return root_dir / "cameras" / "rgb_cameras.npz"


def root_dir_to_target_cameras_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "all_cameras" / f"{cam_id}"


def root_dir_to_image_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "images" / f"{cam_id}"


def collect_frame_files(scene_root_dir: Path, cam_id: int) -> list[Path]:
    image_dir = root_dir_to_image_dir(scene_root_dir, cam_id)
    if image_dir.exists():
        files = sorted(
            f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES
        )
        if files:
            return files

    raise FileNotFoundError(
        "No image files found under images/<cam_id>"
    )


@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15"
    first_frame_number: int = 1
    fname_num_digits: int = 6


def _normalize_intrinsics(intrinsics: np.ndarray) -> np.ndarray:
    if intrinsics.shape == (3, 3):
        return intrinsics[None, ...]
    if intrinsics.shape == (1, 3, 3):
        return intrinsics
    raise ValueError(f"Unexpected intrinsics shape: {intrinsics.shape}")


def _normalize_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape == (3, 4):
        return extrinsics[None, ...]
    if extrinsics.shape == (1, 3, 4):
        return extrinsics
    raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")


def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    scene_root_dir = Path(cfg.scene_root_dir)

    cameras_file = root_dir_to_source_cameras_file(scene_root_dir)
    cameras_data = np.load(cameras_file, allow_pickle=True)
    cam_ids = cameras_data["ids"]
    intrinsics_all = cameras_data["intrinsics"]
    extrinsics_all = cameras_data["extrinsics"]

    if len(cam_ids) != intrinsics_all.shape[0] or len(cam_ids) != extrinsics_all.shape[0]:
        raise ValueError("Camera ids count does not match intrinsics/extrinsics.")

    for idx, cam_id in enumerate(cam_ids):
        cam_id_int = int(cam_id)
        frame_files = collect_frame_files(scene_root_dir, cam_id_int)

        intrinsics = _normalize_intrinsics(intrinsics_all[idx])
        extrinsics = _normalize_extrinsics(extrinsics_all[idx])

        target_dir = root_dir_to_target_cameras_dir(scene_root_dir, cam_id_int)
        target_dir.mkdir(parents=True, exist_ok=True)

        current_frame_number = cfg.first_frame_number
        for _ in frame_files:
            target_file = target_dir / f"{current_frame_number:0{cfg.fname_num_digits}d}.npz"
            np.savez(target_file, intrinsics=intrinsics, extrinsics=extrinsics)
            current_frame_number += 1


if __name__ == "__main__":
    main()
