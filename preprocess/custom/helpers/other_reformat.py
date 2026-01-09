import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro

def root_dir_to_target_format_mask_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "seg" / "img_seg_mask" / f"{cam_id}" / "all"

def root_dir_to_source_format_mask_dir(root_dir: Path) -> Path:
    return root_dir / "masks" / "union"

def root_dir_to_target_format_image_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "images" / f"{cam_id}"

def root_dir_to_source_format_image_dir(root_dir: Path) -> Path:
    return root_dir / "frames"

def root_dir_to_target_format_depth_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "depths" / f"{cam_id}"

def root_dir_to_source_format_depth_dir(root_dir: Path) -> Path:
    return root_dir / "depth_maps" / "raw"


@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance"
    src_cam_id: int = 28
    first_frame_number: int = 1
    fname_num_digits: int = 6


def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    scene_root_dir = Path(cfg.scene_root_dir)

    # Images
    # - prepare directories
    current_image_dir = root_dir_to_source_format_image_dir(scene_root_dir)
    images_dir = root_dir_to_target_format_image_dir(scene_root_dir, cam_id=cfg.src_cam_id)
    images_dir.mkdir(parents=True, exist_ok=True)
    # - copy and rename
    sorted_image_files = sorted(current_image_dir.iterdir())
    current_frame_number = cfg.first_frame_number
    for item in sorted_image_files:
        new_name = f"{current_frame_number:0{cfg.fname_num_digits}d}.jpg"
        current_frame_number += 1
        dest = images_dir / new_name
        shutil.copy2(item, dest)

    # Masks
    # - prepare directories
    current_masks_dir = root_dir_to_source_format_mask_dir(scene_root_dir)
    masks_dir = root_dir_to_target_format_mask_dir(scene_root_dir, cam_id=cfg.src_cam_id)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # - copy and rename
    sorted_mask_files = sorted(current_masks_dir.iterdir())
    current_frame_number = cfg.first_frame_number
    for item in sorted_mask_files:
        new_name = f"{current_frame_number:0{cfg.fname_num_digits}d}.png"
        current_frame_number += 1
        dest = masks_dir / new_name
        shutil.copy2(item, dest)

    # Depths
    curr_depths_dir = root_dir_to_source_format_depth_dir(scene_root_dir)
    depths_dir = root_dir_to_target_format_depth_dir(scene_root_dir, cam_id=cfg.src_cam_id)
    depths_dir.mkdir(parents=True, exist_ok=True)
    # - copy and rename
    sorted_depth_files = sorted(curr_depths_dir.iterdir())
    # -- make sure to only include .npy files
    sorted_depth_files = [f for f in sorted_depth_files if f.suffix == ".npy"]
    current_frame_number = cfg.first_frame_number
    for item in sorted_depth_files:
        new_name = f"{current_frame_number:0{cfg.fname_num_digits}d}.npy"
        current_frame_number += 1
        dest = depths_dir / new_name
        shutil.copy2(item, dest)


if __name__ == "__main__":
    main()
