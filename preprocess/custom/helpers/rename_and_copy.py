import shutil
from dataclasses import dataclass
from pathlib import Path

import tyro

def root_dir_to_target_format_mask_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "seg" / "img_seg_mask" / f"{cam_id}" / "all"

def root_dir_to_source_format_mask_dir(root_dir: Path) -> Path:
    return root_dir / "masks" / "union"

def root_dir_to_source_format_mask_root_dir(root_dir: Path) -> Path:
    return root_dir / "masks"

def root_dir_to_target_format_mask_person_dir(root_dir: Path, cam_id: int, person_id: str) -> Path:
    return root_dir / "seg" / "img_seg_mask" / f"{cam_id}" / f"{person_id}"

def normalize_person_id(person_dir_name: str) -> str:
    if person_dir_name.isdigit():
        return str(int(person_dir_name))
    stripped = person_dir_name.lstrip("0")
    return stripped if stripped != "" else "0"

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

    # Per-person masks (all subdirs except "union")
    masks_root_dir = root_dir_to_source_format_mask_root_dir(scene_root_dir)
    person_mask_dirs = [
        d for d in sorted(masks_root_dir.iterdir())
        if d.is_dir() and d.name != "union"
    ]
    for person_dir in person_mask_dirs:
        person_id = normalize_person_id(person_dir.name)
        person_masks_dir = root_dir_to_target_format_mask_person_dir(
            scene_root_dir,
            cam_id=cfg.src_cam_id,
            person_id=person_id,
        )
        person_masks_dir.mkdir(parents=True, exist_ok=True)
        sorted_person_mask_files = sorted(
            f for f in person_dir.iterdir()
            if f.is_file() and f.suffix == ".png"
        )
        current_frame_number = cfg.first_frame_number
        for item in sorted_person_mask_files:
            new_name = f"{current_frame_number:0{cfg.fname_num_digits}d}.png"
            current_frame_number += 1
            dest = person_masks_dir / new_name
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
