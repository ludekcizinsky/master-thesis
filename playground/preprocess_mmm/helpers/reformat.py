import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro


def root_dir_to_source_format_image_dir(root_dir: Path) -> Path:
    return root_dir / "org_image"

def root_dir_to_target_format_image_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "images" / f"{cam_id}"

def root_dir_to_source_format_dyn_cam_dir(root_dir: Path) -> Path:
    return root_dir / "opt_cam"

def root_dir_to_target_format_dyn_cam_dir(root_dir: Path, cam_id: int) -> Path:
    return root_dir / "all_cameras" / f"{cam_id}"

def root_dir_to_source_format_meshes_dir(root_dir: Path) -> Path:
    return root_dir / "obj" 

def root_dir_to_target_format_meshes_dir(root_dir: Path) -> Path:
    return root_dir / "meshes"


@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/ait_datasets/full/mmm/dance"
    src_cam_id: int = 0
    first_frame_number: int = 1
    fname_num_digits: int = 6
    downscale_factor: int = 2


def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    if cfg.downscale_factor < 1:
        raise ValueError("downscale_factor must be >= 1")
    scene_root_dir = Path(cfg.scene_root_dir)

    # Images
    # - prepare directories
    current_image_dir = root_dir_to_source_format_image_dir(scene_root_dir)
    images_dir = root_dir_to_target_format_image_dir(scene_root_dir, cam_id=cfg.src_cam_id)
    images_dir.mkdir(parents=True, exist_ok=True)
    # - copy and rename
    sorted_image_files = sorted(current_image_dir.iterdir())
    current_frame_number = cfg.first_frame_number
    for item in tqdm(sorted_image_files):
        new_name = f"{current_frame_number:0{cfg.fname_num_digits}d}.jpg"
        current_frame_number += 1
        dest = images_dir / new_name
        if cfg.downscale_factor == 1:
            shutil.copy2(item, dest)
            continue

        with Image.open(item) as image:
            new_size = (
                max(1, image.width // cfg.downscale_factor),
                max(1, image.height // cfg.downscale_factor),
            )
            resized = image.resize(new_size, resample=Image.Resampling.LANCZOS)
            resized.save(dest)

    # Dynamic cameras
    # - prepare directories
    current_dyn_cam_dir = root_dir_to_source_format_dyn_cam_dir(scene_root_dir)
    dyn_cam_dir = root_dir_to_target_format_dyn_cam_dir(scene_root_dir, cam_id=cfg.src_cam_id)
    dyn_cam_dir.mkdir(parents=True, exist_ok=True)
    # - copy and rename
    sorted_dyn_cam_files = sorted(
        f for f in current_dyn_cam_dir.iterdir() if f.suffix == ".npz"
    )
    current_frame_number = cfg.first_frame_number
    for item in tqdm(sorted_dyn_cam_files):
        new_name = f"{current_frame_number:0{cfg.fname_num_digits}d}.npz"
        current_frame_number += 1
        dest = dyn_cam_dir / new_name
        with np.load(item, allow_pickle=True) as data:
            intrinsics = data["K"]
            R_w2c = data["R"]
            t_w2c = data["T"]

        if intrinsics.shape == (3, 3):
            intrinsics = intrinsics[None, ...]
        if R_w2c.shape == (1, 3, 3):
            R_w2c = R_w2c[0]
        if t_w2c.shape == (1, 3):
            t_w2c = t_w2c[0]

        if cfg.downscale_factor != 1:
            intrinsics = intrinsics.copy()
            intrinsics[:, 0, 0] /= cfg.downscale_factor
            intrinsics[:, 1, 1] /= cfg.downscale_factor
            intrinsics[:, 0, 2] /= cfg.downscale_factor
            intrinsics[:, 1, 2] /= cfg.downscale_factor

        extrinsics = np.eye(4, dtype=R_w2c.dtype)
        extrinsics[:3, :3] = R_w2c
        extrinsics[:3, 3] = t_w2c
        extrinsics = extrinsics[None, :3, :]

        np.savez(dest, intrinsics=intrinsics, extrinsics=extrinsics)


    # Meshes
    src_meshes_dir = root_dir_to_source_format_meshes_dir(scene_root_dir)
    tgt_meshes_dir = root_dir_to_target_format_meshes_dir(scene_root_dir)
    tgt_meshes_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(f for f in src_meshes_dir.iterdir() if f.suffix.lower() == ".obj")
    current_frame_number = cfg.first_frame_number

    for frame_file in tqdm(frame_files):
        tgt_mesh_file = tgt_meshes_dir / f"{current_frame_number:0{cfg.fname_num_digits}d}.obj"
        tgt_mesh_file.write_bytes(frame_file.read_bytes())
        current_frame_number += 1

if __name__ == "__main__":
    main()
