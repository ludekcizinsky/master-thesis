from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tyro

@dataclass
class ReformatConfig:
    scene_root_dir: str = "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15"
    first_frame_number: int = 1
    fname_num_digits: int = 6

def root_dir_to_source_format_meshes_dir(root_dir: Path) -> Path:
    return root_dir / "frames" 

def root_dir_to_target_format_meshes_dir(root_dir: Path) -> Path:
    return root_dir / "meshes"


def main() -> None:
    cfg = tyro.cli(ReformatConfig)
    scene_root_dir = Path(cfg.scene_root_dir)
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