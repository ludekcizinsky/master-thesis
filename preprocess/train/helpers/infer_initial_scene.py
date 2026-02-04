from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    video_path: str
    seq_name: str
    cam_id: int
    output_dir: str
    ref_frame_idx: int = 0


def main() -> None:
    args = tyro.cli(Args)
    scene_dir = Path(args.output_dir) / args.seq_name

    cmd = [
        "conda",
        "run",
        "-n",
        "thesis",
        "python",
        "preprocess/train/helpers/init_scene_dir.py",
        "--video-path",
        str(args.video_path),
        "--seq-name",
        str(args.seq_name),
        "--cam-id",
        str(args.cam_id),
        "--output-dir",
        str(args.output_dir),
    ]
    subprocess.run(cmd, check=True)

    if not scene_dir.exists():
        raise RuntimeError(f"Scene dir does not exist after init: {scene_dir}")

    cmd = [
        "bash",
        "submodules/prompthmr/run_inference.sh",
        str(scene_dir),
    ]
    subprocess.run(cmd, check=True)

    cmd = [
        "bash",
        "submodules/da3/run_inference.sh",
        str(scene_dir),
    ]
    subprocess.run(cmd, check=True)

    cmd = [
        "bash",
        "submodules/lhm/run_inference.sh",
        str(scene_dir),
        str(args.ref_frame_idx),
    ]
    subprocess.run(cmd, check=True)

    cmd = [
        "conda",
        "run",
        "-n",
        "thesis",
        "python",
        "preprocess/train/helpers/gen_virtual_cameras.py",
        "--scene-dir",
        str(scene_dir),
        "--num-of-cameras",
        "7",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
