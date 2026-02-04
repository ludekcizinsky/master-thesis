from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class Args:
    scene_dir: str
    ref_frame_idx: int = 0


def main() -> None:
    args = tyro.cli(Args)
    scene_dir = Path(args.scene_dir)
    if not scene_dir.exists():
        raise RuntimeError(f"Scene dir does not exist: {scene_dir}")

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


if __name__ == "__main__":
    main()
