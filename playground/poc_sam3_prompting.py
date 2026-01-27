from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import tyro


@dataclass
class Args:
    scene_name: str
    prompt_frame: int = 0


def main() -> None:
    args = tyro.cli(Args)

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "submodules" / "sam3" / "run_refinement.sh"

    cmd = [
        "bash",
        str(script_path),
        args.scene_name,
        str(args.prompt_frame),
    ]
    subprocess.run(cmd, check=True, cwd=str(repo_root))


if __name__ == "__main__":
    main()
