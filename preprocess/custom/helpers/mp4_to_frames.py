from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro


@dataclass
class Config:
    input_video_path: Path
    scenes_dir: Path
    seq_name: str
    frame_rate: Optional[float] = None
    jpeg_quality: int = 2


def main() -> None:
    cfg = tyro.cli(Config)

    input_path = cfg.input_video_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found in PATH.")

    output_dir = cfg.scenes_dir / cfg.seq_name / "images" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.frame_rate is not None:
        print("Warning: --frame-rate is ignored (using source fps).")

    output_pattern = str(output_dir / "%06d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-start_number",
        "1",
        "-q:v",
        str(cfg.jpeg_quality),
        output_pattern,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
