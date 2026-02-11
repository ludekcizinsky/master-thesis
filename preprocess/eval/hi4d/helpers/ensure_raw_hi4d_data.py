from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro


PAIR_TO_ARCHIVE = {
    "pair15": "pair15_1.tar.gz",
    "pair16": "pair16.tar.gz",
    "pair17": "pair17_1.tar.gz",
    "pair19": "pair19_2.tar.gz",
}


@dataclass
class Config:
    raw_scene_dir: Path
    seq_name: Optional[str] = None
    hf_repo_id: str = "ludekcizinsky/hi4d"
    archive_local_dir: Optional[Path] = None
    dry_run: bool = False


def _has_npz_pose_files(scene_dir: Path) -> bool:
    smpl_dir = scene_dir / "smpl"
    smplx_dir = scene_dir / "smplx"
    has_smpl = smpl_dir.exists() and any(smpl_dir.glob("*.npz"))
    has_smplx = smplx_dir.exists() and any(smplx_dir.glob("*.npz"))
    return bool(has_smpl or has_smplx)


def _is_dir_empty(path: Path) -> bool:
    if not path.exists():
        return True
    return next(path.iterdir(), None) is None


def _infer_pair_key(raw_scene_dir: Path, seq_name: Optional[str]) -> str:
    pair_dir_name = raw_scene_dir.parent.name
    if pair_dir_name in PAIR_TO_ARCHIVE:
        return pair_dir_name

    if seq_name is not None:
        m = re.search(r"(pair\d+)", seq_name)
        if m:
            pair_key = m.group(1)
            if pair_key in PAIR_TO_ARCHIVE:
                return pair_key

    raise ValueError(
        f"Could not infer pair key from raw_scene_dir='{raw_scene_dir}' and seq_name='{seq_name}'."
    )


def _run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _ensure_pair19_layout(raw_root: Path, dry_run: bool) -> None:
    pair19_dir = raw_root / "pair19"
    pair19_dir.mkdir(parents=True, exist_ok=True)

    moves = [
        (raw_root / "piggyback19", pair19_dir / "piggyback19"),
        (raw_root / "highfive19", pair19_dir / "highfive19"),
    ]
    for src, dst in moves:
        if not src.exists() or dst.exists():
            continue
        cmd = ["mv", str(src), str(pair19_dir)]
        _run_cmd(cmd, dry_run=dry_run)


def main() -> None:
    cfg = tyro.cli(Config)
    raw_scene_dir = cfg.raw_scene_dir
    pair_key = _infer_pair_key(raw_scene_dir, cfg.seq_name)
    archive_name = PAIR_TO_ARCHIVE[pair_key]

    raw_root = cfg.archive_local_dir
    if raw_root is None:
        raw_root = raw_scene_dir.parent.parent

    archive_path = raw_root / archive_name

    # Download archive only if the scene still has no npz pose files.
    if not _has_npz_pose_files(raw_scene_dir):
        if not archive_path.exists():
            download_cmd = [
                "hf",
                "download",
                cfg.hf_repo_id,
                archive_name,
                "--repo-type",
                "dataset",
                "--local-dir",
                str(raw_root),
            ]
            _run_cmd(download_cmd, dry_run=cfg.dry_run)

        # Extract only if target scene dir is missing or empty.
        if _is_dir_empty(raw_scene_dir):
            extract_cmd = [
                "tar",
                "-xzf",
                str(archive_path),
                "-C",
                str(raw_root),
            ]
            _run_cmd(extract_cmd, dry_run=cfg.dry_run)
    else:
        print(f"Raw scene already contains npz pose files, skipping download/extract: {raw_scene_dir}")

    # Pair19 archive requires extra moves after extraction.
    if pair_key == "pair19":
        _ensure_pair19_layout(raw_root, dry_run=cfg.dry_run)

    # Final safety check (skip in dry-run mode).
    if not cfg.dry_run and not _has_npz_pose_files(raw_scene_dir):
        raise RuntimeError(
            f"After ensure step, no pose npz files found in {raw_scene_dir}. "
            "Please verify archive contents and scene path."
        )

    print(f"Raw HI4D scene ready: {raw_scene_dir}")


if __name__ == "__main__":
    main()
