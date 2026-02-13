from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi
import tyro


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


def _infer_pair_key(raw_scene_dir: Path, seq_name: Optional[str]) -> str:
    pair_dir_name = raw_scene_dir.parent.name
    if re.fullmatch(r"pair\d+", pair_dir_name):
        return pair_dir_name

    if seq_name is not None:
        match = re.search(r"(pair\d+)", seq_name)
        if match is not None:
            return match.group(1)

    raise ValueError(
        f"Could not infer pair key from raw_scene_dir='{raw_scene_dir}' and seq_name='{seq_name}'."
    )


def _run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _list_pair_archives_from_hf(repo_id: str, pair_key: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    pattern = re.compile(rf"^{re.escape(pair_key)}(?:_(\d+))?\.tar\.gz$")

    ranked: list[tuple[int, str]] = []
    for file_name in files:
        match = pattern.fullmatch(file_name)
        if match is None:
            continue
        suffix = match.group(1)
        rank = 0 if suffix is None else int(suffix)
        ranked.append((rank, file_name))

    ranked.sort(key=lambda item: (item[0], item[1]))
    return [file_name for _rank, file_name in ranked]


def _discover_pose_scene_dirs(raw_root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for smpl_file in raw_root.rglob("smpl/*.npz"):
        candidates.add(smpl_file.parent.parent)
    for smplx_file in raw_root.rglob("smplx/*.npz"):
        candidates.add(smplx_file.parent.parent)
    return sorted(candidates)


def _symlink_target_to_expected_path(expected_scene_dir: Path, target_scene_dir: Path, dry_run: bool) -> None:
    if expected_scene_dir == target_scene_dir:
        return

    if expected_scene_dir.exists():
        if expected_scene_dir.is_symlink():
            cmd = ["rm", str(expected_scene_dir)]
            _run_cmd(cmd, dry_run=dry_run)
        elif expected_scene_dir.is_dir() and not any(expected_scene_dir.iterdir()):
            cmd = ["rmdir", str(expected_scene_dir)]
            _run_cmd(cmd, dry_run=dry_run)
        else:
            raise RuntimeError(
                f"Cannot create scene symlink at {expected_scene_dir}: path exists and is not removable."
            )

    parent_dir = expected_scene_dir.parent
    if not parent_dir.exists():
        cmd = ["mkdir", "-p", str(parent_dir)]
        _run_cmd(cmd, dry_run=dry_run)

    cmd = ["ln", "-s", str(target_scene_dir), str(expected_scene_dir)]
    _run_cmd(cmd, dry_run=dry_run)


def _ensure_expected_scene_path(raw_scene_dir: Path, raw_root: Path, dry_run: bool) -> None:
    if _has_npz_pose_files(raw_scene_dir):
        return

    discovered = _discover_pose_scene_dirs(raw_root)
    if not discovered:
        raise RuntimeError(
            f"No pose scene candidates found under {raw_root} while resolving {raw_scene_dir}."
        )

    name_matches = [candidate for candidate in discovered if candidate.name == raw_scene_dir.name]
    if len(name_matches) == 1:
        target = name_matches[0]
        print(f"Resolved scene path mismatch by linking {raw_scene_dir} -> {target}")
        _symlink_target_to_expected_path(raw_scene_dir, target, dry_run=dry_run)
        return
    if len(name_matches) > 1:
        raise RuntimeError(
            "Ambiguous scene path resolution candidates:\n"
            + "\n".join(f"  - {path}" for path in name_matches)
            + f"\nExpected scene path: {raw_scene_dir}"
        )

    raise RuntimeError(
        f"Could not resolve expected raw scene path '{raw_scene_dir}'. "
        "Found pose scenes:\n"
        + "\n".join(f"  - {path}" for path in discovered)
    )


def main() -> None:
    cfg = tyro.cli(Config)
    raw_scene_dir = cfg.raw_scene_dir
    pair_key = _infer_pair_key(raw_scene_dir, cfg.seq_name)

    raw_root = cfg.archive_local_dir
    if raw_root is None:
        raw_root = raw_scene_dir.parent.parent

    print(f"Ensuring raw HI4D scene: {raw_scene_dir}")
    print(f"Inferred pair key: {pair_key}")
    print(f"Archive root: {raw_root}")

    if _has_npz_pose_files(raw_scene_dir):
        print(f"Raw scene already contains pose npz files, skipping download/extract: {raw_scene_dir}")
    else:
        archives = _list_pair_archives_from_hf(cfg.hf_repo_id, pair_key)
        if len(archives) == 0:
            raise RuntimeError(
                f"No archives found in HF repo '{cfg.hf_repo_id}' for pair '{pair_key}'."
            )

        print("Matching archives:")
        for archive_name in archives:
            print(f"  - {archive_name}")

        for archive_name in archives:
            archive_path = raw_root / archive_name
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

        for archive_name in archives:
            archive_path = raw_root / archive_name
            extract_cmd = [
                "tar",
                "-xzf",
                str(archive_path),
                "-C",
                str(raw_root),
            ]
            _run_cmd(extract_cmd, dry_run=cfg.dry_run)

        _ensure_expected_scene_path(raw_scene_dir, raw_root, dry_run=cfg.dry_run)

    # Final safety check (skip in dry-run mode).
    if not cfg.dry_run and not _has_npz_pose_files(raw_scene_dir):
        raise RuntimeError(
            f"After ensure step, no pose npz files found in {raw_scene_dir}. "
            "Please verify archive contents and scene path."
        )

    print(f"Raw HI4D scene ready: {raw_scene_dir}")


if __name__ == "__main__":
    main()
