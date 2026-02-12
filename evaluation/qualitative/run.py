from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import subprocess
import sys

from rich.console import Console
from rich.table import Table
import tyro


@dataclass
class Args:
    exp_dir: Path
    tasks: str = "nvs"  # nvs, pose
    export: bool = True
    upload: bool = False
    output_dir: Path | None = None

    # Export options
    epoch: str = "all"  # all, latest, epoch_0010, 10
    max_frames_per_camera: int = 0
    max_mesh_frames: int = 0
    spawn_viewer: bool = False

    # Upload options
    upload_selection: str | None = None  # eval_latest,eval_all
    upload_repo_id: str | None = None
    upload_repo_type: str = "dataset"
    upload_branch: str = "main"
    upload_results_root: Path | None = None
    upload_scene_name_includes: str | None = None
    upload_rerun_version: str = "0.29.1"
    upload_dry_run: bool = False
    yes: bool = False

    # Wrapper behavior
    dry_run: bool = False


def _parse_tasks(tasks: str) -> List[str]:
    parsed = [item.strip().lower() for item in tasks.split(",") if item.strip()]
    if not parsed:
        raise ValueError("No tasks provided. Use one or more of: nvs,pose")
    valid = {"nvs", "pose"}
    unknown = [item for item in parsed if item not in valid]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Valid tasks: nvs,pose")
    ordered = []
    for task in ("nvs", "pose"):
        if task in parsed:
            ordered.append(task)
    return ordered


def _run_cmd(command: List[str], dry_run: bool, console: Console) -> None:
    console.print(f"[cyan]$ {' '.join(command)}[/cyan]")
    if dry_run:
        return
    subprocess.run(command, check=True)


def _default_upload_selection(tasks: List[str], epoch: str) -> str:
    if "nvs" not in tasks and "pose" not in tasks:
        return "eval_latest"
    return "eval_latest" if epoch == "latest" else "eval_all"


def main() -> None:
    args = tyro.cli(Args)
    console = Console()
    tasks = _parse_tasks(args.tasks)

    helpers_dir = Path(__file__).resolve().parent / "helpers"
    export_eval_script = helpers_dir / "rerun_export_evaluation.py"
    upload_script = helpers_dir / "upload_rerun_to_hf.py"

    output_dir = args.output_dir if args.output_dir is not None else args.exp_dir / "rerun"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_plan = Table(title="Qualitative Pipeline Plan")
    run_plan.add_column("Step")
    run_plan.add_column("Enabled")
    run_plan.add_column("Details")
    run_plan.add_row("Export", str(args.export), f"tasks={','.join(tasks)}")
    run_plan.add_row("Upload", str(args.upload), f"selection={args.upload_selection or 'auto'}")
    run_plan.add_row("Output Dir", "yes", str(output_dir))
    console.print(run_plan)

    if args.export:
        for task in tasks:
            eval_cmd = [
                sys.executable,
                str(export_eval_script),
                "--exp-dir",
                str(args.exp_dir),
                "--output-dir",
                str(output_dir),
                "--epoch",
                args.epoch,
                "--max-frames-per-camera",
                str(args.max_frames_per_camera),
                "--max-mesh-frames",
                str(args.max_mesh_frames),
            ]
            if task == "nvs":
                eval_cmd.extend(
                    [
                        "--output-prefix",
                        "evaluation_nvs",
                        "--include-nvs",
                        "--no-include-pose-smplx",
                        "--no-include-pose-smpl",
                        "--no-include-meshes",
                    ]
                )
            elif task == "pose":
                eval_cmd.extend(
                    [
                        "--output-prefix",
                        "evaluation_pose",
                        "--no-include-nvs",
                        "--include-pose-smplx",
                        "--include-pose-smpl",
                        "--include-meshes",
                    ]
                )
            if args.spawn_viewer:
                eval_cmd.append("--spawn-viewer")
            _run_cmd(eval_cmd, args.dry_run, console)

    if args.upload:
        scene_name = args.exp_dir.parent.name
        results_root = args.upload_results_root or args.exp_dir.parents[1]
        selection = args.upload_selection or _default_upload_selection(tasks, args.epoch)
        upload_cmd = [
            sys.executable,
            str(upload_script),
            "--exp-name",
            args.exp_dir.name,
            "--results-root",
            str(results_root),
            "--selection",
            selection,
            "--repo-type",
            args.upload_repo_type,
            "--branch",
            args.upload_branch,
            "--rerun-version",
            args.upload_rerun_version,
            "--scene-name-includes",
            args.upload_scene_name_includes or scene_name,
        ]
        if args.upload_repo_id is not None:
            upload_cmd.extend(["--repo-id", args.upload_repo_id])
        if args.upload_dry_run:
            upload_cmd.append("--dry-run")
        if args.yes:
            upload_cmd.append("--yes")
        _run_cmd(upload_cmd, args.dry_run, console)


if __name__ == "__main__":
    main()
