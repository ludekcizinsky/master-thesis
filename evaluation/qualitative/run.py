from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List
import subprocess
import sys

from rich.console import Console
from rich.table import Table
import tyro


@dataclass
class SlurmConfig:
    job_name: str = "qualitative_export"
    slurm_script: Path = Path("evaluation/qualitative/submit.slurm")
    time: str = "01:00:00"
    array_parallelism: int | None = None


@dataclass
class Args:
    exp_dir: Path | None = None
    exp_name: str | None = None
    results_root: Path = Path("/scratch/izar/cizinsky/thesis/results")
    scene_name_includes: str | None = None

    tasks: str = "nvs"  # nvs, pose, trn_nv_generation
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

    # Wrapper behavior
    dry_run: bool = False
    run_all: bool = False
    schedule: bool = False
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def _parse_tasks(tasks: str) -> List[str]:
    parsed = [item.strip().lower() for item in tasks.split(",") if item.strip()]
    if not parsed:
        raise ValueError("No tasks provided. Use one or more of: nvs,pose,trn_nv_generation")
    valid = {"nvs", "pose", "trn_nv_generation"}
    unknown = [item for item in parsed if item not in valid]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Valid tasks: nvs,pose,trn_nv_generation")
    ordered = []
    for task in ("nvs", "pose", "trn_nv_generation"):
        if task in parsed:
            ordered.append(task)
    return ordered


def _run_cmd(command: List[str], dry_run: bool, console: Console) -> None:
    if dry_run:
        console.print(f"[cyan]$ {' '.join(command)}[/cyan]")
        return
    subprocess.run(command, check=True)


def _default_upload_selection(tasks: List[str], epoch: str) -> str:
    if "nvs" not in tasks and "pose" not in tasks:
        return "none"
    return "eval_latest" if epoch == "latest" else "eval_all"


def _resolve_scene_exp_dirs(args: Args) -> List[Path]:
    if args.exp_name is None:
        return []
    if not args.results_root.is_dir():
        raise FileNotFoundError(f"results_root does not exist: {args.results_root}")

    scene_exp_dirs: List[Path] = []
    for scene_dir in sorted(args.results_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name
        if args.scene_name_includes and args.scene_name_includes not in scene_name:
            continue
        exp_dir = scene_dir / args.exp_name
        if exp_dir.is_dir():
            scene_exp_dirs.append(exp_dir)
    return scene_exp_dirs


def _resolve_scene_index() -> int | None:
    env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
    if env_idx is None or env_idx == "":
        return None
    try:
        return int(env_idx)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID='{env_idx}'") from exc


def _confirm_submit(console: Console) -> bool:
    try:
        response = console.input("Press Enter to submit, or type anything to cancel: ")
    except EOFError:
        return False
    return response.strip() == ""


def _forwarded_args(argv: List[str]) -> List[str]:
    blocked_exact = {
        "--schedule",
        "--no-schedule",
        "--dry-run",
        "--no-dry-run",
    }
    blocked_prefix = (
        "--schedule=",
        "--dry-run=",
    )
    forwarded: List[str] = []
    for arg in argv:
        if arg in blocked_exact:
            continue
        if arg.startswith(blocked_prefix):
            continue
        forwarded.append(arg)
    return forwarded


def _submit_array(args: Args, scene_exp_dirs: List[Path], console: Console) -> None:
    if not scene_exp_dirs:
        raise ValueError(
            "No scene experiment directories found to schedule. "
            "Check --exp-name/--results-root/--scene-name-includes."
        )

    repo_root = Path(__file__).resolve().parents[2]
    slurm_script = args.slurm.slurm_script
    if not slurm_script.is_absolute():
        slurm_script = repo_root / slurm_script
    if not slurm_script.exists():
        raise FileNotFoundError(f"Slurm script not found: {slurm_script}")

    array_spec = f"0-{len(scene_exp_dirs) - 1}"
    if args.slurm.array_parallelism is not None:
        array_spec = f"{array_spec}%{args.slurm.array_parallelism}"

    summary = Table(title="Qualitative Scheduling Plan")
    summary.add_column("Index", justify="right")
    summary.add_column("Scene")
    summary.add_column("Exp Dir")
    for idx, exp_dir in enumerate(scene_exp_dirs):
        summary.add_row(str(idx), exp_dir.parent.name, str(exp_dir))
    console.print(summary)
    console.print(
        f"Tasks: [bold]{args.tasks}[/bold] | export={args.export} | upload={args.upload}\n"
        f"Slurm: script={slurm_script} | job_name={args.slurm.job_name} | "
        f"time={args.slurm.time} | array={array_spec}"
    )

    if not _confirm_submit(console):
        console.print("[yellow]Submission cancelled.[/yellow]")
        return

    cmd = [
        "sbatch",
        "--job-name",
        args.slurm.job_name,
        "--time",
        args.slurm.time,
        "--array",
        array_spec,
        "--export",
        "ALL",
        str(slurm_script),
    ]
    cmd.extend(_forwarded_args(sys.argv[1:]))
    _run_cmd(cmd, args.dry_run, console)


def _run_for_exp_dir(args: Args, tasks: List[str], exp_dir: Path, console: Console) -> None:
    helpers_dir = Path(__file__).resolve().parent / "helpers"
    export_eval_script = helpers_dir / "rerun_export_evaluation.py"
    export_trn_nv_script = helpers_dir / "rerun_export_trn_nv_generation.py"
    upload_script = helpers_dir / "upload_rerun_to_hf.py"

    output_dir = args.output_dir if args.output_dir is not None else exp_dir / "rerun"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_plan = Table(title="Qualitative Pipeline Plan")
    run_plan.add_column("Step")
    run_plan.add_column("Enabled")
    run_plan.add_column("Details")
    run_plan.add_row("Exp Dir", "yes", str(exp_dir))
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
                str(exp_dir),
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
                        "--no-include-pose-smpl",
                        "--include-meshes",
                    ]
                )
            elif task == "trn_nv_generation":
                eval_cmd = [
                    sys.executable,
                    str(export_trn_nv_script),
                    "--exp-dir",
                    str(exp_dir),
                    "--output-dir",
                    str(output_dir),
                    "--epoch",
                    args.epoch,
                    "--max-frames",
                    str(args.max_mesh_frames),
                ]
            if args.spawn_viewer:
                eval_cmd.append("--spawn-viewer")
            _run_cmd(eval_cmd, args.dry_run, console)

    if args.upload:
        scene_name = exp_dir.parent.name
        results_root = args.upload_results_root or args.results_root
        selection = args.upload_selection or _default_upload_selection(tasks, args.epoch)
        upload_globs: List[str] = []
        if "trn_nv_generation" in tasks:
            upload_globs.append("trn_nv_generation_epoch_*.rrd")
        upload_cmd = [
            sys.executable,
            str(upload_script),
            "--exp-name",
            exp_dir.name,
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
        for glob_pattern in upload_globs:
            upload_cmd.extend(["--rrd-glob", glob_pattern])
        if args.upload_repo_id is not None:
            upload_cmd.extend(["--repo-id", args.upload_repo_id])
        if args.upload_dry_run:
            upload_cmd.append("--dry-run")
        _run_cmd(upload_cmd, args.dry_run, console)


def main() -> None:
    args = tyro.cli(Args)
    console = Console()
    tasks = _parse_tasks(args.tasks)
    scene_exp_dirs = _resolve_scene_exp_dirs(args)

    if args.schedule:
        if args.exp_name is None:
            raise ValueError("--schedule requires --exp-name.")
        _submit_array(args, scene_exp_dirs, console)
        return

    scene_index = _resolve_scene_index()
    if scene_index is not None:
        if args.exp_name is None:
            raise ValueError("SLURM array execution requires --exp-name.")
        if scene_index < 0 or scene_index >= len(scene_exp_dirs):
            raise IndexError(
                f"Scene index {scene_index} out of range (0..{len(scene_exp_dirs) - 1})."
            )
        _run_for_exp_dir(args, tasks, scene_exp_dirs[scene_index], console)
        return

    if args.exp_dir is not None:
        _run_for_exp_dir(args, tasks, args.exp_dir, console)
        return

    if args.run_all:
        if args.exp_name is None:
            raise ValueError("--run-all requires --exp-name.")
        if not scene_exp_dirs:
            raise ValueError("No scene experiment directories found for --run-all.")
        for exp_dir in scene_exp_dirs:
            _run_for_exp_dir(args, tasks, exp_dir, console)
        return

    if args.exp_name is not None and len(scene_exp_dirs) == 1:
        _run_for_exp_dir(args, tasks, scene_exp_dirs[0], console)
        return

    raise ValueError(
        "Provide --exp-dir for single-scene run, or --exp-name with --run-all/--schedule "
        "(or ensure only one matching scene exists)."
    )


if __name__ == "__main__":
    main()
