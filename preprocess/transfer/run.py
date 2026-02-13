from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import os
import subprocess
import sys

from huggingface_hub import CommitOperationDelete, HfApi, snapshot_download
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
from rich.console import Console
from rich.table import Table
import tyro


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.path_config import ensure_runtime_dirs, load_runtime_paths


Action = Literal["upload", "download", "sync"]
DataKind = Literal["gt", "preprocessed"]

DEFAULT_REPO_BY_KIND = {
    "gt": "ludekcizinsky/thesis-gt-scene-data",
    "preprocessed": "ludekcizinsky/thesis-preprocessed-scenes",
}


@dataclass
class SlurmConfig:
    job_name: str = "preprocess_transfer"
    slurm_script: Path = Path("preprocess/transfer/submit.slurm")
    time: str = "01:00:00"
    array_parallelism: int | None = None
    gres: str | None = "gpu:1"


@dataclass
class Args:
    action: Action
    data_kind: DataKind
    paths_config: Path = Path("configs/paths.yaml")

    repo_id: str | None = None
    repo_type: str = "dataset"
    private: bool = True
    branch: str = "main"
    token: str | None = None

    local_root: Path | None = None
    scene_names: str | None = None
    scene_name_includes: str | None = None
    run_all: bool = False

    yes: bool = False
    dry_run: bool = False
    use_large_upload: bool = True
    num_upload_workers: int | None = None
    hf_progress_bars: bool = False
    schedule: bool = False
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


def _parse_scene_names(scene_names: str | None) -> list[str]:
    if scene_names is None:
        return []
    return [item.strip() for item in scene_names.split(",") if item.strip()]


def _resolve_repo_id(args: Args) -> str:
    if args.repo_id is not None:
        return args.repo_id
    return DEFAULT_REPO_BY_KIND[args.data_kind]


def _resolve_token(args: Args) -> str | None:
    if args.token is not None and args.token.strip() != "":
        return args.token.strip()
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if env_token is not None and env_token.strip() != "":
        return env_token.strip()
    return None


def _resolve_local_root(args: Args) -> Path:
    if args.local_root is not None:
        return args.local_root
    runtime_paths = load_runtime_paths(REPO_ROOT / args.paths_config)
    ensure_runtime_dirs(runtime_paths)
    if args.data_kind == "gt":
        return runtime_paths.canonical_gt_root_dir
    return runtime_paths.preprocessing_root_dir


def _list_local_scene_names(local_root: Path) -> list[str]:
    if not local_root.exists():
        raise FileNotFoundError(f"Local root does not exist: {local_root}")
    # Ignore hidden/system directories like ".cache".
    return sorted(
        [path.name for path in local_root.iterdir() if path.is_dir() and not path.name.startswith(".")]
    )


def _list_remote_scene_names(api: HfApi, repo_id: str, repo_type: str, branch: str) -> list[str]:
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=branch)
    names: set[str] = set()
    for file_path in files:
        if "/" not in file_path:
            continue
        top_level = file_path.split("/", 1)[0].strip()
        if not top_level or top_level.startswith("."):
            continue
        names.add(top_level)
    return sorted(names)


def _select_scene_names(
    available_scene_names: list[str],
    scene_names_arg: str | None,
    scene_name_includes: str | None,
    run_all: bool,
) -> list[str]:
    explicit = _parse_scene_names(scene_names_arg)
    if explicit:
        missing = [name for name in explicit if name not in available_scene_names]
        if missing:
            raise ValueError(f"Requested scene(s) not found: {missing}")
        return explicit

    if scene_name_includes is not None:
        selected = [name for name in available_scene_names if scene_name_includes in name]
        if not selected:
            raise ValueError(f"No scenes matched scene_name_includes='{scene_name_includes}'.")
        return selected

    if run_all:
        return available_scene_names

    if len(available_scene_names) == 1:
        return available_scene_names

    raise ValueError(
        "Scene selection is ambiguous. Use one of: --run-all, --scene-names, --scene-name-includes."
    )


def _index_remote_files_by_scene(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    branch: str,
) -> dict[str, list[str]]:
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=branch)
    grouped: dict[str, list[str]] = {}
    for file_path in files:
        if "/" not in file_path:
            continue
        scene_name, _rest = file_path.split("/", 1)
        grouped.setdefault(scene_name, []).append(file_path)
    return grouped


def _print_plan(
    console: Console,
    args: Args,
    repo_id: str,
    local_root: Path,
    selected_scene_names: list[str],
    scene_stats: dict[str, tuple[int, int]] | None = None,
) -> None:
    total_bytes = 0
    total_files = 0
    if scene_stats is not None:
        for n_files, n_bytes in scene_stats.values():
            total_files += n_files
            total_bytes += n_bytes

    table = Table(title="HF Transfer Plan")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Action", args.action)
    table.add_row("Data Kind", args.data_kind)
    table.add_row("Repo", f"{repo_id} ({args.repo_type})")
    table.add_row("Private", str(args.private))
    table.add_row("Branch", args.branch)
    table.add_row("Local Root", str(local_root))
    table.add_row("Scenes", f"{len(selected_scene_names)}")
    table.add_row("Upload Method", "upload_large_folder" if args.use_large_upload else "upload_folder")
    if args.use_large_upload:
        table.add_row("Upload Workers", str(_resolve_num_upload_workers(args.num_upload_workers)))
    table.add_row("HF Progress Bars", str(args.hf_progress_bars))
    if scene_stats is not None:
        table.add_row("Total Files", str(total_files))
        table.add_row("Total Size", _format_size(total_bytes))
    console.print(table)

    scenes_table = Table(title="Selected Scenes")
    scenes_table.add_column("#", justify="right")
    scenes_table.add_column("Scene")
    if scene_stats is not None:
        scenes_table.add_column("Files", justify="right")
        scenes_table.add_column("Size", justify="right")
    for idx, scene_name in enumerate(selected_scene_names, start=1):
        if scene_stats is None:
            scenes_table.add_row(str(idx), scene_name)
        else:
            n_files, n_bytes = scene_stats[scene_name]
            scenes_table.add_row(str(idx), scene_name, str(n_files), _format_size(n_bytes))
    console.print(scenes_table)


def _confirm(console: Console, yes: bool) -> bool:
    if yes:
        return True
    try:
        response = console.input("Press Enter to continue, or type anything to cancel: ")
    except EOFError:
        return False
    return response.strip() == ""


def _resolve_scene_index() -> int | None:
    env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
    if env_idx is None or env_idx == "":
        return None
    try:
        return int(env_idx)
    except ValueError as exc:
        raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID='{env_idx}'") from exc


def _submit_array(
    args: Args,
    selected_scene_names: list[str],
    runtime_paths,
    console: Console,
) -> None:
    if not selected_scene_names:
        raise ValueError("No scenes selected for scheduling.")

    slurm_script = args.slurm.slurm_script
    if not slurm_script.is_absolute():
        slurm_script = REPO_ROOT / slurm_script
    if not slurm_script.exists():
        raise FileNotFoundError(f"Slurm script not found: {slurm_script}")

    array_spec = f"0-{len(selected_scene_names) - 1}"
    if args.slurm.array_parallelism is not None:
        array_spec = f"{array_spec}%{args.slurm.array_parallelism}"

    scene_csv = ",".join(selected_scene_names)
    script_args: list[str] = [
        "--action",
        args.action,
        "--data-kind",
        args.data_kind,
        "--paths-config",
        str(args.paths_config),
        "--scene-names",
        scene_csv,
        "--yes",
        "--repo-type",
        args.repo_type,
        "--branch",
        args.branch,
    ]
    if args.repo_id is not None:
        script_args.extend(["--repo-id", args.repo_id])
    # Do not pass token via CLI args (would leak in process/logs); rely on env export (HF_TOKEN).
    if args.local_root is not None:
        script_args.extend(["--local-root", str(args.local_root)])
    if args.private:
        script_args.append("--private")
    else:
        script_args.append("--no-private")
    if args.use_large_upload:
        script_args.append("--use-large-upload")
    else:
        script_args.append("--no-use-large-upload")
    if args.num_upload_workers is not None:
        script_args.extend(["--num-upload-workers", str(args.num_upload_workers)])
    if args.hf_progress_bars:
        script_args.append("--hf-progress-bars")
    else:
        script_args.append("--no-hf-progress-bars")
    if args.dry_run:
        script_args.append("--dry-run")

    cmd = [
        "sbatch",
        "--job-name",
        args.slurm.job_name,
        "--time",
        args.slurm.time,
        "--output",
        str(runtime_paths.slurm_dir / "%x.%A_%a.out"),
        "--error",
        str(runtime_paths.slurm_dir / "%x.%A_%a.err"),
        "--array",
        array_spec,
        "--export",
        "ALL",
        str(slurm_script),
        *script_args,
    ]
    if args.slurm.gres is not None and args.slurm.gres.strip() != "":
        cmd[1:1] = ["--gres", args.slurm.gres]

    table = Table(title="Slurm Submission")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Script", str(slurm_script))
    table.add_row("Job Name", args.slurm.job_name)
    table.add_row("Time", args.slurm.time)
    table.add_row("GRES", args.slurm.gres or "none")
    table.add_row("Array", array_spec)
    console.print(table)

    if args.dry_run:
        console.print(f"[cyan]{' '.join(cmd)}[/cyan]")
        return
    subprocess.run(cmd, check=True)


def _upload_scene(
    api: HfApi,
    args: Args,
    repo_id: str,
    local_root: Path,
    scene_name: str,
) -> None:
    scene_dir = local_root / scene_name
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    if args.dry_run:
        return
    if args.use_large_upload:
        api.upload_large_folder(
            folder_path=str(local_root),
            repo_id=repo_id,
            repo_type=args.repo_type,
            revision=args.branch,
            allow_patterns=[f"{scene_name}/**", f"{scene_name}/*"],
            num_workers=_resolve_num_upload_workers(args.num_upload_workers),
            print_report=False,
        )
    else:
        api.upload_folder(
            folder_path=str(scene_dir),
            path_in_repo=scene_name,
            repo_id=repo_id,
            repo_type=args.repo_type,
            revision=args.branch,
            commit_message=f"[{args.data_kind}] upload scene {scene_name}",
        )


def _sync_scene_prune_remote_extras(
    api: HfApi,
    args: Args,
    repo_id: str,
    local_root: Path,
    scene_name: str,
    remote_files_for_scene: list[str],
) -> None:
    scene_dir = local_root / scene_name
    local_rel_files = {
        str(path.relative_to(scene_dir)).replace(os.sep, "/")
        for path in scene_dir.rglob("*")
        if path.is_file()
    }
    remote_rel_files = {
        file_path[len(scene_name) + 1 :]
        for file_path in remote_files_for_scene
        if file_path.startswith(f"{scene_name}/")
    }
    stale_rel_files = sorted(remote_rel_files - local_rel_files)
    if not stale_rel_files or args.dry_run:
        return

    operations = [
        CommitOperationDelete(path_in_repo=f"{scene_name}/{rel_file}")
        for rel_file in stale_rel_files
    ]
    api.create_commit(
        repo_id=repo_id,
        repo_type=args.repo_type,
        revision=args.branch,
        operations=operations,
        commit_message=f"[{args.data_kind}] sync prune stale files in {scene_name}",
    )


def _download_scene(
    args: Args,
    repo_id: str,
    local_root: Path,
    scene_name: str,
) -> None:
    if args.dry_run:
        return
    snapshot_download(
        repo_id=repo_id,
        repo_type=args.repo_type,
        revision=args.branch,
        allow_patterns=[f"{scene_name}/**"],
        local_dir=str(local_root),
        token=args.token,
    )


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


def _collect_scene_stats(local_root: Path, scene_names: list[str]) -> dict[str, tuple[int, int]]:
    stats: dict[str, tuple[int, int]] = {}
    for scene_name in scene_names:
        scene_dir = local_root / scene_name
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        n_files = 0
        n_bytes = 0
        for path in scene_dir.rglob("*"):
            if path.is_file():
                n_files += 1
                n_bytes += path.stat().st_size
        stats[scene_name] = (n_files, n_bytes)
    return stats


def _resolve_num_upload_workers(num_upload_workers: int | None) -> int:
    if num_upload_workers is not None:
        if num_upload_workers <= 0:
            raise ValueError(f"num_upload_workers must be > 0, got {num_upload_workers}")
        return num_upload_workers
    nproc = os.cpu_count() or 1
    return min(max(4, nproc // 2), 16)


def main() -> None:
    args = tyro.cli(Args)
    console = Console()
    if args.hf_progress_bars:
        enable_progress_bars()
    else:
        disable_progress_bars()

    runtime_paths = load_runtime_paths(REPO_ROOT / args.paths_config)
    ensure_runtime_dirs(runtime_paths)
    repo_id = _resolve_repo_id(args)
    token = _resolve_token(args)
    args.token = token
    if args.schedule and args.action in {"upload", "sync"}:
        env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if env_token is None or env_token.strip() == "":
            raise ValueError(
                "Scheduled upload/sync requires HF_TOKEN to be set in the submit shell "
                "(it is forwarded to Slurm via --export ALL)."
            )
    local_root = _resolve_local_root(args)
    local_root.mkdir(parents=True, exist_ok=True)

    if args.action in {"upload", "sync"} and token is None:
        raise ValueError(
            "Missing Hugging Face write token for upload/sync. "
            "Set HF_TOKEN in your environment (recommended for Slurm), "
            "or pass --token."
        )

    api = HfApi(token=token)

    if args.action in {"upload", "sync"}:
        available_scene_names = _list_local_scene_names(local_root)
    else:
        available_scene_names = _list_remote_scene_names(api, repo_id, args.repo_type, args.branch)

    selected_scene_names = _select_scene_names(
        available_scene_names=available_scene_names,
        scene_names_arg=args.scene_names,
        scene_name_includes=args.scene_name_includes,
        run_all=args.run_all,
    )
    if not selected_scene_names:
        console.print("[yellow]No scenes selected.[/yellow]")
        return

    scene_index = _resolve_scene_index()
    is_array_task = scene_index is not None
    if is_array_task:
        if scene_index < 0 or scene_index >= len(selected_scene_names):
            raise IndexError(
                f"Scene index {scene_index} out of range (0..{len(selected_scene_names)-1})."
            )
        selected_scene_names = [selected_scene_names[scene_index]]

    scene_stats = None
    if args.action in {"upload", "sync"}:
        scene_stats = _collect_scene_stats(local_root, selected_scene_names)

    _print_plan(
        console=console,
        args=args,
        repo_id=repo_id,
        local_root=local_root,
        selected_scene_names=selected_scene_names,
        scene_stats=scene_stats,
    )
    if not is_array_task and not _confirm(console, args.yes):
        console.print("[yellow]Cancelled.[/yellow]")
        return
    if args.schedule and not is_array_task:
        _submit_array(args, selected_scene_names, runtime_paths, console)
        return

    if args.action in {"upload", "sync"} and not args.dry_run:
        api.create_repo(
            repo_id=repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )

    if args.action == "upload":
        total = len(selected_scene_names)
        for idx, scene_name in enumerate(selected_scene_names, start=1):
            console.print(f"[cyan]Uploading[/cyan] [{idx}/{total}] {scene_name}")
            _upload_scene(api, args, repo_id, local_root, scene_name)
        console.print("[bold green]Upload finished.[/bold green]")
        return

    if args.action == "download":
        total = len(selected_scene_names)
        for idx, scene_name in enumerate(selected_scene_names, start=1):
            console.print(f"[cyan]Downloading[/cyan] [{idx}/{total}] {scene_name}")
            _download_scene(args, repo_id, local_root, scene_name)
        console.print("[bold green]Download finished.[/bold green]")
        return

    # sync
    remote_files_by_scene = _index_remote_files_by_scene(api, repo_id, args.repo_type, args.branch)
    total = len(selected_scene_names)
    for idx, scene_name in enumerate(selected_scene_names, start=1):
        console.print(f"[cyan]Syncing[/cyan] [{idx}/{total}] {scene_name}")
        _upload_scene(api, args, repo_id, local_root, scene_name)
        _sync_scene_prune_remote_extras(
            api=api,
            args=args,
            repo_id=repo_id,
            local_root=local_root,
            scene_name=scene_name,
            remote_files_for_scene=remote_files_by_scene.get(scene_name, []),
        )
    console.print("[bold green]Sync finished.[/bold green]")


if __name__ == "__main__":
    main()
