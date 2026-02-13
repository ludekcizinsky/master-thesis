from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote
import os
import re
import sys
import tempfile

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.path_config import ensure_runtime_dirs, load_runtime_paths


@dataclass
class CandidateFile:
    scene_name: str
    local_path: Path
    path_in_repo: str
    task: str
    epoch: int


@dataclass
class Args:
    exp_name: str
    paths_config: Path = Path("configs/paths.yaml")
    repo_id: str | None = "ludekcizinsky/rerun-exp-eval"
    repo_type: str = "dataset"  # model, dataset, space
    branch: str = "main"
    results_root: Path | None = None
    scene_name_includes: str | None = None
    selection: str = "eval_latest"  # eval_latest, eval_all
    rrd_glob: List[str] = field(default_factory=list)
    rerun_subdir: str = "rerun"
    rerun_version: str = "0.29.1"
    private: bool = False
    token: str | None = None
    dry_run: bool = False


def _parse_selection(selection: str) -> Tuple[bool, bool]:
    tokens = {token.strip().lower() for token in selection.split(",") if token.strip()}
    valid = {"eval_latest", "eval_all", "none"}
    unknown = tokens - valid
    if unknown:
        raise ValueError(f"Unknown selection token(s): {sorted(unknown)}")
    if "none" in tokens and len(tokens) > 1:
        raise ValueError("selection='none' cannot be combined with other selection tokens.")
    return "eval_latest" in tokens, "eval_all" in tokens


def _sorted_eval_rrds(rerun_dir: Path) -> List[Tuple[int, Path]]:
    pattern = re.compile(r"(evaluation(?:_[a-z0-9]+)?)_epoch_(\d+)\.rrd$")
    files: List[Tuple[int, str, Path]] = []
    for path in rerun_dir.glob("evaluation*_epoch_*.rrd"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        prefix = match.group(1)
        epoch = int(match.group(2))
        files.append((epoch, prefix, path))
    files.sort(key=lambda item: (item[0], item[1]))
    return [(epoch, path) for epoch, _prefix, path in files]


def _parse_task_and_epoch(file_name: str) -> Tuple[str, int]:
    trn = re.fullmatch(r"trn_nv_generation_epoch_(\d+)\.rrd", file_name)
    if trn:
        return "trn_nv_generation", int(trn.group(1))
    pose = re.fullmatch(r"evaluation_pose_epoch_(\d+)\.rrd", file_name)
    if pose:
        return "pose", int(pose.group(1))
    nvs = re.fullmatch(r"evaluation_nvs_epoch_(\d+)\.rrd", file_name)
    if nvs:
        return "nvs", int(nvs.group(1))
    generic = re.fullmatch(r"evaluation_epoch_(\d+)\.rrd", file_name)
    if generic:
        return "nvs", int(generic.group(1))
    fallback = re.search(r"_epoch_(\d+)\.rrd$", file_name)
    if fallback:
        return "other", int(fallback.group(1))
    return "other", -1


def _collect_candidates(args: Args) -> List[CandidateFile]:
    include_eval_latest, include_eval_all = _parse_selection(args.selection)
    candidates: List[CandidateFile] = []

    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root not found: {args.results_root}")

    for scene_dir in sorted(args.results_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene_name = scene_dir.name
        if args.scene_name_includes and args.scene_name_includes not in scene_name:
            continue

        exp_dir = scene_dir / args.exp_name
        rerun_dir = exp_dir / args.rerun_subdir
        if not rerun_dir.is_dir():
            continue

        selected: Dict[Path, None] = {}

        eval_rrds = _sorted_eval_rrds(rerun_dir)
        if include_eval_all:
            for _epoch, path in eval_rrds:
                selected[path] = None
        elif include_eval_latest and eval_rrds:
            max_epoch = max(epoch for epoch, _path in eval_rrds)
            for epoch, path in eval_rrds:
                if epoch == max_epoch:
                    selected[path] = None

        for pattern in args.rrd_glob:
            for path in rerun_dir.glob(pattern):
                if path.is_file() and path.suffix == ".rrd":
                    selected[path] = None

        selected_paths = sorted(selected.keys(), key=lambda p: p.name)
        selected_names = {path.name for path in selected_paths}

        for path in selected_paths:
            task_name, epoch = _parse_task_and_epoch(path.name)

            # If legacy evaluation_epoch_<N>.rrd co-exists with task-specific files
            # for the same epoch, prefer task-specific exports and skip legacy.
            if re.fullmatch(r"evaluation_epoch_(\d+)\.rrd", path.name):
                if epoch >= 0:
                    if (
                        f"evaluation_nvs_epoch_{epoch:04d}.rrd" in selected_names
                        or f"evaluation_pose_epoch_{epoch:04d}.rrd" in selected_names
                    ):
                        continue

            path_in_repo = f"experiments/{args.exp_name}/{scene_name}/rerun_files/{path.name}"
            candidates.append(
                CandidateFile(
                    scene_name=scene_name,
                    local_path=path,
                    path_in_repo=path_in_repo,
                    task=task_name,
                    epoch=epoch,
                )
            )

    return candidates


def _repo_base_url(repo_id: str, repo_type: str) -> str:
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}"
    if repo_type == "space":
        return f"https://huggingface.co/spaces/{repo_id}"
    if repo_type == "model":
        return f"https://huggingface.co/{repo_id}"
    raise ValueError(f"Unsupported repo_type: {repo_type}")


def _resolve_url(repo_base: str, branch: str, path_in_repo: str) -> str:
    return f"{repo_base}/resolve/{branch}/{path_in_repo}"


def _rerun_link(resolve_url: str, rerun_version: str) -> str:
    encoded = quote(resolve_url, safe="")
    return f"https://app.rerun.io/version/{rerun_version}/index.html?url={encoded}"


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def _build_repo_readme(
    args: Args,
    repo_base: str,
    scene_name: str,
    candidates: List[CandidateFile],
) -> str:
    scene_items = sorted(candidates, key=lambda item: (item.task, item.epoch, item.local_path.name))

    def _task_rows_html(scene_items: List[CandidateFile], task_key: str) -> List[str]:
        task_items = [item for item in scene_items if item.task == task_key]
        if not task_items:
            return ["<tr><td>-</td><td>-</td><td>-</td></tr>"]
        rows: List[str] = []
        for item in sorted(task_items, key=lambda it: (it.epoch, it.local_path.name)):
            direct = _resolve_url(repo_base, args.branch, item.path_in_repo)
            rerun_web = _rerun_link(direct, args.rerun_version)
            epoch_text = f"{item.epoch:04d}" if item.epoch >= 0 else "-"
            rows.append(
                "<tr>"
                f"<td><code>{epoch_text}</code></td>"
                f'<td><a href="{direct}">rrd</a></td>'
                f'<td><a href="{rerun_web}">viewer</a></td>'
                "</tr>"
            )
        return rows

    lines: List[str] = []
    lines.append('<div style="margin:0 0 0.6em 0;">')
    lines.append(f"  <b>Scene:</b> <code>{scene_name}</code><br/>")
    lines.append(f"  <b>Rerun viewer version:</b> <code>{args.rerun_version}</code>")
    lines.append("</div>")
    lines.append("")
    lines.append('<hr style="margin:0.4em 0;">')
    lines.append("")
    lines.append('<h3 style="margin:0.6em 0 0.2em 0;">🥽 Qualitative results</h3>')
    lines.append("")
    lines.append("Here, you can visualise qualitative results for the task on which the given scene was evaluated.")
    lines.append("")

    lines.append('<table width="100%">')
    lines.append('<tr valign="top">')
    lines.append('<td width="50%">')
    lines.append("")
    lines.append('<div style="margin:0 0 0.2em 0;"><b>Novel view synthesis</b></div>')
    lines.append("")
    lines.append("<table>")
    lines.append("<tr><th>Epoch</th><th>Download</th><th>Open</th></tr>")
    lines.extend(_task_rows_html(scene_items, "nvs"))
    lines.append("</table>")
    lines.append("")
    lines.append("</td>")
    lines.append('<td width="50%">')
    lines.append("")
    lines.append('<div style="margin:0 0 0.2em 0;"><b>Pose estimation</b></div>')
    lines.append("")
    lines.append("<table>")
    lines.append("<tr><th>Epoch</th><th>Download</th><th>Open</th></tr>")
    lines.extend(_task_rows_html(scene_items, "pose"))
    lines.append("</table>")
    lines.append("")
    lines.append("</td>")
    lines.append("</tr>")
    lines.append("</table>")
    lines.append("")

    lines.append('<h3 style="margin:0.6em 0 0.2em 0;">📹 View Densification using DiFix</h3>')
    lines.append("")
    lines.append(
        "Here, you can rerun the view densification process, i.e., synthesising novel pseudo ground truth views "
        "which are then used in the 3DGS optimisation."
    )
    lines.append("")
    lines.append("<table>")
    lines.append("<tr><th>Epoch</th><th>Download</th><th>Open</th></tr>")
    lines.extend(_task_rows_html(scene_items, "trn_nv_generation"))
    lines.append("</table>")
    lines.append("")

    return "\n".join(lines)


def _print_plan(
    console: Console,
    args: Args,
    repo_base: str,
    candidates: List[CandidateFile],
    readme_paths: List[str],
) -> None:
    total_bytes = sum(item.local_path.stat().st_size for item in candidates)
    console.print(
        Panel.fit(
            f"[bold]Experiment:[/bold] {args.exp_name}\n"
            f"[bold]Repo:[/bold] {args.repo_id} ({args.repo_type})\n"
            f"[bold]Branch:[/bold] {args.branch}\n"
            f"[bold]Files:[/bold] {len(candidates)} (.rrd) + {len(readme_paths)} README(s)\n"
            f"[bold]Total size:[/bold] {_human_size(total_bytes)}\n"
            f"[bold]Repo base URL:[/bold] {repo_base}",
            title="Upload Plan",
            border_style="cyan",
        )
    )

    table = Table(title="Files to Upload")
    table.add_column("#", justify="right")
    table.add_column("Scene")
    table.add_column("Local File")
    table.add_column("Path in Repo")
    table.add_column("Size", justify="right")
    for idx, item in enumerate(candidates, start=1):
        table.add_row(
            str(idx),
            item.scene_name,
            str(item.local_path),
            item.path_in_repo,
            _human_size(item.local_path.stat().st_size),
        )
    console.print(table)
    readme_table = Table(title="README Files to Upload")
    readme_table.add_column("#", justify="right")
    readme_table.add_column("Path in Repo")
    for idx, readme_path in enumerate(sorted(readme_paths), start=1):
        readme_table.add_row(str(idx), readme_path)
    console.print(readme_table)


def _upload(args: Args, candidates: List[CandidateFile], readmes_by_scene: Dict[str, str]) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    operations = [
        CommitOperationAdd(path_in_repo=item.path_in_repo, path_or_fileobj=str(item.local_path))
        for item in candidates
    ]

    temp_readmes: List[Path] = []
    try:
        for scene_name, readme_content in readmes_by_scene.items():
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as handle:
                handle.write(readme_content)
                temp_readme = Path(handle.name)
            temp_readmes.append(temp_readme)
            readme_repo_path = f"experiments/{args.exp_name}/{scene_name}/README.md"
            operations.append(
                CommitOperationAdd(path_in_repo=readme_repo_path, path_or_fileobj=str(temp_readme))
            )
        api.create_commit(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.branch,
            operations=operations,
            commit_message=f"Upload Rerun recordings for {args.exp_name}",
        )
    finally:
        for temp_path in temp_readmes:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    args = tyro.cli(Args)
    console = Console()
    runtime_paths = load_runtime_paths(REPO_ROOT / args.paths_config)
    ensure_runtime_dirs(runtime_paths)
    if args.results_root is None:
        args.results_root = runtime_paths.results_root_dir

    if args.repo_id is None:
        args.repo_id = os.environ.get("HF_RERUN_REPO_ID")
    if args.repo_id is None:
        raise ValueError("Missing repo id. Pass --repo-id or set HF_RERUN_REPO_ID.")

    candidates = _collect_candidates(args)
    if not candidates:
        console.print(
            Panel.fit(
                f"No .rrd files found for experiment '{args.exp_name}'.",
                title="Nothing to Upload",
                border_style="yellow",
            )
        )
        return

    repo_base = _repo_base_url(args.repo_id, args.repo_type)
    scene_to_candidates: Dict[str, List[CandidateFile]] = {}
    for item in candidates:
        scene_to_candidates.setdefault(item.scene_name, []).append(item)
    readmes_by_scene: Dict[str, str] = {
        scene_name: _build_repo_readme(args, repo_base, scene_name, scene_candidates)
        for scene_name, scene_candidates in scene_to_candidates.items()
    }
    readme_paths = [
        f"experiments/{args.exp_name}/{scene_name}/README.md"
        for scene_name in sorted(readmes_by_scene.keys())
    ]

    _print_plan(console, args, repo_base, candidates, readme_paths)

    if args.dry_run:
        console.print("[bold green]Dry run only. No upload performed.[/bold green]")
        return

    _upload(args, candidates, readmes_by_scene)

    links_table = Table(title="Uploaded Links")
    links_table.add_column("Scene")
    links_table.add_column("File")
    links_table.add_column("Rerun Web Link")
    for item in candidates:
        direct = _resolve_url(repo_base, args.branch, item.path_in_repo)
        links_table.add_row(
            item.scene_name,
            item.local_path.name,
            _rerun_link(direct, args.rerun_version),
        )
    console.print(links_table)
    console.print("[bold green]Done.[/bold green] Uploaded scene-level README files:")
    for scene_name in sorted(readmes_by_scene.keys()):
        readme_path = f"experiments/{args.exp_name}/{scene_name}/README.md"
        console.print(f"  - {repo_base}/blob/{args.branch}/{readme_path}")


if __name__ == "__main__":
    main()
