from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import tyro


TASK_FILES: List[Tuple[str, str]] = [
    ("novel_view_results.txt", "Novel View Synthesis"),
    ("smplx_pose_estimation_overall_results.txt", "Pose Estimation (SMPL-X)"),
    ("smpl_pose_estimation_overall_results.txt", "Pose Estimation (SMPL)"),
]


@dataclass
class Args:
    exp_name: str
    results_root: Path = Path("/scratch/izar/cizinsky/thesis/results")
    docs_root: Path = Path("/home/cizinsky/master-thesis/docs/results")
    epoch: str = "latest"
    verbose: bool = True


def _dataset_from_scene(scene_name: str) -> str:
    if "_" not in scene_name:
        return scene_name
    return scene_name.split("_", 1)[0]


def _parse_metrics(results_file: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with results_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def _find_evaluation_dir(scene_dir: Path, exp_name: str) -> Path | None:
    # New layout: <results_root>/<scene>/<exp>/evaluation
    new_layout = scene_dir / exp_name / "evaluation"
    if new_layout.is_dir():
        return new_layout

    # Legacy layout: <results_root>/<scene>/evaluation/<exp>
    old_layout = scene_dir / "evaluation" / exp_name
    if old_layout.is_dir():
        return old_layout

    return None


def _epoch_name_from_arg(epoch_arg: str) -> str:
    if epoch_arg.startswith("epoch_"):
        return epoch_arg
    if epoch_arg.isdigit():
        return f"epoch_{int(epoch_arg):04d}"
    raise ValueError(
        f"Unsupported --epoch value '{epoch_arg}'. Use 'latest', an integer like 30, "
        f"or 'epoch_0030'."
    )


def _resolve_epoch_dir(eval_dir: Path, epoch_arg: str) -> Path | None:
    epoch_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and re.fullmatch(r"epoch_\d+", d.name)]
    if not epoch_dirs:
        return None

    if epoch_arg == "latest":
        epoch_dirs = sorted(epoch_dirs, key=lambda p: int(p.name.split("_", 1)[1]))
        return epoch_dirs[-1]

    target_name = _epoch_name_from_arg(epoch_arg)
    target_dir = eval_dir / target_name
    if target_dir.is_dir():
        return target_dir
    return None


def _collect_by_task_and_dataset(
    args: Args,
) -> Tuple[Dict[str, Dict[str, Dict[str, Dict[str, float]]]], List[str]]:
    # task -> dataset -> scene -> metric -> value
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    warnings: List[str] = []

    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")

    for scene_dir in sorted(args.results_root.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name
        eval_dir = _find_evaluation_dir(scene_dir, args.exp_name)
        if eval_dir is None:
            continue

        epoch_dir = _resolve_epoch_dir(eval_dir, args.epoch)
        if epoch_dir is None:
            warnings.append(
                f"Skipping scene '{scene_name}': no matching epoch folder under {eval_dir}"
            )
            continue

        dataset_name = _dataset_from_scene(scene_name)
        for filename, task_name in TASK_FILES:
            task_file = epoch_dir / filename
            if not task_file.is_file():
                continue
            metrics = _parse_metrics(task_file)
            if not metrics:
                warnings.append(f"Skipping empty metrics file: {task_file}")
                continue

            grouped.setdefault(task_name, {}).setdefault(dataset_name, {})[scene_name] = metrics

    return grouped, warnings


def _metric_columns(scene_to_metrics: Dict[str, Dict[str, float]]) -> List[str]:
    columns: List[str] = []
    seen = set()
    for scene_name in sorted(scene_to_metrics):
        for metric_name in scene_to_metrics[scene_name].keys():
            if metric_name in seen:
                continue
            seen.add(metric_name)
            columns.append(metric_name)
    return columns


def _markdown_table(scene_to_metrics: Dict[str, Dict[str, float]]) -> str:
    columns = _metric_columns(scene_to_metrics)
    if not columns:
        return "_No numeric metrics found._"

    lines: List[str] = []
    lines.append("| scene | " + " | ".join(columns) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(columns)) + " |")

    metric_values: Dict[str, List[float]] = {metric: [] for metric in columns}

    for scene_name in sorted(scene_to_metrics):
        row_values: List[str] = []
        scene_metrics = scene_to_metrics[scene_name]
        for metric_name in columns:
            if metric_name not in scene_metrics:
                row_values.append("-")
                continue
            value = scene_metrics[metric_name]
            row_values.append(f"{value:.4f}")
            metric_values[metric_name].append(value)
        lines.append(f"| {scene_name} | " + " | ".join(row_values) + " |")

    avg_values: List[str] = []
    for metric_name in columns:
        values = metric_values[metric_name]
        if not values:
            avg_values.append("-")
        else:
            avg_values.append(f"{(sum(values) / len(values)):.4f}")
    lines.append("| avg | " + " | ".join(avg_values) + " |")
    return "\n".join(lines)


def _build_markdown(grouped: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], args: Args) -> str:
    lines: List[str] = []
    lines.append(f"# Quantitative Results: `{args.exp_name}`")
    lines.append("")
    lines.append(f"- results root: `{args.results_root}`")
    lines.append(f"- epoch selection: `{args.epoch}`")
    lines.append("")

    if not grouped:
        lines.append("_No quantitative files found for this experiment._")
        return "\n".join(lines)

    task_order = [task_name for _, task_name in TASK_FILES]
    for task_name in task_order:
        if task_name not in grouped:
            continue
        lines.append(f"## {task_name}")
        lines.append("")
        for dataset_name in sorted(grouped[task_name]):
            lines.append(f"### {dataset_name}")
            lines.append("")
            lines.append(_markdown_table(grouped[task_name][dataset_name]))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    console = Console()
    args = tyro.cli(Args)
    with console.status("[bold cyan]Scanning results and aggregating metrics...[/bold cyan]"):
        grouped, warnings = _collect_by_task_and_dataset(args)
        markdown = _build_markdown(grouped, args)

    output_dir = args.docs_root / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "quan_results.md"
    output_path.write_text(markdown, encoding="utf-8")

    console.print(
        Panel.fit(
            f"[bold green]Wrote[/bold green] [cyan]{output_path}[/cyan]",
            title="Quantitative Summary",
            border_style="green",
        )
    )
    if args.verbose:
        if warnings:
            warning_table = Table(
                title="Warnings",
                show_header=True,
                header_style="bold yellow",
                border_style="yellow",
            )
            warning_table.add_column("#", style="dim", width=4, justify="right")
            warning_table.add_column("Message", overflow="fold")
            for idx, warning in enumerate(warnings, start=1):
                warning_table.add_row(str(idx), warning)
            console.print(warning_table)

        discovered_tasks = sorted(grouped.keys())
        discovered_datasets = sorted(
            {dataset for task_data in grouped.values() for dataset in task_data.keys()}
        )
        summary_table = Table(
            title="Discovery",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
        )
        summary_table.add_column("Item", style="bold")
        summary_table.add_column("Values")
        summary_table.add_row(
            "Tasks",
            ", ".join(discovered_tasks) if discovered_tasks else "none",
        )
        summary_table.add_row(
            "Datasets",
            ", ".join(discovered_datasets) if discovered_datasets else "none",
        )
        console.print(summary_table)


if __name__ == "__main__":
    main()
