from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import tyro

@dataclass
class Args:
    exp_name: str
    epoch_str: str
    scene_names: List[str]
    results_root: Path = Path("/scratch/izar/cizinsky/thesis/results")
    output_dir: Path = Path("/home/cizinsky/master-thesis/results")


@dataclass
class TaskSpec:
    name: str
    title: str
    filename: str
    columns: Sequence[Tuple[str, str]]
    avg_formats: Dict[str, str] = field(default_factory=dict)


def _parse_metrics(results_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            metrics[key] = float(value)
    return metrics


def _format_table(df: pd.DataFrame, avg_formats: Dict[str, str]) -> pd.DataFrame:
    records = []
    for scene, row in df.iterrows():
        record: Dict[str, str] = {"scene": str(scene)}
        for col in df.columns:
            value = row[col]
            if scene == "avg":
                fmt = avg_formats.get(col, ".4f")
                record[col] = f"{value:{fmt}}"
            else:
                record[col] = f"{value:.4f}"
        records.append(record)
    return pd.DataFrame(records).set_index("scene")


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = ["scene", *df.columns]
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for scene, row in df.iterrows():
        values: Iterable[str] = [str(scene), *[str(row[col]) for col in df.columns]]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _collect_task_results(args: Args, task: TaskSpec) -> pd.DataFrame:
    rows = []
    for scene in args.scene_names:
        results_path = (
            args.results_root
            / scene
            / "evaluation"
            / args.exp_name
            / f"epoch_{args.epoch_str}"
            / task.filename
        )
        metrics = _parse_metrics(results_path)
        row: Dict[str, float] = {"scene": scene}
        for column, metric_key in task.columns:
            row[column] = metrics[metric_key]
        rows.append(row)

    column_names = [column for column, _ in task.columns]
    df = pd.DataFrame(rows).set_index("scene")[column_names]
    df.loc["avg"] = df.mean(numeric_only=True)
    return _format_table(df, avg_formats=task.avg_formats)


def main(args: Args) -> None:
    tasks = [
        TaskSpec(
            name="nvs",
            title="NVS",
            filename="novel_view_results.txt",
            columns=[("SSIM", "ssim"), ("PSNR", "psnr"), ("LPIPS", "lpips")],
            avg_formats={"SSIM": ".3f", "PSNR": ".1f", "LPIPS": ".4f"},
        ),
        TaskSpec(
            name="pose_estimation",
            title="Pose Estimation",
            filename="pose_estimation_overall_results.txt",
            columns=[
                ("MPJPE_mm", "mpjpe_mm"),
                ("MVE_mm", "mve_mm"),
                ("CD_mm", "cd_mm"),
                ("PCDR", "pcdr"),
            ],
        ),
        TaskSpec(
            name="segmentation",
            title="Segmentation",
            filename="segmentation_overall_results.txt",
            columns=[("IoU", "iou"), ("Recall", "recall"), ("F1", "f1")],
        ),
    ]

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_sections = []
    for task in tasks:
        formatted_df = _collect_task_results(args, task)
        csv_path = out_dir / f"{task.name}.csv"
        formatted_df.to_csv(csv_path)
        md_path = out_dir / f"{task.name}.md"
        md_path.write_text(_to_markdown_table(formatted_df), encoding="utf-8")
        combined_sections.append(f"## {task.title}\n{_to_markdown_table(formatted_df)}")
        print(f"Wrote {csv_path}")
        print(f"Wrote {md_path}")

    combined_path = out_dir / "all_results.md"
    combined_path.write_text("\n\n".join(combined_sections), encoding="utf-8")
    print(f"Wrote {combined_path}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
