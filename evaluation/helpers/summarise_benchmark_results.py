from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Literal

import pandas as pd
import tyro

@dataclass
class Args:
    exp_name: str
    epoch_str: str
    scene_names: List[str]
    src_cam_ids: List[int] = field(default_factory=list)
    include_segmentation_novel_cams: bool = False
    results_root: Path = Path("/scratch/izar/cizinsky/thesis/results")
    output_dir: Path = Path("/home/cizinsky/master-thesis/results")


@dataclass
class TaskSpec:
    name: str
    title: str
    kind: Literal["kv", "csv_camera"]
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


def _results_path(args: Args, scene: str, task: TaskSpec) -> Path:
    return (
        args.results_root
        / scene
        / "evaluation"
        / args.exp_name
        / f"epoch_{args.epoch_str}"
        / task.filename
    )


def _collect_task_results(
    args: Args,
    task: TaskSpec,
    scene_names: Sequence[str],
    src_cam_ids: Sequence[int],
) -> pd.DataFrame:
    if task.kind == "csv_camera" and len(src_cam_ids) != len(scene_names):
        raise ValueError(
            f"--src-cam-ids must have the same length as --scene-names "
            f"({len(src_cam_ids)} vs {len(scene_names)})."
        )

    rows = []
    for idx, scene in enumerate(scene_names):
        results_path = _results_path(args, scene, task)
        if task.kind == "kv":
            metrics = _parse_metrics(results_path)
        else:
            src_cam_id = int(src_cam_ids[idx])
            df = pd.read_csv(results_path)
            if "camera_id" not in df.columns:
                raise ValueError(
                    f"Expected a 'camera_id' column in {results_path}, got {list(df.columns)}"
                )
            row_match = df.loc[df["camera_id"] == src_cam_id]
            if row_match.empty:
                raise ValueError(
                    f"No row found for camera_id={src_cam_id} in {results_path}"
                )
            metrics = row_match.iloc[0].to_dict()

        row: Dict[str, float] = {"scene": scene}
        for column, metric_key in task.columns:
            row[column] = metrics[metric_key]
        rows.append(row)

    column_names = [column for column, _ in task.columns]
    df = pd.DataFrame(rows).set_index("scene")[column_names]
    df.loc["avg"] = df.mean(numeric_only=True)
    return _format_table(df, avg_formats=task.avg_formats)


def _scene_dataset(scene: str) -> str:
    return scene.split("_", 1)[0]


def main(args: Args) -> None:
    tasks = [
        TaskSpec(
            name="nvs",
            title="NVS",
            kind="kv",
            filename="novel_view_results.txt",
            columns=[("SSIM", "ssim"), ("PSNR", "psnr"), ("LPIPS", "lpips")],
            avg_formats={"SSIM": ".3f", "PSNR": ".1f", "LPIPS": ".4f"},
        ),
        TaskSpec(
            name="pose_estimation",
            title="Pose Estimation",
            kind="kv",
            filename="pose_estimation_overall_results.txt",
            columns=[
                ("MPJPE_mm", "mpjpe_mm"),
                ("MVE_mm", "mve_mm"),
                ("CD_mm", "cd_mm"),
                ("PCDR", "pcdr"),
            ],
        ),
        TaskSpec(
            name="reconstruction",
            title="Reconstruction",
            kind="kv",
            filename="reconstruction_overall_results.txt",
            columns=[
                ("V_IoU", "v_iou"),
                ("Chamfer_cm", "chamfer_cm"),
                ("P2S_cm", "p2s_cm"),
                ("Normal_Consistency", "normal_consistency"),
            ],
        ),
    ]

    if args.include_segmentation_novel_cams:
        tasks.append(
            TaskSpec(
                name="segmentation",
                title="Segmentation",
                kind="kv",
                filename="segmentation_overall_results.txt",
                columns=[("IoU", "iou"), ("Recall", "recall"), ("F1", "f1")],
            )
        )

    if args.src_cam_ids:
        tasks.append(
            TaskSpec(
                name="segmentation_src_cam",
                title="Segmentation (Src Cam)",
                kind="csv_camera",
                filename="segmentation_metrics_avg_per_camera.csv",
                columns=[("IoU", "iou"), ("Recall", "recall"), ("F1", "f1")],
            )
        )

    if any(task.kind == "csv_camera" for task in tasks):
        if not args.src_cam_ids:
            raise ValueError(
                "CSV camera tasks require --src-cam-ids (one per scene)."
            )
        if len(args.src_cam_ids) != len(args.scene_names):
            raise ValueError(
                f"--src-cam-ids must have the same length as --scene-names "
                f"({len(args.src_cam_ids)} vs {len(args.scene_names)})."
            )

    scene_indices_by_ds: Dict[str, List[int]] = {}
    for idx, scene in enumerate(args.scene_names):
        ds_name = _scene_dataset(scene)
        scene_indices_by_ds.setdefault(ds_name, []).append(idx)

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_sections = []
    for task in tasks:
        task_sections = []
        for ds_name, indices in scene_indices_by_ds.items():
            dataset_scenes = [args.scene_names[i] for i in indices]
            dataset_src_cam_ids = (
                [args.src_cam_ids[i] for i in indices] if args.src_cam_ids else []
            )
            missing = [
                scene
                for scene in dataset_scenes
                if not _results_path(args, scene, task).is_file()
            ]
            if missing:
                print(f"Skipping task {task.name} for dataset {ds_name}")
                continue
            formatted_df = _collect_task_results(
                args, task, dataset_scenes, dataset_src_cam_ids
            )
            csv_path = out_dir / f"{ds_name}_{task.name}.csv"
            formatted_df.to_csv(csv_path)
            md_path = out_dir / f"{ds_name}_{task.name}.md"
            md_path.write_text(_to_markdown_table(formatted_df), encoding="utf-8")
            task_sections.append(
                f"#### {ds_name}\n{_to_markdown_table(formatted_df)}"
            )
            print(f"Wrote {csv_path}")
            print(f"Wrote {md_path}")
        if task_sections:
            combined_sections.append(f"### {task.title}\n" + "\n\n".join(task_sections))

    combined_path = out_dir / "all_results.md"
    combined_path.write_text("\n\n".join(combined_sections), encoding="utf-8")
    print(f"Wrote {combined_path}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
