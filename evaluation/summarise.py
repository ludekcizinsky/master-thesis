from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas is required to summarise metrics. Install it with `pip install pandas` and re-run."
    ) from exc


METRIC_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("rendering", ("ssim", "psnr", "lpips")),
    ("segmentation", ("segm_iou", "segm_f1", "segm_recall")),
    ("pose", ("pa-mpjpe",)),
)
METRIC_HIGHER_IS_BETTER = {
    "ssim": True,
    "psnr": True,
    "lpips": False,
    "segm_iou": True,
    "segm_f1": True,
    "segm_recall": True,
    "pa-mpjpe": False,
}
SEPARATOR_LABEL = " "


def _read_metrics(csv_path: Path) -> Dict[str, float]:
    """Return a metric -> value mapping for a given scene file."""
    metrics_df = pd.read_csv(csv_path)
    if "metric" not in metrics_df or "value" not in metrics_df:
        raise ValueError(f"{csv_path} does not contain the required columns 'metric' and 'value'")
    series = metrics_df.set_index("metric")["value"]
    return series.to_dict()


def _collect_metrics(metrics_root: Path) -> pd.DataFrame:
    """Collect all scene metrics from the directory tree into a flat DataFrame."""
    records: List[Dict[str, object]] = []

    for dataset_dir in sorted(metrics_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        for method_dir in sorted(dataset_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name

            for csv_path in sorted(method_dir.glob("*.csv")):
                scene_name = csv_path.stem
                metrics = _read_metrics(csv_path)
                record = {"dataset": dataset_name, "scene": scene_name, "method": method_name}
                record.update(metrics)
                records.append(record)

    if not records:
        raise RuntimeError(f"No metrics found under {metrics_root}")

    return pd.DataFrame.from_records(records)


def _ordered_metrics(metric_columns: Sequence[str]) -> Tuple[List[str], List[int]]:
    """Return ordered metrics plus separator insertion points between groups."""
    ordered: List[str] = []
    separators: List[int] = []

    for group_idx, (_, metrics) in enumerate(METRIC_GROUPS):
        present = [m for m in metrics if m in metric_columns]
        if not present:
            continue
        ordered.extend(present)
        if any(
            any(m in metric_columns for m in next_metrics)
            for _, next_metrics in METRIC_GROUPS[group_idx + 1 :]
        ):
            separators.append(len(ordered))

    remaining = [m for m in metric_columns if m not in ordered]
    ordered.extend(remaining)
    return ordered, separators


def _scene_tables(dataset_df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, List[int]]]:
    """Return a mapping of scene name to tables plus separator placement."""
    tables: Dict[str, Tuple[pd.DataFrame, List[int]]] = {}
    for scene_name, scene_df in dataset_df.groupby("scene"):
        metric_columns = [col for col in scene_df.columns if col not in {"dataset", "scene", "method"}]
        ordered_metrics, separator_positions = _ordered_metrics(metric_columns)
        if ordered_metrics:
            table = scene_df.set_index("method")[ordered_metrics]
        else:
            table = scene_df.set_index("method")
        tables[scene_name] = (table, separator_positions)
    return tables


def _format_scene_table(table: pd.DataFrame, separators: Sequence[int]) -> str:
    """Return a markdown table string with separators and bold best metrics."""
    columns = list(table.columns)
    separator_set = set(separators)

    display_columns: List[str] = []
    for idx, column in enumerate(columns, start=1):
        display_columns.append(column)
        if idx in separator_set:
            display_columns.append(SEPARATOR_LABEL)

    header = ["method"] + display_columns
    align = [":--"] + ["--:" if col != SEPARATOR_LABEL else ":-:" for col in display_columns]

    best_values: Dict[str, float] = {}
    for col in columns:
        series = pd.to_numeric(table[col], errors="coerce").dropna()
        if series.empty:
            continue
        better_high = METRIC_HIGHER_IS_BETTER.get(col, True)
        best_values[col] = series.max() if better_high else series.min()

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
    ]

    for method, row in table.iterrows():
        cells = [str(method)]
        for idx, col in enumerate(columns, start=1):
            value = row[col]
            if pd.isna(value):
                formatted = "-"
            else:
                formatted = f"{value:.3f}"
                best_value = best_values.get(col)
                if best_value is not None:
                    tolerance = max(1e-9, abs(best_value) * 1e-6)
                    if abs(float(value) - float(best_value)) <= tolerance:
                        formatted = f"**{formatted}**"
            cells.append(formatted)
            if idx in separator_set:
                cells.append("")

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _dataset_markdown(dataset_name: str, dataset_df: pd.DataFrame) -> str:
    """Build markdown containing one table per scene."""
    tables = _scene_tables(dataset_df)
    lines: List[str] = [f"# {dataset_name} metrics", ""]
    for scene_name in sorted(tables):
        lines.append(f"## {scene_name}")
        lines.append("")
        table, separators = tables[scene_name]
        lines.append(_format_scene_table(table, separators))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def summarise_metrics(metrics_root: Path, output_dir: Path) -> Mapping[str, Path]:
    """
    Build per-dataset markdown summaries of all scene metrics.

    Parameters
    ----------
    metrics_root:
        Root directory that contains dataset/method/scene.csv files.
    output_dir:
        Directory where the markdown tables will be written.

    Returns
    -------
    Mapping[str, Path]
        A mapping from dataset name to the markdown file that stores its summary.
    """
    metrics_root = Path(metrics_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = _collect_metrics(metrics_root)
    outputs: Dict[str, Path] = {}

    for dataset_name, dataset_df in metrics_df.groupby("dataset"):
        markdown = _dataset_markdown(dataset_name, dataset_df)
        output_path = output_dir / f"{dataset_name}_summary.md"
        output_path.write_text(markdown)
        outputs[dataset_name] = output_path

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise evaluation metrics into markdown tables.")
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("/scratch/izar/cizinsky/thesis/evaluation/metrics"),
        help="Root directory of the dataset/method metrics tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/summaries"),
        help="Directory to store the generated markdown tables.",
    )
    args = parser.parse_args()

    outputs = summarise_metrics(args.metrics_root, args.output_dir)
    for dataset, path in outputs.items():
        print(f"Wrote {dataset} summary to {path}")


if __name__ == "__main__":
    main()
