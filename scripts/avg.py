from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import tyro

@dataclass
class Args:
    exp_name: str
    epoch_str: str
    scene_names: List[str]
    results_root: Path = Path("/scratch/izar/cizinsky/thesis/results")
    output_dir: Path = Path("/home/cizinsky/master-thesis/results")

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


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scene, row in df.iterrows():
        record: Dict[str, str] = {"scene": str(scene)}
        for col in df.columns:
            value = row[col]
            if scene == "avg":
                if col == "SSIM":
                    record[col] = f"{value:.3f}"
                elif col == "PSNR":
                    record[col] = f"{value:.1f}"
                else:
                    record[col] = f"{value:.4f}"
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


def main(args: Args) -> None:
    rows = []
    for scene in args.scene_names:
        results_path = (
            args.results_root
            / scene
            / "evaluation"
            / args.exp_name
            / f"epoch_{args.epoch_str}"
            / "novel_view_results.txt"
        )
        metrics = _parse_metrics(results_path)
        rows.append(
            {
                "scene": scene,
                "SSIM": metrics["ssim"],
                "PSNR": metrics["psnr"],
                "LPIPS": metrics["lpips"],
            }
        )

    columns = ["SSIM", "PSNR", "LPIPS"]
    df = pd.DataFrame(rows).set_index("scene")[columns]
    df.loc["avg"] = df.mean(numeric_only=True)
    formatted_df = _format_table(df)

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "nvs.csv"
    formatted_df.to_csv(csv_path)
    md_path = out_dir / "nvs.md"
    md_path.write_text(_to_markdown_table(formatted_df), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
