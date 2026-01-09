from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont
import tyro


@dataclass
class Args:
    exp_names: List[str]
    camera_id: str
    frame_idx_str: str
    epoch_str: str
    scene_name: str
    inputs_root: Path = Path("/scratch/izar/cizinsky/thesis/results")
    results_root: Path = Path("/scratch/izar/cizinsky/thesis/nvs_renders_comparison")


def main(args: Args) -> None:
    render_images = []
    gt_image = None
    for exp_name in args.exp_names:
        image_path = (
            args.inputs_root
            / args.scene_name
            / "evaluation"
            / exp_name
            / f"epoch_{args.epoch_str}"
            / args.camera_id
            / "render_vs_gt"
            / f"{args.frame_idx_str}.jpg"
        )
        image = Image.open(image_path)
        width, height = image.size
        mid = width // 2
        render = image.crop((0, 0, mid, height))
        gt = image.crop((mid, 0, width, height))
        render_images.append((exp_name, render))
        if gt_image is None:
            gt_image = gt

    if gt_image is None:
        raise RuntimeError("No images found to extract ground truth.")

    tiles = render_images + [("GT", gt_image)]
    tile_width, tile_height = tiles[0][1].size
    output_width = tile_width * len(tiles)
    output_height = tile_height
    output = Image.new("RGB", (output_width, output_height))

    draw = ImageDraw.Draw(output)
    font_size = 20 
    font = None
    for font_name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for idx, (label, tile) in enumerate(tiles):
        x_offset = idx * tile_width
        output.paste(tile, (x_offset, 0))
        if font is not None:
            text = label
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            padding = 4
            rect = (
                x_offset,
                0,
                x_offset + text_width + padding * 2,
                text_height + padding * 2,
            )
            draw.rectangle(rect, fill="black")
            draw.text(
                (x_offset + padding, padding),
                text,
                fill="white",
                font=font,
            )

    out_name = f"side_by_side.jpg"
    out_path = args.results_root / args.scene_name / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
