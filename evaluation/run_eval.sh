#!/usr/bin/env bash
set -euo pipefail

# Default input directories (override by exporting variables before calling the script)
IMAGES_PATH=${IMAGES_PATH:-/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/image}
GT_MASKS_PATH=${GT_MASKS_PATH:-/scratch/izar/cizinsky/multiply-output/training/original_taichi/pseudo_gt_masks}
CONDA_ENV=${CONDA_ENV:-thesis}

# Default list of render directories (space-separated string override via RENDERS_PATHS)
DEFAULT_RENDER_PATHS=(
  "/scratch/izar/cizinsky/multiply-output/training/original_taichi/joined_fg"
  "/scratch/izar/cizinsky/thesis/output/taichi/training/robust-frost-232_vzoxt51b/visualizations/all_humans/epoch_0400/rgb"
)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY_SCRIPT="$SCRIPT_DIR/evaluate_metrics.py"

module load gcc ffmpeg 

if [[ -n "${RENDERS_PATHS:-}" ]]; then
  read -r -a RENDER_PATH_LIST <<< "$RENDERS_PATHS"
else
  RENDER_PATH_LIST=("${DEFAULT_RENDER_PATHS[@]}")
fi

for RENDERS_PATH in "${RENDER_PATH_LIST[@]}"; do
  echo "Evaluating renders at: $RENDERS_PATH"
  conda run -n "$CONDA_ENV" python "$PY_SCRIPT" \
    --images-path "$IMAGES_PATH" \
    --gt-masks-path "$GT_MASKS_PATH" \
    --renders-path "$RENDERS_PATH"

  MASKED_DIR="$RENDERS_PATH/masked_renders"
  OUTPUT_VIDEO="$MASKED_DIR/masked_renders.mp4"
  echo "Creating masked video at: $OUTPUT_VIDEO"
  ffmpeg -hide_banner -loglevel error -y -framerate 10 \
    -i "$MASKED_DIR/%04d.png" \
    -c:v libx264 -pix_fmt yuv420p "$OUTPUT_VIDEO"

  echo "============================================================"
done
