#!/usr/bin/env bash
set -euo pipefail

# Absolute paths to frame directories
run_id="feasible-flower-193_29fw9obj"
epoch="0250"
ORIGINAL_FRAMES="/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/image"
FG_FRAMES="/scratch/izar/cizinsky/thesis/output/taichi/training/${run_id}/visualizations/all_humans/epoch_${epoch}/rgb"

# Directory where all outputs will be stored (defaults to current working directory)
output_folder="/scratch/izar/cizinsky/thesis/output/taichi/training/${run_id}/visualizations"
mkdir -p "${output_folder}"

# Output video filenames (stored inside ${output_folder})
ORIGINAL_VIDEO="${output_folder}/taichi-original.mp4"
FG_VIDEO="${output_folder}/feasible-flower-fg.mp4"
STACKED_VIDEO="${output_folder}/taichi-vs-fg.mp4"

# Generate original clip
ffmpeg -y -framerate 10 -i "${ORIGINAL_FRAMES}/%04d.png" \
  -c:v libx264 -pix_fmt yuv420p "${ORIGINAL_VIDEO}"

# Generate foreground render clip
ffmpeg -y -framerate 10 -i "${FG_FRAMES}/%04d.png" \
  -c:v libx264 -pix_fmt yuv420p "${FG_VIDEO}"

# Stack the two clips side by side
ffmpeg -y -i "${ORIGINAL_VIDEO}" -i "${FG_VIDEO}" \
  -filter_complex "[0:v]setsar=1[left];[1:v]setsar=1[right];[left][right]hstack=inputs=2[out]" \
  -map "[out]" -c:v libx264 -pix_fmt yuv420p "${STACKED_VIDEO}"

echo "Side-by-side video saved to: ${STACKED_VIDEO}"