#!/bin/bash

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

exp_names_list=(
  "difix_v1_baseline"
  "difix_v2_baseline"
)
camera_id="76"
frame_idx_str="000011"
epoch_str="0150"
scene_name="hi4d_pair16_jump"

python scripts/compare_nvs_renders.py \
  --exp-names "${exp_names_list[@]}" \
  --camera-id "$camera_id" \
  --frame-idx-str "$frame_idx_str" \
  --epoch-str "$epoch_str" \
  --scene-name "$scene_name" \
