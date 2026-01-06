#!/bin/bash

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

exp_name="v1_est_masks_est_smplx_h1_use_rgb_render_based_masks_and_h3r_smplx_no_tune"
epoch_str="0150"

scene_names=(
  "hi4d_pair15_fight"
  "hi4d_pair16_jump"
  "hi4d_pair17_dance"
  "hi4d_pair19_piggyback"
)


python scripts/summarise_benchmark_results.py \
  --exp-name "$exp_name" \
  --epoch-str "$epoch_str" \
  --scene-names "${scene_names[@]}" \
  --results-root "/scratch/izar/cizinsky/thesis/results"
