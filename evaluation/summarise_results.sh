#!/bin/bash

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

exp_name="v1_pretrain_all_scenes_eval"
epoch_str="0000"

scene_names=(
  "hi4d_pair15_fight"
  "hi4d_pair16_jump"
  "hi4d_pair17_dance"
  "hi4d_pair19_piggyback"
  "mmm_dance"
  "mmm_lift"
  "mmm_walkdance"
)

src_cam_ids=(4 4 28 4 0 0 0)

python evaluation/helpers/summarise_benchmark_results.py \
  --exp-name "$exp_name" \
  --epoch-str "$epoch_str" \
  --scene-names "${scene_names[@]}" \
  --src-cam-ids "${src_cam_ids[@]}" \
  --results-root "/scratch/izar/cizinsky/thesis/results"
