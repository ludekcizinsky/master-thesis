#!/bin/bash

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

exp_eval_dir=$1
# eg. /scratch/izar/cizinsky/thesis/results/mmm_lift/evaluation/v1_all_scenes_tune_after_hi4d_reprocess/epoch_0015

inputs_eval_dir=$2
# eg. /scratch/izar/cizinsky/thesis/input_data/test/mmm_lift/v1_all_scenes_tune_after_hi4d_reprocess

cam_id=$3 # e.g. 0

max_frames=${4:-10} # default 10

python evaluation/helpers/visualise_meshes_in_2d.py \
  --exp-eval-dir $exp_eval_dir \
  --inputs-eval-dir $inputs_eval_dir \
  --cam-id $cam_id \
  --max-frames $max_frames