#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# Baseline experiment however evaluated on all tasks and scenes
exp_name="v975_A0b_baseline_combo_all_scenes_eval"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name