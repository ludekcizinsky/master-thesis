#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

exp_name="v1_all_scenes_tune_after_hi4d_reprocess"
python /home/cizinsky/master-thesis/schedule.py --exp_name $exp_name --job_name_prefix $exp_name
