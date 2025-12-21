#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

python /home/cizinsky/master-thesis/schedule.py --exp_name refactor_difix_v2 --job_name_prefix refactor_difix_v2
