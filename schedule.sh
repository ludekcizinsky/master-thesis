#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

exp_name="v402_mc_mesh_extraction_with_better_hyperparams"
python /home/cizinsky/master-thesis/schedule.py --exp_name $exp_name --job_name_prefix $exp_name 
