#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

exp_name="v2_est_masks_gt_smplx_h2_smplx_mesh_based"
python /home/cizinsky/master-thesis/schedule.py --exp_name $exp_name --job_name_prefix $exp_name