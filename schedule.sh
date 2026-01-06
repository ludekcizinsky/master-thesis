#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

exp_name="v1_gt_masks_est_smplx_h1_use_h3r_no_smplx_tune_during_training_and_dont_apply_mask_to_nvs_renders"
python /home/cizinsky/master-thesis/schedule.py --exp_name $exp_name --job_name_prefix $exp_name