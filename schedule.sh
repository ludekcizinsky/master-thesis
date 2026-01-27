#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

exp_name="v708_pose_eval_refactor_use_genders_for_gt_smpl"
python /home/cizinsky/master-thesis/schedule.py --exp_name $exp_name --job_name_prefix $exp_name --scene_name_includes "hi4d"
