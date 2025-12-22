#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

seq_name="hi4d_pair17_dance"
num_persons=2
exp_name="est_smplx_dev"
source_cam_id=28
target_cam_ids="[4,28,52]"
root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=false