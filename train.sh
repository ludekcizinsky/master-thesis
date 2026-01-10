#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

# ------- Hi4d
# seq_name="hi4d_pair15_fight"
# num_persons=2
# exp_name="v2_of_pose_est_metrics"
# source_cam_id=4
# target_cam_ids="[76,4,28]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=true difix.trn_enable=false

seq_name="hi4d_pair16_jump"
num_persons=2
exp_name="v1_large_refactor_check"
source_cam_id=4
target_cam_ids="[76,4,28]"
root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=true difix.trn_enable=false

#seq_name="hi4d_pair17_dance"
#num_persons=2
#exp_name="est_smplx_dev"
#source_cam_id=28
#target_cam_ids="[4,28,52]"
#root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
#python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=false


# ------- MMM
# seq_name="mmm_dance"
# num_persons=4
# exp_name="v1_trying_to_run_mmm"
# source_cam_id=0
# target_cam_ids="[]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/mmm/dance
# python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id  nvs_eval.target_camera_ids=$target_cam_ids  wandb.enable=false eval_pretrain=false difix.trn_enable=false
