#!/bin/bash
#SBATCH --job-name=ev_lhm_fix
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Change to the directory where the script is located
cd /home/cizinsky/master-thesis
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg

# --- LHM eval on hi4d
#seq_name="hi4d_pair15_fight"
#num_persons=2
#exp_name="zero_shot_eval_with_difix"
#source_cam_id=4
#root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
#python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id wandb.enable=false 

#seq_name="hi4d_pair16_jump"
#num_persons=2
#exp_name="zero_shot_eval_with_difix"
#source_cam_id=4
#root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
#python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id wandb.enable=false 

seq_name="hi4d_pair17_dance"
num_persons=2
exp_name="zero_shot_eval_with_difix"
source_cam_id=28
root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id wandb.enable=false 

#seq_name="hi4d_pair19_piggyback"
#num_persons=2
#exp_name="zero_shot_eval_with_difix"
#source_cam_id=4
#root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
#python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id wandb.enable=false 

# --- LHM finetune
# seq_name="hi4d_pair19_piggyback"
# num_persons=2
# exp_name="v3"
# source_cam_id=4
# target_cam_ids="[4,28,76]"
# root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# python LHM/train_multi_humans.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id nvs_eval.target_camera_ids=$target_cam_ids eval_pretrain=true epochs=50 eval_every_epoch=25

seq_name="hi4d_pair17_dance"
num_persons=2
exp_name="difix_tune_all"
source_cam_id=28
# target_cam_ids="[4,16,28,40,52,64,76,88]"
# train_params="[shs]"
target_cam_ids="[4,28,52]"
root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
python LHM/train_multi_humans.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id nvs_eval.target_camera_ids=$target_cam_ids