#!/bin/bash
#SBATCH --job-name=difix_v8_baseline
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

# Shared across scenes
exp_name="difix_v8_baseline"
num_persons=2

# --- LHM finetune
seq_name="hi4d_pair15_fight"
source_cam_id=4
root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id 

#seq_name="hi4d_pair16_jump"
#source_cam_id=4
#root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
#python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id

#seq_name="hi4d_pair17_dance"
#source_cam_id=28
#root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
#python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id 

#seq_name="hi4d_pair19_piggyback"
#source_cam_id=4
#root_gt_dir_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
#python training/simple_multi_human_trainer.py scene_name=$seq_name exp_name=$exp_name num_persons=$num_persons nvs_eval.root_gt_dir_path=$root_gt_dir_path nvs_eval.source_camera_id=$source_cam_id
