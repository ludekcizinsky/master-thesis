#!/bin/bash
#SBATCH --job-name=preprocess_part1
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# navigate to the directory containing the preprocessing script
cd /home/cizinsky/master-thesis

path_to_script=preprocess/custom/get_trn_data_part1.sh

# --- Hi4d
#seq_name=hi4d_pair15_fight
#src_camera_id=4
#gt_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
#bash $path_to_script $seq_name $src_camera_id $gt_root_dir

#seq_name=hi4d_pair16_jump
#src_camera_id=4
#gt_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
#bash $path_to_script $seq_name $src_camera_id $gt_root_dir

# seq_name=hi4d_pair17_dance
# src_camera_id=28
# gt_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# bash $path_to_script $seq_name $src_camera_id $gt_root_dir

# seq_name=hi4d_pair19_piggyback
# src_camera_id=4
# gt_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# bash $path_to_script $seq_name $src_camera_id $gt_root_dir

# --- MMM
#seq_name=mmm_dance
#src_camera_id=0
#gt_root_dir=/scratch/izar/cizinsky/ait_datasets/full/mmm/dance
#bash $path_to_script $seq_name $src_camera_id $gt_root_dir

# seq_name=mmm_lift
# src_camera_id=0
# gt_root_dir=/scratch/izar/cizinsky/ait_datasets/full/mmm/lift
# bash $path_to_script $seq_name $src_camera_id $gt_root_dir

# seq_name=mmm_walkdance
# src_camera_id=0
# gt_root_dir=/scratch/izar/cizinsky/ait_datasets/full/mmm/walkdance
# bash $path_to_script $seq_name $src_camera_id $gt_root_dir

# --- In the wild
#seq_name=taichi
#src_camera_id=0
#gt_root_dir=/scratch/izar/cizinsky/thesis/in_the_wild/scenes/taichi
#bash $path_to_script $seq_name $src_camera_id $gt_root_dir

seq_name=frank_vs_luka
src_camera_id=0
gt_root_dir=/scratch/izar/cizinsky/thesis/in_the_wild/scenes/frank_vs_luka
bash $path_to_script $seq_name $src_camera_id $gt_root_dir