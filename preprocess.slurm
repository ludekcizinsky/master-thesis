#!/bin/bash
#SBATCH --job-name=lhm
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

seq_name=hi4d_pair17_dance
src_camera_id=28
gt_root_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
bash preprocess.sh $seq_name $src_camera_id $gt_root_dir