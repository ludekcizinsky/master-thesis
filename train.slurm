#!/bin/bash
#SBATCH --job-name=modric_vs_ribberi
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

OUTPUT_ERROR="/scratch/izar/cizinsky/thesis/slurm/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"
OUTPUT_OUT="/scratch/izar/cizinsky/thesis/slurm/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Change to the directory where the script is located
cd /home/cizinsky/master-thesis

# Training
# python training/run.py scene_name=modric_vs_ribberi
# python training/run.py scene_name=modric_vs_ribberi mask_type=bbox
# python training/run.py scene_name=modric_vs_ribberi mask_type=bbox_crop
# python training/run.py scene_name=modric_vs_ribberi mask_type=tight_mask

# python training/run.py scene_name=modric_vs_ribberi mask_type=tight_mask ma_mode=nn
# python training/run.py scene_name=modric_vs_ribberi mask_type=tight_mask ma_free_radius=0.01 
# python training/run.py scene_name=modric_vs_ribberi mask_type=tight_mask ma_scale_aware=true 