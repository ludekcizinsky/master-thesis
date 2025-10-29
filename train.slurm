#!/bin/bash
#SBATCH --job-name=v2_runs
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Change to the directory where the script is located
cd /home/cizinsky/master-thesis

# Full scene (default setup)
GROUP_NAME=v2_default
# - dynamic
# python training/run.py scene_name=football_high_res tids=[0,1] train_bg=false resume=false logger.tags=[v2] group_name=$GROUP_NAME
# - static
# python training/run.py scene_name=football_high_res tids=[] train_bg=true resume=true logger.tags=[v2] group_name=$GROUP_NAME
# - dynamic+static
# python training/run.py scene_name=football_high_res tids=[0,1] train_bg=true resume=true logger.tags=[v2] group_name=$GROUP_NAME gs_refine_start_iter=2000 gs_refine_stop_iter=4000

python training/run.py scene_name=football_high_res tids=[0,1] train_bg=false resume=true logger.tags=[v2] group_name=$GROUP_NAME debug=true save_freq=30000
