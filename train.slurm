#!/bin/bash
#SBATCH --job-name=v1_runs
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=05:00:00
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
# - dynamic
python training/run.py scene_name=football_high_res tids=[0] train_bg=false resume=false logger.tags=[v1] group_name=v1_default
python training/run.py scene_name=football_high_res tids=[1] train_bg=false resume=false logger.tags=[v1] group_name=v1_default
# - static
python training/run.py scene_name=football_high_res tids=[] train_bg=true resume=false logger.tags=[v1] opacity_reg=0.001 scale_reg=0.01 cloud_downsample=10 gs_refine_every=100 gs_refine_stop_iter=15000 group_name=v1_default

# - joint  dynamic+static
python training/run.py scene_name=football_high_res tids=[0,1] train_bg=true resume=true logger.tags=[v1] opacity_reg=0.001 scale_reg=0.01 cloud_downsample=10 gs_refine_every=500 gs_refine_start_iter=5000 gs_refine_stop_iter=7000 group_name=v1_default