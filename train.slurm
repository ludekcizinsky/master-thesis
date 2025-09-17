#!/bin/bash
#SBATCH --job-name=3dgs_params
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=08:00:00
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

# Refinement frequency 100
python training/run.py scene_name=modric_vs_ribberi gs_refine_every=100 'logger.tags=[h1, refine100]'
python training/run.py scene_name=modric_vs_ribberi gs_refine_every=200 'logger.tags=[h1, refine200]'
python training/run.py scene_name=modric_vs_ribberi gs_refine_every=300 'logger.tags=[h1, refine300]'

# Refinement stop iteration 5000
python training/run.py scene_name=modric_vs_ribberi gs_refine_stop_iter=5000 'logger.tags=[h1, refstop5k]'
python training/run.py scene_name=modric_vs_ribberi gs_refine_stop_iter=10000 'logger.tags=[h1, refstop10k]'
python training/run.py scene_name=modric_vs_ribberi gs_refine_stop_iter=15000 'logger.tags=[h1, refstop15k]'