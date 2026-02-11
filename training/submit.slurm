#!/bin/bash
#SBATCH --job-name=train_scene
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%A_%a.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=master

cd /home/cizinsky/master-thesis
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg

python training/run.py "$@"
