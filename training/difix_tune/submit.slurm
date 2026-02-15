#!/bin/bash
#SBATCH --job-name=difix_tune
#SBATCH --output=/scratch/izar/cizinsky/thesis/misc/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/misc/slurm/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=master

cd /home/cizinsky/master-thesis
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate "${THESIS_CONDA_ENV:-thesis}"
module load gcc ffmpeg

python training/difix_tune/run.py "$@"
