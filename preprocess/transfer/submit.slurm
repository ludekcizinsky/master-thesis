#!/bin/bash
#SBATCH --job-name=preprocess_transfer
#SBATCH --output=/scratch/izar/cizinsky/thesis/misc/slurm/%x.%A_%a.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/misc/slurm/%x.%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=master

cd /home/cizinsky/master-thesis
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

python preprocess/transfer/run.py "$@"
