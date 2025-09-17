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


cd /home/cizinsky/master-thesis

OUTPUT_ERROR="/scratch/izar/cizinsky/thesis/slurm/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"
OUTPUT_OUT="/scratch/izar/cizinsky/thesis/slurm/${SLURM_JOB_NAME}.${SLURM_JOB_ID}.out"
echo "Error output will be written to ${OUTPUT_ERROR} and standard output to ${OUTPUT_OUT}"

bash train.sh