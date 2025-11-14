#!/bin/bash
#SBATCH --job-name=prep_hi4d
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
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


# paths=(
    # "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00/images/76"
    # "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01/images/76"
    # "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15/images/4"
    # "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16/images/4"
    # "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17/images/28"
    # "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19/images/4"
# )

# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair00_dance00_cam76 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00/images/76
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair01_hug01_cam76 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair01/pair01/hug01/images/76
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair15_fight15_cam4 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15/images/4
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair16_jump16_cam4 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16/images/4
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair17_dance17_cam28 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17/images/28
# bash preprocess/multiply_pipeline/run_preprocessing_multiply.sh --seq hi4d_pair19_piggyback19_cam4 --images-dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19/images/4

bash compute_align_transformations.sh
