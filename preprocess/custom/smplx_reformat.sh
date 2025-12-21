#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

preprocessing_root=/scratch/izar/cizinsky/thesis/preprocessing
# Hi4D
python preprocess/custom/smplx_reformat.py --scene-root-dir $preprocessing_root/hi4d_pair15_fight --first-frame-number 1 
