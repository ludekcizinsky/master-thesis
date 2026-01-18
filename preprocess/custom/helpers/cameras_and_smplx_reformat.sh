#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

preprocessing_root=/scratch/izar/cizinsky/thesis/preprocessing
script_path=preprocess/custom/helpers/cameras_and_smplx_reformat.py

# Hi4D
python $script_path --scene-root-dir $preprocessing_root/hi4d_pair15_fight --first-frame-number 1 --src-cam-id 4
python $script_path --scene-root-dir $preprocessing_root/hi4d_pair16_jump --first-frame-number 11 --src-cam-id 4
python $script_path --scene-root-dir $preprocessing_root/hi4d_pair17_dance --first-frame-number 1 --src-cam-id 28
python $script_path --scene-root-dir $preprocessing_root/hi4d_pair19_piggyback --first-frame-number 1 --src-cam-id 4

# # MMM
# python $script_path --scene-root-dir $preprocessing_root/mmm_dance --first-frame-number 1 --src-cam-id 0
# python $script_path --scene-root-dir $preprocessing_root/mmm_lift --first-frame-number 1 --src-cam-id 0
# python $script_path --scene-root-dir $preprocessing_root/mmm_walkdance --first-frame-number 1 --src-cam-id 0

# # In-the-wild
# python $script_path --scene-root-dir $preprocessing_root/taichi --first-frame-number 1 --src-cam-id 0