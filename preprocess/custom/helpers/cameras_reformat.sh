#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

preprocessing_root=/scratch/izar/cizinsky/thesis/preprocessing

echo "Reformatting camera data..."

# Hi4D
echo "Reformatting Hi4D camera data..."
python preprocess/custom/helpers/cameras_reformat.py --scene-root-dir $preprocessing_root/hi4d_pair15_fight --src-cam-id 4 
python preprocess/custom/helpers/cameras_reformat.py --scene-root-dir $preprocessing_root/hi4d_pair16_jump --src-cam-id 4 --first-frame-number 11
python preprocess/custom/helpers/cameras_reformat.py --scene-root-dir $preprocessing_root/hi4d_pair17_dance --src-cam-id 28 
python preprocess/custom/helpers/cameras_reformat.py --scene-root-dir $preprocessing_root/hi4d_pair19_piggyback --src-cam-id 4 

# MMM
echo "Reformatting MMM camera data..."
python preprocess/custom/helpers/cameras_reformat.py --scene-root-dir $preprocessing_root/mmm_dance --src-cam-id 0 
python preprocess/custom/helpers/cameras_reformat.py --scene-root-dir $preprocessing_root/mmm_lift --src-cam-id 0
python preprocess/custom/helpers/cameras_reformat.py --scene-root-dir $preprocessing_root/mmm_walkdance --src-cam-id 0

# In-the-wild
echo "Reformatting In-the-wild camera data..."
python preprocess/custom/helpers/cameras_reformat.py --scene-root-dir $preprocessing_root/taichi --src-cam-id 0