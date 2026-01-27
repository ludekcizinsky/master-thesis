#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

preprocessing_root=/scratch/izar/cizinsky/thesis/preprocessing
script_path=preprocess/custom/helpers/rename_and_copy.py

echo "Starting rename_and_copy.sh (images, masks and depths)"

# Hi4D
echo "Processing Hi4D scenes..."
python $script_path --scene-root-dir $preprocessing_root/hi4d_pair15_fight --src-cam-id 4 --first-frame-number 1 
python $script_path --scene-root-dir $preprocessing_root/hi4d_pair16_jump --src-cam-id 4 --first-frame-number 11 
python $script_path --scene-root-dir $preprocessing_root/hi4d_pair17_dance --src-cam-id 28 --first-frame-number 1 
python $script_path --scene-root-dir $preprocessing_root/hi4d_pair19_piggyback --src-cam-id 4 --first-frame-number 1 

# MMM
echo "Processing MMM scenes..."
python $script_path --scene-root-dir $preprocessing_root/mmm_dance --src-cam-id 0 --first-frame-number 1 
python $script_path --scene-root-dir $preprocessing_root/mmm_lift --src-cam-id 0 --first-frame-number 1
python $script_path --scene-root-dir $preprocessing_root/mmm_walkdance --src-cam-id 0 --first-frame-number 1

# In-the-wild
echo "Processing In-the-wild scenes..."
python $script_path --scene-root-dir $preprocessing_root/taichi --src-cam-id 0 --first-frame-number 1

echo "Finished rename_and_copy.sh"