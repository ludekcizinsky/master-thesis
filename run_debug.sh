#!/bin/bash
set -e # exit on error

# Visualiation debug
# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

estimated_scene_dir=/scratch/izar/cizinsky/thesis/v2_preprocessing/hi4d_pair19_piggyback
python preprocess/vis/check_scene_in_3d.py --scene-dir $estimated_scene_dir --src-cam-id 4 --frame-idx-range 100 150