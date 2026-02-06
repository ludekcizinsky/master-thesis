#!/bin/bash
set -e # exit on error

# Visualiation debug
# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

# debug

estimated_scene_dir=/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair19_piggyback
# estimated_scene_dir=/scratch/izar/cizinsky/thesis/debug/sample_hi4d_pair19_piggyback
# bash submodules/prompthmr/run_inference.sh $estimated_scene_dir
python preprocess/vis/check_scene_in_3d.py --scene-dir $estimated_scene_dir --src-cam-id 4 --frame-idx-range 50 100
