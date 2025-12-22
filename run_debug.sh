#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

python training/smplx_debug.py --hi4d_scene_dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17 --human3r_scene_dir /scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance --src_cam_id 28 --device cuda