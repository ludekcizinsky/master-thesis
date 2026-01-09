#!/bin/bash

# configuration from user
seq_name=$1  # e.g., hi4d_pair15_fight
camera_id=$2  # e.g., 0

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to master thesis dir
cd /home/cizinsky/master-thesis

# set paths
preprocess_dir=/scratch/izar/cizinsky/thesis/preprocessing

# derive scene dir
scenes_dir=$preprocess_dir/$seq_name

# run rendering check script
python preprocess/vis/helpers/check_rendering.py --scenes-dir $scenes_dir --camera_id $camera_id
