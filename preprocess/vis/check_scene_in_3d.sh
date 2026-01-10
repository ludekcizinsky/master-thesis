#!/bin/bash

# configuration from user
seq_name=$1  # e.g., hi4d_pair15_fight
camera_id=$2  # e.g., 4
show_3dgs=$3  # optional, default: false

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
vis_3dgs_flag=""
if [[ "$show_3dgs" == "true" || "$show_3dgs" == "1" || "$show_3dgs" == "yes" ]]; then
    vis_3dgs_flag="--vis-3dgs"
fi

python preprocess/vis/helpers/check_scene_in_3d.py --scenes-dir $scenes_dir --src_cam_id $camera_id $vis_3dgs_flag
