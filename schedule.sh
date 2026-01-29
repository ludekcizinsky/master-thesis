#!/bin/bash

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# Baseline using TSDF to get meshes
exp_name="v975_A0b_baseline_combo_all_scenes_eval"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --overrides gs_to_mesh_method=tsdf


exp_name="v984_mc_for_3dgs_to_mesh"
python /home/cizinsky/master-thesis/schedule.py \
    --exp_name $exp_name \
    --job_name_prefix $exp_name \
    --overrides gs_to_mesh_method=mc
