#!/bin/bash
set -e # exit on error

# Visualiation debug
# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

# python training/smplx_debug.py --hi4d_scene_dir /scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17 --human3r_scene_dir /scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair17_dance --src_cam_id 28 --device cuda --selected-frame-idx 70



# SMPLX to SMPL conversion debug
scene_dir=/scratch/izar/cizinsky/thesis/input_data/test/hi4d_pair15_fight/v3_est_masks_gt_smplx_h6_rgb_smplx_band_and_apply_mask_to_renders
# - conversion
# bash /home/cizinsky/master-thesis/submodules/smplx/tools/run_smplx_to_smpl.sh  $scene_dir
# - check conversion
python playground/smplx_to_smpl_check.py --scene-dir $scene_dir --model-folder /home/cizinsky/body_models