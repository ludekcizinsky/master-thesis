#!/bin/bash
set -e # exit on error

# Visualiation debug
# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

# # Compare SMPLX from two different sources
# src_a_dir=/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair16_jump
# src_a_name="estimated"
# src_b_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# src_b_name="ground_truth"
# body_model_kind="smpl"
# python playground/different_sources_body_models_check.py --source_a_scene_dir $src_a_dir --source_a_name $src_a_name --source_b_scene_dir $src_b_dir --source_b_name $src_b_name --src_cam_id 4 --body_model_kind $body_model_kind


# # SMPLX to SMPL conversion debug
# scene_dir=/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair19_piggyback
# # - conversion
# bash /home/cizinsky/master-thesis/submodules/smplx/tools/run_conversion.sh $scene_dir smplx smpl
# # - check conversion
# python playground/body_model_conversion_check.py --scene-dir $scene_dir --model-folder /home/cizinsky/body_models --src-cam-id 4 --image-width 940 --image-height 1280

# # SMPL to SMPLX conversion debug
# scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# # - conversion
# bash /home/cizinsky/master-thesis/submodules/smplx/tools/run_conversion.sh $scene_dir smpl smplx
# # - check conversion
# python playground/body_model_conversion_check.py --scene-dir $scene_dir --model-folder /home/cizinsky/body_models --src-cam-id 4 --image-width 940 --image-height 1280


# # Reformat and align debug
# bash preprocess/custom/run_reformat_and_align.sh
# src_a_dir=/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair16_jump
# src_a_name="estimated"
# src_b_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# src_b_name="ground_truth"
# body_model_kind="smplx"
# python playground/different_sources_body_models_check.py --source_a_scene_dir $src_a_dir --source_a_name $src_a_name --source_b_scene_dir $src_b_dir --source_b_name $src_b_name --src_cam_id 4 --body_model_kind $body_model_kind


# 3DGS to mesh and visualisation debug
root_save_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/evaluation/v1_reconstruction_debug/epoch_0000
path_to_posed_3dgs=$root_save_dir/posed_3dgs_per_frame
path_to_posed_meshes=$root_save_dir/posed_meshes_per_frame
python playground/3dgs_to_mesh.py \
  --posed-3dgs-dir $path_to_posed_3dgs \
  --output-dir $path_to_posed_meshes \
  --grid-size 96 \
  --truncation 2.0 \
  --sigma-scale 0.7 \
  --max-frames 1000 --overwrite


path_to_gt_meshes=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16/seg
python playground/visualise_sequence_of_posed_3dgs.py --posed-3dgs-dir $path_to_posed_3dgs --port 8080 --scaling-mode auto --max-scale 0.02 --max-gaussians 80000 --posed-meshes-dir $path_to_posed_meshes --mesh-opacity 0.5 --gt-meshes-dir $path_to_gt_meshes --gt-mesh-opacity 0.5