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


# # 3DGS to mesh and visualisation debug
# root_save_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair19_piggyback/evaluation/v3_est_masks_est_smplx_test_recon_eval/epoch_0015
# path_to_posed_3dgs=$root_save_dir/posed_3dgs_per_frame
# path_to_posed_meshes=$root_save_dir/posed_meshes_per_frame

# # # bash train.sh
# # # # python playground/3dgs_to_mesh.py \
  # # # # --posed-3dgs-dir $path_to_posed_3dgs \
  # # # # --output-dir $path_to_posed_meshes \
  # # # # --grid-size 96 \
  # # # # --truncation 2.0 \
  # # # # --sigma-scale 0.7 \
  # # # # --max-frames 1000 --overwrite

# path_to_gt_meshes=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19/seg
# python playground/visualise_sequence_of_posed_3dgs.py --posed-3dgs-dir $path_to_posed_3dgs --port 8080 --max-scale 0.02 --max-gaussians 80000 --posed-meshes-dir $path_to_posed_meshes --mesh-opacity 0.5 --gt-meshes-dir $path_to_gt_meshes --gt-mesh-opacity 0.5

# # Visualise MMM mesh
# mesh_data_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15/meshes
# python playground/visualise_mmm_mesh.py --mmm-data-dir $mesh_data_dir --port 8080 

# Running preprocess for MMM

# # 1. reformat
# bash preprocess/ait_datasets/mmm/other_reformat.sh

# # 2. compute additional data (e.g., smplx params, masks etc.)
# seq_name=taichi
# src_camera_id=0
# gt_root_dir=/scratch/izar/cizinsky/thesis/in_the_wild/scenes/taichi
# bash preprocess.sh $seq_name $src_camera_id $gt_root_dir

# # 3. check h3r output
# preprocess_dir=/scratch/izar/cizinsky/thesis/preprocessing/frank_vs_luka
# python preprocess/custom/check_h3r_output.py --scene-dir $preprocess_dir --model-folder /home/cizinsky/body_models


# # Checking quality of preprocessing
# # 1. check rendering
# bash preprocess/vis/check_rendering.sh hi4d_pair15_fight 4 
# bash preprocess/vis/check_rendering.sh mmm_lift 0 
# bash preprocess/vis/check_rendering.sh mmm_dance 0 
# bash preprocess/vis/check_rendering.sh taichi 0 

# # 2. check in 3D
# bash preprocess/vis/check_scene_in_3d.sh hi4d_pair15_fight 4 false
# bash preprocess/vis/check_scene_in_3d.sh taichi 0 false
# bash preprocess/vis/check_scene_in_3d.sh mmm_dance 0 false
# bash preprocess/vis/check_scene_in_3d.sh mmm_lift 0 false
# bash preprocess/vis/check_scene_in_3d.sh mmm_walkdance 0 false

# src_cam_id=4
# scenes_dir="/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15"
# python preprocess/vis/helpers/check_scene_in_3d.py --scenes-dir $scenes_dir --src_cam_id $src_cam_id

# ------- Check inputs to the reconstur tion evaluation script
# root_pred_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/evaluation/v3_large_refactor_check/epoch_0000
# path_to_posed_3dgs=$root_pred_dir/posed_3dgs_per_frame
# path_to_posed_meshes=$root_pred_dir/posed_meshes_per_frame
# path_to_gt_meshes=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16/meshes
# python evaluation/visualise_sequence_of_posed_3dgs.py --posed-3dgs-dir $path_to_posed_3dgs --port 8080 --max-scale 0.02 --max-gaussians 80000 --posed-meshes-dir $path_to_posed_meshes --mesh-opacity 0.5 --gt-meshes-dir $path_to_gt_meshes --gt-mesh-opacity 0.5

# root_pred_dir=/scratch/izar/cizinsky/thesis/results/mmm_dance/evaluation/v3_large_refactor_check/epoch_0000
# path_to_posed_3dgs=$root_pred_dir/posed_3dgs_per_frame
# path_to_posed_meshes=$root_pred_dir/aligned_posed_meshes_per_frame
# path_to_gt_meshes=/scratch/izar/cizinsky/ait_datasets/full/mmm/dance/meshes
# python evaluation/visualise_sequence_of_posed_3dgs.py --posed-3dgs-dir $path_to_posed_3dgs --port 8080 --max-scale 0.02 --max-gaussians 80000 --posed-meshes-dir $path_to_posed_meshes --mesh-opacity 0.5 --gt-meshes-dir $path_to_gt_meshes --gt-mesh-opacity 0.5

# exp_name=v301_tsdf_mesh
# root_pred_dir=/scratch/izar/cizinsky/thesis/results/mmm_lift/evaluation/$exp_name/epoch_0000
# path_to_posed_3dgs=$root_pred_dir/posed_3dgs_per_frame
# path_to_posed_meshes=$root_pred_dir/aligned_posed_meshes_per_frame
# path_to_gt_meshes=/scratch/izar/cizinsky/ait_datasets/full/mmm/lift/meshes
# python evaluation/visualise_sequence_of_posed_3dgs.py --posed-3dgs-dir $path_to_posed_3dgs --port 8080 --max-scale 0.02 --max-gaussians 80000 --posed-meshes-dir $path_to_posed_meshes --mesh-opacity 0.5 --gt-meshes-dir $path_to_gt_meshes --gt-mesh-opacity 0.5

# root_pred_dir=/scratch/izar/cizinsky/thesis/results/mmm_walkdance/evaluation/v3_large_refactor_check/epoch_0000
# path_to_posed_3dgs=$root_pred_dir/posed_3dgs_per_frame
# path_to_posed_meshes=$root_pred_dir/aligned_posed_meshes_per_frame
# path_to_gt_meshes=/scratch/izar/cizinsky/ait_datasets/full/mmm/walkdance/meshes
# python evaluation/visualise_sequence_of_posed_3dgs.py --posed-3dgs-dir $path_to_posed_3dgs --port 8080 --max-scale 0.02 --max-gaussians 80000 --posed-meshes-dir $path_to_posed_meshes --mesh-opacity 0.5 --gt-meshes-dir $path_to_gt_meshes --gt-mesh-opacity 0.5



# ------ Visualise normal maps of posed meshes in 2D

# -- Hi4d pair17 dance
# exp_name=v202_white_bg
# exp_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair17_dance/evaluation/$exp_name/epoch_0000
# inputs_eval_dir=/scratch/izar/cizinsky/thesis/input_data/test/hi4d_pair17_dance/$exp_name
# cam_id=28
# max_frames=10
# bash evaluation/visualise_meshes_in_2d.sh $exp_eval_dir $inputs_eval_dir $cam_id $max_frames

# -- Hi4d pair19 piggyback
# exp_name=v203_core_ablation_set_a4_use_lhm_with_view_densification_and_difix
# exp_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair19_piggyback/evaluation/$exp_name/epoch_0015
# inputs_eval_dir=/scratch/izar/cizinsky/thesis/input_data/test/hi4d_pair19_piggyback/$exp_name
# cam_id=4
# max_frames=200
# bash evaluation/visualise_meshes_in_2d.sh $exp_eval_dir $inputs_eval_dir $cam_id $max_frames


# -- MMM lift
# exp_name=v203_core_ablation_set_a4_use_lhm_with_view_densification_and_difix
# exp_eval_dir=/scratch/izar/cizinsky/thesis/results/mmm_lift/evaluation/$exp_name/epoch_0015
# inputs_eval_dir=/scratch/izar/cizinsky/thesis/input_data/test/mmm_lift/$exp_name
# cam_id=0
# max_frames=200
# bash evaluation/visualise_meshes_in_2d.sh $exp_eval_dir $inputs_eval_dir $cam_id $max_frames

# ------ Visualise in the wild posed 3dgs
# scene_eval_dir="/scratch/izar/cizinsky/thesis/results/taichi/evaluation/v100_initial_run/epoch_0000"
# python evaluation/visualise_posed_3dgs.py --posed-3dgs-dir $scene_eval_dir/posed_3dgs_per_frame --port 8080 --max-scale 0.5 --max-gaussians 200000

# ------ Visualise in the wild posed meshes
# scene_eval_dir=/scratch/izar/cizinsky/thesis/results/taichi/evaluation/v101_improved_version/epoch_0000/posed_meshes_per_frame
# python evaluation/visualise_meshes_in_3d.py --mesh-dir=$scene_eval_dir


# ------ Visualise the eval output of in the wild scene in 3d
bash train.sh
scene_eval_dir=/scratch/izar/cizinsky/thesis/results/taichi/evaluation/v303_tsdf_mesh/epoch_0000
python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --port 8080 --max-scale 0.5 --max-gaussians 200000

# python playground/debug_tsdf.py \
  # --tsdf-debug-dir /scratch/izar/cizinsky/thesis/results/taichi/evaluation/v303_tsdf_mesh/epoch_0000/debug/tsdf_ply \
  # --person-id 1 \
  # --frame 1
