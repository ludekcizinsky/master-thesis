#!/bin/bash
set -e # exit on error

# Visualiation debug
# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis


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


# ------ Visualise the qualitattive output
# - hi4d pair15 fight
# scene_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/v605_first_tune_pose_then_3dgs/epoch_0000
# python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --no-is-minus-y-up --source-camera-id 4 --frame-index 0

# - hi4d pair16 jump
# bash train.sh
# scene_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/evaluation/v600_debugging_recon_v3/epoch_0000
# python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --no-is-minus-y-up --source-camera-id 4

# - taichi
# scene_eval_dir=/scratch/izar/cizinsky/thesis/results/taichi/evaluation/v605_first_tune_pose_then_3dgs/epoch_0000
# python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --frame-index 0


# ------ Visualise predicted and ground truth meshes for debug of 3d recon quality
# # -- Hi4d pair15 fight
# # baseline experiment
# baseline_exp_name=v402_tsdf_mesh_extraction
# pred_meshes_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/$baseline_exp_name/epoch_0000/posed_meshes_per_frame
# # comparison experiment
# comp_exp_name=v402_tsdf_mesh_extraction 
# comp_meshes_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/$comp_exp_name/epoch_0015/aligned_posed_meshes_per_frame
# # gt meshes
# gt_meshes_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15/meshes
# # run visualisation
# python playground/debug_vis_recon_eval.py \
  # --pred-aligned-meshes-dir $pred_meshes_dir \
  # --gt-meshes-dir $gt_meshes_dir \
  # --frame-index 0 \
  # --other-pred-aligned-meshes-dir $comp_meshes_dir


# -- mmm dance
# # baseline experiment
# baseline_exp_name=v402_mc_mesh_extraction_with_better_hyperparams
# pred_meshes_dir=/scratch/izar/cizinsky/thesis/results/mmm_dance/evaluation/$baseline_exp_name/epoch_0015/aligned_posed_meshes_per_frame
# # comparison experiment
# comp_exp_name=v402_tsdf_mesh_extraction 
# comp_meshes_dir=/scratch/izar/cizinsky/thesis/results/mmm_dance/evaluation/$comp_exp_name/epoch_0015/aligned_posed_meshes_per_frame
# # gt meshes
# gt_meshes_dir=/scratch/izar/cizinsky/ait_datasets/full/mmm/dance/meshes
# # run visualisation
# python playground/debug_vis_recon_eval.py \
  # --pred-aligned-meshes-dir $pred_meshes_dir \
  # --other-pred-aligned-meshes-dir $comp_meshes_dir \
  # --gt-meshes-dir $gt_meshes_dir \
  # --frame-index 0

# ----- Pose eval debug visualisation
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# pred_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/evaluation/v602_pose_est_refactor_loading/epoch_0000
# comp_pred_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/evaluation/v602_pose_est_refactor_loading/epoch_0015
# python playground/debug_vis_pose_eval.py \
  # --gt-scene-dir $gt_scene_dir \
  # --pred-scene-dir $pred_scene_dir \
  # --comp-pred-scene-dir $comp_pred_scene_dir \
  # --no-is-minus-y-up


# ------ Debug pose conversion
# gt_scene_dir=/scratch/izar/cizinsky/thesis/debug/jump16_new_conversion
# bash submodules/smplx/tools/run_conversion.sh $gt_scene_dir smpl smplx

# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# python playground/debug_pose_conversion.py \
  # --scene-dir $gt_scene_dir \
  # --no-is-minus-y-up \
  # --frame-idx-range 10 40


# ------ Check scene preprocessed scene in 3D
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19

python preprocess/vis/helpers/check_scene_in_3d.py --scenes-dir $gt_scene_dir --src_cam_id 4 --frame-idx-range 30 40