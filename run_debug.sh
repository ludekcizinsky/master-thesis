#!/bin/bash
set -e # exit on error

# Visualiation debug
# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

# ------ Check scene preprocessed scene in 3D
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19

# estimated_scene_dir=/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair19_piggyback
# python preprocess/vis/helpers/check_scene_in_3d.py --scenes-dir $estimated_scene_dir --src_cam_id 4 --frame-idx-range 30 40


# ------ Visualise scene in 2D
# --- hi4d pair 15
# exp_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0030
# python evaluation/visualise_scene_in_2d.py \
  # --exp-eval-dir $exp_eval_dir \
  # --cam-id 4 \
  # --max-frames 90

# -- mmm piggybac
# exp_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair19_piggyback/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0030
# python evaluation/visualise_scene_in_2d.py \
  # --exp-eval-dir $exp_eval_dir \
  # --cam-id 4 \
  # --max-frames 140

# -- mmm lift
# exp_eval_dir=/scratch/izar/cizinsky/thesis/results/mmm_lift/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0030
# python evaluation/visualise_scene_in_2d.py \
  # --exp-eval-dir $exp_eval_dir \
  # --cam-id 0 \
  # --max-frames 60

# # -- taichi
# exp_eval_dir=/scratch/izar/cizinsky/thesis/results/taichi/evaluation/v605_first_tune_pose_then_3dgs/epoch_0000
# python evaluation/visualise_scene_in_2d.py \
  # --exp-eval-dir $exp_eval_dir \
  # --cam-id 0 \
  # --max-frames 5

# ------ Visualise the qualitattive output
# - hi4d pair15 fight
# scene_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0030
# python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --no-is-minus-y-up --source-camera-id 4 --frame-index 0

# - hi4d pair16 jump
# exp_name=v860_test_speed
# scene_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/evaluation/$exp_name/epoch_0000
# python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --no-is-minus-y-up --source-camera-id 4

# - hi4d pair17 dance
# scene_eval_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair17_dance/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0030
# python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --no-is-minus-y-up --source-camera-id 28 --frame-index 80

# - mmm dance
scene_eval_dir=/scratch/izar/cizinsky/thesis/results/mmm_dance/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0030
python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --source-camera-id 0 --frame-index 90

# - taichi
# scene_eval_dir=/scratch/izar/cizinsky/thesis/results/taichi/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0000
# python evaluation/visualise_scene_in_3d.py --eval-scene-dir $scene_eval_dir --frame-index 20


# ------ Visualise predicted and ground truth meshes for debug of 3d recon quality
# -- Hi4d pair15 fight
# # baseline experiment
# baseline_exp_name=v984_mc_for_3dgs_to_mesh
# pred_meshes_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/$baseline_exp_name/epoch_0030/aligned_posed_meshes_per_frame

# # compare experiment
# comp_exp_name=v975_A0b_baseline_combo_all_scenes_eval
# comp_pred_meshes_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/$comp_exp_name/epoch_0030/aligned_posed_meshes_per_frame

# # gt meshes
# gt_meshes_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15/meshes
# # run visualisation
# python playground/debug_vis_recon_eval.py \
  # --pred-aligned-meshes-dir $pred_meshes_dir \
  # --other_pred_aligned_meshes_dir $comp_pred_meshes_dir \
  # --gt-meshes-dir $gt_meshes_dir \
  # --frame-index 60


# -- mmm dance
# # baseline experiment
# baseline_exp_name=v984_mc_for_3dgs_to_mesh
# pred_meshes_dir=/scratch/izar/cizinsky/thesis/results/mmm_dance/evaluation/$baseline_exp_name/epoch_0000/aligned_posed_meshes_per_frame

# # compare experiment
# comp_exp_name=v975_A0b_baseline_combo_all_scenes_eval
# comp_meshes_dir=/scratch/izar/cizinsky/thesis/results/mmm_dance/evaluation/$comp_exp_name/epoch_0000/aligned_posed_meshes_per_frame

# # gt meshes
# gt_meshes_dir=/scratch/izar/cizinsky/ait_datasets/full/mmm/dance/meshes

# # run visualisation
# python playground/debug_vis_recon_eval.py \
  # --pred-aligned-meshes-dir $pred_meshes_dir \
  # --other_pred_aligned_meshes_dir $comp_meshes_dir \
  # --gt-meshes-dir $gt_meshes_dir \
  # --frame-index 50

# ----- Pose eval debug visualisation
# # -- Hi4d pair15 fight
# exp_name=v937_lrHigh
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15
# pred_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/$exp_name/epoch_0000
# comp_pred_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/$exp_name/epoch_0030

# python playground/debug_vis_pose_eval.py \
  # --gt-scene-dir $gt_scene_dir \
  # --no-is-minus-y-up \
  # --pred-scene-dir $pred_scene_dir \
  # --comp-pred-scene-dir $comp_pred_scene_dir

# hi4d pair16 jump
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair16/pair16/jump16
# pred_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/evaluation/v605_first_tune_pose_then_3dgs/epoch_0000
# comp_pred_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair16_jump/evaluation/v605_first_tune_pose_then_3dgs/epoch_0015
# python playground/debug_vis_pose_eval.py \
  # --gt-scene-dir $gt_scene_dir \
  # --no-is-minus-y-up \
  # --pred-scene-dir $pred_scene_dir \
  # --comp-pred-scene-dir $comp_pred_scene_dir

# hi4d pair19 piggyback
# gt_scene_dir=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair19_2/piggyback19
# pred_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair19_piggyback/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0000
# # comp_pred_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair19_piggyback/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0015
# python playground/debug_vis_pose_eval.py \
  # --gt-scene-dir $gt_scene_dir \
  # --no-is-minus-y-up \
  # --pred-scene-dir $pred_scene_dir \
  # --frame-index 140


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


# ------ Run mask refinement for a scene
# scene_name=taichi
# prompt_frame=0
# python playground/poc_sam3_prompting.py --scene-name $scene_name --prompt-frame $prompt_frame