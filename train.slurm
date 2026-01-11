#!/bin/bash
#SBATCH --job-name=difix_v9_baseline
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Change to the directory where the script is located
cd /home/cizinsky/master-thesis
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg

# Shared across scenes (can be overridden via sbatch --export)
exp_name="${EXP_NAME:-difix_v8_baseline}"
num_persons="${NUM_PERSONS:-2}"
seq_name="${SEQ_NAME:-}"
source_cam_id="${SOURCE_CAM_ID:-}"
root_gt_dir_path="${ROOT_GT_DIR_PATH:-}"
target_cam_ids="${TARGET_CAMERA_IDS:-}"
test_masks_scene_dir="${TEST_MASKS_SCENE_DIR:-}"
test_smpl_params_scene_dir="${TEST_SMPL_PARAMS_SCENE_DIR:-}"
smpl_params_scene_dir="${SMPL_PARAMS_SCENE_DIR:-}"
test_smplx_params_scene_dir="${TEST_SMPLX_PARAMS_SCENE_DIR:-}"
cameras_scene_dir="${CAMERAS_SCENE_DIR:-}"

if [[ -z "$seq_name" || -z "$source_cam_id" || -z "$root_gt_dir_path" ]]; then
  echo "Missing required vars. Provide SEQ_NAME, SOURCE_CAM_ID, ROOT_GT_DIR_PATH via sbatch --export."
  exit 1
fi

extra_args=()
if [[ -n "$target_cam_ids" ]]; then
  if [[ "$target_cam_ids" == \[* ]]; then
    target_cam_ids_formatted="$target_cam_ids"
  else
    target_cam_ids_formatted="[${target_cam_ids//:/,}]"
  fi
  extra_args+=("nvs_eval.target_camera_ids=$target_cam_ids_formatted")
fi
if [[ -n "$test_masks_scene_dir" ]]; then
  extra_args+=("test_masks_scene_dir=$test_masks_scene_dir")
fi
if [[ -n "$test_smpl_params_scene_dir" ]]; then
  extra_args+=("test_smpl_params_scene_dir=$test_smpl_params_scene_dir")
fi
if [[ -n "$smpl_params_scene_dir" ]]; then
  extra_args+=("smpl_params_scene_dir=$smpl_params_scene_dir")
fi
if [[ -n "$test_smplx_params_scene_dir" ]]; then
  extra_args+=("test_smplx_params_scene_dir=$test_smplx_params_scene_dir")
fi
if [[ -n "$cameras_scene_dir" ]]; then
  extra_args+=("cameras_scene_dir=$cameras_scene_dir")
fi

python training/simple_multi_human_trainer.py \
  scene_name="$seq_name" \
  exp_name="$exp_name" \
  num_persons="$num_persons" \
  root_gt_dir_path="$root_gt_dir_path" \
  nvs_eval.source_camera_id="$source_cam_id" \
  "${extra_args[@]}"
