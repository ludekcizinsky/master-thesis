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

if [[ -z "$seq_name" || -z "$source_cam_id" || -z "$root_gt_dir_path" ]]; then
  echo "Missing required vars. Provide SEQ_NAME, SOURCE_CAM_ID, ROOT_GT_DIR_PATH via sbatch --export."
  exit 1
fi

python training/new_simple_multi_human_trainer.py \
  scene_name="$seq_name" \
  exp_name="$exp_name" \
  num_persons="$num_persons" \
  root_gt_dir_path="$root_gt_dir_path" \
  nvs_eval.source_camera_id="$source_cam_id"
