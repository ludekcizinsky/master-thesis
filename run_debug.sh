#!/bin/bash
set -e # exit on error

# Visualiation debug
# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

# Training template (Hydra overrides)
# Uncomment and edit paths/values as needed.
#
# scene_name="hi4d_pair15_fight"
# exp_name="debug_template"
#
# python training/simple_multi_human_trainer.py \
#   scene_name="${scene_name}" \
#   exp_name="${exp_name}"

# Actual debug call
scene_name=hi4d_pair16_jump
exp_name=102_major_refactor_debug
python training/simple_multi_human_trainer.py scene_name=$scene_name exp_name=$exp_name wandb.enable=false

# eval_scene_dir=/scratch/izar/cizinsky/thesis/results/$scene_name/$exp_name/evaluation/epoch_0000
# python evaluation/visualise_scene_in_3d.py \
  # --eval-scene-dir $eval_scene_dir \
  # --frame-index 0 \
  # --source-camera-id 4 \
  # --no-is-minus-y-up
