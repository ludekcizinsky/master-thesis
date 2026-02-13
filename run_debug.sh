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
# scene_name=hi4d_pair16_jump
# exp_name=105_major_refactor_debug
# python training/simple_multi_human_trainer.py \
  # scene_name=$scene_name \
  # exp_name=$exp_name \
  # wandb.enable=false \
  # trn_nv_gen.num_cameras=5 \
  # pose_eval.eval_smpl=false \
  # epochs=10 \
  # eval_pretrain=true \
  # eval_every_epoch=5 \
  # pose_tuning.start_epoch=0 \
  # pose_tuning.end_epoch=4 \
  # gs_tuning.start_epoch=5 \
  # gs_tuning.end_epoch=9 \
  # nv_gen_epoch=5


scene_dir=/scratch/izar/cizinsky/thesis/gt_scene_data/hi4d_pair00_dance
python preprocess/vis/check_scene_in_3d.py \
  --scene-dir $scene_dir \
  --frame-idx-range 0 50 \
