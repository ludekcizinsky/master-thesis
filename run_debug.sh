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
#   shared.scene_name="${scene_name}" \
#   shared.exp_name="${exp_name}"

# Actual debug call
# scene_name=hi4d_pair16_jump
# exp_name=105_major_refactor_debug
# python training/simple_multi_human_trainer.py \
  # shared.scene_name=$scene_name \
  # shared.exp_name=$exp_name \
  # shared.wandb.enable=false \
  # train.nv_generation.trn_nv_gen.num_cameras=5 \
  # train.evaluation.pose_eval.eval_smpl=false \
  # train.optimization.epochs=10 \
  # train.evaluation.eval_pretrain=true \
  # train.evaluation.eval_every_epoch=5 \
  # train.pose_tuning.start_epoch=0 \
  # train.pose_tuning.end_epoch=4 \
  # train.gs_tuning.start_epoch=5 \
  # train.gs_tuning.end_epoch=9 \
  # train.nv_generation.nv_gen_epoch=5

# DiFix data-generation smoke run (single scene generate + aggregate)
# scene_name=hi4d_pair00_dance
# exp_name=v0_difix_data_smoke
# python training/simple_multi_human_trainer.py \
  # run_mode=difix_data_generation_generate \
  # shared.scene_name=$scene_name \
  # shared.exp_name=$exp_name \
  # difix_data_generation.frame_stride=10 \
  # difix_data_generation.max_samples_per_camera=20 \
  # difix_data_generation.target_cameras.mode=explicit \
  # difix_data_generation.target_cameras.explicit_ids=[16,28] \
  # difix_data_generation.filtering.min_mask_coverage=0.0

# python training/simple_multi_human_trainer.py \
  # run_mode=difix_data_generation_aggregate \
  # shared.scene_name=$scene_name \
  # shared.exp_name=$exp_name

scene_dir=/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair00_dance
python preprocess/vis/check_scene_in_3d.py \
  --scene-dir $scene_dir \
  --frame-idx-range 0 50 \
