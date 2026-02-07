#!/bin/bash
set -e # exit on error

# Visualiation debug
# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

# navigate to project directory
cd /home/cizinsky/master-thesis

# debug
# estimated_scene_dir=/scratch/izar/cizinsky/thesis/debug/sample_hi4d_pair19_piggyback
# bash submodules/prompthmr/run_inference.sh $estimated_scene_dir
# python preprocess/vis/check_scene_in_3d.py --scene-dir $estimated_scene_dir --src-cam-id 4 --frame-idx-range 0 10

# estimated_scene_dir=/scratch/izar/cizinsky/thesis/preprocessing/hi4d_pair15_fight
# python preprocess/vis/check_scene_in_3d.py --scene-dir $estimated_scene_dir --src-cam-id 4 --frame-idx-range 100 150


eval_scene_dir=/scratch/izar/cizinsky/thesis/results/taichi/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0030
python evaluation/visualise_scene_in_3d.py \
  --eval-scene-dir $eval_scene_dir \
  --frame-index 30 \
  --source-camera-id 0
