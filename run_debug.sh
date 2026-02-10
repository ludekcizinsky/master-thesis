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

# estimated_scene_dir=/scratch/izar/cizinsky/thesis/v2_preprocessing/wild_ilia_malinin
# python preprocess/vis/check_scene_in_3d.py --scene-dir $estimated_scene_dir --src-cam-id 0 --frame-idx-range 40 41 --vis-3dgs

eval_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0030
python evaluation/visualise_scene_in_3d.py \
  --eval-scene-dir $eval_scene_dir \
  --frame-index 96 \
  --source-camera-id 4 \
  --no-is-minus-y-up

# codex, add your call here:
# Remote usage:
# 1) run this script on the server
# 2) from your local machine:
#    ssh -L 9876:127.0.0.1:9876 -L 9877:127.0.0.1:9877 <user>@<server>
# 3) open:
#    http://localhost:9876/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9877%2Fproxy
# If needed once: pip install rerun-sdk
# The script will reclaim (terminate) processes already listening on 9876/9877.
# eval_scene_dir=/scratch/izar/cizinsky/thesis/results/hi4d_pair15_fight/evaluation/v975_A0b_baseline_combo_all_scenes_eval/epoch_0000
# # Points3D + gsplat single-frame profile (VistaDream-style)
# python evaluation/rerun_dynamic_3dgs.py \
  # --eval-scene-dir $eval_scene_dir \
  # --frame-idx-range 0 150 \
  # --subsample-rate 1 \
  # --src-cam-id 4 \
  # --splat-primitive points \
  # --include-rgb \
  # --include-gsplat-render \
  # --min-opacity 0.02 \
  # --max-gaussians-per-person 120000 \
  # --point-radius-scale 0.06 \
  # --max-point-radius 0.12 \
  # --gsplat-max-side 960 \
  # --gsplat-rasterize-mode antialiased \
  # --gsplat-background black
