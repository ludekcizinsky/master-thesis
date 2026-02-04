#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

repo_dir=/home/cizinsky/master-thesis
cd $repo_dir

preprocess_dir_path=/scratch/izar/cizinsky/thesis/v2_preprocessing
infer_scene_dir_script_path=preprocess/train/helpers/infer_initial_scene.py


# video_path=/scratch/izar/cizinsky/thesis/debug/sample_hi4d_pair15_fight
# video_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair15_1/pair15/fight15/images/4
# python $infer_scene_dir_script_path \
  # --video-path $video_path \
  # --seq-name hi4d_pair15_fight \
  # --cam-id 4 \
  # --output-dir $preprocess_dir_path \
  # --ref-frame-idx 0


video_path=/scratch/izar/cizinsky/thesis/in_the_wild/soccernet/frank_vs_luka.mp4
python $infer_scene_dir_script_path \
  --video-path $video_path \
  --seq-name luka_vs_frank \
  --cam-id 0 \
  --output-dir $preprocess_dir_path \
  --ref-frame-idx 63

