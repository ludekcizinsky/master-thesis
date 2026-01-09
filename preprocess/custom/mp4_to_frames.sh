#!/bin/bash

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

scenes_dir=/scratch/izar/cizinsky/thesis/in_the_wild/scenes

# bilibili
path_to_bilibili_dir=/scratch/izar/cizinsky/thesis/in_the_wild/bilibili

seq_name=taichi
python preprocess/custom/helpers/mp4_to_frames.py --input-video-path $path_to_bilibili_dir/$seq_name.mp4 --scenes-dir $scenes_dir --seq-name $seq_name
