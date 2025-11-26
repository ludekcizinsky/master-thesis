#!/bin/bash
# This script runs a playground environment for testing and development.

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/master-thesis/playground

# prepare data for LHM
input_mask_dir="/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/sam2_masks"
frame_folder="/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/image"
output_dir="/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/lhm_input"
python prepare_lhm_data.py \
    --input_mask_dir $input_mask_dir \
    --input_img_dir $frame_folder \
    --output_dir $output_dir \
    --threshold 0.5

# frame_folder="/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/image"
# output_video="/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/taichi_10fps.mp4"
# ffmpeg -framerate 10 -start_number 0 -i "$frame_folder/%04d.png" \
  # -c:v libx264 -pix_fmt yuv420p -crf 18 $output_video
