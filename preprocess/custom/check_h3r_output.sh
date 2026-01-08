#!/bin/bash

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

preprocess_dir=$1

python preprocess/custom/check_h3r_output.py --scene-dir $preprocess_dir --model-folder /home/cizinsky/body_models