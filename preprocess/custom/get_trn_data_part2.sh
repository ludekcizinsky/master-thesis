#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/master-thesis

# configurable settings
seq_name=$1
gt_root_dir=$2 # currently expects hi4d format root dir

# for now default settings
lhm_input_frame_idx=0

echo "--- Running inference.sh to obtain canonical 3dgs models for each human"
bash submodules/lhm/run_inference.sh $seq_name $lhm_input_frame_idx $gt_root_dir