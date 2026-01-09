#!/bin/bash
set -e # exit on error

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/master-thesis

# configurable settings
seq_name=$1
src_camera_id=$2 
gt_root_dir=$3 # currently expects hi4d format root dir

# for now default settings
sam_frame_idx=60

# derived paths
input_frames_path=$gt_root_dir/images/$src_camera_id
preprocess_dir=/scratch/izar/cizinsky/thesis/preprocessing/$seq_name
mkdir -p $preprocess_dir

echo "--- [1/4] Preparing frame folder"
frame_folder=$preprocess_dir/frames
mkdir -p $frame_folder
cp -r $input_frames_path/* $frame_folder/

echo "--- [2/4] Running Human3R to obtain: smplx parameters and camera poses"
bash submodules/human3r/run_inference.sh $seq_name
 
echo "--- [3/4] Running Depth Anything 3 to generate depth maps"
bash submodules/da3/run_inference.sh $seq_name

echo "--- [4/4] Running SAM3 to generate masks and masked images"
bash submodules/sam3/run_inference.sh $seq_name $sam_frame_idx