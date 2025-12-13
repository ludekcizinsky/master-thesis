#!/bin/bash
set -e # exit on error

# Example usage:
#   bash preprocess.sh taichi 

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg

# navigate to project directory
cd /home/cizinsky/master-thesis

# configurable settings
seq_name=$1

# for now default settings
default_ref_frame_idx=60
input_frames_path=/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair17_1/pair17/dance17/images/28

# derived paths
results_dir=/scratch/izar/cizinsky/thesis/results/$seq_name
mkdir -p $results_dir

#echo "--- [1/?] Preparing frame folder"
#frame_folder=$results_dir/frames
#mkdir -p $frame_folder
#cp -r $input_frames_path/* $frame_folder/

#echo "--- [2/?] Running Human3R to obtain: smplx parameters and camera poses"
#bash submodules/human3r/run_inference.sh $seq_name

#echo "--- [3/?] Running Depth Anything 3 to generate depth maps"
#bash submodules/da3/run_inference.sh $seq_name

#echo "--- [4/?] Running SAM3 to generate masks and masked images"
#bash submodules/sam3/run_inference.sh $seq_name $default_ref_frame_idx

# echo "--- [5/?] Running inference.sh to obtain canonical 3dgs models for each human"
# conda deactivate && conda activate lhm
# bash inference.sh $seq_name 0 $default_ref_frame_idx LHM-1B
# bash inference.sh $seq_name 1 $default_ref_frame_idx LHM-1B