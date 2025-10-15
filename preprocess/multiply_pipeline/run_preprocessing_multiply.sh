#!/bin/bash
set -e

# parameter setup
folder_path="/scratch/izar/cizinsky/multiply-output/preprocessing" # absolute path of preprocessing folder
source="custom" # "custom" if use custom data
seq="pushups_smpl" # name of the sequence
seq_path="/home/cizinsky/zurihack/iphone_vids/$seq.mov" # path to the seq
number=1 # number of people
rm -rf ~/.cache/torch/kernels/* # remove cached torch kernels to avoid this weird error saying Torch.prod produces RuntimeError: CUDA driver error: invalid

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc cuda/11.8 ffmpeg
trace_env="trace"
vitpose_env="vitpose_legacy"
aitviewer_env="aitv"
multiply_env="multiply"
cd $folder_path

echo "---- Extracting frames from the video"
mkdir $folder_path/raw_data/$seq
mkdir $folder_path/raw_data/$seq/frames
ffmpeg -y -i "$seq_path" -vf fps=15 -vsync 0 "$folder_path/raw_data/$seq/frames/%04d.png"

echo "---- Running Trace"
conda activate $trace_env
scene_dir=$folder_path/trace_results/$seq
mkdir -p $scene_dir
trace2 -i $folder_path/raw_data/$seq/frames --subject_num=$number --results_save_dir=$scene_dir --save_video --time2forget=40

echo "---- Reformatting Trace output"
cd /home/cizinsky/MultiPly/preprocessing
conda deactivate && conda activate $aitviewer_env
python reformat_trace_output.py --seq $seq --output_folder $folder_path

echo "---- Getting projected SMPL masks"
conda deactivate && conda activate $multiply_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode mask

echo "---- Running VitPose to get 2D keypoints"
conda deactivate && conda activate $vitpose_env
python vitpose_trace.py \
  --pose_checkpoint /scratch/izar/cizinsky/pretrained/vitpose-h-multi-coco.pth \
  --img-root $folder_path/raw_data/$seq/frames \
  --kpt-thr 0.3

echo "---- Refining poses offline"
conda deactivate && conda activate $multiply_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode refine --vitpose

echo "---- Scaling images and centering human in 3D space"
conda deactivate && conda activate $multiply_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode final --scale_factor 2

echo "--- Normalize cameras such that all cameras are within the sphere of radius 3"
conda deactivate && conda activate $multiply_env
python normalize_cameras_trace.py --input_cameras_file $folder_path/data/$seq/cameras.npz \
                            --output_cameras_file $folder_path/data/$seq/cameras_normalize.npz \
                            --max_human_sphere_file $folder_path/data/$seq/max_human_sphere.npy
