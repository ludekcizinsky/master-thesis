#!/bin/bash
set -e

# parameter setup
scripts_path="/home/cizinsky/master-thesis/preprocess/multiply_pipeline" # absolute path of preprocessing scripts
folder_path="/scratch/izar/cizinsky/multiply-output/preprocessing" # absolute path of preprocessing folder
source="custom" # "custom" if use custom data
seq="football_high_res" # name of the sequence
number=2 # number of people
rm -rf ~/.cache/torch/kernels/* # remove cached torch kernels to avoid this weird error saying Torch.prod produces RuntimeError: CUDA driver error: invalid
cd $scripts_path # change directory to the preprocessing scripts folder

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc cuda/11.8 ffmpeg
trace_env="trace"
vitpose_env="vitpose"
thesis_env="thesis"
unidepth_env="unidepth"

echo "---- Running Trace"
conda activate $trace_env
scene_dir=$folder_path/trace_results/$seq
mkdir -p $scene_dir
trace2 -i $folder_path/data/$seq/image --subject_num=$number --results_save_dir=$scene_dir --save_video --time2forget=40

echo "---- Reformatting Trace output"
conda deactivate && conda activate $trace_env
python reformat_trace_output.py --seq $seq --output_folder $folder_path

echo "---- Getting projected SMPL masks"
conda deactivate && conda activate $thesis_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode mask

echo "---- Running VitPose to get 2D keypoints"
conda deactivate && conda activate $vitpose_env
python vitpose_trace.py \
  --pose_checkpoint /scratch/izar/cizinsky/pretrained/vitpose-h-multi-coco.pth \
  --img-root $folder_path/raw_data/$seq/frames \
  --kpt-thr 0.3

echo "---- Refining poses offline"
conda deactivate && conda activate $thesis_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode refine --vitpose

echo "---- Scaling images and centering human in 3D space"
conda deactivate && conda activate $thesis_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode final --scale_factor 2

echo "--- Normalize cameras such that all cameras are within the sphere of radius 3"
conda deactivate && conda activate $thesis_env
python normalize_cameras_trace.py --input_cameras_file $folder_path/data/$seq/cameras.npz \
                            --output_cameras_file $folder_path/data/$seq/cameras_normalize.npz \
                            --max_human_sphere_file $folder_path/data/$seq/max_human_sphere.npy

echo "---- Running unidepth to obtain per frame depth maps"
conda activate $unidepth_env
export PYTHONPATH="${PYTHONPATH}:/home/cizinsky/master-thesis/preprocess/multiply_pipeline/unidepth"
CUDA_VISIBLE_DEVICES=0 python unidepth/scripts/demo_mega-sam.py \
--scene-name $seq \
--img-path $folder_path/data/$seq/image \
--outdir $folder_path/data/$seq/unidepth

echo "---- Running mask refinement with SAM"
conda deactivate && conda activate $thesis_env
cd /home/cizinsky/master-thesis
python training/run.py scene_name=$seq tids=[0,1] train_bg=false resume=false group_name=dev debug=true is_preprocessing=true 

echo "---- Converting unidepth to point clouds"
conda deactivate && conda activate thesis
cd /home/cizinsky/master-thesis/preprocess/multiply_pipeline
python unidepth_to_cloud.py \
  --preprocess_dir /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq \
  --unidepth_dir   /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/unidepth \
  --mask_dir       /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/sam2_masks \
  --output_npz /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/unidepth_cloud_static_scaled.npz \
  --depth_scale 0.2

echo "---- Running visualization of joined results"
python joined_viz.py --sequence $seq