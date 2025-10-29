#!/bin/bash
set -e

# parameter setup
folder_path="/scratch/izar/cizinsky/multiply-output/preprocessing" # absolute path of preprocessing folder
source="custom" # "custom" if use custom data
seq="taichi" # name of the sequence
number=2 # number of people
rm -rf ~/.cache/torch/kernels/* # remove cached torch kernels to avoid this weird error saying Torch.prod produces RuntimeError: CUDA driver error: invalid

source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc cuda/12.1 ffmpeg
trace_env="trace2"
vitpose_env="vitpose_legacy"
aitviewer_env="aitv"
multiply_env="multiply"
# cd $folder_path

# echo "---- Running Trace"
# conda activate $trace_env
# scene_dir=$folder_path/$seq/trace_results
# mkdir -p $scene_dir
# trace2 -i $folder_path/$seq/image --subject_num=$number --results_save_dir=$scene_dir --save_video --time2forget=40

# echo "---- Reformatting Trace output"
# cd /home/cizinsky/MultiPly/preprocessing
# conda deactivate && conda activate $aitviewer_env
# python reformat_trace_output.py --seq $seq --output_folder $folder_path

# echo "---- Getting projected SMPL masks"
# conda deactivate && conda activate $multiply_env
# python preprocessing_multiple_trace.py --source custom --seq $seq --mode mask

# echo "---- Running VitPose to get 2D keypoints"
# conda deactivate && conda activate $vitpose_env
# python vitpose_trace.py \
  # --pose_checkpoint /scratch/izar/cizinsky/pretrained/vitpose-h-multi-coco.pth \
  # --img-root $folder_path/raw_data/$seq/frames \
  # --kpt-thr 0.3

# echo "---- Refining poses offline"
# conda deactivate && conda activate $multiply_env
# python preprocessing_multiple_trace.py --source custom --seq $seq --mode refine --vitpose

# echo "---- Scaling images and centering human in 3D space"
# conda deactivate && conda activate $multiply_env
# python preprocessing_multiple_trace.py --source custom --seq $seq --mode final --scale_factor 2

# echo "--- Normalize cameras such that all cameras are within the sphere of radius 3"
# conda deactivate && conda activate $multiply_env
# python normalize_cameras_trace.py --input_cameras_file $folder_path/data/$seq/cameras.npz \
                            # --output_cameras_file $folder_path/data/$seq/cameras_normalize.npz \
                            # --max_human_sphere_file $folder_path/data/$seq/max_human_sphere.npy

echo "---- Running unidepth to obtain per frame depth maps"
conda activate mega_sam
export PYTHONPATH="${PYTHONPATH}:/home/cizinsky/master-thesis/preprocess/multiply_pipeline/unidepth"
CUDA_VISIBLE_DEVICES=0 python unidepth/scripts/demo_mega-sam.py \
--scene-name $seq \
--img-path $folder_path/data/$seq/image \
--outdir $folder_path/data/$seq/unidepth

echo "---- Running mask refinement with SAM"
conda deactivate && conda activate thesis
cd /home/cizinsky/master-thesis
python training/run.py scene_name=$seq tids=[0,1] train_bg=false resume=false group_name=dev debug=true is_preprocessing=true mask_refinement.sam2.max_points=6

echo "---- Converting unidepth to point clouds"
conda deactivate && conda activate thesis
cd /home/cizinsky/master-thesis/preprocess/multiply_pipeline
python unidepth_to_cloud.py \
  --preprocess_dir /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq \
  --unidepth_dir   /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/unidepth \
  --mask_dir       /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/sam2_masks \
  --output_npz /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/unidepth_cloud_static_scaled.npz \
  --max_frames 30 \
  --depth_scale 0.4

# echo "---- Running visualization of joined results"
python joined_viz.py --sequence $seq