#!/bin/bash
set -e

# parameter setup
usage() {
  echo "Usage: $0 --seq <sequence_name> --images-dir <path_to_images> [--gt-smpl-dir <path_to_gt_smpl_npz_dir>]"
  exit 1
}

seq=""
images_folder_path=""
gt_smpl_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seq)
      seq="$2"
      shift 2
      ;;
    --images-dir|--images_folder_path)
      images_folder_path="$2"
      shift 2
      ;;
    --gt-smpl-dir)
      gt_smpl_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

if [[ -z "$seq" || -z "$images_folder_path" ]]; then
  echo "Error: --seq and --images-dir are required."
  usage
fi

number=2 # number of people
source="custom" # "custom" if use custom data
scripts_path="/home/cizinsky/master-thesis/preprocess/multiply_pipeline" # absolute path of preprocessing scripts
folder_path="/scratch/izar/cizinsky/multiply-output/preprocessing" # absolute path of preprocessing folder
trace_file_name="$(basename "$images_folder_path").npz"
init_mask_path="$folder_path/raw_data/$seq/init_mask"
vitpose_output_path="$folder_path/data/$seq/vitpose"
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
trace2 -i $images_folder_path --subject_num=$number --results_save_dir=$scene_dir --save_video --time2forget=40

echo "---- Reformatting Trace output"
conda deactivate && conda activate $trace_env
python reformat_trace_output.py --seq $seq --output_folder $folder_path --trace_file "$trace_file_name"

echo "---- Getting projected SMPL masks"
conda deactivate && conda activate $thesis_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode mask --images_dir "$images_folder_path"

echo "---- Running VitPose to get 2D keypoints"
conda deactivate && conda activate $vitpose_env
python vitpose_trace.py \
  --pose_checkpoint /scratch/izar/cizinsky/pretrained/vitpose-h-multi-coco.pth \
  --img-root "$images_folder_path" \
  --kpt-thr 0.3 \
  --init-mask-path "$init_mask_path" \
  --output-dir "$vitpose_output_path"

echo "---- Refining poses offline"
conda deactivate && conda activate $thesis_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode refine --vitpose --images_dir "$images_folder_path"

echo "---- Scaling images and centering human in 3D space"
conda deactivate && conda activate $thesis_env
python preprocessing_multiple_trace.py --source custom --seq $seq --mode final --scale_factor 1 --images_dir "$images_folder_path"

echo "--- Normalize cameras such that all cameras are within the sphere of radius 3"
conda deactivate && conda activate $thesis_env
python normalize_cameras_trace.py --input_cameras_file $folder_path/data/$seq/cameras.npz \
                            --output_cameras_file $folder_path/data/$seq/cameras_normalize.npz \
                            --max_human_sphere_file $folder_path/data/$seq/max_human_sphere.npy

echo "---- Running mask refinement with SAM"
conda deactivate && conda activate $thesis_env
cd /home/cizinsky/master-thesis
python training/run.py scene_name=$seq tids=[0,1] train_bg=false resume=false group_name=dev debug=true is_preprocessing=true 

if [[ -n "$gt_smpl_dir" ]]; then
  echo "---- Aligning canonical Trace SMPL to metric GT SMPL"
  conda deactivate && conda activate $thesis_env
  cd /home/cizinsky/master-thesis
  python preprocess/multiply_pipeline/align_trace_to_gt.py \
    --preprocess_dir $folder_path/data/$seq \
    --gt_smpl_dir "$gt_smpl_dir"
fi

# echo "---- Running unidepth to obtain per frame depth maps"
# conda activate $unidepth_env
# export PYTHONPATH="${PYTHONPATH}:/home/cizinsky/master-thesis/preprocess/multiply_pipeline/unidepth"
# CUDA_VISIBLE_DEVICES=0 python unidepth/scripts/demo_mega-sam.py \
# --scene-name $seq \
# --img-path $folder_path/data/$seq/image \
# --outdir $folder_path/data/$seq/unidepth

# echo "---- Converting unidepth to point clouds"
# conda deactivate && conda activate thesis
# cd /home/cizinsky/master-thesis/preprocess/multiply_pipeline
# python unidepth_to_cloud.py \
  # --preprocess_dir /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq \
  # --unidepth_dir   /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/unidepth \
  # --mask_dir       /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/sam2_masks \
  # --output_npz /scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq/unidepth_cloud_static_scaled.npz \
  # --depth_scale 0.8

# echo "---- Running visualization of joined results"
# python joined_viz.py --sequence $seq
