#/bin/bash

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg 

# Parse CLI arguments
seq_name=$1
gt_dir=$2
cam_id=$3

# Construct paths from CLI arguments
eval_dir_output=/scratch/izar/cizinsky/thesis/evaluation
preprocess_dir_path=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq_name
images_path=$preprocess_dir_path/image
test_results_path=/scratch/izar/cizinsky/multiply-output/training/$seq_name/visualisations/
render_path=$test_results_path/test_fg_rendering/-1
gt_masks_path=$gt_dir/seg/img_seg_mask/$cam_id/all
gt_masks_ds_type=multicolor_png
pred_masks_path=$test_results_path/test_mask/-1
pred_masks_ds_type=binary_png
gt_joints_path=$gt_dir/smpl
gt_joints_ds_type=hi4d
pred_joints_path=/scratch/izar/cizinsky/multiply-output/training/$seq_name/checkpoints
pred_joints_ds_type=multiply
metrics_output_path=$test_results_path
python run.py \
  --images-path $images_path \
  --renders-path $render_path \
  --gt-masks-path $gt_masks_path \
  --gt-masks-ds-type $gt_masks_ds_type \
  --pred-masks-path $pred_masks_path \
  --pred-masks-ds-type $pred_masks_ds_type \
  --gt-joints-path $gt_joints_path \
  --gt-joints-ds-type $gt_joints_ds_type \
  --pred-joints-path $pred_joints_path \
  --pred-joints-ds-type $pred_joints_ds_type \
  --transformations-dir-path $preprocess_dir_path \
  --metrics-output-path $metrics_output_path

mkdir -p ${eval_dir_output}/metrics/multiply
cp $metrics_output_path/metrics.csv ${eval_dir_output}/metrics/multiply/${seq_name}.csv
echo "Metrics CSV saved to: ${eval_dir_output}/metrics/multiply/${seq_name}.csv"

mkdir -p ${eval_dir_output}/videos/masked_renders/multiply
masked_dir=$render_path/../masked_renders
output_video=${eval_dir_output}/videos/masked_renders/multiply/${seq_name}.mp4
ffmpeg -hide_banner -loglevel error -y -framerate 20 \
  -i "$masked_dir/%04d.png" \
  -c:v libx264 -pix_fmt yuv420p $output_video 
echo "Video saved to: $output_video"