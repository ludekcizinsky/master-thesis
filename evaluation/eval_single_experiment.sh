#/bin/bash

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg 

# Parse CLI arguments
seq_name=$1
gt_dir=$2
cam_id=$3
exp_version=$4

# Construct paths from CLI arguments
eval_dir_output=/scratch/izar/cizinsky/thesis/evaluation
preprocess_dir_path=/scratch/izar/cizinsky/multiply-output/preprocessing/data/$seq_name
images_path=$preprocess_dir_path/image
render_path=/scratch/izar/cizinsky/thesis/output/$seq_name/checkpoints/${exp_version}_$seq_name/fg_render/all/rgb
gt_masks_path=$gt_dir/seg/img_seg_mask/$cam_id/all
gt_masks_ds_type=multicolor_png
pred_masks_path=/scratch/izar/cizinsky/thesis/output/$seq_name/checkpoints/${exp_version}_$seq_name/progressive_sam
pred_masks_ds_type=progressive_sam
gt_joints_path=$gt_dir/smpl
gt_joints_ds_type=hi4d
pred_joints_path=/scratch/izar/cizinsky/thesis/output/$seq_name/checkpoints/${exp_version}_$seq_name/smpl
pred_joints_ds_type=ours
metrics_output_path=/scratch/izar/cizinsky/thesis/output/$seq_name/checkpoints/${exp_version}_$seq_name
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

cp $metrics_output_path/metrics.csv ${eval_dir_output}/metrics/${exp_version}_${seq_name}.csv

masked_dir=$render_path/../masked_renders
output_video=${eval_dir_output}/videos/masked_renders/${exp_version}_${seq_name}.mp4
ffmpeg -hide_banner -loglevel error -y -framerate 20 \
  -i "$masked_dir/%04d.png" \
  -c:v libx264 -pix_fmt yuv420p $output_video 
echo "Video saved to: $output_video"