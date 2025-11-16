#!/bin/bash
#SBATCH --job-name=v7_eval
#SBATCH --output=/scratch/izar/cizinsky/thesis/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/thesis/slurm/%x.%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg 

# Change to the directory where the script is located
cd /home/cizinsky/master-thesis

# V7 / Taichi evaluation / MultiPly
images_path="/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi/image"
gt_masks_path="/scratch/izar/cizinsky/multiply-output/training/original_taichi/pseudo_gt_masks"
gt_masks_ds_type="binary_png"
render_path="/scratch/izar/cizinsky/multiply-output/training/original_taichi/joined_fg"
metrics_output_path="/scratch/izar/cizinsky/multiply-output/evaluation/original_taichi"
python evaluation/run.py \
  --images-path $images_path \
  --gt-masks-path $gt_masks_path \
  --gt-masks-ds-type $gt_masks_ds_type \
  --renders-path $render_path \
  --metrics-output-path $metrics_output_path

masked_dir=$render_path/masked_renders
output_video=$masked_dir/masked_renders.mp4
ffmpeg -hide_banner -loglevel error -y -framerate 10 \
  -i "$masked_dir/%04d.png" \
  -c:v libx264 -pix_fmt yuv420p $output_video 
echo "Video saved to: $output_video"