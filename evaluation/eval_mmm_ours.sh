#/bin/bash

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg 

# Parse CLI arguments
seq_name=$1
exp_version=$2

# Construct paths from CLI arguments
eval_dir_output=/scratch/izar/cizinsky/thesis/evaluation
eval_dir_path=/scratch/izar/cizinsky/thesis/output/$seq_name/evaluation/${exp_version}

# Derive other paths
render_path=$eval_dir_path/fg_render/all/rgb

# Run evaluation
python run.py --renders-path $render_path

# Generate video from original renders
mkdir -p ${eval_dir_output}/videos/renders/custom/ours
render_dir=$render_path/../renders
output_video=${eval_dir_output}/videos/renders/custom/ours/${seq_name}.mp4
ffmpeg -hide_banner -loglevel error -y -framerate 20 \
  -i "$render_dir/%04d.png" \
  -c:v libx264 -pix_fmt yuv420p $output_video 
echo "Renders video saved to: $output_video"