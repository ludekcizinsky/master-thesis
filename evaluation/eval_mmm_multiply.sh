#/bin/bash

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis
module load gcc ffmpeg 

# Parse CLI arguments
seq_name=$1

# Construct paths from CLI arguments
eval_dir_output=/scratch/izar/cizinsky/thesis/evaluation

# Derive other paths
test_results_path=/scratch/izar/cizinsky/multiply-output/training/$seq_name/visualisations/
render_path=$test_results_path/test_fg_rendering/-1

# Run evaluation
python run.py --renders-path $render_path

# Generate video from original renders
mkdir -p ${eval_dir_output}/videos/renders/custom/multiply
render_dir=$render_path/../renders
output_video=${eval_dir_output}/videos/renders/custom/multiply/${seq_name}.mp4
ffmpeg -hide_banner -loglevel error -y -framerate 20 \
  -i "$render_dir/%04d.png" \
  -c:v libx264 -pix_fmt yuv420p $output_video 
echo "Renders video saved to: $output_video"