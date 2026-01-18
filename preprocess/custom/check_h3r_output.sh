#!/bin/bash

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

preprocess_dir=""
case "$1" in
    --scene_dir|--scene-dir)
        preprocess_dir=$2
        ;;
    *)
        preprocess_dir=$1
        ;;
esac

if [[ -z "$preprocess_dir" ]]; then
    echo "Usage: $0 [--scene_dir|--scene-dir] <path_to_scene_dir>"
    exit 1
fi

python preprocess/custom/helpers/check_h3r_output.py --scene-dir "$preprocess_dir" --model-folder /home/cizinsky/body_models
