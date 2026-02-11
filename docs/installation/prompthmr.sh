# !/bin/bash

# Things to set
pretrained_model_dir=/scratch/izar/cizinsky/pretrained
root_repo_path=/home/cizinsky/master-thesis

# Activate conda 
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

# CD into the submodule 
cd $root_repo_path
cd submodules/prompthmr/

# Create conda env and install dependencies
bash scripts/install.sh --pt_version=2.4 --world-video=true
bash scripts/fetch_data.sh
