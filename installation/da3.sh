# !/bin/bash

# Things to set
pretrained_model_dir=/scratch/izar/cizinsky/pretrained
root_repo_path=/home/cizinsky/master-thesis

# Activate conda 
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

# Create conda env
conda create -n da3 python=3.10

# Navigate to da3 directory
cd $root_repo_path/submodules/da3

# Activate conda env
conda activate da3

# Install dependencies
# - basics
pip install xformers torch\>=2 torchvision
pip install . # Basic