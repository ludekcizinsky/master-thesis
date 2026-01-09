# !/bin/bash

# Things to set
root_repo_path=/home/cizinsky/master-thesis

# Activate conda 
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

# Create conda env
conda create -n smplx python=3.9

# Navigate to smplx directory
cd $root_repo_path/submodules/smplx

# Activate conda env
conda activate smplx

# Install dependencies
cd transfer_model
pip install -r requirements.txt

# trust ncg
pip install --no-build-isolation "git+https://github.com/vchoutas/torch-trust-ncg.git"

# chumpy
pip install --no-build-isolation git+https://github.com/mattloper/chumpy

# Follow the instructions [here](https://github.com/ludekcizinsky/smplx/tree/main/transfer_model#data) 
# to download and extract the neccesary transfer data.