#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Training
python training/run.py scene_name=modric_vs_ribberi
