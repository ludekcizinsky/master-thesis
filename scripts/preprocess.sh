#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status


# Conda init
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh

python preprocess/run.py