#!/bin/bash
set -e # exit on error

cd /home/cizinsky/master-thesis

bash preprocess/custom/other_reformat.sh
bash preprocess/custom/cameras_reformat.sh
bash preprocess/custom/smplx_reformat.sh
bash preprocess/custom/smplx_align.sh