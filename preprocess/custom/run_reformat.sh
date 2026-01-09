#!/bin/bash
set -e # exit on error

cd /home/cizinsky/master-thesis

bash preprocess/custom/helpers/other_reformat.sh
bash preprocess/custom/helpers/cameras_reformat.sh
bash preprocess/custom/helpers/smplx_reformat.sh