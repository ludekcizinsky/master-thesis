#!/bin/bash
set -e # exit on error

cd /home/cizinsky/master-thesis

bash preprocess/custom/helpers/rename_and_copy.sh
bash preprocess/custom/helpers/cameras_reformat.sh
bash preprocess/custom/helpers/smplx_reformat.sh
bash preprocess/custom/helpers/smplx_align.sh