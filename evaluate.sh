#!/bin/bash

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Evaluate
python training/evaluate.py internal_run_id=cool-sea-51_f2rqogjf split=train
python training/evaluate.py internal_run_id=usual-shape-52_hxjuecfe split=train