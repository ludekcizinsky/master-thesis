#!/bin/bash

# activate conda environment
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
module load gcc ffmpeg
conda activate thesis

cd /home/cizinsky/master-thesis

part=$1
if [[ -z "$part" ]]; then
    echo "Usage: $0 <1|2|part1|part2>"
    exit 1
fi

case "$part" in
    1|part1)
        part_arg="part1"
        ;;
    2|part2)
        part_arg="part2"
        ;;
    *)
        echo "Invalid part: $part (expected 1, 2, part1, or part2)"
        exit 1
        ;;
esac

python preprocess/custom/helpers/schedule_get_trn_data.py --part "$part_arg"
