#!/bin/bash
JOB_ID=$(sbatch --parsable train.slurm)
echo "Submitted job $JOB_ID"
OUT="/scratch/izar/cizinsky/thesis/output/slurm/modric_vs_ribberi.${JOB_ID}.out"
ERR="/scratch/izar/cizinsky/thesis/output/slurm/modric_vs_ribberi.${JOB_ID}.err"
echo "Stdout will go to: $OUT"
echo "Stderr will go to: $ERR"