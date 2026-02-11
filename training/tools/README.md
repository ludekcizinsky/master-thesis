# Training Tools

## watch_slurm_logs.py

Live viewer for Slurm logs with split panes:

- left pane: `.out`
- right pane: `.err`

It can:

- list running jobs from `squeue -u <user>` and let you pick one
- accept a direct job id via `--jobid`
- resolve logs under `/scratch/izar/cizinsky/thesis/slurm`
- handle array jobs (`<jobid>_<taskid>`)
- render job selection and startup info with `rich` (falls back to plain text if unavailable)

### Usage

Interactive (list jobs first):

```bash
python training/tools/watch_slurm_logs.py
```

Specify user:

```bash
python training/tools/watch_slurm_logs.py --user cizinsky
```

Watch a specific job task:

```bash
python training/tools/watch_slurm_logs.py --jobid 2788165_2
```

Tune refresh and tail length:

```bash
python training/tools/watch_slurm_logs.py --jobid 2788165 --lines 80 --refresh 0.3
```

### Controls

- `q`: quit
