#!/usr/bin/env bash
#SBATCH --job-name=dac-cw
#SBATCH --gpus=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/slurm_logs/cw_%j.log
#SBATCH --chdir=/data/groups/bci/chris/workspace/bci_descript-audio-codec

# One Slurm job running several experiment configs on a single GPU.
# Submitted by submit_cw_sweep.py; not meant to be run by hand.
#
# Edit codebook_sizes, loss_combinations, dataset_names, run_cntr, and
# jobs_per_GPU in submit_cw_sweep.py, then from the repo root:
#   Preview:  /data/groups/bci/chris/workspace/venv_dac/bin/python scripts/submit_cw_sweep.py --dry-run
#   Submit:   /data/groups/bci/chris/workspace/venv_dac/bin/python scripts/submit_cw_sweep.py
#
# Follow an experiment:
#   tail -f runs/slurm_logs/crt_cw_026_<jobid>.log
#
# Usage: sbatch scripts/train_cw_slurm.sh CONFIG.yml [CONFIG.yml ...]
# Each config sets save_path and WandB.name.

set -euo pipefail

REPO_DIR=/data/groups/bci/chris/workspace/bci_descript-audio-codec
PY=/data/groups/bci/chris/workspace/venv_dac/bin/python

cd "$REPO_DIR"

if [[ $# -eq 0 ]]; then
  echo "usage: sbatch $0 CONFIG.yml [CONFIG.yml ...]" >&2
  exit 2
fi

# Slurm supplies CUDA_VISIBLE_DEVICES; leave its GPU allocation unchanged.
pids=()
for CONFIG in "$@"; do
  NAME="${CONFIG##*/}"
  RUN_NAME="${NAME%.yml}_${SLURM_JOB_ID}"
  echo "== run: $RUN_NAME | config: $CONFIG | python: $PY | node: $(hostname) =="
  "$PY" scripts/train_cw.py --args.load "$CONFIG" \
    > "runs/slurm_logs/${RUN_NAME}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
