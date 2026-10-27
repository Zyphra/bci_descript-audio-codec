#!/usr/bin/env bash
#SBATCH --job-name=dac-eeg-eval
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/slurm_logs/eeg_eval_%j.log
#SBATCH --chdir=/data/groups/bci/chris/workspace/bci_descript-audio-codec

# Usage: sbatch scripts/evaluate_eeg_slurm.sh CHECKPOINT [CHECKPOINT ...] OPTIONS
# Accepts all evaluate_eeg.py arguments, including --checkpoint-list and --inputs.
# Example (first checkpoint is the paired baseline):
# sbatch scripts/evaluate_eeg_slurm.sh \
#   runs/vq_long_000/latest runs/vq_long_003/latest runs/vq_long_006/latest \
#   --labels c1_b1_s0 c2_b1_s0 c1_b4_s0 \
#   --windows 2500 --output analysis/vq_long_spectral_seed0
# Use a new/empty output directory. Relative paths resolve from REPO_DIR.
set -euo pipefail

REPO_DIR=/data/groups/bci/chris/workspace/bci_descript-audio-codec
PY=/data/groups/bci/chris/workspace/venv_dac/bin/python
cd "$REPO_DIR"

if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch $0 CHECKPOINT [CHECKPOINT ...] --output OUTPUT_DIR [EVALUATOR_OPTIONS]" >&2
    echo "Use --help for evaluator options, or --checkpoint-list FILE for a checkpoint list." >&2
    exit 2
fi

# Keep runtime files on node-local /tmp, outside the evaluation output directory.
runtime_dir="$(mktemp -d /tmp/dac-eeg-eval.XXXXXXXX)"
trap 'rm -rf -- "$runtime_dir"' EXIT
mkdir -p "$runtime_dir/tmp" "$runtime_dir/matplotlib"
export TMPDIR="$runtime_dir/tmp"
export MPLCONFIGDIR="$runtime_dir/matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Slurm sets CUDA_VISIBLE_DEVICES; cuda:0 is the allocated GPU. Explicit user
# options come last so they can override these defaults. Do not exec: the EXIT
# trap must run after Python exits, including when evaluation fails.
"$PY" -u scripts/evaluate_eeg.py \
    --device cuda:0 --batch-size 8 --threads "$OMP_NUM_THREADS" "$@"
