#!/usr/bin/env bash
#SBATCH --job-name=dac-train
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=runs/slurm_logs/train_%j.log
#SBATCH --chdir=/data/groups/bci/jonas/workspace/bci_descript-audio-codec
#
#   sbatch scripts/train_jm_slurm.sh [conf/base_jm_eeg.yml]
#   tail -f runs/slurm_logs/train_<jobid>.log

CONFIG="${1:-conf/base_jm_eeg.yml}"

set -euo pipefail
cd /data/groups/bci/jonas/workspace/bci_descript-audio-codec
mkdir -p runs/slurm_logs

if [ -x /scratch/jonas/venv_dac_audio/bin/python ]; then
  PY=/scratch/jonas/venv_dac_audio/bin/python
else
  PY=/data/groups/bci/jonas/venv_dac_audio/bin/python
fi
export NUMPY_MADVISE_HUGEPAGE=0

echo "== config: $CONFIG | python: $PY | node: $(hostname) =="
"$PY" scripts/train_jm.py --args.load "$CONFIG"
