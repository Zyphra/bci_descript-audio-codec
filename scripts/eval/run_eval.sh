#!/usr/bin/env bash
#SBATCH --job-name=dac-eegbci-eval
#SBATCH --partition=gpu           # big pool, no time limit ('dev' = nodes 042/045, 10h cap)
#SBATCH --gpus=1                  # Slurm picks a free GPU and hides all others from us
#SBATCH --cpus-per-task=16        # matches classify's parallelism
#SBATCH --mem=64G
#SBATCH --time=02:00:00           # quick pass fits easily; raise for the full sweep
#SBATCH --output=scripts/eval/logs/slurm_%j.log
#SBATCH --chdir=/data/groups/bci/jonas/workspace/bci_descript-audio-codec
#SBATCH --nodelist=dgxh100-042    # node with our local venv copy in /scratch
#
# EEGBCI classifier eval: for each checkpoint, codec the labeled windows, train
# classifiers, print + save results. Edit the block below (SUBJ_FRACTION = how many
# subjects), save, then from the repo root:
#
#   cd /data/groups/bci/jonas/workspace/bci_descript-audio-codec
#   sbatch scripts/eval/run_eval.sh              # prints: Submitted batch job <jobid>
#   tail -f scripts/eval/logs/slurm_<jobid>.log  # watch it live (Ctrl-C stops the tail only)
#
#   (or run directly without slurm:  bash scripts/eval/run_eval.sh)
#
# Results land in scripts/eval/results/<arch>_<hash>[_s<N>]/ — one folder per checkpoint,
# finished checkpoints are cached and skipped, so re-running is cheap.

# ========================== EDIT ME ==========================
DEVICE="cuda:1"    # GPU when run directly on vp42 (ours: 1 and 4). Ignored under Slurm.
LIGHT=0            # 1 = smoke test: eyes task only, quarter of the windows
SUBJ_FRACTION=55   # keep every N-th subject (all their trials).
                   #   55 -> 2 subjects, smoke-speed  |  6 -> quick pass  |  1 -> full sweep
FRACTION=1         # keep every N-th window. Leave at 1: thinning windows starves CSP.
FORCE=0            # 1 = recompute everything, ignore caches
CKPTS=(            # run-folder names (uses their latest) or exact paths to a weights.pth
  1k_1X16
  1k_1X64
  1k_1X256
  1k_1X1024
  1k_3X16
  1k_5X16
  10k_1X16
  10k_1X64
  10k_1X256
  10k_1X1024
  10k_3X16
  10k_5X16
)
# =============================================================

set -euo pipefail
# Fixed path, not $(dirname $0): Slurm executes a spooled COPY of this script,
# so "where am I" tricks point at /var/spool/slurm there.
REPO="/data/groups/bci/jonas/workspace/bci_descript-audio-codec"
EVAL="$REPO/scripts/eval"

# Under Slurm the allocated GPU is always visible as cuda:0.
[ -n "${SLURM_JOB_ID:-}" ] && DEVICE="cuda:0"

# Only one eval run at a time (two would fight over CPU and result folders).
exec 9>"$EVAL/.run.lock"
if ! flock -n 9; then
  echo "!! another run_eval.sh is already running — wait, or: pkill -f run_eval.sh" >&2
  exit 1
fi
trap 'kill 0' INT TERM EXIT   # killing this script kills its python children too

[ "$#" -gt 0 ] && CKPTS=("$@")
TASKS="eyes,fist_lr,fists_feet"
[ "$LIGHT" = "1" ] && { TASKS="eyes"; FRACTION=4; }
FORCE_FLAG=""
[ "$FORCE" = "1" ] && FORCE_FLAG="--force"
echo "== tasks=$TASKS subject-fraction=$SUBJ_FRACTION window-fraction=$FRACTION device=$DEVICE =="

# a checkpoint entry may be a folder name in runs/ or a direct path to a weights.pth
resolve() {
  if [ -f "$1" ]; then echo "$1"
  elif [ -f "$REPO/runs/$1/latest/dac/weights.pth" ]; then echo "$REPO/runs/$1/latest/dac/weights.pth"
  else echo "NOT_FOUND"
  fi
}

# Node-local venv, selected by EXPLICIT interpreter path: the copied activate script
# hardcodes its original Lustre path, so `source .../scratch/.../activate` silently
# re-selects the Lustre python (found by codex review). The binary itself resolves its
# prefix from its own location, so invoking it directly is correct.
if [ -x /scratch/jonas/venv_dac_audio/bin/python ]; then
  PY=/scratch/jonas/venv_dac_audio/bin/python
else
  PY=/data/groups/bci/jonas/venv_dac_audio/bin/python
fi
echo "== python: $PY =="

# NumPy requests transparent huge pages for large allocations; on this fragmented node
# (compact_fail 98%) that triggers synchronous compaction: 8 MiB touch measured at
# 2.18 s with THP requests vs 4-58 ms without. Must be set before python starts.
export NUMPY_MADVISE_HUGEPAGE=0

cd "$REPO"

# Work on node-local /scratch (immune to Lustre stalls), publish each finished
# checkpoint to scripts/eval/results/ with one sequential rsync. Pre-seed the local
# root from any already-published results so caching keeps working across nodes.
if [ -d /scratch/jonas ]; then
  export EVAL_RESULTS_ROOT=/scratch/jonas/eval_results
  mkdir -p "$EVAL_RESULTS_ROOT" "$EVAL/results"
  rsync -a "$EVAL/results/" "$EVAL_RESULTS_ROOT/" 2>/dev/null || true
else
  export EVAL_RESULTS_ROOT="$EVAL/results"
fi

DIRS=()
for RAW in "${CKPTS[@]}"; do
  CKPT="$(resolve "$RAW")"
  if [ "$CKPT" = "NOT_FOUND" ]; then echo "!! skipping '$RAW': no checkpoint found"; continue; fi
  echo ""
  echo "############################################################"
  echo "# $RAW  ->  $CKPT"
  echo "############################################################"
  "$PY" "$EVAL/dump.py" --ckpt "$CKPT" --tasks "$TASKS" --device "$DEVICE" \
         --fraction "$FRACTION" --subject-fraction "$SUBJ_FRACTION" $FORCE_FLAG
  RESULTS="$(readlink -f "$EVAL_RESULTS_ROOT/_latest")"
  "$PY" "$EVAL/classify.py" --results "$RESULTS" --tasks "$TASKS" $FORCE_FLAG
  "$PY" "$EVAL/plots.py" --results "$RESULTS"
  if [ "$EVAL_RESULTS_ROOT" != "$EVAL/results" ]; then
    rsync -a "$RESULTS" "$EVAL/results/"   # publish to /data in one sequential pass
  fi
  DIRS+=("$RESULTS")
done

if [ "${#DIRS[@]}" -gt 1 ]; then
  echo ""
  echo "############################################################"
  echo "# ALL RUNS — combined"
  echo "############################################################"
  for d in "${DIRS[@]}"; do echo ""; echo ">>> $(basename "$d")"; sed -n '/Accuracy per task/,$p' "$d/summary.txt"; done
fi
echo ""
echo "done. per-run outputs in scripts/eval/results/<name>/ (summary.txt, accuracy.csv, bands.csv, plots/)"
