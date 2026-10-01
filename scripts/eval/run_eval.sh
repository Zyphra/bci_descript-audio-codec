#!/usr/bin/env bash
#SBATCH --job-name=dac-eegbci-eval
#SBATCH --partition=gpu           # big pool, no time limit ('dev' = nodes 042/045, 10h cap)
#SBATCH --gpus=1                  # Slurm picks a free GPU and hides all others from us
#SBATCH --cpus-per-task=16        # matches classify's parallelism
#SBATCH --mem=64G
#SBATCH --time=02:00:00           # quick pass fits easily; raise for the full sweep
#SBATCH --output=scripts/eval/logs/slurm_%j.log
#SBATCH --chdir=/data/groups/bci/jonas/workspace/bci_descript-audio-codec


# HOW TO RUN: 
#
# SBATCH: sbatch scripts/eval/run_eval.sh
# sbatch helpers: 
#   squeue -u jonas 
#   tail -f scripts/eval/logs/slurm_<jobid>.log
#   scancel <jobid>  
# RESERVED: DEVICE=cuda:0 bash scripts/eval/run_eval.sh

# figures rebuild automatically at the end; to refresh them alone (no GPU needed): python scripts/eval/probe_plots.py
#
# RESULTS: scripts/eval/results/<run>/ 

# best tasks: fist_lr + ssvep --> together 9 min per checkpoint



# ========================== EDIT ME ==========================
[ -n "${DEVICE:-}" ] && DEVICE_EXPLICIT=1
DEVICE="${DEVICE:-cuda:1}"    # GPU for direct runs; slurm jobs get their allocated GPU (cuda:0)
LIGHT=0            # 1 = smoke test: eyes task only, quarter of the windows
SUBJECTS="${SUBJECTS:-tier}" # "tier" = the battery tier per dataset (results merge
                             # seamlessly with all previous runs and plots); or a
                             # number (approx subjects per dataset), or "all".
FRACTION=1         # keep every N-th window. Leave at 1: thinning windows starves CSP.
FORCE=0            # 1 = recompute everything, ignore caches
# ---- tasks: 1 = run, 0 = skip (grouped by the dataset they come from) ----
# default = the bread-and-butter probe (fist_lr + ssvep, ~9 min/ckpt); flip on
# p300/motor_4class (+~20 min) for milestone-depth evals — all plots pick them up.
TASK_EYES=0        # eegbci    2-class  eyes open/closed (gain-confounded canary)
TASK_FIST_LR=1     # eegbci    2-class  left/right fist (spatial axis)
TASK_FISTS_FEET=0  # eegbci    2-class  fists/feet (real+imagined)
TASK_P300=0        # erpbci    2-class  target/nontarget flashes (timing axis)
TASK_MOTOR4=0      # bciciv2a  4-class  motor imagery (harder spatial axis)
TASK_EMOTION=0     # seed      3-class  film emotion
TASK_SSVEP=1       # sandiego 12-class  flicker frequency (spectral axis)
CHRIS=/data/groups/bci/chris/workspace/bci_descript-audio-codec/runs
CKPTS=(            # full paths to weights.pth (run-folder names also work, see resolve())
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/1k_1X16/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/1k_1X64/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/1k_1X256/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/1k_1X1024/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/1k_3X16/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/1k_5X16/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/10k_1X16/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/10k_1X64/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/10k_1X256/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/10k_1X1024/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/10k_3X16/latest/dac/weights.pth
  # /data/groups/bci/jonas/workspace/bci_descript-audio-codec/runs/10k_5X16/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_21/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_22/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_23/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_24/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_25/latest/dac/weights.pth
  /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_041/latest/dac/weights.pth
  /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_042/latest/dac/weights.pth
  /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_043/latest/dac/weights.pth
  /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_044/latest/dac/weights.pth
  /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_045/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_046/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_047/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_048/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_049/latest/dac/weights.pth
  # /data/groups/bci/chris/workspace/bci_descript-audio-codec/runs/crt_cw_050/latest/dac/weights.pth
)
# =============================================================

set -euo pipefail
# Fixed path, not $(dirname $0): Slurm executes a spooled COPY of this script,
# so "where am I" tricks point at /var/spool/slurm there.
REPO="/data/groups/bci/jonas/workspace/bci_descript-audio-codec"
EVAL="$REPO/scripts/eval"

# Under Slurm the allocated GPU is always visible as cuda:0.
[ -n "${SLURM_JOB_ID:-}" ] && [ -z "${DEVICE_EXPLICIT:-}" ] && DEVICE="cuda:0" || true

# Only one eval run at a time (two would fight over CPU and result folders).
exec 9>/tmp/dac_eval_run.lock
if ! flock -n 9; then
  echo "!! another run_eval.sh is already running — wait, or: pkill -f run_eval.sh" >&2
  exit 1
fi
# On Ctrl-C/TERM: first drop this trap (kill 0 signals our own group, which would
# otherwise re-fire the trap forever), then take the python children down with us.
trap 'trap - INT TERM; kill 0' INT TERM

[ "$#" -gt 0 ] && CKPTS=("$@")
# TASKS env (comma list of task names) overrides the toggle block entirely
if [ -n "${TASKS:-}" ]; then
  TASK_EYES=0; TASK_FIST_LR=0; TASK_FISTS_FEET=0; TASK_P300=0
  TASK_MOTOR4=0; TASK_EMOTION=0; TASK_SSVEP=0
  for t in ${TASKS//,/ }; do
    case "$t" in
      eyes) TASK_EYES=1 ;; fist_lr) TASK_FIST_LR=1 ;; fists_feet) TASK_FISTS_FEET=1 ;;
      p300) TASK_P300=1 ;; motor_4class|motor4) TASK_MOTOR4=1 ;;
      emotion) TASK_EMOTION=1 ;; ssvep) TASK_SSVEP=1 ;;
      *) echo "unknown task '$t' (eyes fist_lr fists_feet p300 motor_4class emotion ssvep)"; exit 1 ;;
    esac
  done
fi

# build "dataset:task,task" pairs from the toggles
PAIRS=()
t=""
[ "$TASK_EYES" = "1" ] && t="eyes"
[ "$TASK_FIST_LR" = "1" ] && t="$t,fist_lr"
[ "$TASK_FISTS_FEET" = "1" ] && t="$t,fists_feet"
t="${t#,}"; [ -n "$t" ] && PAIRS+=("eegbci:$t")
[ "$TASK_P300" = "1" ] && PAIRS+=("erpbci:p300")
[ "$TASK_MOTOR4" = "1" ] && PAIRS+=("bciciv2a:motor_4class")
[ "$TASK_EMOTION" = "1" ] && PAIRS+=("seed:emotion")
[ "$TASK_SSVEP" = "1" ] && PAIRS+=("ssvep:ssvep")
[ "$LIGHT" = "1" ] && { PAIRS=("eegbci:eyes"); FRACTION=4; }
[ "${#PAIRS[@]}" -gt 0 ] || { echo "no tasks enabled"; exit 1; }
FORCE_FLAG=""
[ "$FORCE" = "1" ] && FORCE_FLAG="--force"
echo "== tasks=${PAIRS[*]} subjects=$SUBJECTS device=$DEVICE =="

# a checkpoint entry may be a run-folder name — searched in our runs/, chris's runs/,
# then our own results snapshots (survives the original run being deleted) — or a
# direct path to a weights.pth
resolve() {
  local snap
  if [ -f "$1" ]; then echo "$1"
  elif [ -f "$REPO/runs/$1/latest/dac/weights.pth" ]; then echo "$REPO/runs/$1/latest/dac/weights.pth"
  elif [ -f "$CHRIS/$1/latest/dac/weights.pth" ]; then echo "$CHRIS/$1/latest/dac/weights.pth"
  else
    snap=$(ls "$EVAL/results/$1"/*/weights_snapshot.pth 2>/dev/null | head -1)
    if [ -n "$snap" ]; then echo "$snap"; else echo "NOT_FOUND"; fi
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
# checkpoint to scripts/eval/results/ with one sequential rsync. Scratch is a
# DISPOSABLE MIRROR of results/ (--delete): scripts/eval/results/ is the single
# source of truth, so deleting results/<run>/ (or all of results/) deletes its
# cache too — scratch is wiped to match on the next launch.
if [ -d /scratch/jonas ]; then
  export EVAL_RESULTS_ROOT=/scratch/jonas/eval_results
  mkdir -p "$EVAL_RESULTS_ROOT" "$EVAL/results"
  rsync -a --delete "$EVAL/results/" "$EVAL_RESULTS_ROOT/" 2>/dev/null || true
else
  export EVAL_RESULTS_ROOT="$EVAL/results"
fi

BATCH=()   # run names processed by THIS invocation (for the latest-batch figure)
for RAW in "${CKPTS[@]}"; do
  CKPT="$(resolve "$RAW")"
  if [ "$CKPT" = "NOT_FOUND" ]; then echo "!! skipping '$RAW': no checkpoint found"; continue; fi
  echo ""
  echo "############################################################"
  echo "# $RAW  ->  $CKPT"
  echo "############################################################"
  for PAIR in "${PAIRS[@]}"; do
    DS="${PAIR%%:*}"; DSTASKS="${PAIR#*:}"
    if [ "$SUBJECTS" = "all" ]; then SF=1
    elif [ "$SUBJECTS" = "tier" ]; then
      case "$DS" in eegbci) SF=9 ;; erpbci|seed) SF=2 ;; *) SF=1 ;; esac
    else
      SF=$("$PY" - "$DS" "$SUBJECTS" <<'PYSF'
import sys, pandas as pd
ds, want = sys.argv[1], int(sys.argv[2])
n = pd.read_parquet(f"/data/groups/bci/datasets/processed/v8_sets/classifier_eval/{ds}/labels.parquet",
                    columns=["subject"]).subject.nunique()
print(max(1, round(n / max(1, min(want, n)))))
PYSF
)
    fi
    "$PY" "$EVAL/dump.py" --ckpt "$CKPT" --tasks "$DSTASKS" --device "$DEVICE" --dataset "$DS" \
           --fraction "$FRACTION" --subject-fraction "$SF" $FORCE_FLAG
    RESULTS="$(readlink -f "$EVAL_RESULTS_ROOT/_latest")"
    "$PY" "$EVAL/classify.py" --results "$RESULTS" --tasks "$DSTASKS" $FORCE_FLAG
    RUN_NAME="$(basename "$(dirname "$RESULTS")")"      # results/<run>/<dataset dir>
    if [ "$EVAL_RESULTS_ROOT" != "$EVAL/results" ]; then
      mkdir -p "$EVAL/results/$RUN_NAME"
      rsync -a "$RESULTS" "$EVAL/results/$RUN_NAME/"   # publish to /data sequentially
    fi
    case " ${BATCH[*]:-} " in *" $RUN_NAME "*) ;; *) BATCH+=("$RUN_NAME") ;; esac
  done
done

echo ""
echo "############################################################"
echo "# figures + ranking (all cached checkpoints)"
echo "############################################################"
"$PY" "$EVAL/probe_plots.py"
if [ "${#BATCH[@]}" -gt 0 ]; then
  echo ""
  echo "############################################################"
  echo "# latest-batch figure (this invocation: ${BATCH[*]})"
  echo "############################################################"
  "$PY" "$EVAL/probe_plots.py" --name latest "${BATCH[@]}"
fi
echo ""
echo "done. results/<run>/ has the caches + card.png; results/_overview/ has probe_summary.png"
echo "(all checkpoints) and probe_summary_latest.png (just this invocation's batch)"
