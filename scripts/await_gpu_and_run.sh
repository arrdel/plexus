#!/usr/bin/env bash
# Wait for a 4090 to actually free up, then run one PLEXUS job on it.
#
# PLEXUS cannot timeshare a GPU: the frozen 8B backbone alone is 16.0 GB and a LoRA step at
# S=2048 peaks near 21 GB of a 23.6 GiB card (RESEARCH_PLAN.md §5, §6). So this waits for a
# *nearly empty* card rather than for "enough" free memory, and it will not evict anything.
#
#   ./scripts/await_gpu_and_run.sh                        # default: the smoke run
#   ./scripts/await_gpu_and_run.sh --task nc --mask-family star --out experiments/E0.1_s0
#
# Env:
#   MIN_FREE_MIB   memory a card must have free to be claimed (default 22000)
#   STABLE_CHECKS  consecutive passing polls required (default 3) -- avoids grabbing a card
#                  during the gap between two of our own queued jobs
#   POLL_SECONDS   seconds between polls (default 60)
#   MAX_WAIT_HOURS give up after this long (default 72)
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"

MIN_FREE_MIB=${MIN_FREE_MIB:-22000}
STABLE_CHECKS=${STABLE_CHECKS:-3}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-72}

export HF_HOME=${HF_HOME:-/tmp/adelechinda/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PLEXUS_DATA_ROOT=${PLEXUS_DATA_ROOT:-/tmp/adelechinda/plexus_data}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

PYTHON="$REPO/.venv/bin/python"
LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/await_gpu_${STAMP}.log"

if [[ $# -eq 0 ]]; then
  ARGS=(--smoke --out "$REPO/experiments/smoke_${STAMP}")
else
  ARGS=("$@")
fi

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "waiting for a GPU with >= ${MIN_FREE_MIB} MiB free, ${STABLE_CHECKS} consecutive polls"
log "will run: run_experiment.py ${ARGS[*]}"

# Fail early if the checkpoint is gone -- a reboot empties /tmp (E0.0f). No point waiting hours
# for a card only to fall over on model load.
if ! "$PYTHON" -c "from plexus.models.llada import assert_snapshot_present as a; print(a())" \
     >>"$LOG" 2>&1; then
  log "FATAL: checkpoint missing; see $LOG"
  exit 1
fi

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
declare -A streak=()

while (( $(date +%s) < deadline )); do
  claimed=""
  while IFS=, read -r idx used total; do
    idx=$(echo "$idx" | tr -d ' ')
    used=$(echo "$used" | tr -d ' MiB')
    total=$(echo "$total" | tr -d ' MiB')
    free=$(( total - used ))
    if (( free >= MIN_FREE_MIB )); then
      streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
      if (( ${streak[$idx]} >= STABLE_CHECKS )); then
        claimed=$idx
        break
      fi
    else
      streak[$idx]=0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)

  if [[ -n "$claimed" ]]; then
    log "claiming GPU $claimed"
    export CUDA_VISIBLE_DEVICES=$claimed
    # The process sees the claimed card as device 0.
    log "launching (output appended to $LOG)"
    if "$PYTHON" scripts/run_experiment.py "${ARGS[@]}" --device cuda:0 >>"$LOG" 2>&1; then
      log "RUN COMPLETED"
      exit 0
    else
      code=$?
      log "RUN FAILED with exit $code -- see $LOG"
      exit $code
    fi
  fi

  status=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')
  log "no card free yet ($status)"
  sleep "$POLL_SECONDS"
done

log "gave up after ${MAX_WAIT_HOURS}h without a free card"
exit 2
