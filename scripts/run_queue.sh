#!/usr/bin/env bash
# Work through a queue of PLEXUS jobs, waiting for a nearly empty GPU before each one.
#
# The one-shot watcher caught a single window and exited. On a machine where cards free for
# minutes at a time, the scarce resource is *windows*, so the queue persists: it takes the next
# unfinished job, waits, runs it, records the outcome, and moves on. Jobs are ordered so that the
# cheapest and most decisive run first -- if only one window ever appears, we want the timing
# benchmark and the fulcrum comparison, not the tail of a seed sweep.
#
#   ./scripts/run_queue.sh scripts/queue_e0.txt
#
# Queue file format: one job per line, `<name> <TAB or spaces> <command line>`. Blank lines and
# `#` comments ignored. A job whose marker file exists in logs/queue/ is skipped, so the queue is
# resumable and safe to relaunch.
#
# Several workers may share one queue file. Each claims a job by creating a lock *directory*, which
# is atomic, so two workers never take the same job. Pin each worker to its own card:
#
#   ONLY_GPU=0 ./scripts/run_queue.sh scripts/queue_e0.txt &
#   ONLY_GPU=1 ./scripts/run_queue.sh scripts/queue_e0.txt &
#
# A worker killed mid-job leaks its lock; clear it with `rmdir logs/queue/<name>.lock`.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
QUEUE=${1:-scripts/queue_e0.txt}

MIN_FREE_MIB=${MIN_FREE_MIB:-21000}
STABLE_CHECKS=${STABLE_CHECKS:-3}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-96}
ONLY_GPU=${ONLY_GPU:-}
WORKER=${WORKER:-${ONLY_GPU:-0}}
FAST_FAIL_SECONDS=${FAST_FAIL_SECONDS:-120}
MAX_FAST_FAILURES=${MAX_FAST_FAILURES:-2}

export HF_HOME=${HF_HOME:-/tmp/adelechinda/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PLEXUS_DATA_ROOT=${PLEXUS_DATA_ROOT:-/tmp/adelechinda/plexus_data}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
# Do not rely on `pip install -e .` having been run in whichever interpreter this host uses.
# A rebuilt venv without it fails at the first import, and because that failure takes one second
# the queue then burns through every remaining job at one second each and marks them all attempted.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# ai-lab2 has a venv in the repo; titan3 runs from a conda env. Both hosts must use interpreters
# with the same torch build, or a difference between arms could be a difference between libraries.
PYTHON=${PLEXUS_PYTHON:-"$REPO/.venv/bin/python"}
if [[ ! -x "$PYTHON" ]]; then
  echo "no interpreter at $PYTHON; set PLEXUS_PYTHON" >&2
  exit 1
fi
MARKS="$REPO/logs/queue"
mkdir -p "$MARKS" "$REPO/logs"
MAIN_LOG="$REPO/logs/queue_w${WORKER}_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] [w$WORKER] $*" | tee -a "$MAIN_LOG"; }

# A reboot empties /tmp (E0.0f). Fail now rather than after waiting hours for a card.
if ! "$PYTHON" -c "from plexus.models.llada import assert_snapshot_present as a; a()" \
     >>"$MAIN_LOG" 2>&1; then
  log "FATAL: checkpoint missing"
  exit 1
fi

# Wait for a card and echo its index, or return 1 on timeout.
claim_gpu() {
  local deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
  declare -A streak=()
  while (( $(date +%s) < deadline )); do
    while IFS=, read -r idx used total; do
      idx=${idx// /}; used=${used// MiB/}; total=${total// MiB/}
      used=${used// /}; total=${total// /}
      [[ -n "$ONLY_GPU" && "$idx" != "$ONLY_GPU" ]] && continue
      if (( total - used >= MIN_FREE_MIB )); then
        streak[$idx]=$(( ${streak[$idx]:-0} + 1 ))
        if (( ${streak[$idx]} >= STABLE_CHECKS )); then echo "$idx"; return 0; fi
      else
        streak[$idx]=0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader)
    sleep "$POLL_SECONDS"
  done
  return 1
}

total=0; done_already=0; ran=0; failed=0; fast_failures=0
while IFS= read -r line; do
  [[ -z "${line// /}" || "${line#\#}" != "$line" ]] && continue
  name=$(echo "$line" | awk '{print $1}')
  cmd=$(echo "$line" | cut -d' ' -f2- | sed 's/^[[:space:]]*//')
  total=$(( total + 1 ))

  if [[ -f "$MARKS/$name.done" ]]; then
    log "skip $name (already done)"
    done_already=$(( done_already + 1 ))
    continue
  fi

  # mkdir is the claim: it succeeds for exactly one worker even if several test the same job in
  # the same instant, which a -e test followed by a touch would not.
  if ! mkdir "$MARKS/$name.lock" 2>/dev/null; then
    log "skip $name (claimed by another worker)"
    done_already=$(( done_already + 1 ))
    continue
  fi
  echo "$WORKER $$ $(date '+%F %T')" > "$MARKS/$name.lock/owner"

  log "next job: $name"
  log "  waiting for a card with >= ${MIN_FREE_MIB} MiB free"
  if ! gpu=$(claim_gpu); then
    log "gave up waiting after ${MAX_WAIT_HOURS}h; ${total} jobs seen, stopping"
    exit 2
  fi
  log "  claimed GPU $gpu"

  job_log="$REPO/logs/${name}.log"
  start=$(date +%s)
  if CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" $cmd --device cuda:0 >"$job_log" 2>&1; then
    log "  $name COMPLETED in $(( $(date +%s) - start ))s -> $job_log"
    date > "$MARKS/$name.done"
    ran=$(( ran + 1 ))
  else
    code=$?
    elapsed=$(( $(date +%s) - start ))
    log "  $name FAILED exit $code after ${elapsed}s -> $job_log"
    log "  $(tail -3 "$job_log" | tr '\n' ' ')"
    failed=$(( failed + 1 ))
    # A job that dies in seconds died on import or on argument parsing, which is a property of
    # this host and not of this job. Continuing would consume the whole queue at a second apiece
    # and leave every job marked attempted, which is exactly what happened once. Slow failures
    # are different -- those are per-job, and the queue should carry on past them.
    if (( elapsed < FAST_FAIL_SECONDS )); then
      fast_failures=$(( fast_failures + 1 ))
      if (( fast_failures >= MAX_FAST_FAILURES )); then
        log "stopping: $fast_failures failures in under ${FAST_FAIL_SECONDS}s each look like a"
        log "broken environment rather than broken jobs. Fix it, clear logs/queue/*.lock, rerun."
        exit 3
      fi
    else
      fast_failures=0
    fi
  fi
done < "$QUEUE"

log "queue finished: $ran ran, $failed failed, $done_already skipped, $total total"
exit 0
