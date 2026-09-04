#!/usr/bin/env bash
# Push code to the other training host, and verify the two now agree.
#
#   ./scripts/sync_hosts.sh ai-lab2
#
# Every exclude is anchored with a leading slash, and that is the whole point of this script
# existing rather than being an rsync line typed from memory. An unanchored `--exclude data`
# matches *any* directory named `data` at any depth, so it silently skipped `plexus/data/` -- the
# package holding the dataset loaders -- while reporting success. The remote ran stale loader code
# for hours and only a test import failure revealed it.
#
# With arms of one experiment split across two machines, code drift between hosts is
# indistinguishable from an effect of the mask being tested, so this script ends by comparing
# checksums rather than trusting rsync's exit status.
set -uo pipefail

cd "$(dirname "$0")/.."
REMOTE=${1:-ai-lab2}
REMOTE_DIR=${2:-projects/plexus}

# No --delete and no --delete-excluded. --delete-excluded deletes the *excluded* paths from the
# destination, which is the opposite of what excluding them is for: it wiped the remote's venv,
# its 15 GB checkpoint, its logs and its experiments/ directory, destroying two eight-hour runs
# that had already finished. Excluding a path here means "this host owns its own copy", so this
# script only ever adds and overwrites.
rsync -az \
  --exclude '/.git' \
  --exclude '/.venv' \
  --exclude '/experiments' \
  --exclude '/logs' \
  --exclude '/baselines' \
  --exclude '/checkpoints' \
  --exclude '/data' \
  --exclude '/submission/main.pdf' \
  --exclude '__pycache__' \
  --exclude '*.egg-info' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  ./ "$REMOTE:$REMOTE_DIR/"

echo "rsync done; verifying the two hosts agree on every tracked .py"

# `find`, not `git ls-files`: the remote is a plain directory, since .git is not transferred.
# Listing files the same way on both sides is the only comparison that means anything.
LIST='find plexus scripts tests -name "*.py" -not -path "*/__pycache__/*" | sort | xargs md5sum'

local_sums=$(eval "$LIST")
remote_sums=$(ssh -o BatchMode=yes "$REMOTE" "cd $REMOTE_DIR && $LIST")

if [[ "$local_sums" == "$remote_sums" ]]; then
  n=$(echo "$local_sums" | wc -l)
  echo "OK: $n python files identical on both hosts"
else
  echo "MISMATCH between local and $REMOTE:" >&2
  diff <(echo "$local_sums") <(echo "$remote_sums") >&2
  exit 1
fi
