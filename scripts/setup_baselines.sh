#!/usr/bin/env bash
# Clone upstream baseline repos into baselines/ (git-ignored).
#
# Keep these pristine -- never patch upstream in place. Anything PLEXUS needs to change goes in
# plexus/ as an adapter, so `git -C baselines/<repo> status --porcelain` stays empty and every
# reproduction number we report stays trustworthy.
#
# Note: TAG-DLM (arXiv 2606.31166) has NO public code release as of 2026-09-03. Our
# reimplementation lives in plexus/tagdlm/ and is validated against the paper's reported numbers
# in E0.1. Treat every TAG-DLM number we produce as "our reimplementation" until that gate passes.
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p baselines

clone() {
  local url=$1 dest=$2 why=$3
  if [[ -d "baselines/$dest/.git" ]]; then
    echo "  [skip]  $dest"
    return 0
  fi
  echo "  [clone] $dest  -- $why"
  if ! git clone --depth 1 -q "$url" "baselines/$dest" 2>/dev/null; then
    echo "  [FAIL]  $dest  ($url)  -- resolve manually, do not silently proceed"
    return 1
  fi
}

echo "Cloning baselines..."
clone https://github.com/ML-GSAI/LLaDA.git \
      LLaDA    "the TAG-DLM backbone: LLaDA-8B, generation + likelihood scoring reference"
clone https://github.com/VITA-Group/LLaGA.git \
      LLaGA    "strongest LLM+graph baseline; TAG-DLM borrows its dataset preprocessing"
clone https://github.com/kuleshov-group/mdlm.git \
      mdlm     "reference MDLM formulation and the SUBS parameterisation"
clone https://github.com/kuleshov-group/bd3lms.git \
      bd3lms   "block diffusion primitives, if the block-masking direction is taken up"
clone https://github.com/XiaoxinHe/TAPE.git \
      TAPE     "TAG benchmark preprocessing + the strong-text-feature GNN comparison"

echo
echo "Verify they are unmodified:"
echo '  for r in baselines/*/; do echo "$r"; git -C "$r" status --porcelain; done'
