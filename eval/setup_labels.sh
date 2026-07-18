#!/usr/bin/env bash
# One-time: create the eval:* labels the PR bot applies. Idempotent (--force upserts).
#   eval/setup_labels.sh [owner/repo]
set -euo pipefail
REPO="${1:-gittensor-ai-lab/sparkinfer}"
declare -A C=( [XL]=0E8A16 [L]=1D76DB [M]=5319E7 [S]=FBCA04 [XS]=BFD4F2
               [none]=C5DEF5 [REJECT]=B60205 [BASELINE]=D4C5F9 [infra-error]=F9A825 )
for k in "${!C[@]}"; do
  gh label create "eval:$k" -R "$REPO" --color "${C[$k]}" \
     --description "sparkinfer auto-eval verdict: $k" --force >/dev/null
done

# subsystem / emission-weight labels — assigned deterministically from changed paths (no AI)
declare -A AC=( [kernels]=006B75 [runtime]=0052CC [moe]=8250DF [bench]=C2E0C6 )
declare -A AW=( [kernels]=0.42 [runtime]=0.26 [moe]=0.21 [bench]=0.11 )
for k in "${!AC[@]}"; do
  gh label create "area:$k" -R "$REPO" --color "${AC[$k]}" \
     --description "subsystem (emission weight ${AW[$k]})" --force >/dev/null
done

# UI-only: the context where the PR showed its strongest measured improvement. This does not affect
# the eval score; label.py already computed the reward tier from the selected context.
declare -A CC=( [128-context]=D14D72 [512-context]=7B5DFF [4k-context]=0E8A16 [16k-context]=B8860B [32k-context]=6F42C1 )
for k in "${!CC[@]}"; do
  gh label create "$k" -R "$REPO" --color "${CC[$k]}" \
     --description "UI-only: strongest measured context in sparkinfer eval" --force >/dev/null
done

declare -A RC=( [regression-128]=F4A3A8 [regression-512]=E7828A [regression-4k]=D95D67 [regression-16k]=B60205 [regression-32k]=6A1B9A )
for k in "${!RC[@]}"; do
  gh label create "$k" -R "$REPO" --color "${RC[$k]}" \
     --description "sparkinfer eval regression marker for this context" --force >/dev/null
done
echo "eval:*, area:*, *-context, and regression-* labels ready on $REPO"
