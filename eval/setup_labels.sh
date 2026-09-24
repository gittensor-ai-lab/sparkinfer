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
  gh label create "eval-prefill:$k" -R "$REPO" --color "${C[$k]}" \
     --description "sparkinfer prefill speed tier: $k (may annotate REJECT headlines)" --force >/dev/null
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

# DFlash speculative-decode bot (eval/pr_dflash_bot.py) — separate from AR eval:*
for k in "${!C[@]}"; do
  gh label create "eval-dflash:$k" -R "$REPO" --color "${C[$k]}" \
     --description "sparkinfer DFlash vs-main speed tier: $k" --force >/dev/null
done
gh label create "dflash-merge-first" -R "$REPO" --color "0E8A16" \
   --description "round winner: biggest verified DFlash speedup — auto-merge candidate" --force >/dev/null
gh label create "dflash-needs-rebase" -R "$REPO" --color "FBCA04" \
   --description "verified DFlash speedup but not this round's dflash-merge-first" --force >/dev/null

# Muse Glimmer bot (eval/pr_museglimmer_bot.py) — 128-token AR decode only, scored + accuracy-gated
# separately from eval:*/eval-dflash:* (see pr_museglimmer_bot.py's apply_result() for why it does
# NOT mirror to eval:*).
for k in "${!C[@]}"; do
  gh label create "eval-museglimmer:$k" -R "$REPO" --color "${C[$k]}" \
     --description "sparkinfer Muse Glimmer 128-decode vs-main speed tier: $k" --force >/dev/null
done
gh label create "museglimmer-merge-first" -R "$REPO" --color "0E8A16" \
   --description "round winner: biggest verified Muse Glimmer 128-decode speedup — auto-merge candidate" --force >/dev/null
gh label create "museglimmer-needs-rebase" -R "$REPO" --color "FBCA04" \
   --description "conflicts with main — rebase before the Muse Glimmer bot can evaluate it" --force >/dev/null

# Qwen3.8-27B bot (eval/pr_qwen38_bot.py) — decode@128 on the NVFP4 checkpoint, differential
# accuracy gate, Qwen3.6 no-regression guard. Mirrors its tier to eval:* (SN74 scoring reads
# eval:*), unlike the Muse Glimmer labels above.
for k in "${!C[@]}"; do
  gh label create "eval-qwen38:$k" -R "$REPO" --color "${C[$k]}" \
     --description "sparkinfer Qwen3.8-27B decode@128 vs-main speed tier: $k" --force >/dev/null
done
gh label create "qwen38-merge-first" -R "$REPO" --color "0E8A16" \
   --description "round winner: biggest verified Qwen3.8-27B decode@128 speedup — auto-merge candidate" --force >/dev/null
gh label create "qwen38-needs-rebase" -R "$REPO" --color "FBCA04" \
   --description "conflicts with main — rebase before the Qwen3.8-27B bot can evaluate it" --force >/dev/null

# Ternary-Bonsai-2-27B bot (eval/pr_bonsai_bot.py) — decode + prefill at 128..32k and concurrent
# decode on the PTQ1_0 GGUF, differential accuracy gates. Mirrors its tier to eval:*.
for k in "${!C[@]}"; do
  gh label create "eval-bonsai:$k" -R "$REPO" --color "${C[$k]}" \
     --description "sparkinfer Ternary-Bonsai-2-27B vs-main speed tier: $k" --force >/dev/null
done
gh label create "bonsai-merge-first" -R "$REPO" --color "0E8A16" \
   --description "round winner: biggest verified Ternary-Bonsai-2-27B speedup — auto-merge candidate" --force >/dev/null
gh label create "bonsai-needs-rebase" -R "$REPO" --color "FBCA04" \
   --description "conflicts with main, or lost this round's bonsai-merge-first" --force >/dev/null

# Noise-ban penalty labels (`eval:XL-p`, `eval-qwen38:M-p`, ...) are deliberately NOT created
# here. eval/noise_penalty.py upserts one the first time a ban actually parks that tier, so the
# repo only ever carries penalty labels it has really used -- which is also what keeps the
# restore sweep cheap (it walks the penalty labels that exist). See .github/noise-ban-list.txt.

echo "eval:*, eval-dflash:*, eval-museglimmer:*, eval-qwen38:*, eval-bonsai:*, area:*, *-context, regression-*, dflash-merge-*, museglimmer-merge-*, qwen38-merge-*, bonsai-* labels ready on $REPO"
