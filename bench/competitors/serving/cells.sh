#!/bin/bash
# The standard serving cells, sparkinfer against vLLM, one server per engine:
#   chat      1024 -> 256 tokens at 4 / 16 / 32 concurrent requests
#   8k        8192 -> 256 tokens at 4 / 16
# AIPerf streaming chat, ignore_eos (fixed-length answers), default sampling, no draft.
#
#   SI_M=<sparkinfer model> VL_M=<vLLM model> TOK=<tokenizer dir> UNIQ=1 ./cells.sh
#
# A GGUF-only model runs against llama.cpp instead: WHICH="sparkinfer llamacpp" (LC_M, default SI_M;
# 32 slots sharing LC_CTX tokens of KV, default 32 x 8,704). AIP_EXTRA adds AIPerf arguments to
# every cell, e.g. a reasoning model whose thinking a server hides unless the request asks for it:
#   AIP_EXTRA='--extra-inputs chat_template_kwargs:{"enable_thinking":true}'
# (AIPerf counts the streamed text, so hidden reasoning would read as missing output.)
#
# More environment: WHICH ("sparkinfer vllm"), ONLYCHAT / ONLY8K, DECODE_ONLY (32 -> 256 tokens at
# 16 / 32, pure decode), SI_EXTRA (extra server args, e.g. "--draft-model DIR"), VL_EXTRA (extra
# vllm serve args, e.g. "--kv-cache-dtype fp8"), CTX (32768).
# See common.sh for binaries and paths.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${SI_M:?SI_M: the model sparkinfer serves}"; VL_M=${VL_M:-$SI_M}; TOK=${TOK:-$SI_M}
[ -d "$TOK" ] || { echo "TOK must be a directory with tokenizer.json"; exit 2; }
CTX=${CTX:-32768}; WHICH=${WHICH:-"sparkinfer vllm"}; VL_EXTRA=${VL_EXTRA:-}
OUT=$OUT_ROOT/cells_$(date +%m%d_%H%M); mkdir -p "$OUT"
take_lock

cell() {  # engine port name isl osl c
    aiperf_cell "$OUT" "$1_$3_c$6" "$2" "$TOK" "$6" "${UNIQ:+$(( $4 * 100 + $6 ))}" \
        --synthetic-input-tokens-mean "$4" --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean "$5" --output-tokens-stddev 0 \
        --extra-inputs ignore_eos:true --extra-inputs "max_tokens:$5" $AIP_EXTRA
}
run_load() {  # engine port
    if [ -n "$DECODE_ONLY" ]; then for c in 16 32; do cell "$1" "$2" tiny 32 256 $c; done; return; fi
    [ -z "$ONLY8K" ] && for c in 4 16 32; do cell "$1" "$2" chat 1024 256 $c; done
    [ -z "$ONLYCHAT" ] && for c in 4 16; do cell "$1" "$2" 8k 8192 256 $c; done
}

if [[ $WHICH == *sparkinfer* ]]; then
    # shellcheck disable=SC2086
    start_sparkinfer "$SI_M" "$TOK" "$CTX" "$OUT/sparkinfer_srv.log" $SI_EXTRA && run_load sparkinfer "$SI_PORT"
    stop_server "$SP"
fi
if [[ $WHICH == *vllm* ]]; then
    # shellcheck disable=SC2086
    start_vllm "$VL_M" "$CTX" "$OUT/vllm_srv.log" --gpu-memory-utilization 0.92 --max-num-seqs 64 $VL_EXTRA &&
        run_load vllm "$VL_PORT"
    stop_server "$VP"
fi
if [[ $WHICH == *llamacpp* ]]; then
    start_llamacpp "${LC_M:-$SI_M}" "${LC_CTX:-278528}" 32 "$OUT/llamacpp_srv.log" && run_load llamacpp "$LC_PORT"
    stop_server "$LP"
fi
echo "ALLDONE $OUT"
