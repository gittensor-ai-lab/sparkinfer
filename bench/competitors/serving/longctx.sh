#!/bin/bash
# Long prompts, one request at a time: time to first token (prefill) and decode speed.
# 128-token answers, ignore_eos, prefix caching off on both engines so every prompt is prefilled.
#
#   SI_M=<sparkinfer model> VL_M=<vLLM model> TOK=<tokenizer dir> ./longctx.sh
#
# LENS ("32768 65536 120000"), WHICH ("sparkinfer vllm", or "sparkinfer llamacpp" for a GGUF-only model), AIP_EXTRA (see cells.sh).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${SI_M:?SI_M: the model sparkinfer serves}"; VL_M=${VL_M:-$SI_M}; TOK=${TOK:-$SI_M}
WHICH=${WHICH:-"sparkinfer vllm"}; LENS=${LENS:-"32768 65536 120000"}
OUT=$OUT_ROOT/longctx_$(date +%m%d_%H%M); mkdir -p "$OUT"
take_lock

cell() {  # engine port isl
    aiperf_cell "$OUT" "$1_L$3" "$2" "$TOK" 1 "$(( $3 + 7 ))" \
        --synthetic-input-tokens-mean "$3" --synthetic-input-tokens-stddev 0 \
        --output-tokens-mean 128 --output-tokens-stddev 0 \
        --extra-inputs ignore_eos:true --extra-inputs max_tokens:128 $AIP_EXTRA
}

if [[ $WHICH == *sparkinfer* ]]; then
    SPARKINFER_PREFIX_CACHE=0 start_sparkinfer "$SI_M" "$TOK" 131072 "$OUT/sparkinfer_srv.log" &&
        for L in $LENS; do cell sparkinfer "$SI_PORT" "$L"; done
    stop_server "$SP"
fi
if [[ $WHICH == *vllm* ]]; then
    start_vllm "$VL_M" 131072 "$OUT/vllm_srv.log" --gpu-memory-utilization 0.92 --max-num-seqs 16 \
        --no-enable-prefix-caching && for L in $LENS; do cell vllm "$VL_PORT" "$L"; done
    stop_server "$VP"
fi
if [[ $WHICH == *llamacpp* ]]; then
    # One slot holding the longest prompt; --no-cache-prompt matches the other engines' cache-off.
    start_llamacpp "${LC_M:-$SI_M}" 131072 1 "$OUT/llamacpp_srv.log" --no-cache-prompt &&
        for L in $LENS; do cell llamacpp "$LC_PORT" "$L"; done
    stop_server "$LP"
fi
echo "ALLDONE $OUT"
