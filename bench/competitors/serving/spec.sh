#!/bin/bash
# Speculative decoding at low concurrency: Qwen3.8 + a DFlash draft on both engines, plus
# sparkinfer without the draft for reference. Sampled chat (T=0.7, top_k 20, top_p 0.95),
# ignore_eos, 1 / 2 / 4 / 8 concurrent requests.
#
#   SI_M=<target> DRAFT=<draft dir> UNIQ=1 ./spec.sh
#
# DATASET=synthetic (default: 1024 -> 256 tokens) or sharegpt (AIPerf's public set; it sets each
# answer's length). ARMS ("si_draft si_plain vllm_draft"). vLLM keeps the bf16 draft resident beside
# its KV cache, so it runs at --max-model-len 6144 and 0.95 memory (it does not start at 8K or more).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${SI_M:?SI_M: the target model}"; : "${DRAFT:?DRAFT: the draft model directory}"
VL_M=${VL_M:-$SI_M}; TOK=${TOK:-$SI_M}
ARMS=${ARMS:-"si_draft si_plain vllm_draft"}; DATASET=${DATASET:-synthetic}
OUT=$OUT_ROOT/spec_${DATASET}_$(date +%m%d_%H%M); mkdir -p "$OUT"
take_lock

if [ "$DATASET" = sharegpt ]; then
    data=(--public-dataset sharegpt)
else
    data=(--synthetic-input-tokens-mean 1024 --synthetic-input-tokens-stddev 0)
fi
run() {  # arm port
    for c in 1 2 4 8; do
        aiperf_cell "$OUT" "$1_c$c" "$2" "$TOK" "$c" "${UNIQ:+$(( 500 + c ))}" "${data[@]}" \
            --output-tokens-mean 256 --output-tokens-stddev 0 \
            --extra-inputs ignore_eos:true --extra-inputs max_tokens:256 \
            --extra-inputs temperature:0.7 --extra-inputs top_p:0.95 --extra-inputs top_k:20 $AIP_EXTRA
    done
}

for arm in $ARMS; do
    case $arm in
    si_draft)  start_sparkinfer "$SI_M" "$TOK" 32768 "$OUT/${arm}_srv.log" --draft-model "$DRAFT" && run $arm "$SI_PORT"; stop_server "$SP" ;;
    si_plain)  start_sparkinfer "$SI_M" "$TOK" 32768 "$OUT/${arm}_srv.log" && run $arm "$SI_PORT"; stop_server "$SP" ;;
    vllm_draft)
        start_vllm "$VL_M" 6144 "$OUT/vllm_srv.log" --gpu-memory-utilization 0.95 --max-num-seqs 64 \
            --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":7}" &&
            run $arm "$VL_PORT"
        stop_server "$VP" ;;
    esac
done
echo "ALLDONE $OUT"
