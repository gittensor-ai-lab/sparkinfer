#!/bin/bash
# Shared setup for the serving head-to-head scripts in this directory. Source it; do not run it.
#
# Environment (all optional):
#   SI_BIN     sparkinfer_server binary          (default: <repo>/build/server/sparkinfer_server)
#   AIPERF     AIPerf CLI                        (default: aiperf on PATH)
#   VLLM       vLLM CLI                          (default: vllm on PATH)
#   LLAMA_SERVER  llama.cpp's llama-server       (default: llama-server on PATH)
#   OUT_ROOT   where result directories go       (default: ./results)
#   LOCK       flock file serializing GPU runs   (default: /tmp/sparkinfer_bot.lock; empty = none)
#   UNIQ       set to give each cell its own AIPerf seed (distinct prompts per cell; recommended)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SI_BIN=${SI_BIN:-$REPO/build/server/sparkinfer_server}
AIPERF=${AIPERF:-aiperf}
VLLM=${VLLM:-vllm}
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
OUT_ROOT=${OUT_ROOT:-$PWD/results}
LOCK=${LOCK-/tmp/sparkinfer_bot.lock}
SI_PORT=${SI_PORT:-18301}
VL_PORT=${VL_PORT:-18303}
LC_PORT=${LC_PORT:-18305}

# One GPU job at a time: a timed-out lock aborts the run instead of overlapping two servers.
take_lock() {
    [ -z "$LOCK" ] && return 0
    exec 9>"$LOCK"
    flock -w 3600 9 || { echo "LOCK_TIMEOUT ($LOCK)"; exit 1; }
}

wait_up() {  # port
    for _ in $(seq 1 600); do curl -sf "localhost:$1/health" >/dev/null && return 0; sleep 2; done
    return 1
}

# start_sparkinfer <model> <tokenizer dir> <ctx> <log> [extra server args...]; sets SP
start_sparkinfer() {
    local m=$1 tok=$2 ctx=$3 log=$4; shift 4
    "$SI_BIN" -m "$m" --tokenizer "$tok/tokenizer.json" --model-name q --ctx "$ctx" --port "$SI_PORT" "$@" > "$log" 2>&1 &
    SP=$!
    wait_up "$SI_PORT"
}

# start_vllm <model> <max-model-len> <log> [extra vllm serve args...]; sets VP
start_vllm() {
    local m=$1 len=$2 log=$3; shift 3
    # MAX_JOBS=4: flashinfer's JIT otherwise runs enough compilers to exhaust host memory.
    MAX_JOBS=4 "$VLLM" serve "$m" --served-model-name q --max-model-len "$len" \
        --max-num-batched-tokens 8192 --limit-mm-per-prompt '{"image":0,"video":0}' \
        --port "$VL_PORT" "$@" > "$log" 2>&1 &
    VP=$!
    wait_up "$VL_PORT"
}

# start_llamacpp <gguf> <total ctx> <slots> <log> [extra llama-server args...]; sets LP
# --ctx is the KV cache shared by all slots: give it slots x the longest request.
start_llamacpp() {
    local m=$1 ctx=$2 np=$3 log=$4; shift 4
    "$LLAMA_SERVER" -m "$m" -ngl 99 -fa on --jinja -np "$np" -c "$ctx" --port "$LC_PORT" "$@" > "$log" 2>&1 &
    LP=$!
    wait_up "$LC_PORT"
}

stop_server() {  # pid
    kill "$1" 2>/dev/null
    wait "$1" 2>/dev/null
    sleep 5
}

# aiperf_cell <outdir> <tag> <port> <tokenizer> <concurrency> <seed> [aiperf args...]
# Streaming chat; prints one summary line (see parse.py).
aiperf_cell() {
    local out=$1 tag=$2 port=$3 tok=$4 c=$5 seed=$6; shift 6
    local n=$(( c * 4 < 8 ? 8 : c * 4 ))
    "$AIPERF" profile -m q --url "http://127.0.0.1:$port" --endpoint-type chat --streaming \
        --concurrency "$c" --request-count "$n" --warmup-request-count "$c" \
        ${seed:+--random-seed $seed} --tokenizer "$tok" --artifact-dir "$out/$tag" "$@" \
        > "$out/$tag.log" 2>&1
    python3 "$HERE/parse.py" "$out/$tag" "$tag" "$c"
}
