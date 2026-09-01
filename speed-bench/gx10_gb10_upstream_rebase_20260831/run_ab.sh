#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/speed-bench/gx10_gb10_upstream_rebase_20260831"
MODEL="$ROOT/ds4flash.gguf"
PROMPT="$ROOT/speed-bench/promessi_sposi.txt"
BASELINE_BIN=${BASELINE_BIN:?set BASELINE_BIN to the ds4-bench built from baseline_revision}
CANDIDATE_BIN=${CANDIDATE_BIN:-$ROOT/ds4-bench}

if systemctl --user is-active --quiet ds4-gx10.service; then
    echo "ds4-gx10.service must be inactive before benchmarking" >&2
    exit 1
fi
if pgrep -f '(^|/)(ds4|ds4-server|ds4-agent|ds4-bench)( |$)' >/dev/null; then
    echo "another DS4 process is active" >&2
    exit 1
fi

run_one() {
    local name=$1
    local bin=$2
    DS4_BENCH_FORCE_SNAPSHOT=1 DS4_CUDA_DECODE_GRAPHS=0 \
        "$bin" --cuda -m "$MODEL" --prompt-file "$PROMPT" \
        --ctx-start 2048 --ctx-max 65536 --step-incr 2048 \
        --gen-tokens 128 --csv "$OUT/$name.csv" \
        >"$OUT/$name.stdout" 2>"$OUT/$name.stderr"
}

run_one baseline "$BASELINE_BIN"
run_one candidate "$CANDIDATE_BIN"
