#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${BASELINE_BIN:-}" || -z "${CANDIDATE_BIN:-}" ]]; then
    echo "usage: BASELINE_BIN=/path/to/old/ds4-bench CANDIDATE_BIN=/path/to/new/ds4-bench $0" >&2
    exit 2
fi

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
artifact_dir=$(cd "$(dirname "$0")" && pwd)
model=${MODEL:-$root_dir/ds4flash.gguf}
prompt=${PROMPT_FILE:-$root_dir/speed-bench/promessi_sposi.txt}

if [[ $(systemctl --user show ds4-gx10.service -p ActiveState --value) != inactive ]]; then
    echo "ds4-gx10.service must remain inactive during the benchmark" >&2
    exit 1
fi

if pgrep -af '/(ds4|ds4-server|ds4-agent|ds4-bench)( |$)' >/dev/null; then
    echo "a DS4 process is already running" >&2
    exit 1
fi

run_one() {
    local label=$1
    local iteration=$2
    local binary=$3
    local csv="$artifact_dir/decode_${label}_${iteration}.csv"
    local stderr="$artifact_dir/decode_${label}_${iteration}.stderr"

    echo "[$(date --iso-8601=seconds)] $label iteration $iteration"
    DS4_BENCH_FORCE_SNAPSHOT=1 \
    DS4_CUDA_DECODE_GRAPHS=0 \
        "$binary" --cuda -m "$model" --prompt-file "$prompt" \
        --ctx-start 32768 --ctx-max 32768 --step-incr 2048 \
        --gen-tokens 256 --csv "$csv" \
        >"$artifact_dir/decode_${label}_${iteration}.stdout" 2>"$stderr"
}

# Alternate the order to reduce bias from temperature and background drift.
for iteration in 1 2 3; do
    if (( iteration % 2 == 1 )); then
        run_one baseline "$iteration" "$BASELINE_BIN"
        run_one candidate "$iteration" "$CANDIDATE_BIN"
    else
        run_one candidate "$iteration" "$CANDIDATE_BIN"
        run_one baseline "$iteration" "$BASELINE_BIN"
    fi
done

echo "[$(date --iso-8601=seconds)] repeated decode benchmark complete"
