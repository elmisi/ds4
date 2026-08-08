#!/usr/bin/env bash
# Reproduce the current upstream GB10 README sweep on dgx-performance.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
report_dir=${1:-"$root/speed-bench/gx10_gb10_dgx_performance_main_style_20260808"}
model="$root/ds4flash.gguf"
prompt="$root/speed-bench/promessi_sposi.txt"

mkdir -p "$report_dir"
cd "$root"

for process in ds4-server ds4-agent ds4 ds4-bench; do
    if pgrep -x "$process" >/dev/null; then
        echo "refusing benchmark: $process is already running" >&2
        exit 1
    fi
done

{
    echo "revision=$(git rev-parse HEAD)"
    echo "branch=$(git branch --show-current)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "model=$(readlink -f "$model")"
    echo "prompt=$(readlink -f "$prompt")"
    echo "build=CUDA_ARCH=sm_121"
    echo "command=DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128"
    nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu --format=csv,noheader
} >"$report_dir/provenance.txt"

make -B -j8 ds4-bench CUDA_ARCH=sm_121 >"$report_dir/build.log" 2>&1

DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench \
    -m ds4flash.gguf \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 2048 \
    --ctx-max 65536 \
    --step-incr 2048 \
    --gen-tokens 128 \
    >"$report_dir/dgx_performance.csv" \
    2>"$report_dir/benchmark.log"

expected_header='ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes'
if [[ $(head -n 1 "$report_dir/dgx_performance.csv") != "$expected_header" ]]; then
    echo "unexpected CSV header; see $report_dir/benchmark.log" >&2
    exit 1
fi
if [[ $(($(wc -l <"$report_dir/dgx_performance.csv") - 1)) -ne 32 ]]; then
    echo "expected 32 benchmark rows; see $report_dir/benchmark.log" >&2
    exit 1
fi

python3 speed-bench/plot_speed.py "$report_dir/dgx_performance.csv" \
    --title "DGX Spark GB10 dgx-performance t/s" \
    >"$report_dir/plot.log" 2>&1

awk -F, 'BEGIN { print "| Context | Prefill | Generation |"; print "| ---: | ---: | ---: |" } NR > 1 && ($1 == 2048 || $1 == 16384 || $1 == 32768 || $1 == 65536) { printf "| %s | %s t/s | %s t/s |\\n", $1, $3, $5 }' \
    "$report_dir/dgx_performance.csv" >"$report_dir/readme_table.md"

date --iso-8601=seconds >>"$report_dir/provenance.txt"
echo "report_dir=$report_dir"
