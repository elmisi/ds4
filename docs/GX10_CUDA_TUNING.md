# DGX Spark / GB10 CUDA tuning notes

These notes record the CUDA tuning pass run on an ASUS Ascent GX10 / NVIDIA
GB10 with another GPU workload (`sd-server`) left resident. The goal was to
improve generated tokens/sec without materially hurting prefill or increasing
memory-crash risk.

## Baseline

The baseline used `ds4-bench` with the public-domain *I Promessi Sposi* prompt:

```sh
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 \
  --gen-tokens 128 \
  --csv tuning/gx10-20260514/baseline_default.csv
```

With `sd-server` using about 10.5 GiB GPU memory, the baseline averaged:

- prefill: 366.99 t/s
- generation: 13.16 t/s

## Accepted changes

- GB10/sm_121 now uses the regular unordered one-token F16 matmul path by
  default, including the F16 pair helper. The previous ordered F16 decode
  kernels remain available with `DS4_CUDA_FORCE_ORDERED_F16_MATMUL=1`.
- Several CUDA hot-path environment switches are cached during backend
  initialization instead of being read repeatedly during decode.
- CUDA uses `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=0` by default. The old split
  default is kept on Apple/Metal, where it is useful for command scheduling.

The full env-equivalent sweep for the unordered F16 path improved average
generation from 13.16 t/s to 13.89 t/s while keeping prefill effectively flat.
After the F16 pair helper was aligned with the same GB10 decision, the local
8k/64 check measured 14.44 t/s generation. A post-rebase check on the updated
main measured 14.26 t/s generation at the same 8k/64 frontier.

## Rejected knobs

The following local knobs were tested and were slower or not useful for this
machine:

- `DS4_CUDA_SERIAL_ROUTER=1`
- `DS4_CUDA_NO_WARP_ROUTER_SELECT=1`
- `DS4_CUDA_MOE_NO_DECODE_LUT_GATE=1`
- `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6=1`
- `DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR=1`
- `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=72`
- `DS4_CUDA_DIRECT_MODEL=1`
- `DS4_CUDA_DISABLE_QKV_RMS_FUSED=1`
- `DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN=5`

## Fork scan

The relevant fork ideas were:

- `spmurrayzzz/ds4`: CUDA decode fast-path experiments. The CUDA no-split idea
  was safe to port; the larger patches target an older baseline and overlap
  with current upstream work.
- `reffdev/ds4` branch `fused-matmul-mtp`: MTP/tiny-batch fused matmul work.
  It is useful background for MTP, but not for the current no-MTP greedy decode
  bottleneck.
- `Entrpi/ds4-on-spark`: useful GX10/Spark methodology notes, including the
  difference between first-token-inclusive CLI/bench numbers and pure
  steady-state decode measurements.

## MTP

The optional MTP GGUF loaded and ran, but did not help on this setup. With
`--mtp-draft 2`, a 4k/64 CLI session test measured 3.66 t/s with speculative
decode disabled and 3.70 t/s with speculative decode enabled. Timing logs showed
frequent first-draft misses or one-token commits and verifier costs around
72-139 ms per speculative step, so MTP is not recommended for GB10 production
use yet.

## Verification

The branch was verified with:

```sh
make cuda-spark CUDA_HOME=/usr/local/cuda-13.0
make ds4_test CUDA_HOME=/usr/local/cuda-13.0
./ds4_test --server
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 \
  --gen-tokens 64
```
