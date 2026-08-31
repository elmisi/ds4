# GB10 upstream rebase validation — 2026-08-31

This package validates the strict fast-forward-plus-rebase workflow used on
this machine. Upstream-only `main` advanced from `c1d4597` to `ec7642c`, then
the 35 `dgx-performance` commits were rebased from `4caae72` to `35ae260`.
No merge commit was created. The recoverable pre-rebase ref is
`backup/dgx-performance-pre-rebase-20260831`.

The upstream range is dominated by GLM 5.3 and ROCm work. The directly useful
DGX additions are CUDA GLM prefill acceleration, integrated-CUDA vision weight
mapping, fixed-length greedy server decode, and live steering. Those features
are not assumed to speed up the operational DeepSeek V4 Flash model; the A/B
below measures the complete rebased binary on that exact model instead.

## Performance verdict

The identical 32-frontier sweep measured a **+1.817% mean prefill** change and
a **-0.281% mean generation** change. Because the generation delta was small,
a second alternating-order test used three fresh processes per revision at
32,768 context tokens and 256 generated tokens. That controlled check measured
**+2.856% mean prefill** and **+0.498% mean generation** for the candidate.
The small negative generation value in the first sweep is therefore not a
reproducible regression.

| Context | Baseline prefill | Candidate prefill | Baseline generation | Candidate generation |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 821.02 t/s | 839.92 t/s | 22.08 t/s | 21.95 t/s |
| 16,384 | 866.29 t/s | 883.43 t/s | 17.84 t/s | 17.79 t/s |
| 32,768 | 850.14 t/s | 866.24 t/s | 16.96 t/s | 16.91 t/s |
| 65,536 | 819.97 t/s | 834.42 t/s | 16.12 t/s | 16.10 t/s |

All 32 KV-cache fields match. Both legs used 474 aligned CUDA artifacts
(78.71 GiB), the same 0731 GGUF and prompt, forced post-frontier snapshots,
and `DS4_CUDA_DECODE_GRAPHS=0`. No fallback, OOM, illegal access, capture
failure, or non-finite value appears in the logs.

## Quality gate

- `make -j8 cuda CUDA_ARCH=sm_121` produced all five ARM64 CUDA binaries;
  `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent` pass `--help`.
- CUDA long-context, fused-decode, split-K, official logprob-vector, and local
  golden-vector tests pass.
- The 4,096-token local golden vector retains top-1 identity, 5/5 top-5 and
  20/20 top-20 agreement.
- A real greedy CUDA prompt returns exactly `DGX_REBASE_OK`.
- The operational service was inactive before the work and remains inactive;
  no benchmark overlapped another DS4 process.

## Reproduction

Build an exact baseline `ds4-bench` at `4caae72`, build the candidate at
`35ae260`, then run:

```sh
BASELINE_BIN=/path/to/4caae72/ds4-bench \
  CANDIDATE_BIN=$PWD/ds4-bench \
  ./speed-bench/gx10_gb10_upstream_rebase_20260831/run_ab.sh

BASELINE_BIN=/path/to/4caae72/ds4-bench \
  CANDIDATE_BIN=$PWD/ds4-bench \
  ./speed-bench/gx10_gb10_upstream_rebase_20260831/run_decode_repeat.sh

./speed-bench/gx10_gb10_upstream_rebase_20260831/summarize.sh
```

Raw CSV/stdout/stderr, `comparison.csv`, `summary.csv`, SVG plots, test logs,
binary checksums, exact commands, and host/model provenance are retained here.
