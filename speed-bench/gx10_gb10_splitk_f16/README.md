# GX10 / GB10 F16 split-K experiment

This benchmark evaluates an Entrpi-inspired split-K F16 decode matmul path on
the `dgx-performance` branch.

The implementation adds a deterministic two-pass split-K kernel for one-token
F16 matmuls and a 4 MiB CUDA scratch buffer allocated at `ds4_gpu_init()` time.
It is intentionally **opt-in** via:

```sh
DS4_CUDA_F16_SPLITK=1
```

The final branch default remains the existing unordered one-token F16 matmul,
because the real-model benchmark below shows that it is faster on this branch.
`DS4_CUDA_ORDERED_F16_MATMUL=1` is retained for legacy ordered A/B checks.

## Scope

This tranche borrows the split-K idea and benchmark discipline from Entrpi's
DGX Spark work. It does not port Entrpi's full per-layer CUDA Graph capture:
that path changes the graph ABI, device-side live scalars, capture streams, and
many `ds4.c` call sites. Importing it into this fused-HC branch should be a
separate patch with its own eager-vs-captured token and logprob gates.

## Method

Branch revision at benchmark time:

```text
a1edf72 docs: add DGX benchmark data
```

Build:

```sh
make ds4-bench
```

Model:

```text
/home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```

Benchmark shape:

```sh
DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench --cuda \
  -m /home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 8192 \
  --ctx-alloc 65700 \
  --gen-tokens 128 \
  --csv <output.csv>
```

Variants:

| CSV | Runtime flags | Meaning |
| --- | --- | --- |
| [`no_splitk.csv`](no_splitk.csv) | `DS4_CUDA_NO_F16_SPLITK=1` | Final branch default behavior after the gate change |
| [`default_splitk.csv`](default_splitk.csv) | `DS4_CUDA_F16_SPLITK=1` equivalent; collected before the final gate change | Split-K opt-in |
| [`ordered_f16.csv`](ordered_f16.csv) | `DS4_CUDA_ORDERED_F16_MATMUL=1` | Legacy ordered F16 path |

All runs reported `80.24 GiB` of model tensor spans covered by the CUDA model
cache and `1743.17 MiB` of context buffers at `ctx=65700`.

## Results

| Context | Final default gen t/s | Split-K gen t/s | Ordered gen t/s | Split-K vs final | Final vs ordered | KV equal |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2,048 | 18.92 | 18.63 | 18.19 | -1.53% | +4.01% | yes |
| 10,240 | 16.03 | 15.81 | 15.48 | -1.37% | +3.55% | yes |
| 18,432 | 15.98 | 15.77 | 15.44 | -1.31% | +3.50% | yes |
| 26,624 | 15.81 | 15.60 | 15.28 | -1.33% | +3.47% | yes |
| 34,816 | 15.19 | 15.00 | 14.70 | -1.25% | +3.33% | yes |
| 43,008 | 15.03 | 14.85 | 14.56 | -1.20% | +3.23% | yes |
| 51,200 | 14.88 | 14.70 | 14.43 | -1.21% | +3.12% | yes |
| 59,392 | 14.72 | 14.54 | 14.26 | -1.22% | +3.23% | yes |
| 65,536 | 14.61 | 14.42 | 14.15 | -1.30% | +3.25% | yes |

Average generation throughput across the nine frontiers:

| Variant | Avg gen t/s | Avg prefill t/s |
| --- | ---: | ---: |
| Final default | 15.69 | 332.63 |
| Split-K opt-in | 15.48 | 331.77 |
| Ordered F16 | 15.17 | 331.97 |

Split-K is useful as an opt-in diagnostic and still beats the ordered legacy
path by **+3.3%** on average, but it is **-1.3%** on average versus the existing
unordered GB10 path in this branch. The default therefore stays on the faster
measured path.

## Memory

The split-K implementation reserves one fixed 4 MiB CUDA scratch allocation
when enabled. It does not change KV-cache sizing: all three benchmark runs have
equal `kvcache_bytes` at every recorded frontier. The refreshed throughput run
did not include a separate `/usr/bin/time -v` RSS A/B; the only intentional
memory delta remains the 4 MiB CUDA scratch buffer when split-K is enabled.

## Validation

Passed:

```sh
make cuda-regression
```

This runs `cuda_long_context_smoke`, `cuda_fused_decode_smoke`, and the new
`cuda_splitk_smoke`. The split-K smoke forces `DS4_CUDA_F16_SPLITK=1` and
compares the opt-in kernel against the non-split path and a CPU reference:
`max_ref=1.54972e-06`, `max_vs_nosplit=3.12924e-07`.

The real-model official logprob vector test was also checked with and without
split-K:

```sh
DS4_TEST_MODEL=/home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  DS4_CUDA_F16_SPLITK=1 ./ds4_test --logprob-vectors

DS4_TEST_MODEL=/home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  ./ds4_test --logprob-vectors
```

Both runs failed in the same existing CUDA case,
`short_code_completion step 1 selected token mismatch`. That failure is not
attributable to split-K, but it also means this opt-in path should not be
treated as a strict logprob-golden quality path until the CUDA golden mismatch
is resolved independently.

## Artifacts

| File | Description |
| --- | --- |
| [`comparison_summary.csv`](comparison_summary.csv) | Per-context deltas across the three variants |
| [`no_splitk.csv`](no_splitk.csv) | Final default benchmark CSV |
| [`default_splitk.csv`](default_splitk.csv) | Split-K opt-in benchmark CSV |
| [`ordered_f16.csv`](ordered_f16.csv) | Ordered F16 benchmark CSV |
| [`no_splitk_ts.svg`](no_splitk_ts.svg) | Final default chart |
| [`default_splitk_ts.svg`](default_splitk_ts.svg) | Split-K opt-in chart |
| [`ordered_f16_ts.svg`](ordered_f16_ts.svg) | Ordered F16 chart |
