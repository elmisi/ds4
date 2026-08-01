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

Branch base at benchmark time:

```text
d557259 docs: refresh GX10 fused decode benchmarks
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
./ds4-bench --cuda \
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

All runs reported `80.76 GiB` of model tensor spans covered by the CUDA model
cache and `1743.17 MiB` of context buffers at `ctx=65700`.

## Results

| Context | Final default gen t/s | Split-K gen t/s | Ordered gen t/s | Split-K vs final | Final vs ordered | KV equal |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2,048 | 15.73 | 15.70 | 15.04 | -0.19% | +4.59% | yes |
| 10,240 | 15.86 | 15.77 | 14.87 | -0.57% | +6.66% | yes |
| 18,432 | 15.65 | 15.52 | 14.68 | -0.83% | +6.61% | yes |
| 26,624 | 15.24 | 15.10 | 14.32 | -0.92% | +6.42% | yes |
| 34,816 | 14.42 | 14.29 | 13.59 | -0.90% | +6.11% | yes |
| 43,008 | 14.21 | 14.08 | 13.41 | -0.91% | +5.97% | yes |
| 51,200 | 13.94 | 13.81 | 13.17 | -0.93% | +5.85% | yes |
| 59,392 | 13.63 | 13.51 | 12.90 | -0.88% | +5.66% | yes |
| 65,536 | 13.43 | 13.29 | 12.70 | -1.04% | +5.75% | yes |

Average generation throughput across the nine frontiers:

| Variant | Avg gen t/s | Avg prefill t/s |
| --- | ---: | ---: |
| Final default | 14.68 | 345.23 |
| Split-K opt-in | 14.56 | 349.95 |
| Ordered F16 | 13.85 | 346.81 |

Split-K is useful as an opt-in diagnostic and still beats the ordered legacy
path by **+5.1%** on average, but it is **-0.8%** on average versus the existing
unordered GB10 path in this branch. The default therefore stays on the faster
measured path.

## Memory

The split-K implementation reserves one fixed 4 MiB CUDA scratch allocation
when enabled. It does not change KV-cache sizing: all three benchmark runs had
identical `kvcache_bytes` at every frontier, including `926033292` bytes at
65,536 tokens. `/usr/bin/time -v` max RSS was also effectively unchanged:

| Variant | Max RSS |
| --- | ---: |
| Final default | 1,630,484 KB |
| Split-K opt-in | 1,630,584 KB |
| Ordered F16 | 1,631,112 KB |

The small RSS differences are below run-to-run noise for this workload; the
only intentional memory delta is the 4 MiB CUDA scratch buffer when split-K is
enabled.

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
