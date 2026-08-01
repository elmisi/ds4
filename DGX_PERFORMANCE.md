# DGX performance branch

`dgx-performance` is the performance-oriented development branch for
NVIDIA DGX Spark / GB10 and related CUDA deployments. It deliberately keeps
[`main`](README.md) available as a clean upstream mirror. This document records
the functional differences introduced on this branch; it is not a claim that
every change should be merged upstream as a single patch.

## Scope

The branch has two complementary goals:

- improve one-token CUDA decode performance for DeepSeek V4 Flash on GB10
  without increasing KV-cache allocation;
- make large GLM 5.2 deployments practical through selected-expert SSD
  streaming, bounded GPU weight caching, and a distributed DGX launcher.

The benchmark artefacts are part of the branch and are linked below. Results
are hardware- and model-specific; use them as reproducible A/B references, not
as universal performance claims.

## CUDA decode path

The default CUDA decode path includes the following GB10-focused changes:

- Fused HC decode preparation combines the HC mixing/split work with RMS norm.
- The decode graph fuses Q normalization with RoPE, KV RoPE with the raw FP8 KV
  store, and the low-rank attention-output work with RoPE where applicable.
- The one-token F16 path uses an unordered pair matmul and GB10-oriented launch
  tuning.
- The Q8 decode input is cached for the one-token path.
- The graph driver avoids an unnecessary token-split synchronization on CUDA.

These fusions are enabled by default. They can be isolated for diagnostics with
the following *disable* switches:

| Switch | Effect |
| --- | --- |
| `DS4_CUDA_NO_Q_NORM_ROPE_FUSED=1` | Use the non-fused Q-normalization/RoPE path. |
| `DS4_CUDA_NO_FUSED_HC_PRE=1` | Use the non-fused HC decode preparation. |
| `DS4_CUDA_NO_KV_ROPE_STORE_FUSED=1` | Split RoPE from the decode KV store. |
| `DS4_CUDA_NO_ATTN_OUTPUT_ROPE_LOW_FUSED=1` | Split RoPE from the low-rank attention output path. |

### Optional CUDA experiments

Two experiments are intentionally opt-in rather than part of the default
decode path:

| Switch | Implementation | Branch decision |
| --- | --- | --- |
| `DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN=1` | Tile-8 row-span MoE down-projection kernel. | Retained as an opt-in prefill tuning knob. |
| `DS4_CUDA_F16_SPLITK=1` | Deterministic two-pass split-K one-token F16 matmul, with a fixed 4 MiB CUDA scratch buffer. | Retained for A/B and diagnostics; the unordered F16 default measured faster. |
| `DS4_CUDA_ORDERED_F16_MATMUL=1` | Legacy ordered F16 path. | Retained only as an A/B reference. |

The split-K experiment does not change KV-cache sizing. Its real-model
logprob-vector run exposed the same pre-existing CUDA golden mismatch with and
without split-K, so it should not be considered a strict logprob-golden path
until that independent issue is resolved.

## GLM 5.2 SSD streaming and memory behaviour

The branch extends the CUDA GLM path so that selected routed experts can be
streamed from SSD while using a bounded GPU weight cache. It adds cache recycling
between layers, selected-expert cache management during batch prefill, and the
supporting GPU interface. The Metal implementation supplies a no-op recycler so
the shared graph flow stays portable.

The GLM prefill flow also:

- honors the full-layer prefill threshold in the indexed-prefill path;
- reuses/evicts selected-expert cache entries at the configured budget instead
  of failing on overflow views;
- loads IQ2 dequantization LUTs unconditionally in the batch MoE expert-tile
  kernels.

`run-glm52-server-dgx.sh` is an operational launcher for a two-host DGX setup.
It selects coordinator or worker role, assigns the layer range, enables
SSD streaming, provides bounded-cache and memory-guard defaults, and can run
inside a transient user systemd service. Every setting is overridable through
its documented `DS4_*` environment variables.

## Build and validation

`make cuda-spark` remains the DGX Spark build target. Its default omits an
explicit `nvcc -arch`; `CUDA_SPARK_ARCH=sm_N` is available for controlled A/B
builds. `make cuda-generic` remains the generic local CUDA target.

The CUDA regression target now includes:

```sh
make cuda-regression
```

- `cuda_long_context_smoke`
- `cuda_fused_decode_smoke`
- `cuda_splitk_smoke`

The latter two validate the branch-specific fused decode and split-K paths
against their reference behaviour.

## Measured results

All figures below are recorded ASUS GX10 / NVIDIA GB10 runs from the versioned
artefacts in this branch. They use the same model and benchmark command within
each A/B; the linked benchmark README records the exact model path, build flags,
context allocation, prompt, and raw CSV files. Throughput is tokens per second.

### Fused decode versus upstream main

On the recorded ASUS GX10 / NVIDIA GB10 benchmark, over 49 common contexts
from 2k to 100k tokens, the fused decode path measured **+12.4% average
generation throughput** with **-0.3% average prefill throughput** and no
KV-cache-size delta.

| Context | Main generation | DGX branch generation | Generation delta | Main prefill | DGX branch prefill | KV delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 13.95 | 15.92 | +14.1% | 383.55 | 387.07 | 0 |
| 32,768 | 12.86 | 14.51 | +12.8% | 343.50 | 343.37 | 0 |
| 65,536 | 11.98 | 13.38 | +11.7% | 293.50 | 291.42 | 0 |
| 100,000 | 11.20 | 12.41 | +10.8% | 249.76 | 249.14 | 0 |

The method, raw CSV files, and charts are in
[`speed-bench/gx10_gb10_fused_hc/`](speed-bench/gx10_gb10_fused_hc/).

### F16 split-K experiment

Across nine measured frontiers, split-K was **-0.8%** in generation throughput
versus the final unordered default, while still outperforming the legacy
ordered F16 path. That is why split-K remains opt-in.

| Context | Final default generation | Split-K generation | Ordered F16 generation | Split-K vs default | Default vs ordered |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 15.73 | 15.70 | 15.04 | -0.19% | +4.59% |
| 10,240 | 15.86 | 15.77 | 14.87 | -0.57% | +6.66% |
| 18,432 | 15.65 | 15.52 | 14.68 | -0.83% | +6.61% |
| 26,624 | 15.24 | 15.10 | 14.32 | -0.92% | +6.42% |
| 34,816 | 14.42 | 14.29 | 13.59 | -0.90% | +6.11% |
| 43,008 | 14.21 | 14.08 | 13.41 | -0.91% | +5.97% |
| 51,200 | 13.94 | 13.81 | 13.17 | -0.93% | +5.85% |
| 59,392 | 13.63 | 13.51 | 12.90 | -0.88% | +5.66% |
| 65,536 | 13.43 | 13.29 | 12.70 | -1.04% | +5.75% |

| Variant | Average generation | Average prefill |
| --- | ---: | ---: |
| Final unordered default | 14.68 | 345.23 |
| Split-K opt-in | 14.56 | 349.95 |
| Legacy ordered F16 | 13.85 | 346.81 |

KV-cache sizing was identical in every split-K run. The opt-in path reserves
one fixed 4 MiB CUDA scratch allocation; measured maximum RSS was effectively
unchanged (1,630,484 KB default, 1,630,584 KB split-K, and 1,631,112 KB
ordered).

The full experiment, memory accounting, validation notes, CSV files, and charts
are in [`speed-bench/gx10_gb10_splitk_f16/`](speed-bench/gx10_gb10_splitk_f16/).

### CUDA architecture A/B

On the recorded GB10 run, an explicit `sm_121` build was generation-neutral but
averaged **-2.4% prefill throughput** beyond the short-context point. The
no-explicit-architecture build is therefore the `cuda-spark` default.

| Context | `sm_121` generation | No-arch generation | Generation delta | `sm_121` prefill | No-arch prefill | Prefill delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 19.56 | 19.08 | +2.5% | 392.60 | 386.62 | +1.5% |
| 10,240 | 16.14 | 16.13 | +0.1% | 358.14 | 370.20 | -3.3% |
| 18,432 | 16.08 | 16.09 | -0.1% | 345.21 | 356.51 | -3.2% |
| 26,624 | 15.89 | 15.91 | -0.1% | 333.86 | 344.50 | -3.1% |
| 34,816 | 15.31 | 15.29 | +0.1% | 321.11 | 330.85 | -2.9% |
| 43,008 | 15.16 | 15.13 | +0.2% | 305.19 | 313.71 | -2.7% |
| 51,200 | 15.01 | 14.98 | +0.2% | 295.12 | 303.07 | -2.6% |
| 59,392 | 14.84 | 14.80 | +0.3% | 285.47 | 292.58 | -2.4% |
| 65,536 | 14.71 | 14.69 | +0.1% | 279.41 | 287.12 | -2.7% |

The method and artefacts are in
[`speed-bench/gx10_gb10_sm121/`](speed-bench/gx10_gb10_sm121/).

## Change map

| Area | Main files |
| --- | --- |
| CUDA kernels, decode fusions, Q8 cache, MoE tuning, SSD weight cache | `ds4_cuda.cu`, `ds4_gpu.h` |
| Decode-graph integration, GLM prefill/decode flow, memory guard | `ds4.c` |
| Shared cache-recycler compatibility | `ds4_metal.m` |
| DGX build target and CUDA smoke-test targets | `Makefile` |
| Distributed GLM server operation | `run-glm52-server-dgx.sh` |
| Focused CUDA validation | `tests/cuda_fused_decode_smoke.c`, `tests/cuda_splitk_smoke.c` |
| Benchmark data and methodology | `speed-bench/gx10_gb10_*/` |

For the exact code history, compare the branch against `main`:

```sh
git log --oneline main..dgx-performance
git diff --stat main...dgx-performance
```
