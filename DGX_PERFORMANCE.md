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

### CUDA experiments and GB10 defaults

The tile-8 row-span MoE down-projection kernel is enabled by default on
GB10/sm_121. It improves prompt prefill and leaves generation throughput and
KV-cache sizing effectively unchanged. It can be disabled for an A/B with
`DS4_CUDA_NO_MOE_DOWN_TILE8_ROWSPAN=1`. The positive switch remains available
to test the same kernel on other CUDA devices.

The F16 experiments below remain intentionally opt-in rather than part of the
default decode path:

| Switch | Implementation | Branch decision |
| --- | --- | --- |
| `DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN=1` | Tile-8 row-span MoE down-projection kernel on non-GB10 CUDA devices. | Default on GB10/sm_121; use `DS4_CUDA_NO_MOE_DOWN_TILE8_ROWSPAN=1` to opt out. |
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

### Speed-compatible DGX Spark snapshot

This is the same `Machine`, `Backend`, `Context`, `Prefill`, and `Generation`
table published in the [`README.md` Speed section](README.md#speed). It uses the
standard *Promessi sposi* input, 2,048-token context steps, and 128 greedy
generation tokens at every frontier. The complete sweeps are in
[`speed-bench/m5_max.csv`](speed-bench/m5_max.csv) and
[`speed-bench/gb10.csv`](speed-bench/gb10.csv).

| Machine | Backend | Context | Prefill | Generation |
| --- | --- | ---: | ---: | ---: |
| MacBook Pro M5 Max, 128 GB | Metal | 2048 | 790.18 t/s | 39.35 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 16384 | 572.53 t/s | 36.14 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 32768 | 557.04 t/s | 34.36 t/s |
| MacBook Pro M5 Max, 128 GB | Metal | 65536 | 398.50 t/s | 27.64 t/s |
| DGX Spark GB10, 128 GB | CUDA | 2048 | 825.76 t/s | 18.05 t/s |
| DGX Spark GB10, 128 GB | CUDA | 16384 | 872.44 t/s | 15.10 t/s |
| DGX Spark GB10, 128 GB | CUDA | 32768 | 855.94 t/s | 14.43 t/s |
| DGX Spark GB10, 128 GB | CUDA | 65536 | 822.98 t/s | 13.84 t/s |

### Fused decode versus upstream main

On the refreshed ASUS GX10 / NVIDIA GB10 benchmark, over 49 common contexts
from 2k to 100k tokens, the fused decode path measured **+5.0% average
generation throughput** with **-0.09% average prefill throughput** and no
KV-cache-size delta. The benchmark uses a forced post-frontier snapshot to
avoid prompt replay between frontiers; the timed throughput measurement remains
an ordinary CUDA run.

| Context | Main generation | DGX branch generation | Generation delta | Main prefill | DGX branch prefill | KV delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 15.30 | 16.11 | +5.29% | 365.45 | 366.64 | 0 |
| 32,768 | 14.55 | 15.28 | +5.02% | 330.19 | 329.05 | 0 |
| 65,536 | 13.95 | 14.63 | +4.87% | 283.82 | 283.43 | 0 |
| 100,000 | 13.40 | 14.03 | +4.70% | 246.45 | 245.63 | 0 |

The method, raw CSV files, and charts are in
[`speed-bench/gx10_gb10_fused_hc/`](speed-bench/gx10_gb10_fused_hc/).

### Tile-8 MoE down projection on GB10

With the current DeepSeek V4 Flash 0731 GGUF, a fresh-process A/B at a 2,048
token frontier and a 262,144-token allocated context measured **+5.6% average
prefill throughput** (369.18 to 389.98 t/s). Generation was neutral (18.30 to
18.34 t/s); tile8 is therefore a prompt/context-ingestion optimization rather
than a decode-speed claim. Both variants planned 85.31 GiB of model, KV, and
buffer memory. The inputs, raw samples, and reproduction command are in
[`speed-bench/gx10_gb10_tile8/`](speed-bench/gx10_gb10_tile8/).

### Upstream `main` MMQ comparison

On 2026-08-04, the same 0731 IQ2XXS/w2Q2K Flash GGUF and 262,144-token
allocation were compared with upstream `main` at `b7e9f00`. The upstream raw
MMQ path improved prefill from 386.33 to **441.93 t/s** (**+14.4%**) versus
the deployed tile8 branch, but reduced steady decode from 18.34 to 17.14 t/s
(**-6.5%**) and increased first-token latency from 70.37 to 75.76 ms.

Upstream's default aligned-artifact path built 78.71 GiB of derived CUDA
weights, then did not produce a benchmark result. No OOM or NVIDIA Xid was
logged, but that default is not a deployment candidate on this configuration.
The successful MMQ run deliberately disabled those artifacts; its quality
smoke prompt produced the same correct deterministic answer as the branch.
Keep `dgx-performance` as the service baseline until the artifact failure and
decode regression are resolved. The commands, exact environments, CSV, and
quality-smoke result are in
[`speed-bench/gx10_gb10_main_mmq/`](speed-bench/gx10_gb10_main_mmq/).

### F16 split-K experiment

Across nine refreshed measured frontiers, split-K was **-1.3%** in generation throughput
versus the final unordered default, while still outperforming the legacy
ordered F16 path. That is why split-K remains opt-in.

| Context | Final default generation | Split-K generation | Ordered F16 generation | Split-K vs default | Default vs ordered |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 18.92 | 18.63 | 18.19 | -1.53% | +4.01% |
| 10,240 | 16.03 | 15.81 | 15.48 | -1.37% | +3.55% |
| 18,432 | 15.98 | 15.77 | 15.44 | -1.31% | +3.50% |
| 26,624 | 15.81 | 15.60 | 15.28 | -1.33% | +3.47% |
| 34,816 | 15.19 | 15.00 | 14.70 | -1.25% | +3.33% |
| 43,008 | 15.03 | 14.85 | 14.56 | -1.20% | +3.23% |
| 51,200 | 14.88 | 14.70 | 14.43 | -1.21% | +3.12% |
| 59,392 | 14.72 | 14.54 | 14.26 | -1.22% | +3.23% |
| 65,536 | 14.61 | 14.42 | 14.15 | -1.30% | +3.25% |

| Variant | Average generation | Average prefill |
| --- | ---: | ---: |
| Final unordered default | 15.69 | 332.63 |
| Split-K opt-in | 15.48 | 331.77 |
| Legacy ordered F16 | 15.17 | 331.97 |

KV-cache sizing was identical in every refreshed split-K run. The opt-in path
reserves one fixed 4 MiB CUDA scratch allocation. This refresh measures
throughput only; it does not repeat the separate `/usr/bin/time -v` RSS A/B.

The full experiment, memory accounting, validation notes, CSV files, and charts
are in [`speed-bench/gx10_gb10_splitk_f16/`](speed-bench/gx10_gb10_splitk_f16/).

### CUDA architecture A/B

On the refreshed GB10 run, an explicit `sm_121` build was generation-neutral
(**+0.5% average**) but averaged **-3.3% prefill throughput**. The
no-explicit-architecture build is therefore the `cuda-spark` default.

| Context | `sm_121` generation | No-arch generation | Generation delta | `sm_121` prefill | No-arch prefill | Prefill delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 19.37 | 18.92 | +2.4% | 388.30 | 388.38 | -0.0% |
| 10,240 | 16.06 | 16.03 | +0.2% | 355.08 | 372.74 | -4.7% |
| 18,432 | 15.99 | 15.98 | +0.1% | 342.19 | 358.07 | -4.4% |
| 26,624 | 15.82 | 15.81 | +0.1% | 331.04 | 345.37 | -4.1% |
| 34,816 | 15.23 | 15.19 | +0.3% | 318.59 | 331.13 | -3.8% |
| 43,008 | 15.09 | 15.03 | +0.4% | 303.06 | 313.70 | -3.4% |
| 51,200 | 14.93 | 14.88 | +0.3% | 293.57 | 303.70 | -3.3% |
| 59,392 | 14.77 | 14.72 | +0.3% | 284.39 | 293.46 | -3.1% |
| 65,536 | 14.65 | 14.61 | +0.3% | 278.45 | 287.16 | -3.0% |

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
