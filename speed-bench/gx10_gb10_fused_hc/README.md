# ASUS GX10 / GB10 fused decode benchmark

This benchmark compares upstream `main` with a focused CUDA decode optimization
branch on ASUS GX10 / NVIDIA GB10.

![ASUS GX10 / GB10 generation comparison](gx10_gb10_fused_hc_vs_main_ts.svg)

## Code under test

| Variant | Revision | Notes |
| --- | --- | --- |
| Upstream main | `ad0209f` (`Fix PRO routed MoE expert mapping`) | Clean `origin/main` worktree |
| Fused HC decode | `c0d4130` (`cuda: fuse GB10 decode HC and RoPE paths`) | Branch `gx10-decode-fused-hc` |

The optimized branch keeps the quality-preserving GB10 decode changes that
measured as useful without increasing KV/cache memory. It intentionally does
not include the experimental SoA Q8 cache path, which used more CUDA memory and
failed with CUDA graph capture in agent testing.

## Method

Both variants were built with:

```sh
make -j$(nproc) ds4-bench CUDA_ARCH=
```

The benchmark command was:

```sh
./ds4-bench --cuda \
  -m /home/alessandro/projects/ds4/ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 100000 \
  --step-incr 2048 \
  --ctx-alloc 100200 \
  --gen-tokens 128 \
  --csv <output.csv>
```

No runtime performance environment variables were enabled for the optimized
run. The CUDA allocation reported by `nvidia-smi` during both runs was
`107275 MiB`; `kvcache_bytes` is identical at every measured context point.

## Results

| Context | Upstream gen t/s | Optimized gen t/s | Gain | Upstream prefill t/s | Optimized prefill t/s | KV delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 13.57 | 15.40 | +13.5% | 385.99 | 380.75 | 0 |
| 32,768 | 12.49 | 14.10 | +12.9% | 343.19 | 341.30 | 0 |
| 65,536 | 11.63 | 13.03 | +12.0% | 292.39 | 291.67 | 0 |
| 100,000 | 10.90 | 12.11 | +11.1% | 249.47 | 249.13 | 0 |

Across the 49 common context points from 2k to 100k, the optimized path improves
generation throughput by **+12.4% average**. Prefill is close to neutral,
averaging **-0.5%**.

## Artifacts

| File | Description |
| --- | --- |
| [`upstream_main.csv`](upstream_main.csv) | Raw upstream main benchmark CSV |
| [`optimized_fused_hc.csv`](optimized_fused_hc.csv) | Raw optimized benchmark CSV |
| [`comparison_summary.csv`](comparison_summary.csv) | Per-context delta table |
| [`upstream_main_ts.svg`](upstream_main_ts.svg) | Upstream-only speed graph |
| [`optimized_fused_hc_ts.svg`](optimized_fused_hc_ts.svg) | Optimized-only speed graph |
| [`gx10_gb10_fused_hc_vs_main_ts.svg`](gx10_gb10_fused_hc_vs_main_ts.svg) | Generation comparison graph |
