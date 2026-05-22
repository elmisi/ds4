# GX10 tuning run - 2026-05-14

Device constraint: keep a concurrent GPU workload resident while benchmarking.

## System

- Linux aarch64 on NVIDIA GB10, CUDA driver/runtime 13.0.
- The concurrent GPU workload stayed active during tests, using about 10.5 GiB GPU memory.
- Full cached ds4 runs used about 103 GiB GPU memory in `nvidia-smi`, leaving only a small safety margin.
- Disk/RAM were sufficient to also test the optional 3.6 GiB MTP model.

## Baseline

Command:

```sh
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 32768 --step-incr 2048 \
  --gen-tokens 128 \
  --csv tuning/gx10-20260514/baseline_default.csv
```

Summary:

- Average prefill: 366.99 t/s
- Average generation: 13.16 t/s
- Generation range: 12.34-13.88 t/s

## Implemented changes

1. GB10/sm_121 now defaults to the regular unordered one-token F16 matmul path, including the F16 pair helper.

   - File: `ds4_cuda.cu`
   - Rationale: the ordered one-token F16 path regressed decode on GB10 while not helping batched prefill.
   - Override: `DS4_CUDA_FORCE_ORDERED_F16_MATMUL=1` restores the old path.
   - Override: `DS4_CUDA_NO_ORDERED_F16_MATMUL=1` still forces the unordered path on any CUDA device.
   - Full sweep result from the env-equivalent test: average generation improved from 13.16 to 13.89 t/s, +5.54%.
   - Average prefill stayed neutral/slightly positive: +0.28%.
   - Final 8k/64 after also routing the F16 pair helper through the same decision: 14.44 t/s generation.

2. Hot CUDA environment flags are cached at backend initialization.

   - File: `ds4_cuda.cu`
   - Rationale: avoid repeated `getenv()` checks in decode hot paths.
   - Effect: small/noise on benchmark throughput, but token profile showed slightly lower host encode/launch overhead.

3. CUDA no longer uses the Metal token split default.

   - File: `ds4.c`
   - Rationale: `ds4_gpu_flush_commands()` is a CUDA device synchronize on this backend. The split is useful for Metal command scheduling, but on CUDA it inserts a per-token barrier.
   - Default is now `split_after_layers=4` on Apple/Metal and `0` elsewhere.
   - A/B at 8k/64 was effectively neutral: 13.99 t/s default vs 13.98 t/s with `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=4`.

## Current final check

Command before the final F16 pair alignment:

```sh
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 \
  --gen-tokens 64 \
  --csv tuning/gx10-20260514/final_short_8k_g64.csv
```

Result with the concurrent GPU workload still running:

- Prefill: 390.78 t/s
- Generation: 13.87 t/s

Final command after aligning the F16 pair helper:

```sh
./ds4-bench -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 \
  --gen-tokens 64 \
  --csv tuning/gx10-20260514/final_pairfix_short_8k_g64.csv
```

Result:

- Prefill: 389.83 t/s
- Generation: 14.44 t/s

This is the best measured no-MTP result in the run.

## Negative local tests

- `DS4_CUDA_SERIAL_ROUTER=1`: generation dropped to 11.44 t/s.
- `DS4_CUDA_NO_WARP_ROUTER_SELECT=1`: generation dropped to 12.84 t/s.
- `DS4_CUDA_MOE_NO_DECODE_LUT_GATE=1`: generation dropped to 12.01 t/s.
- `DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6=1`: no useful improvement.
- `DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR=1`: slight regression.
- `DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=72`: startup model cache failed.
- `DS4_CUDA_DIRECT_MODEL=1`: much lower memory use, but too slow for the target use case.
- `DS4_CUDA_DISABLE_QKV_RMS_FUSED=1`: no meaningful improvement.
- `DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN=5`: no meaningful improvement for greedy no-MTP decode.

## Fork scan

Checked GitHub forks from `https://api.github.com/repos/antirez/ds4/forks` and inspected the relevant forks/branches. Most forks are identical or platform ports. Relevant findings:

- `spmurrayzzz/ds4`
  - Has CUDA decode fast-path work: device top-1, avoiding decode split synchronization, and larger fast-path patches.
  - The safe split-sync idea was ported into `ds4.c`.
  - The large CUDA patches are not a clean drop-in here: they are from an older baseline, overlap with upstream work, and target RTX Pro/SM120 measurements. Saved patches:
    - `spmurrayzzz_decode_fastpaths_ds4_cuda.patch`
    - `spmurrayzzz_decode_fastpaths_ds4_c.patch`
    - `spmurrayzzz_bd4c01c5_device_top1.patch`
    - `spmurrayzzz_540a7e4a_avoid_sync.patch`

- `reffdev/ds4` branch `fused-matmul-mtp`
  - Has fused matmul work for tiny MTP/batch verification and a cuBLAS threshold tweak.
  - Saved patches:
    - `reffdev_3ba9d28d_ds4_cuda.patch`
    - `reffdev_00532b98_ds4_cuda.patch`
  - The cuBLAS threshold tweak was tested via env and did not help normal greedy decode.
  - The fused matmul patch is mostly MTP-specific, not the current no-MTP decode bottleneck.

- `Entrpi/ds4-on-spark`
  - Useful GX10/Spark notes and methodology caveat.
  - Their older MTP parity gap notes mention missing CUDA Q4_K routed expert support; this local code already has Q4_K routed MTP paths, so that exact blocker appears fixed here.
  - They also call out that first-token-inclusive `ds4`/`ds4-bench` numbers can look much lower than steady-state decode-only numbers.

- `Tonoken3/ds4-sm120`
  - Mostly SM120 / multi-GPU / mapping work. Not directly useful for this single GB10 CUDA run.

- `berschmitt/ds4`
  - Mostly docs/config notes around cache reserve. No code change worth porting for this target.

- ROCm/OpenVINO/Metal-specialized forks
  - Not relevant to this CUDA GB10 target unless the project direction changes backend.

## MTP test

Downloaded:

```sh
./download_model.sh mtp
```

Test prompt: `speed-bench/quick_bench.txt`, `--ctx 4096`, `--nothink`, greedy, `-n 64`, with the concurrent GPU workload still running.

Baseline within the same CLI session path:

```sh
DS4_MTP_SPEC_DISABLE=1 ./ds4 -m ds4flash.gguf \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 --ctx 4096 --nothink --temp 0 -n 64 \
  --prompt-file speed-bench/quick_bench.txt
```

Result:

- Prefill: 79.16 t/s
- Generation: 3.66 t/s

MTP active:

```sh
DS4_MTP_TIMING=1 DS4_MTP_SPEC_LOG=1 ./ds4 -m ds4flash.gguf \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --mtp-draft 2 --ctx 4096 --nothink --temp 0 -n 64 \
  --prompt-file speed-bench/quick_bench.txt
```

Result:

- Prefill: 79.41 t/s
- Generation: 3.70 t/s

Conclusion: MTP is not useful on this setup yet. It loads and runs, but the verifier/draft costs dominate. Timing logs show frequent first-draft misses or one-token commits, with verifier costs around 72-139 ms per speculative step. This is much slower than the normal no-MTP decode path around 13.9 t/s.

## Verification

Passed after code changes, including the F16 pair alignment:

```sh
make cuda-spark CUDA_HOME=/usr/local/cuda-13.0
make ds4_test CUDA_HOME=/usr/local/cuda-13.0
./ds4_test --server
```

Greedy smoke with patched default and forced ordered path both returned `ok`.

## Current recommendation

Keep the GB10 unordered F16 matmul default. Keep the CUDA no-split default because it is logically correct and neutral in tests. Do not enable MTP for production on this GX10 while the concurrent GPU workload is resident. The remaining meaningful headroom is likely in porting/reworking specific CUDA decode kernels, especially MoE and attention-output stages, not in environment knobs.
