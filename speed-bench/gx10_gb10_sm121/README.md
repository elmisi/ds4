# GX10 / GB10 nvcc -arch A/B: sm_121 vs no explicit arch

This benchmark measures whether building the CUDA backend with an explicit
`nvcc -arch=sm_121` helps on ASUS GX10 / NVIDIA GB10, after the branch was
rebased onto upstream main (session batching, tensor parallelism, new decode
paths).

## Method

Both binaries were built from the same tree (branch tip after the rebase):

```sh
make -B ds4-bench CUDA_ARCH=sm_121   # ds4-bench-sm121
make -B ds4-bench CUDA_ARCH=         # ds4-bench-noarch
```

Benchmark shape (same as the split-K writeup):

```sh
./ds4-bench --cuda \
  -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 8192 \
  --ctx-alloc 65700 --gen-tokens 128 --csv <output.csv>
```

## Results

| Context | sm_121 gen t/s | no-arch gen t/s | gen delta | sm_121 prefill t/s | no-arch prefill t/s | prefill delta |
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

Generation is neutral (**+0.4% average**, entirely from the 2k point). Prefill
is consistently slower with `sm_121` (**-2.4% average**, -2.4%..-3.3% beyond
the first point). `kvcache_bytes` is identical at every frontier.

`make cuda-spark` therefore keeps the no-arch default; `CUDA_SPARK_ARCH=sm_N`
remains available as an override for future A/Bs.

## Rebase effect (cross-session reference)

Compared with the pre-rebase branch measurements in
[`../gx10_gb10_splitk_f16/no_splitk.csv`](../gx10_gb10_splitk_f16/no_splitk.csv)
(same no-arch build flags, different day — treat as indicative):
generation improved from **+1.7% at 10k to +9.4% at 65k** (~+6% average),
while prefill regressed ~3.5-5%. The upstream decode work and this branch's
GB10 fusions compound on generation.

## Artifacts

| File | Description |
| --- | --- |
| [`sm121.csv`](sm121.csv) | sm_121 build benchmark CSV |
| [`noarch.csv`](noarch.csv) | no-arch build benchmark CSV |
| [`sm121.log`](sm121.log) | sm_121 run log |
| [`noarch.log`](noarch.log) | no-arch run log |
