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
DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench --cuda \
  -m /home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 8192 \
  --ctx-alloc 65700 --gen-tokens 128 --csv <output.csv>
```

## Results

| Context | sm_121 gen t/s | no-arch gen t/s | gen delta | sm_121 prefill t/s | no-arch prefill t/s | prefill delta |
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

Generation is effectively neutral (**+0.5% average**, largely from the 2k
point). Prefill is consistently slower with `sm_121` (**-3.3% average**,
-3.0%..-4.7% beyond the first point). `kvcache_bytes` is identical at every
frontier.

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
