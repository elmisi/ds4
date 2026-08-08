# GB10 tile8 branch versus upstream main MMQ

This is a one-sample, fresh-process comparison between the deployed
`dgx-performance` baseline and upstream `main` after the MMQ integration. It
uses the DeepSeek V4 Flash 0731 IQ2XXS/w2Q2K GGUF used by the DGX service.
The purpose is to decide whether upstream is deployable on this GB10, not to
make a universal CUDA performance claim.

## Environment

- Host GPU: NVIDIA GB10 / DGX Spark.
- `dgx-performance`: `8251c0a` (`cuda: enable tile8 MoE down on GB10`), built
  with `CUDA_ARCH=`.
- Upstream `main`: `b7e9f00`, built with `CUDA_ARCH=sm_121`; its Makefile
  emits `sm_121a` and enables the MMQ objects.
- Model:
  `/home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`.
- The serving unit `ds4-gx10.service` was stopped before the runs and restored
  afterwards.

## Reproduction

The common benchmark command is:

```sh
DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench --cuda \
  -m /home/alessandro/projects/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 2048 --ctx-alloc 262144 \
  --gen-tokens 128 --prefill-chunk 2048 --gpu-vram auto
```

Run it from `/home/alessandro/projects/ds4-gx10` after
`make -B -j8 ds4-bench CUDA_ARCH=` for the baseline. For upstream, build from
`/home/alessandro/projects/ds4` with `make -B -j8 ds4-bench CUDA_ARCH=sm_121`.

The raw-MMQ run adds:

```sh
DS4_CUDA_MOE_NO_IQ2_ALIGNED=1 \
DS4_CUDA_MOE_NO_Q2K_ALIGNED=1 \
DS4_CUDA_Q8_NO_ALIGNED=1
```

Those switches intentionally disable upstream's derived aligned artifacts,
not MMQ itself. Add `DS4_CUDA_MMQ=0` to obtain the upstream legacy fallback.

## Result

| Variant | Prefill t/s | Steady decode t/s | First token | Outcome |
| --- | ---: | ---: | ---: | --- |
| `dgx-performance` tile8 | 386.33 | 18.34 | 70.37 ms | Baseline |
| `main` MMQ raw | **441.93** (+14.4%) | 17.14 (-6.5%) | 75.76 ms | Completes |
| `main`, MMQ disabled | 408.87 (+5.8%) | 17.05 (-7.0%) | 73.30 ms | Completes |
| `main` default artifacts | — | — | — | No final CSV row |

The raw-MMQ delta against upstream without MMQ is +8.1% prefill. The complete
numeric record is [`main_mmq_comparison.csv`](main_mmq_comparison.csv).

## Default artifact-path result

On the default upstream environment, startup built 474 aligned CUDA artifacts
(78.71 GiB) and replaced raw expert residency. It reached the messages
`dense Q8 prefill using aligned D2R` and `routed MoE using aligned CUDA
artifacts`, but did not emit the benchmark CSV data row. There was no kernel
OOM or NVIDIA Xid in the contemporaneous logs. Treat this as an unresolved
runtime failure, not as a zero-throughput result.

## Quality smoke

Both successful builds were run with `--cuda --gpu-vram auto -c 8192`, greedy
sampling, and a real Italian prompt asking for five provinces beginning with
"B" and their regions. Both returned Bergamo, Biella, Bologna, Bolzano, and
Brescia with the correct regions. This confirms a basic deterministic output
smoke only; it is not a full quality or logprob acceptance suite.

## Decision

Do not replace the `dgx-performance` service binary. Raw MMQ is promising for
prefill-heavy workloads, but the default upstream artifact path must complete
and decode must recover before it can be promoted over the quality-validated
tile8 baseline.
