# GB10 `sm_121` safe comparison — 2026-08-08

This is the quality-gated, like-for-like comparison of `dgx-performance`
against upstream `main`.  Both runs use the same 0731 Flash GGUF, the standard
*Promessi sposi* prompt, 2,048-token incremental prefill steps, 128 greedy
generation tokens, `CUDA_ARCH=sm_121`, and CUDA decode-graph capture disabled.

`main` at `b030961` needs `DS4_CUDA_DECODE_GRAPHS=0` for the last condition.
`dgx-performance` at `03fcd84` makes that the GB10 default after reproducing an
incomplete long-context response with capture enabled.  This retains MMQ/D2R
prefill and the DGX decode kernels.

## Result

| Context | main safe prefill | dgx prefill | Delta | main safe generation | dgx generation | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 781.35 t/s | 781.45 t/s | +0.01% | 18.98 t/s | 20.79 t/s | +9.54% |
| 16384 | 814.22 t/s | 814.98 t/s | +0.09% | 15.59 t/s | 16.79 t/s | +7.70% |
| 32768 | 800.66 t/s | 804.14 t/s | +0.43% | 14.89 t/s | 16.03 t/s | +7.66% |
| 65536 | 773.21 t/s | 771.95 t/s | -0.16% | 14.20 t/s | 15.24 t/s | +7.32% |
| **Mean (32 rows)** | **797.12 t/s** | **798.29 t/s** | **+0.15%** | **15.12 t/s** | **16.29 t/s** | **+7.73%** |

The prefill difference is within run noise while `dgx-performance` preserves a
repeatable decode advantage.  Do not compare these values directly with the
older README GB10 row: that row was collected with decode capture enabled and
does not meet this quality gate.

## Quality gate

- The 30,474-token assignment-recall prompt returned all 16 exact `Name=number`
  lines with D2R active.  Its output is byte-for-byte equal to the MMQ-off
  control apart from the informational GPU-budget line.
- `./ds4_test --long-context` passed.
- `./ds4_test --logprob-vectors --local-golden-vectors` passed.  The local
  long-story golden check kept the same top-1 token and full top-5 overlap.
- Neither complete sweep reported a D2R fallback, a CUDA launch failure, or a
  decode-graph capture failure.

## Artifacts

- `dgx_performance.csv`: `dgx-performance` 32-row sweep.
- `main_sm121_safe.csv`: fresh 32-row `main` control.
- `benchmark.log`, `build.log`, `provenance.txt`, and `readme_table.md`:
  reproduction details for the candidate run.
- `../gx10_gb10_sm121_quality_20260808/`: long-context outputs and deterministic
  quality-test logs.
