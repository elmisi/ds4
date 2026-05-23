# GX10 MMQ/MMVQ Port Track

Date: 2026-05-23

Branch: `gx10-mmqv-port`

Worktree: `/home/alessandro/projects/ds4-mmqv-port`

Base branch: `gx10-cuda-graph-decode` at `7fee9f1`

Source reference: `Entrpi/ds4:mmq-step-A-full-layer-graphs`

## Goal

Evaluate a real tensor-core/MMQ/MMVQ route for exact single-token decode without
polluting the main tuning branch. This track exists because the remaining
full-quality speed work is structural: repeated local MoE and Q8 micro-probes
have not crossed the promotion gate.

Target remains full-quality decode near 20 t/s with memory under control.

## Import Policy

Only take the parts needed for a controlled experiment:

- import `cuda/mmq/` as a vendored kernel scaffold;
- build and link it beside the current CUDA backend;
- gate every runtime call behind an opt-in env flag;
- benchmark one narrow Q8 route before touching broader dispatch;
- keep layer graphs, VMM, MoE graphs, MTP, and policy changes out until a
  smaller MMQ/MMVQ call proves useful.

The Entrpi branch reports lower public GB10 numbers than this branch, so this is
an idea/source import, not a performance assumption.

## Checkpoints

### 2026-05-23 scaffold import

Imported `cuda/mmq/` from `Entrpi/ds4:mmq-step-A-full-layer-graphs` and added
MMQ objects to the local Makefile:

- `cuda/mmq/ds4_ggml_stubs.o`;
- `cuda/mmq/ds4_mmq.o`;
- `cuda/mmq/quantize.o`;
- `cuda/mmq/mmid.o`;
- `cuda/mmq/mmvq.o`.

Build command:

```sh
make -j$(nproc) cuda-spark
```

Build result: success. All standard CUDA Spark binaries link:

- `ds4`;
- `ds4-server`;
- `ds4-bench`;
- `ds4-eval`;
- `ds4-agent`.

Runtime status: inactive. No dispatcher path calls MMQ/MMVQ yet.

### 2026-05-23 runtime probe pass

Added opt-in runtime routes for the imported scaffold:

- `DS4_CUDA_MMQ_Q8_DENSE_VEC_ATTN_Q_B=1`: single-token Q8 `attn_q_b`
  dense-vector path through MMVQ;
- `DS4_CUDA_MMQ_Q81_PERSISTENT=1`: persistent Q8_1 scratch buffer for MMVQ
  activation quantization;
- `DS4_CUDA_MMQ_MOE_GATE_UP=1`: generic MMQ IQ2 routed gate/up pair with local
  exact clamp/router SwiGLU;
- `DS4_CUDA_MMVQ_MOE_GATE_UP=1`: fused MMVQ IQ2 routed gate/up with in-kernel
  DS4 clamp and router weights.

Build command:

```sh
make -j$(nproc) cuda-spark
```

Build result: success.

Benchmarks used the standard 8192/128 smoke against same-run `exact_fast`:

| Row | Same-run control | Candidate gen t/s | Decision |
| --- | ---: | ---: | --- |
| `mmq_q8_dense_vec_attn_q_b` | 16.02 | **9.65** | reject |
| `mmq_q8_dense_vec_attn_q_b_persist` | 15.95 | **15.72** | reject, near but slower |
| `mmq_moe_gate_up` | 15.96 | **2.59** | reject |
| `mmvq_moe_gate_up_persist` | 16.37 | **10.15** | reject |

The persistent Q8_1 scratch proved that the worst Q8-vector regression was
allocation overhead, but it still did not beat the native warp8 path. Generic
MMQ and fused MMVQ routed MoE both lose badly to the existing qwarp/LUT gate-up
kernel on single-token decode.

Runtime decision: park these routes as negative diagnostics. The imported
scaffold remains useful for reference, but the tested calls are not promotion
candidates.

## Next Gate

Do not broaden the generic MMQ/MMVQ dispatcher. A future tensor-core route must
start from a kernel/layout redesign with a clear profiler reason, not another
wrapper around the imported dense or MoE vector calls.
