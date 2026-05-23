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

## Next Gate

Add one opt-in single-token Q8 projection route, then compare:

- same-run `exact_fast`;
- candidate with the new MMQ/MMVQ env flag;
- context 8192, allocation 100000, 128 generated tokens;
- deterministic/logprob or `ds4-eval` only if the speed gate clears.

Abort or park the route if it is slower, graph-capture-hostile, or shows quality
drift without a meaningful speed margin.
