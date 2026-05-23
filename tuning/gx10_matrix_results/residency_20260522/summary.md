# Residency Probe - 2026-05-22

Context: `ctx_alloc=100000`, `ctx=8192`, `gen_tokens=128`, graph+SoA exact-fast
environment.

| Probe | Artifact | Result | Decision |
| --- | --- | --- | --- |
| `DS4_CUDA_DIRECT_MODEL=1` | `direct_model_exact_fast_bench.csv` | no usable row; CUDA graph decode failed after a long direct-model run | reject |
| `DS4_CUDA_NO_FD_CACHE=1` | none | no CSV; command timed out after 300s | reject |

See also `../model_residency_20260522/copy_model_exact_fast_bench.csv` for the
full device-owned model copy probe: 15.75 t/s after a 457.539s 80.76 GiB copy.
