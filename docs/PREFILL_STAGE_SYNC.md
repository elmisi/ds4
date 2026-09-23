# V4.1 prefill stage synchronization experiment

2026-09-23, GB10, worktree `ds4-engram-external`, branch
`feature/external-engram-gguf`. Experimental, not deployed.

## Hypothesis and controls

Stage profiling inserted seven GPU barriers and appeared faster than ordinary
prefill. `DS4_CUDA_V41_PREFILL_STAGE_SYNC=1` isolates those barriers without
stage clocks/logging. Default OFF; CUDA only. Arithmetic/dispatch unchanged.

All runs: 65536 populated tokens, 262144 allocated context, teacher-forced
256-token decode, cache argument72GB, Engram reader pool8, decode graphs0.
Internal expert/model GGUF, alternate Engram GGUF on Lexar. Native sm_121a.
No filesystem cache dropping, swap-policy changes or service/alias deployment.

Binary SHA256:
`2c92d5b2e9b1fc2a5a92e96424ca9bd415e2346dc4e6b6042b7cab790c17ba0f`.
Runtime source committed at53e680b9; subsequent commits only change harness/docs.
Raw root: `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw`.
Prefix: `prefill-stage-sync64k-v1`.

## Exactness gate

Both gate runs wrote all256 float32 vocabulary vectors, 129280 values/vector.
OFF matches the pre-change native reference; ON matches OFF byte-for-byte.
SHA256: `9dee9a5d2cc7d4a254716e50e07cd995e58b2c885fdea7fcdd36b36eb47a8298`.
This does not replace a populated near256K quality gate.

## Unprofiled timing

| Pair / order | OFF prefill tok/s | ON prefill tok/s | Process swap | Use |
|---|---:|---:|---|---|
| 1 ON/OFF |329.26|392.20|OFF85588KiB|Exclude entire pair|
| 2 OFF/ON |350.17|355.01|0 both|Clean, +1.38%|
| 3 ON/OFF |364.82|405.57|0 both|Clean, +11.17%|

Clean means: OFF357.495 / ON380.29 tok/s, +6.38%. Only two pairs with large
variation: promising, not a robust promoted gain. Decode stays around9.5tok/s.
Clean pair2 wall218.11/216.11s OFF/ON; pair3 wall210.14/192.13s OFF/ON.
Do NOT quote the original two-pair suite's unfiltered +9.98% aggregate.
`prefill_timing_summary.py` retains all samples and excludes a whole pair on
failed/stopped runs, missing/unknown process swap, or nonzero process swap.
The suite now uses this filtered summary automatically.

NVMe temperatures observed about54..65C; no established thermal explanation.
Host disk readings during pair1 OFF: ~206.88GiB internal, ~4.79GiB Lexar over
228s. These are host-wide counters and do not attribute latency to either disk.

## Next attribution

Nsight Systems CUDA/OS-runtime trace, CPU sampling/context-switch collection
disabled (no permission changes). Keep gen-tokens1 to focus on prefill.
Use separate names `prefill-stage-nsys64k-{off,on}-v1`. Profiled rates must not
be pooled with unprofiled timing. The resource runner observes the nsys parent;
use `prefill_telemetry.py --process ds4-bench` for actual child RSS/swap evidence.
Require long-context quality validation before promoting the option.

Both traces completed successfully, with actual model VmSwap0 in the separate
process monitor (94 OFF/83 ON sampled model observations). Same2,106,477 GPU
kernel events. Summed kernel durations OFF143.336s/ON143.641s. Whole-trace GPU
activity union (kernels+copies+memsets, NOT an isolated prefill range):
OFF146.518s active/48.217s gaps; ON145.259s active/26.713s gaps.
The shorter elapsed span is predominantly fewer gaps in recorded GPU activity,
not faster kernel arithmetic. This does not prove CPU versus disk causation.
API wait time shifts from memcpy calls to explicit synchronization; do not
interpret summed API durations as transfer or disk time. CUDA copy time ~4s.
Profiled prefill rates341.12/386.94 are diagnostic, not clean benchmark samples.

Trace also exposes a separate opportunity: per-row causal top-k launches cost
~19.1s in chunk+merge kernels and hundreds of thousands of launches. The existing
streaming exact top512 kernel can be specialized for per-row visible bounds.
New independent experiment `DS4_CUDA_V41_TOPK_BATCH=1` is default-OFF, single-GPU,
rows>=32 and physical width>8192. It preserves the packed score/index total order,
never reads future entries, and needs no extra score buffer. Existing disable
flags remain respected. Unit oracle extended through visible131071, ratios1/2,
row-count dispatch boundaries, ties, infinities, NaNs, signed zeros, attractive
future entries and output sentinel. Initial NaN test failed because packed-key
sort and comparison sort order NaNs differently. Fixed with a visible-only
integer-bit NaN scan and original-path fallback; flag read introduces one
stream synchronization, retained in all timing. Test `prefill-topk-unit-on-v2`
passes the expanded causal oracle. No model correctness/speed claim until gates.
