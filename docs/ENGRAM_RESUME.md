# Engram / GB10 work checkpoint

Updated 2026-09-23. Active worktree: `/home/alessandro/projects/ds4-engram-external`.
Branch: `feature/external-engram-gguf`, remote: `origin` (`elmisi/ds4`).

## Active prefill investigation (resumed 2026-09-23)

Latest: stage-sync clean pairs2/3 mean +6.38% (large variation). Nsight OFF/ON
same kernel count/time, whole-trace idle gaps48.22->26.71s; no disk causation
claimed. Full evidence: `PREFILL_STAGE_SYNC.md`. Stage option remains default OFF.

New independent candidate `DS4_CUDA_V41_TOPK_BATCH=1`: causal batched exact top512
using the existing streaming selection with per-row visible bounds. Single GPU,
rows32..65535, width>8192; option must equal1, disable flags honored.
Initial NaN ordering test failed (`prefill-topk-unit-on-v1`); fixed with visible
NaN scan and fallback to original per-row implementation. This introduces a
small flag read/synchronization whose cost MUST remain in timing. No extra score
matrix allocation. `prefill-topk-unit-on-v2` PASSED all expanded causal tests,
including visible131071, both ratios, ties, NaNs, infinities, signed zero,
future-score masking and output bounds. All five native binaries rebuilt.
Full GPU test suite is next/current, then new exact-gated64K model suite with
`prefill_stage_suite.py --toggle DS4_CUDA_V41_TOPK_BATCH`; stage-sync stays OFF
to isolate this candidate. Do not deploy either option yet. Fresh quota68%
at10:11:54 UTC (historical, always refresh).

User explicitly asked to continue investigating prefill with the 60% quota floor.
Fresh telemetry at start: 73% (historical, recheck before further work).
Completed `prefill-stage-isolate64k-v1`: existing final native binary,
only `DS4_METAL_V41_STAGE_PROFILE=1` enabled among profiling flags, pool8,
65536 populated/262144 allocated context, cache72, 256 full decode vectors.
All 256 full vectors equal `engram-native-64k-v1-gate-on8/decode.f32`.
Prefill375.30/decode9.52tok/s, wall206.11s, no process swap. This is one profiled
point, not a promoted gain. Added CUDA-only default-OFF experimental option
`DS4_CUDA_V41_PREFILL_STAGE_SYNC=1`: same seven stage barriers without clocks or
logging; 0/unset retains the normal path. Other backends unchanged.
All five binaries built. Host suite-planning/comparator tests pass.
Next/current suite `prefill-stage-sync64k-v1` uses `prefill_stage_suite.py`:
OFF gate must match pre-change native reference; ON gate must match OFF; only
then run two clean timing pairs ordered ON/OFF/OFF/ON. Both use pool8/cache72,
graphs0, no profiling, 64K populated/256K allocated. Keep binary unchanged while
the suite runs. Any significant candidate requires further long-context gating.

Implementation/harness committed and pushed as `53e680b9`. Both gates PASS:
all 256 full vectors identical, including OFF versus the pre-change reference.
Gate rates OFF349.58/ON367.67 prefill tok/s. Clean timing1 ON392.20; timing1 OFF
329.26 but peak process swap85588KiB, so exclude the ENTIRE first pair from
performance claims. Pair2 completed clean: OFF350.17/ON355.01 (+1.382%),
decode9.48/9.49. This is much smaller than the contaminated aggregate.
Pair3 ON/OFF is being run as a replacement, with the identical binary SHA256
`2c92d5b2e9b1fc2a5a92e96424ca9bd415e2346dc4e6b6042b7cab790c17ba0f`.
Source is 53e680b9 plus later harness/docs-only commits; no rebuild since gates.
Do not use the original suite's unfiltered
arithmetic mean as a clean gain: use `prefill_timing_summary.py --raw RAW
--prefix prefill-stage-sync64k-v1 --telemetry RAW/prefill-stage-sync64k-v1-telemetry.jsonl`.
This preserves all samples and rejects failed/incomplete/swapped pairs.
The suite now uses this filtered reporter automatically for future summaries.
Read-only sysfs monitor `prefill_telemetry.py` is running for1100s from around
09:35 UTC, sampling temperatures and disk sector counts. NVMe ~54..65C observed;
no thermal cause established. No swap settings or desktop changes.
Nsight Systems2025.3.2 available; CPU sampling forbidden by perf_event_paranoid4.
If used after the suite, choose CUDA/OS-runtime trace with sample/cpuctxsw NONE;
monitor the actual ds4-bench child, not just the nsys wrapper, for memory/swap.

## Completed work and prior evidence

The user explicitly authorized continued profiling and optimization, regular
commits/pushes, and persistent checkpoints. Check fresh quota with
`/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/check_quota.py` before and
after bounded steps; stop below 60% or on unknown/stale (>600 s) telemetry.
Do not reuse the historical percentage in this file as authorization telemetry.

Completed: alternate Engram GGUF support (459942ff), persistent decode reader
pool (opt-in `DS4_CUDA_ENGRAM_READERS=8`), exact 4K/32 and 64K/256 decode gates,
ASan/UBSan/TSan tests, builds and one clean timing pair (+6.50% decode on Lexar).
Full evidence and commands: `ENGRAM_PARALLEL_DECODE.md`.

Pool implementation is committed as `44cdb00e`. Publication unblocked on
2026-09-23 after the user renewed GitHub login/setup: push to
`origin/feature/external-engram-gguf` succeeded, with remote `e571a916` verified
identical to local HEAD before this checkpoint update. Upstream tracking is set.
Earlier HTTPS attempts had missing/rejected credentials; SSH host verification
also failed. The user reports frequent reauthentication: investigate separately,
without assuming the keyring is the cause or changing credentials implicitly.

Profiling committed as `3498f905`. Completed `engram-profile-64k-v1`: 228.13 s
wall, 333.06 prefill tok/s, 9.57 decode tok/s, no process swap, 256 decode
vectors. IMPORTANT: this is a profiled, partially rebuilt binary, not a clean
throughput comparison. `cuobjdump --list-elf` exposed stale sm_75 objects in
the previous binaries despite linking with CUDA_ARCH=sm_121. Recompiling the
instrumented backend changed it to sm_121a, but the MMQ objects remained sm_75.
The new full logits differ from the old reference; do not claim exactness or
promote this architecture change without further quality evaluation. Old pool
OFF/ON comparisons remain internally valid, but do not represent native GB10.
Preserved `ds4-bench-hybrid` and `ds4-agent-old-sm75` in that run's raw directory.
Scope: verified the actual `ds4-agent-ds41` alias still launches
`/home/alessandro/projects/ds4-gx10/ds4-agent` (72GB cache, no Engram alternate).
Its cuobjdump output is ALREADY entirely sm_121a. The stale build problem was
in this experimental worktree; do NOT promise a 3.6x prefill gain for the user's
currently configured agent, nor imply it was running the broken sm_75 image.

Build invalidation fix committed `6ef951de`; ALL 9 cubins verified sm_121a.
Regression test covers first build, reuse, arch changes, switch back, compiler/
flags/includes changes. V4.1 CUDA and MXFP4 numerical unit tests passed.
Current guarded suite `engram-native-64k-v1`: OFF gate completed (218.11 s wall,
352.92 prefill, 8.89 decode tok/s, no process swap). ON gate passed: all 256 full
vectors byte-identical, SHA256
`9dee9a5d2cc7d4a254716e50e07cd995e58b2c885fdea7fcdd36b36eb47a8298`.
Suite complete. Clean ON timing: 357.92 prefill, 9.54 decode, 9.61 steady tok/s,
214.12 s wall; OFF: 354.18 / 9.09 / 9.16, 218.12 s wall. Decode +4.95% in one
clean pair. Full details and limitations in `ENGRAM_NATIVE_GB10.md`.

Verified correctness fix: `ds4_cuda_attn_tokentile_arch_ok` checked only
the physical device, not compiled kernel PTX. Pre-Ampere compilation contains
zero/no-op HMMA/cp.async branches, so add binary/PTX >=80 checks. Reproduce with
the existing `tests/test_deepseek41_cuda --tp-attention` test using isolated
sm_75 builds before/after guard; do not let these GPU tests compete with the
running suite. Pre-guard source preserved in
`/tmp/ds4-cuda-arch-audit.rZJ8DF/ds4_cuda_before_guard.cu`.
`engram-arch-guard-audit-v1` completed successfully: unguarded sm_75 fails the
split-head oracle at 129 rows, guarded sm_75 passes through 2048 rows. Commands,
isolated sources/objects, logs and result.json are preserved in its raw directory.
Guard committed `23c5af8b`. Native rebuild complete (all nine cubins sm_121a),
full `engram-native-guard-unit-v1` CUDA suite PASS. Final benchmark SHA256
`ca5164fc3b43b9482390fcd10626bafa3e801cac2be5f85d357556a15c83ce8a`;
CLI SHA256 `2564b032cb07fbbabc8087c01c251bdf3e386f936e556e32d4e686c9feb295bf`.
`engram-native-profile-64k-v1` completed: all 256 vectors byte-identical to
`engram-native-64k-v1-gate-on8/decode.f32`. Native profile: Engram waits 0.02313
ms/token, expert loads 29.92 ms/token; report updated with stage breakdown.
`engram-native-recall-256k-v1` PASSED strict checker, 16/16 associations at
254902 actual tokens, final CLI, Lexar pool8/cache72/temp0/nothink. Prefill237.50,
generation7.86 tok/s, wall1086.56s, no process swap, min available19.16GiB.
Checker report in RUN/recall.json. This is quality evidence, not a paired speedup.
Final run `engram-native-final-clean64k-v1` completed: unprofiled final-build
355.59 prefill / 9.55 decode / 9.63 steady tok/s, wall216.13s, no process swap.
No benchmark jobs remain running. Native profiling's399.73 prefill tok/s is
NOT a promoted speedup. Next bounded research lead: isolate stage-profile GPU
barriers from other profiling flags, then repeated exact-gated A/Bs before any
opt-in scheduling implementation. Keep Q8/queue disabled; do not repeat the
completed old queue/cache72 suites. Deployment was not requested/performed.

Old-to-hybrid full-logit diagnostic: 0/256 exact vectors, 144/256 matching top1,
max absolute difference 22.8622, RMS 2.51945, mean KL(old||hybrid) 1.72623.
These differences are NOT acceptable numerical noise; resolve the build
correctness issue before treating either architecture as an output oracle.

Profile mean decode: 104.38 ms total; Engram wait0+wait1 ~0.00037 ms;
expert cache load 29.43 ms (includes reads/uploads/victim selection), misses
4.5573%, logical expert reads 25.96 GiB/256 tokens. Prefill measured stage sums:
attention core/index 72.36 s, shared/routed FFN 62.67 s, attention output 23.34 s.
These are nested timings and do not prove a speedup achievable by parallel I/O.
Added opt-in `DS4_CUDA_V41_DECODE_PROFILE` and `DS4_CUDA_SSD_CACHE_PROFILE`;
reuse existing `DS4_METAL_GRAPH_PREFILL_PROFILE`, `DS4_METAL_V41_STAGE_PROFILE`,
and `DS4_CUDA_SSD_PREFETCH_PROFILE`. Interpret nested intervals carefully.
Summarize with `python3 speed-bench/engram_profile_summary.py RUN/stderr.log`.
Compare RUN/decode.f32 to the preserved `engram-pool-64k-v1-gate-on8` reference;
`speed-bench/compare_decode_logits.py` reports numeric/top1/KL differences,
but these diagnostics are not by themselves a quality acceptance gate.

Existing raw artifacts remain under
`/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/engram-pool-*`.
Do not overwrite them. Full raw logits and model files are local, not pushed;
commit code, tests, small summaries and reproduction instructions only.

Aliases/services have not been deployed or restarted. Never compete with a live
model service or stop one without authorization. Other worktrees, including
`dgx-performance` and the older `ds4-ds41` research branch, remain separate.
