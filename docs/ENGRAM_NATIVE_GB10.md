# Engram pool: native GB10 build and architecture audit

Date: 2026-09-23. Worktree `ds4-engram-external`, branch
`feature/external-engram-gguf`. No alias/service deployment.

## Build provenance matters

The first Engram pool suite accidentally reused CUDA objects compiled for
`sm_75`; linking with `CUDA_ARCH=sm_121` did not rebuild them. The profiling
change rebuilt `ds4_cuda.o` for `sm_121a` while leaving MMQ objects stale.
`cuobjdump --list-elf`, not the final link command, revealed the mismatch.

Commit `6ef951de` adds a content-dependent Makefile configuration stamp covering
the CUDA compiler, resolved flags, and MMQ includes. The regression test verifies
reuse and invalidation, including switching back to a previous architecture.
All nine cubins in the rebuilt benchmark are now `sm_121a`.

The actual user alias launches `/home/alessandro/projects/ds4-gx10/ds4-agent`.
That binary was separately verified to contain only `sm_121a` images already.
**The ~98 -> ~354 prefill tok/s correction in this experimental worktree is not
a measured speedup over the user's configured agent.** The older pool OFF/ON
exactness results remain valid for those binaries, but they are not a native
GB10 performance baseline or an independently verified model-quality oracle.

## Native OFF/ON results

Binary SHA256: `9d5b4164b66f14198c856022ed0aa130dd81ca911509f0c29b3b64e51a6e977d`.
Source commit `6ef951de`; later commits during this suite only added reports and
host tests. An attention guard was being edited separately but was NOT rebuilt
into this binary during the suite.

Conditions: 65,536 populated prompt tokens, 262,144 allocated context, 256
teacher-forced target-only decode tokens, 72GB expert-cache argument (64.88 GiB
actual, 6,999 slots), external Engram GGUF on Lexar, internal expert GGUF.
Decode graphs OFF; queue/Q8 experimental flags unset. OS page cache uncontrolled.

| Run | Engram readers | Prefill tok/s | Decode tok/s | Steady tok/s | Wall s |
|---|---:|---:|---:|---:|---:|
| gate-off | 0 | 352.92 | 8.89 | 8.97 | 218.11 |
| gate-on8 | 8 | 350.14 | 9.53 | 9.62 | 218.12 |
| timing-on8 | 8 | 357.92 | 9.54 | 9.61 | 214.12 |
| timing-off | 0 | 354.18 | 9.09 | 9.16 | 218.12 |

Full-vector gate: all 256 x 129,280 float32 logits byte-identical OFF/ON.
Both 132,382,720-byte files have SHA256
`9dee9a5d2cc7d4a254716e50e07cd995e58b2c885fdea7fcdd36b36eb47a8298`.

The clean pair has **+4.95% decode throughput** (+4.91% steady) with the pool.
This is one timing pair, not a confidence interval. Prefill does not use this
decode-only pool: the +1.06% prefill difference is not evidence of a pool benefit.
Whole-run wall is ~1.83% lower in this pair; prefill dominates the request.
No process swap; minimum available RAM across the suite was ~19.12 GiB.

Raw evidence:
`/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/engram-native-64k-v1-*`.
Each run includes command, resources, quota checks, summary and source provenance.
Reproduce with `speed-bench/engram_pool_suite.py` and the same guarded runner/model
paths documented in `ENGRAM_PARALLEL_DECODE.md`. Do not overwrite existing runs.

## Profiling and quality caveats

The hybrid profiling run (`engram-profile-64k-v1`) measured ~104.38 ms per decode
token, of which ~29.43 ms was expert-cache load (victim selection, disk reads,
uploads and completion included). Engram waits totaled only ~0.00037 ms/token
with the pool. Miss fraction 4.56%; logical expert reads 25.96 GiB/256 tokens.
Prefill attention core/index was 72.36 s and shared/routed FFN 62.67 s. These
are nested intervals, and the profile is neither a clean timing nor a fully
native MMQ build: repeat a native profile before choosing the next optimization.

Changing compiled architectures was NOT numerically exact:

| Comparison | Matching top1 / 256 | RMS logit difference | Mean KL(left || right) |
|---|---:|---:|---:|
| old sm_75 -> hybrid | 144 | 2.51945 | 1.72623 |
| hybrid -> native | 240 | 0.36526 | 0.01844 |

These diagnostics cannot certify answer quality. Native V4.1 CUDA numerical
tests and MXFP4 parity tests pass. A further bug was found in the token-tile
attention dispatch: it checked the physical GPU but not compiled PTX capability,
although pre-Ampere device code substitutes zeros/no-ops for HMMA/cp.async.
The isolated sm_75 before/after audit **reproduced and fixed the bug**:
unguarded code failed the full-head versus split-head attention oracle at 129
rows; the guarded build passed 1/31/32/33/129/257/2048 rows, with max split
difference zero and the 2048-row double-oracle error 6.70e-8. Both builds use
the same native MMQ objects, and differ only in the attention guard in the
sm_75 main backend. This isolates a correctness defect, not an architecture
rounding comparison. Audit: `engram-arch-guard-audit-v1/arch-audit/result.json`.
Reproduce using the guarded operational runner and
`speed-bench/cuda_arch_guard_audit.py --before-ref 6ef951de --output RUN`.

The guard now checks both `cudaFuncAttributes.binaryVersion` and `ptxVersion`
>=80 before selecting the token-tile kernel, falling back safely otherwise.
The final native rebuild checks are recorded below; real-output validation is
tracked in `ENGRAM_RESUME.md`. Do not promote an architecture transition based
only on the throughput numbers above.

## Final native guard/profile verification

Commit `23c5af8b` rebuilt all binaries for sm_121a; the full numerical V4.1 CUDA
suite passed again (`engram-native-guard-unit-v1`). Benchmark SHA256
`ca5164fc3b43b9482390fcd10626bafa3e801cac2be5f85d357556a15c83ce8a`.
`engram-native-profile-64k-v1` dumped 256 full vectors identical to the native
pre-guard OFF/ON reference above, despite enabling all profiling flags. Thus
the guard and these profiling settings preserve the native output on this gate.

This profile supersedes the hybrid profile for bottleneck assessment:

- Mean decode 105.06 ms; Engram wait0+wait1 **0.02313 ms/token** (~0.022%).
- Expert-cache load **29.92 ms/token** (~28.5%, includes I/O/uploads/victims),
  4.619% misses, 26.31 GiB logical expert reads across 256 decode tokens.
- Prefill stage sums: attention core/index **71.96 s**, attention output
  **23.37 s**, shared/routed FFN **27.92 s**. Foreground expert-prefetch wait
  **12.76 s**, Engram interval **4.00 s**; timings are nested, not additive.

More Engram readers cannot materially reduce the already hidden decode wait.
Attention/prefill scheduling and expert-cache load overlap are better candidates,
but the cache load number must not be mistaken for pure disk latency.

Profile throughput was 399.73 prefill / 9.51 decode tok/s, 194.16 s wall,
no process swap. It includes extra GPU barriers and is not a clean performance
claim: the higher prefill rate versus the earlier clean runs needs a controlled
post-guard timing before attributing it to scheduling, build or run variability.
Real-output long-context recall passed (details below).

## Real-output quality at near-full context

`engram-native-recall-256k-v1`: **16/16 exact name/value associations**, checked
by the existing strict `check_recall.py` against the hash-pinned fixture.
Actual prompt length **254,902 tokens**, allocated context 262,144; native final
CLI, alternate Engram GGUF on Lexar, pool8, cache72, target-only temp0/nothink.
CLI SHA256 `2564b032cb07fbbabc8087c01c251bdf3e386f936e556e32d4e686c9feb295bf`.

Prefill 237.50 tok/s, generation 7.86 tok/s, wall 1086.56 s. No process swap;
minimum available RAM 20,090,008 KiB (~19.16 GiB). These are quality-run timings,
not a paired throughput comparison with the older agent or an OFF run.
Exact stdout and strict checker report are preserved in the run directory.

Final unprofiled 64K timing (`engram-native-final-clean64k-v1`): **355.59 prefill,
9.55 decode, 9.63 steady tok/s**, wall216.13s, no process swap. This agrees with
the earlier native clean results, so the guard did not explain the faster
399.73 tok/s profiled prefill. The next bounded lead is to isolate
`DS4_METAL_V41_STAGE_PROFILE` (extra GPU barriers) from the other profiling flags
and run repeated clean A/Bs before considering an independent opt-in scheduling
change. No new prefill speedup or synchronization optimization is promoted.

Code and small reports through `e571a916` were pushed and verified on
`origin/feature/external-engram-gguf` after renewed GitHub authentication.
See the live checkpoint for publication status and the pending login diagnosis.
Aliases, running services and other performance branches were not changed.
