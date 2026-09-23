# Engram / GB10 work checkpoint

Updated 2026-09-23. Active worktree: `/home/alessandro/projects/ds4-engram-external`.
Branch: `feature/external-engram-gguf`, remote: `origin` (`elmisi/ds4`).

## Prefill investigation: current checkpoint

User explicitly resumed prefill research on2026-09-23 with quota floor60%.
Run fresh `check_quota.py` before/after bounded work; unknown/stale telemetry
or any window below60% means stop. Do not treat percentages in this file as live.
Latest observed64% around11:00UTC. Keep the desktop and unrelated services intact.

Runtime implementation2452f795; later commits update harness/report/checkpoints.
All five binaries built native sm_121a. Bench SHA256:
`e47291805525ecf8bf8a203f1643760fa29e9b721a35b934277e2f0db9cdda18`.
No alias, service, OS swap setting or deployment was changed.

### Verified top-k candidate (default OFF)

`DS4_CUDA_V41_TOPK_BATCH=1` batches exact causal top512, single GPU,
rows32..65535,width>8192; existing disable flags remain respected.
Visible NaNs use the original path; retain the scan/read synchronization in
timing. No extra score matrix. Detailed evidence: `PREFILL_TOPK_BATCH.md`.

- Expanded exact-ID tests PASS through visible131071, both ratios, dispatch
  boundaries, NaNs/infinities/signed zero/ties/future masking/output sentinel.
- Full GPU numeric suite PASS. Memcheck PASS,0errors,444.17s.
- `prefill-topk64k-v1`: all256 full float32 vocabulary vectors match OFF/ON
  and pre-change reference exactly. Both clean counterbalanced timing pairs
  passed without swap: mean364.305->404.795tok/s prefill (+11.114%), decode
  essentially unchanged.64Kpopulated/256Kallocated,cache72,pool8,graphs0,
  stage-sync OFF. Do not call this populated256K throughput evidence.
- CUDA attribution: GPU kernel count2,106,477->1,800,574; topk-named kernel
  durations21.851->3.169s; actual model swap0. Profiled rates are not benchmarks.

Long-context gate COMPLETE: `prefill-topk-recall-256k-v1`, same actual254902-token
fixture/parameters as native reference. Strict checker PASS,16/16 associations,
exit0,swap0,minavailable18.96GiB,wall966.42s. Prefill267.60tok/s is diagnostic
quality-run timing, not a paired long speedup. All jobs/monitors have ended.
Raw root: `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw`.
Run metadata's git_commit describes runner checkout; use added source provenance
for the correct runtime source. No deployment: flags remain opt-in and the
existing alias still uses its previous worktree/configuration. Further research
must refresh quota; do not rerun completed gates unnecessarily. A combined
top-k/stage-sync A/B or projection-launch investigation would be new work;
neither has an established additional gain. Final report and memory checkpoint
are saved; source and evidence are consolidated on this branch.

### Separate stage-sync experiment (default OFF, not promoted)

`DS4_CUDA_V41_PREFILL_STAGE_SYNC=1` inserts seven barriers without profiling
clocks/logging.64K/256-vector gates exact. Two clean pairs mean+6.38%, but highly
variable. Initial swapped pair excluded entirely; never quote its contaminated
aggregate. Same kernel time under Nsight, smaller whole-trace gaps48.22->26.71s;
no causal disk or thermal claim. Do not add this gain to top-k's gain.
Detailed chronology/artifacts: `PREFILL_STAGE_SYNC.md`. All its jobs/monitors ended.

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
