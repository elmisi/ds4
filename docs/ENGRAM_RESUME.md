# Engram / GB10 work checkpoint

Updated 2026-10-26. Active worktree: `/home/alessandro/projects/ds4-engram-external`.
Branch: `feature/external-engram-gguf`, remote: `origin` (`elmisi/ds4`).

Public showcase: [PR #10](https://github.com/elmisi/ds4/pull/10), intentionally
kept open for visibility and **never to be merged into `main`**, matching the
separate DGX performance showcase [PR #9](https://github.com/elmisi/ds4/pull/9).

## KV checkpoint maintenance 2026-10-16

User authorized a bounded KV-cache investigation, patch, and live validation in
this runtime worktree. The older queue/Q8 and open-ended Engram research remain
paused. Local base: `0ce0033fd53f93f0d29d16e6fc48f34892c5159b`.

The continued-checkpoint scheduler previously required exact multiples of the
20480-token interval. V4.1 progress callbacks arrive at 32768-token frontiers,
and intermediate callbacks can precede a valid final checkpoint. A long prompt
ending at 254896 tokens was consequently never persisted before generation;
its longer post-answer snapshot could not satisfy an identical prompt.

The scheduler now accepts the first valid frontier after an interval boundary,
preserving the interval grid and the actual payload length. The server skips
callbacks whose session checkpoint is not valid. No model kernels, compression
format, intermediate checkpoint validity, or sampling behavior were changed.

Validation artifacts are local to `speed-bench/ds41-kv-audit-20261016/`:

- Host server suite passed, including the new off-grid/duplicate/restore tests.
  The UBSan policy reproduction reports one valid 254896-token save instead of
  zero saves and one invalid attempt in the baseline.
- `raw/payload-roundtrip-254k-v1`: actual 254000-token prefill; real disk save
  and restore; 257 full F32 vocabulary vectors (129280 values each) and 256
  token logprobs bit-identical before/after restore. Append to 258096 passed
  position/prefix checks with finite full logits. Minimum available RAM
  10.576 GiB; process swap zero. This append check is not a full-logit equality
  comparison against an independent 258096-token prefill reference.
- GPU prefill 876.256s, checkpoint save 2.022s, immediate restore 0.367s.
  These are single-run diagnostics; the restore is not a cold-disk measurement.
- HTTP cold/repeated 254896-token strict recall passed 16/16 twice, with
  identical normalized output SHA256
  `61d7160f499bedcb55dd1b8c3b04dac1c5d4ef1484cd04f7868769e418b2df8d`.
  Cold request: 920.450s wall, cache_write_tokens254896, cached_tokens0;
  actual prefill908.388s / 280.60t/s. Repeat: 6.713s wall,
  cached_tokens254896, cache_write_tokens0. The journal confirms a real
  disk-text hit (435ms load) after the expected live-cache rollback miss,
  then zero tokens to prefill (0.019s prompt stage). Minimum available RAM
  10.439GiB; process swap zero; service restarts zero.
  These timings compare a cold prompt with its immediate cached repeat,
  not a paired model-throughput benchmark or a cold-disk restore.
  Raw results: `validation-20261016/http-recall-254k/summary.json`.

The candidate server is installed and the service active via `dgx-ctl`;
API readiness, running executable hash, profile, and restart count verified.
SHA256: `e4a8bb850ae7f7a442c368c9145f1034081b22af42cba83ac9d850be1015eb10`.
The validated October 8 engine/CUDA objects were reused after comparing all
228 tracked build inputs; changed C frontends were recompiled. All nine CUDA
cubins are sm_121a. Build provenance, native cubin listing, and the previous
server backup are under `validation-20261016/`.
Combined evidence: `validation-20261016/validation-report.json`; its deployment
and Git fields record the validation snapshot taken before this commit.
The source fix, regression harnesses, fixtures and text validation artifacts
are committed together. Generated KV/logit payloads and build/rollback binaries
remain on disk, excluded from Git with sizes and SHA256 hashes recorded in
`local-artifacts-manifest.json`. Nothing was pushed or merged; PR #10 remains
a permanent showcase and must never be merged into `main`.

Current service overrides: expert cache80GiB (72.88GiB dynamic plus 7.12GiB
prefill reserve), persistent KV budget64GiB, ctx262144, readers8, topk1,
graphs0, stage-sync0. Unit changes live in amygdala. Swap remains disabled
after the user's manual swapoff. The 64GiB setting is disk-cache capacity;
it is not a 64GiB GPU KV allocation. Historical profiles below are unchanged.

Latest upstream SSD-read result: [ordered CUDA SSD expert reads A/B, 2026-10-26](SSD_ORDERED_READS.md).
On the same 30.5K-token V4.1 service request, the default ordered-read path
reduced mean prefill time by 6.8% across two runs per mode, with byte-identical
16-line recall output. Treat this as a small exploratory sample, not a general
speedup claim.

## Maintenance rebase 2026-10-08

User requested reviewing upstream GB10 changes and rebasing the service branch.
This maintenance does not resume the open-ended research paused below.

- Upstream `main`: `fc80bd695da76ee14bcb86820bf900a5cdfda806` (October 7),
  verified unchanged remotely after validation. Nine upstream commits added.
- All 21 local commits retained; pre-rebase backup:
  `backup/external-engram-pre-main-20261008` at `28b819f2`.
- Tested runtime source: `c516520d`, including the maintenance fix extending
  scalar NaN-order fallback to upstream's newly batched small causal rows.
  Rebase conflicts retained upstream's causal kernel and batching, the local
  wide-row flag/NaN guard, and both sets of frontier tests. The future-NaN
  test pattern is now actually executed. Source worktree is clean.
- Alternate external GGUF, persistent Engram readers, native architecture
  guard and compiler-configuration invalidation retained. Service profile:
  cache72GiB, ctx262144, readers8, topk1, graphs0, stage-sync0, KV32GiB.
  Lexar is mounted ext4/noatime and negotiated UAS at 20Gb/s.

Validation: native build; all four installed binaries contain nine sm_121a
cubins each. Engram and alternate-GGUF tests, ASan/UBSan Engram tests, full
V4.1 CUDA numeric suite, causal top-k ON/OFF through visible131071 (both
ratios, NaNs, future masking, ties and output tails), shared-expert,
token-tile and streaming-top-k tests passed. Top-k CUDA memcheck: zero errors.

- `raw/rebase-20261008-logits64k-v2`: 65536 populated / 262144 allocated,
  256 complete F32 vocabulary vectors byte-identical to
  `prefill-topk64k-v1-gate-on/decode.f32`; model swap0. SHA256:
  `9dee9a5d2cc7d4a254716e50e07cd995e58b2c885fdea7fcdd36b36eb47a8298`.
- `raw/rebase-20261008-recall254k`: actual254902 tokens, strict recall16/16,
  output byte-identical to `prefill-topk-recall-256k-v1`; model swap0,
  minimum available18.63GiB, wall962.41s. Single-run diagnostic rates:
  prefill268.66t/s, generation7.96t/s. No paired speedup claim.
- Raw root: `/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw`.
  Each new run has separate runtime source provenance: the runner's own
  checkout commit is not the tested runtime commit. The first logits run
  failed writing a literal environment placeholder; v2 fixed the path.

Service was stopped using `dgx-ctl stop ds4-ds41` as requested. Validated
`ds4`, `ds4-server`, `ds4-bench`, `ds4-agent` binaries were installed into
the service worktree; the service remains inactive and uses them next start.
Build logs, cubin lists, hashes and rollback binaries:
`/home/alessandro/projects/ds4/logs/rebase-ds41-20261008/`.
No launcher/unit changes. After renewed GitHub authentication, the rebased
branch was pushed at `5dd72b0c` using `--force-with-lease` against the verified
old remote tip `28b819f2`; the pre-rebase backup above remains local.
PR #10 publishes the implementation, validation and reproduction reports.
Upstream now includes causal batching too: the historical local
+11.114% cannot be added to, or claimed against, this new upstream baseline.

## PAUSED — handoff 2026-09-23 evening (supersedes older deployment notes)

User requested saving research and resuming another time. No further research,
benchmarks or delegation until explicit resume. Global quota floor is now **50%**,
not 60%. Invoke `check_quota.py --threshold 50` explicitly (script default may
still be 60). Require fresh <=600s valid telemetry; unknown/future/stale stops.
Last observed 61% is historical, not future authorization. Child-session telemetry
can be missing/stale: obtain a fresh parent guard, never assume quota is enough.

### Operational handoff

The manual `ds4-agent-ds41` alias and `dgx-ctl` profile `ds4-ds41` now use
this worktree's existing native sm_121a binaries, runtime implementation2452f795.
No inference source was changed or speculative fork optimization deployed.
Configuration: CUDA SSD streaming, expert cache72GB, ctx262144, internal V4.1
Flash Q2 GGUF, alternate identical GGUF under
`/mnt/ds4-models/models/DeepSeek-V4.1-Flash-Q2.gguf`, readers8, top-k1,
decode graphs0, stage-sync0. Q8/queue experiments remain unpromoted.

Agent: `~/.zshrc:53`. Server launcher and unit:
`/home/alessandro/elmisi/amygdala/scripts/start-ds4-ds41.sh` and
`systemd-user/ds4-ds41.service`, installed in `~/.config/systemd/user/`.
Server retains existing host0.0.0.0:8000, persistent KV32GiB and continued20480.
Other service profiles unchanged. Existing unrelated dirty amygdala changes
were preserved; operational edits there remain uncommitted with that worktree.
Both binaries contain nine cubins, all sm_121a. SHA256:
- agent: aa4bfb4620b5bbf0751785b9af4591122a09eb578b0d13f84cbea94b3153cde4
- server: a13bfc44fa3b8df381163eb9f3ceb559f833e62ad8c4f043a1aca0ef52961058

Shell syntax and systemd unit verification passed, daemon-reload performed.
A faulty dry-run wrapper accidentally launched the real server, which reached
listening with external Engram/cache72/context262144; it was immediately stopped
via SIGTERM with orderly shutdown. No other agent/server was found running before
that check. This is startup evidence only, NOT an inference/quality benchmark.
Final service state inactive, autostart disabled; no session was restarted.

### Next experiments: V4.1 ONLY

1. Reproduce agent incremental prefill on a retained KV context with append1/2/4K;
   separate KV reuse/recomputed tokens, prefill, first-response latency, decode.
   Compare top-k ON with stage-sync OFF/ON, repeated counterbalanced clean pairs.
   64K is preliminary; populated near256K is the final decision workload.
2. Profile V4.1 attention projections, then audit upstream PR979 commit
   ad19000e06ed: https://github.com/antirez/ds4/pull/979 . Luna found a plausible
   shared Q8 attention-output entry point, NOT proof of applicability or novelty.
   Reported +10.65% belongs to V4 Flash Q2, not V4.1. Verify actual diff/dispatch,
   layout/rounding and hotspot before any opt-in port. No promised gain.
3. Decode: split expert-load cost into disk reads, transfers and cache management.
   Previous native profile: expert loads29.92ms/token vs Engram waits0.02313ms;
   increasing Lexar readers is low priority. Keep cache72, desktop and swap safety.
4. Secondary ideas from Luna read-only scan (no implementations/benchmarks):
   - https://github.com/antirez/ds4/pull/822 : small continuation fragment at
     prefill-cap boundary; Metal/ROCm idea needs V4.1 CUDA adaptation.
   - https://github.com/Entrpi/ds4/commit/574ccbba3228cd3ab47a388cf47f522bfe10ec :
     RMS+F16 projection fusion; CUDA HC path currently separate, Apple fused.
     Numerical equivalence and scratch memory require verification.
   Selected-expert-cache PR1031 and quantum/batching were reported already present;
   don't reimplement without checking. Backend/model mismatches are not evidence.

All candidates need exact full-vocabulary logit comparison, real output,
long-context recall and appropriate sanitizer gates. Do not compound individual
speedup percentages or compare incremental agent70t/s with bulk405t/s directly.
Observed agent22.6K context generation7.0t/s was a live TUI sample, not controlled
A/B. Its output was accessible read-only with tmux capture-pane (previously %14);
rediscover pane/PID on resume, never read /dev/pts directly. No trace file enabled.

Historical sections below describe earlier stages and retain their evidence;
their statements about no deployment, old alias and quota60 are superseded here.

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
