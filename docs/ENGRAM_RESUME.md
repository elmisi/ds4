# Engram / GB10 work checkpoint

Updated 2026-09-23. Active worktree: `/home/alessandro/projects/ds4-engram-external`.
Branch: `feature/external-engram-gguf`, remote: `origin` (`elmisi/ds4`).

The user explicitly authorized continued profiling and optimization, regular
commits/pushes, and persistent checkpoints. Check fresh quota with
`/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/check_quota.py` before and
after bounded steps; stop below 60% or on unknown/stale (>600 s) telemetry.
Do not reuse the historical percentage in this file as authorization telemetry.

Completed: alternate Engram GGUF support (459942ff), persistent decode reader
pool (opt-in `DS4_CUDA_ENGRAM_READERS=8`), exact 4K/32 and 64K/256 decode gates,
ASan/UBSan/TSan tests, builds and one clean timing pair (+6.50% decode on Lexar).
Full evidence and commands: `ENGRAM_PARALLEL_DECODE.md`.

Pool implementation is committed as `44cdb00e`. Push attempted, blocked:
HTTPS credentials unavailable; `gh auth status` reports invalid token for elmisi;
SSH fails host-key verification. User has been asked asynchronously to restore
GitHub login. Do not claim remote backup until push is verified.

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

Current step: fix Makefile configuration invalidation and rebuild ALL CUDA
objects for sm_121a; regression test covers first build, reuse, arch changes,
switch back, compiler/flags/includes changes. Next run native pool OFF/ON exact
64K/256 gates and clean timing, then evaluate architecture-induced differences
and real output. Keep Q8 and the older queue experiment disabled.

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
