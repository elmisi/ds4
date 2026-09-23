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

Current next step: profile the validated pool-enabled path at 65536 populated
tokens, 262144 allocated context, cache argument 72GB, decode graphs OFF.
Separate Engram waits, selected expert loads/cache misses, GPU compute, and
prefill stage/prefetch waits before choosing a further optimization. Preserve
exact numerical output. Keep Q8 and the older queue experiment disabled.

Existing raw artifacts remain under
`/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/engram-pool-*`.
Do not overwrite them. Full raw logits and model files are local, not pushed;
commit code, tests, small summaries and reproduction instructions only.

Aliases/services have not been deployed or restarted. Never compete with a live
model service or stop one without authorization. Other worktrees, including
`dgx-performance` and the older `ds4-ds41` research branch, remain separate.
