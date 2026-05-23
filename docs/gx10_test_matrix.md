# GX10 Test Matrix

This matrix is the reproducible follow-up to `docs/road_to_20_ts.md`.  Its
purpose is to separate three questions:

1. How much did the current branch gain over `origin/main`?
2. Which exact, quality-preserving toggles still help on the current branch?
3. Which research/tradeoff toggles are worth re-testing only when a related
   kernel changes?

The current manual baseline is:

| Build | Agent throughput | Interpretation |
| --- | ---: | --- |
| `origin/main` worktree at `8d57664` | 14.6 t/s | upstream-after-rebase baseline, including native `ds4-agent` |
| current branch, plain `ds4-agent` | 16.3 t/s | most branch optimizations are compiled in |
| current branch, `ds4-agent-exact-fast` | 16.7 t/s | branch plus graph decode and Q8 SoA cache |

Current branch state after the upstream rebase: `gx10-cuda-graph-decode` is
based on `upstream/main` / `origin/main` at `8d57664` and carries the local GX10
work on top. The baseline above is therefore not the old pre-agent upstream; it
already includes the 2026-05-21 upstream server/agent/KV work.

Latest matrix artifacts:

| Run | Artifact | Key result |
| --- | --- | --- |
| Core 8192/128 pass | `tuning/gx10_matrix_results/core_20260522_1537/summary.md` | `soa` 16.07 t/s and `exact_fast` 16.05 t/s are effectively tied in `ds4-bench` |
| Exact sweep 2048..65536 | `tuning/gx10_matrix_results/exact_20260522_1540/summary.md` | only `soa_b_forced` beat `exact_fast`, by +0.98% mean; repeat before any promotion |
| Residency probes | `tuning/gx10_matrix_results/model_residency_20260522/`, `tuning/gx10_matrix_results/residency_20260522/` | full model copy, direct model and no-FD placement did not beat exact-fast |
| Arena placement probes | `tuning/gx10_matrix_results/arena_20260522/summary.md` | default arena chunk won; 1024 MiB failed cache, 4096/8192 MiB were slower |

Fork reconnaissance is tracked separately in `docs/gx10_fork_recon.md`. Rows
below are only rows that are executable on this branch. Fork ideas that require
porting, such as Entrpi's `cuda/mmq` + layer-graph stack, are not represented as
env rows until the code exists here.

## Run Harness

Use `tuning/gx10_matrix.py` to keep environment cleanup consistent across rows.

```sh
python3 tuning/gx10_matrix.py list
python3 tuning/gx10_matrix.py bench exact_fast --ctx-alloc 100000 --ctx-start 8192 --ctx-max 8192 --gen-tokens 128
python3 tuning/gx10_matrix.py bench-suite core --ctx-alloc 100000 --ctx-start 8192 --ctx-max 8192 --gen-tokens 128
python3 tuning/gx10_matrix.py summary
python3 tuning/gx10_matrix.py eval exact_fast --ctx 100000 --repeat 1 --canary
python3 tuning/gx10_matrix.py ds-eval exact_fast --questions 4 --tokens 1024 --nothink --seed 1
```

For server-only manual testing:

```sh
python3 tuning/gx10_matrix.py server exact_fast --ctx 100000 --port 8106
python3 tuning/coding_eval_extended.py --base-url http://127.0.0.1:8106 --label exact_fast_manual --repeat 1
```

The runner sanitizes known `DS4_*` experiment flags before applying a row. This
matters because most variants are opt-in env toggles and stale shell state can
invalidate an A/B result.

## Systematic Benchmark Protocol

The default speed pass is intentionally small enough to run often:

```sh
python3 tuning/gx10_matrix.py bench-suite core \
  --ctx-alloc 100000 --ctx-start 8192 --ctx-max 8192 --gen-tokens 128
```

This writes one CSV per row under `tuning/gx10_matrix_results/` and produces:

- `summary.csv`
- `summary.json`
- `summary.md`

When a row beats `exact_fast` by at least 1-2%, expand the sweep:

```sh
python3 tuning/gx10_matrix.py bench-suite exact \
  --ctx-alloc 100000 --ctx-start 2048 --ctx-max 65536 \
  --gen-tokens 128
```

When a speed result survives the expanded sweep, run the coding canary:

```sh
python3 tuning/gx10_matrix.py eval-suite exact_fast <candidate-row> \
  --ctx 100000 --canary --repeat 1
```

Also run the `ds4-eval` capability smoke before a long gate:

```sh
python3 tuning/gx10_matrix.py ds-eval-suite exact_fast <candidate-row> \
  --questions 12 --tokens 4096 --think --seed 1
```

Only after canary parity and `ds4-eval` smoke parity, run the full repeated
coding gate:

```sh
python3 tuning/gx10_matrix.py eval-suite exact_fast <candidate-row> \
  --ctx 100000 --repeat 3
```

The long `ds4-eval` gate is reserved for candidates that survive all earlier
speed and quality filters:

```sh
python3 tuning/gx10_matrix.py ds-eval-suite exact_fast <candidate-row> \
  --questions 92 --tokens 16000 --think --seed 1
```

Record every executed pass in `docs/road_to_20_ts.md`: command, date, context,
output artifact path, best/worst throughput, pass rate, and decision.

## Promotion Rules

Use the same prompt, context, seed and token budget for every row in a batch.

| Stage | Command | Promote when |
| --- | --- | --- |
| Speed smoke | `bench` at `ctx_alloc=100000`, `ctx_start=8192`, `ctx_max=8192`, `gen_tokens=128` | at least 1-2% faster than `exact_fast` |
| Coding canary | `eval --canary --repeat 1` | no new failures vs `exact_fast` |
| `ds4-eval` smoke | `ds-eval-suite exact_fast <candidate> --questions 12 --tokens 4096 --think --seed 1` | pass count no worse than same-run `exact_fast` |
| Full coding gate | `eval --repeat 3` | matches or beats the `30/36` repeated baseline and keeps speed gain |
| Long `ds4-eval` gate | `ds-eval-suite exact_fast <candidate> --questions 92 --tokens 16000 --think --seed 1` | no pass-count regression vs same-run `exact_fast` |
| Agent check | manual `ds4-agent` run | no obvious UX/output regression, memory still fits the target deployment |

Rows already known to regress speed or quality should not consume a full coding
gate unless their underlying kernel changed.

## Best-Effort Timebox

For the next full-quality attempt, stop after two targeted kernel prototypes or
about one short working day of implementation/profiling, whichever comes first.
Stop earlier if no prototype clears at least +3% in the 8192/128 speed smoke, or
if the only faster prototype fails deterministic logprob, coding canary, or
`ds4-eval` smoke. This keeps the 20 t/s push disciplined without closing the
door on hard but plausible kernel work.

## Matrix Rows

### Baselines

| Row | Category | Env delta | Status from `road_to_20_ts.md` | Next action |
| --- | --- | --- | --- | --- |
| `plain` | branch baseline | none after sanitization | current branch already includes most CUDA wins | measure against `origin/main` only |
| `graph` | exact-safe isolate | `DS4_CUDA_GRAPH_DECODE=1` | useful but not enough alone | speed smoke |
| `soa` | exact-safe isolate | `DS4_CUDA_Q8_SOA_CACHE=1` | positive memory-for-speed path | speed smoke |
| `exact_fast` | production exact-safe | `graph + SoA` | recommended production-safe default, about 17.4-17.9 t/s server decode in warm logs | baseline for all branch A/B |
| `soa_b_forced` | exact-safe diagnostic | force `attn_output_b` through generic SoA decode | latest exact sweep was +0.98% mean vs `exact_fast`, below promotion threshold | repeat `exact_fast` vs this row before any canary |

### Exact Diagnostic Rows

| Row | Env delta over `exact_fast` | Prior result | Next action |
| --- | --- | --- | --- |
| `soa_qb` | `DS4_CUDA_Q8_SOA_QB=1` | microbench positive, full decode neutral/negative; 2026-05-23 recheck 16.00 vs 16.06 control | diagnostic only; do not promote |
| `soa_qkv` | `DS4_CUDA_Q8_SOA_QKV=1` | q/k/v microbench positive, fused full path regressed | speed smoke only |
| `soa_shared` | `DS4_CUDA_Q8_SOA_SHARED=1` | memory-safe, neutral, not byte-identical in stored logprob JSON | do not promote without quality gate |
| `soa_hc_expand` | `DS4_CUDA_Q8_SOA_HC_EXPAND=1` | 16.01 vs 16.06 t/s same-run control; initial capture-preload issue fixed | diagnostic only; do not promote |
| `attn_qb_hwarp16` | `DS4_CUDA_ATTN_Q_B_HWARP16=1` | 15.92 vs 16.13 t/s same-run control; changes reduction shape | diagnostic only; do not promote |
| `attn_qb_soa_hwarp16` | `DS4_CUDA_Q8_SOA_QB=1 DS4_CUDA_ATTN_Q_B_HWARP16=1` | 16.03 vs 16.13 t/s same-run control; changes reduction shape | diagnostic only; do not promote |
| `attn_qb_b32_special` | `DS4_CUDA_ATTN_Q_B_B32_SPECIAL=1` | exact-order decode-shape specialization, 16.07 vs 16.12 t/s control | diagnostic only; do not promote |
| `attn_qkv_pair_shape` | `DS4_CUDA_ATTN_QKV_PAIR_SHAPE=1` | exact-order QKV pair shape specialization, 16.07 vs 16.12 t/s control and higher register use | diagnostic only; do not promote |
| `attn_b_shape4096` | `DS4_CUDA_ATTENTION_OUTPUT_B_SHAPE4096=1` | exact-order attention-output-B shape specialization, 15.90 vs 16.12 t/s control and higher register use | diagnostic only; do not promote |
| `soa_qkv_no_pair` | `DS4_CUDA_Q8_SOA_QKV=1 DS4_METAL_DISABLE_QKV_PAIR_PROJ=1` | disabling the pair projection did not rescue QKV SoA: 15.70 vs 15.94 t/s control | diagnostic only; do not promote |
| `hc_expand_nhc4_special` | `DS4_CUDA_HC_EXPAND_NHC4_SPECIAL=1` | exact-shape HC-expand specialization, 16.11 vs 16.14 t/s control | diagnostic only; do not promote |
| `hc_expand_no_block_out` | `DS4_CUDA_HC_EXPAND_NO_BLOCK_OUT=1` | skips auxiliary `attn_out` store in fused HC-expand, 16.09 vs 16.17 t/s control | diagnostic only; do not promote |
| `shared_gate_up_noaux` | `DS4_CUDA_SHARED_GATE_UP_NOAUX=1` | shared expert mid-only kernel, 16.01 vs 16.15 t/s control | diagnostic only; do not promote |
| `shared_gate_up_cache_x` | `DS4_CUDA_SHARED_GATE_UP_CACHE_X=1` | shared activation staging for fused shared gate/up SwiGLU was slower: 15.88 vs 16.00 t/s control | diagnostic only; do not promote |
| `f16_pair_fast_reduce` | `DS4_CUDA_F16_PAIR_FAST_REDUCE=1` | exact-order final-warp sync reduction for compressor F16 pair was neutral/slower: 16.04 vs 16.07 t/s control | diagnostic only; do not promote |
| `compressor_pair_off` | `DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ=1` | disabling paired F16 compressor projections was slower: 15.85 vs 16.09 t/s control | diagnostic only; keep pair path |
| `soa_cache_x` | `DS4_CUDA_Q8_SOA_CACHE_X=1` | noisy/unstable and slower | re-test only after kernel change |
| `output_top1` | `DS4_CUDA_OUTPUT_TOP1=1` | exact but slower than full logits in A/B | diagnostic only |
| `output_q8_warp8` | `DS4_CUDA_OUTPUT_Q8_WARP8=1` | graph nsys showed full-logits output head on generic Q8, but warp8 routing was slower: 15.93 vs 16.13 and changes reduction order | diagnostic only; do not promote |
| `attn_b_cublas_min1` | `DS4_CUDA_ATTENTION_OUTPUT_B_CUBLAS_MIN=1` | did not improve refreshed profile | diagnostic only |
| `attn_a_hwarp16` | `DS4_CUDA_ATTENTION_OUTPUT_A_HWARP16=1` | negative in 8192/128 smoke; changes reduction order | diagnostic only; do not promote |
| `attn_a_shape8192` | `DS4_CUDA_ATTENTION_OUTPUT_A_SHAPE8192=1` | exact-order DS4-shape SoA kernel; no-unroll resource was `REG:36`, but 256-token run was negative: 15.95 vs 16.04 | diagnostic only; do not promote |
| `attn_a_cache_x16` | `DS4_CUDA_ATTENTION_OUTPUT_A_CACHE_X16=1` | full-warp 16-row shared-x shape probe was slower: 15.90 vs 15.94 and `REG:64` | diagnostic only; do not promote |
| `moe_h16` | `DS4_CUDA_MOE_DECODE_GATE_H16=1` | negative | re-test only after MoE kernel change |
| `moe_noaux` | `DS4_CUDA_MOE_DECODE_GATE_NOAUX=1` | neutral/negative | re-test only after MoE kernel change |
| `moe_pair2` | `DS4_CUDA_MOE_DECODE_GATE_PAIR2=1` | byte-identical but slower | re-test only after MoE kernel change |
| `moe_fused_midq` | `DS4_CUDA_MOE_DECODE_FUSED_MIDQ=1` | byte-identical but slower | re-test only after MoE kernel change |
| `moe_down_meta_cache` | `DS4_CUDA_MOE_DOWN_SUM6_META_CACHE=1` | tiny/noisy +0.6% in one smoke, below gate | diagnostic only; do not promote |
| `moe_gate_weight_cache` | `DS4_CUDA_MOE_DECODE_GATE_WEIGHT_CACHE=1` | negative in smoke | diagnostic only; do not promote |
| `moe_span128_template` | `DS4_CUDA_MOE_DECODE_GATE_SPAN128_TEMPLATE=1` | explicit span<128> template, 16.00 vs 16.16 t/s control | diagnostic only; do not promote |
| `moe_global_lut` | `DS4_CUDA_MOE_DECODE_GATE_GLOBAL_LUT=1` | strongly negative, 13.89 vs 16.16 t/s control | diagnostic only; do not promote |
| `moe_gate_maxr48` | `DS4_CUDA_MOE_DECODE_GATE_MAXR48=1` | real reg cap, but noisy: 16.18 vs 16.09 then 16.03 vs 16.10 | diagnostic only; do not promote |
| `moe_gate_ldg` | `DS4_CUDA_MOE_DECODE_GATE_LDG=1` | read-only-cache loads for routed gate/up weights were slower: 15.93 vs 16.15 t/s control | diagnostic only; do not promote |
| `moe_down_ldg` | `DS4_CUDA_MOE_DOWN_SUM6_LDG=1` | read-only-cache loads for routed down weights were slower: 15.96 vs 16.15 t/s control | diagnostic only; do not promote |
| `moe_ldg_weights` | both `*_LDG` flags | combined read-only-cache path was slower: 15.86 vs 16.15 t/s control | diagnostic only; do not promote |
| `moe_gate_shape2048` | `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1` | exact and byte-identical; small positive signal: 16.24 vs 16.05 at 128 tokens, 16.22 vs 15.87 at 256 tokens | candidate for retest/combination; not promoted alone |
| `moe_gate_shape2048_conststride` | `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTSTRIDE=1` | exact and byte-identical; small positive: 16.23 vs 16.09 at 128 tokens, 16.12 vs 15.80 at 256 tokens | minor candidate; not promoted alone |
| `moe_gate_shape2048_constclamp` | `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTCLAMP=1` | hardcoded DS4 clamp on top of const-stride was slower: 15.97 vs 16.07 t/s control | diagnostic only; do not promote |
| `moe_gate_shape2048_splitup` | `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_SPLITUP=1` | gate-first/up-second scheduling had no resource win and was slower: 16.08 vs 16.14 t/s control | diagnostic only; do not promote |
| `moe_gate_conststride_soa_b_forced` | `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTSTRIDE=1 DS4_CUDA_Q8_SOA_ATTN_OUTPUT_B_DECODE=1` | combo did not compose: 15.95 vs 16.08 t/s control; `soa_b_forced` alone was 15.85 | diagnostic only; do not promote |
| `moe_gate_prefer_l1` | `DS4_CUDA_MOE_GATE_PREFER_L1=1` | CUDA prefer-L1 cache config was slower: 16.02 vs 16.09; shape2048+L1 did not improve shape2048 | diagnostic only; do not promote |
| `moe_down_shape4096` | `DS4_CUDA_MOE_DOWN_SUM6_SHAPE4096=1` | neutral: 16.04 vs 16.05 t/s control | diagnostic only; do not promote |
| `moe_shape_special` | both MoE shape flags | below gate: 16.18 vs 16.05 t/s control at 128 tokens; down shape diluted the gate-only signal | diagnostic only; do not promote |
| `shared_gate_up_shape2048` | `DS4_CUDA_SHARED_GATE_UP_SHAPE2048=1` | too small: 16.07 vs 15.98 t/s control | diagnostic only; do not promote |
| `shape_gate_shared` | routed gate shape + shared gate shape | too small in 256-token recheck: 16.03 vs 15.95 t/s control | diagnostic only; do not promote |
| `shape_gate_attn_a` | routed gate shape + attention-output-A shape | did not compound: 16.16 vs 16.18 for routed gate shape alone | diagnostic only; do not promote |
| `indexer_topk_chunk8192` | `DS4_CUDA_TOPK_CHUNK8192=1` | long-context top-k chunking negative: 13.36 vs 13.46 t/s at frontier 65536; `uint32_t` 8192 chunk exceeded shared memory | diagnostic only; do not promote |
| `graph_no_presync` | `DS4_CUDA_GRAPH_DECODE_NO_SYNC=1` | normal decode graph capture without pre-sync was slower: 16.03 vs 16.18 t/s control | diagnostic only; do not promote |
| `weight_tensor_align2m` | `DS4_CUDA_WEIGHT_TENSOR_ALIGN_MB=2` | 2 MiB tensor-base alignment in the local CUDA arena was slower: 15.95 vs 16.04 t/s control | diagnostic only; do not promote |
| `q8_batch1_cache_x` | `DS4_CUDA_Q8_BATCH1_CACHE_X=1` | cached-x warp8 for one-token blocks<=32 Q8 projections was neutral/negative: 16.09 vs 16.13 t/s control | diagnostic only; do not promote |
| `sample_cache_probs` | `DS4_SAMPLE_CACHE_PROBS=1` | default sampled CLI output was byte-identical for seed 1, but throughput stayed 18.08 vs 18.08 t/s | diagnostic only; do not promote |
| `ffn_parallel_shared` | `DS4_CUDA_FFN_PARALLEL_SHARED=1` | shared gate/up overlap on second CUDA stream was slower: 15.26 vs 16.14 t/s same-run control | diagnostic only; do not promote |
| `ffn_shared_first` | `DS4_CUDA_FFN_SHARED_FIRST=1` | shared gate/up before routed MoE failed decode graph capture with CUDA synchronize-not-permitted | diagnostic only; do not promote |
| `moe_meta_cache` | both metadata-cache flags | negative in smoke | diagnostic only; do not promote |

### Residency And Placement Rows

These are not normal promotion rows because they affect model residency or
startup allocation rather than math kernels. They were tested after the roadmap
recheck because they were not covered by the exact env matrix.

| Probe | Env delta over `exact_fast` | Result | Decision |
| --- | --- | --- | --- |
| full model copy | `DS4_CUDA_COPY_MODEL=1` | copied 80.76 GiB in 457.539s, then measured 15.75 t/s | rejected |
| arena 1024 | `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=1024` | failed complete model cache with OOM at tensor span 116 | rejected |
| arena 4096 | `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=4096` | 15.80 t/s vs 16.06 adjacent default | rejected |
| arena 8192 | `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=8192` | 15.88 t/s vs 16.06 adjacent default | rejected |
| direct model | `DS4_CUDA_DIRECT_MODEL=1` | no usable CSV; graph decode failed after a long direct-model run | rejected |
| no FD cache | `DS4_CUDA_NO_FD_CACHE=1` | no CSV; timed out after 300s | rejected |

### Quality Tradeoff Rows

These rows can cross or approach 20 t/s, but they are not exact full-K decode.
They must never replace `exact_fast` without an explicit quality decision.

| Row | Env delta over `exact_fast` | Prior result | Next action |
| --- | --- | --- | --- |
| `k3_renorm` | `DS4_MOE_ACTIVE_EXPERTS=3`, `DS4_MOE_ACTIVE_EXPERTS_RENORM=1` | crossed 20 t/s server decode on coding prompts, not consistently better quality | research only |
| `k2_renorm` | K=2 with renorm | fastest, but coding unsafe in smoke | rejected for agent default |
| `k6_0_2_k3_renorm` | K=6 on layers 0-2, K=3 elsewhere with renorm | first 4-task smoke passed and hit target, 12-task eval regressed on different tasks | research only |

### MTP Research Rows

MTP is useful as a regression harness, not the current path to 20 t/s.

| Row | Env and args | Prior result | Next action |
| --- | --- | --- | --- |
| `mtp_quality` | current guarded prefix1 + draft2-skip candidate, `--mtp ... --mtp-draft 2` | quality-preserving candidate, but below no-MTP exact-fast throughput | keep as research benchmark |
| `mtp_attn_b_soa_batch2` | `mtp_quality` plus narrow `attn_output_b` SoA batch2 | tied but did not beat best MTP/no-MTP path | diagnostic only |

### External Port Candidates

These are not runnable by `tuning/gx10_matrix.py` on the current branch. They
come from the fork scan and require explicit code import before they become
matrix rows.

| Candidate | Source branch | Why it matters | Current decision |
| --- | --- | --- | --- |
| CUDA MMQ + layer graphs | `Entrpi/ds4:mmq-step-A-full-layer-graphs` | vendored llama.cpp MMQ/MMVQ, per-layer CUDA graph replay, VMM arena, proof harness | invasive port branch only; public GB10 CSV reports 13.74 t/s at ctx=8192/128 and 11.70 t/s at ctx=65536/128, below current exact-fast |
| MTP prefix/fused verifier | `reffdev/ds4:fused-matmul-mtp`, Entrpi MTP commits | alternative MTP verifier architecture and prefix-K ideas | keep as research input; no simple env drop-in here |
| KV self-eviction guard | `audreyt/ds4:feat/kv-cache-guard-fresh-cold-saves` | one-commit server cache correctness guard | small import candidate if reproduces locally |
| ROCm/HIP prequant ideas | `ejpir/ds4-hip`, `chiefnoah/ds4` | prequant/f16 scratch and launch-overhead work | architecture-specific; mine for concepts only |
| Expert sharding | `mirkodandrea/ds4:moe-expert-sharding` | reduces local resident expert memory through remote/CPU shards | memory/distribution route, not single-GB10 decode speed |
| Steering/agent feature work | `Chida82/ds4`, `audreyt/ds4` | expert/directional steering, tool-safe steering, agent polish | feature/quality axis, not speed matrix |

## Coverage Assessment

The env-toggle space has been explored broadly:

- graph decode and default Q8 SoA are the only exact-safe toggles promoted;
- broad SoA extensions, activation caches, Q8/cuBLAS swaps, top1 output, MoE
  micro-kernel toggles, and residency/arena placement variants were neutral,
  negative, or operationally worse;
- reduced-K modes buy speed by changing active experts, so they are quality
  tradeoffs;
- MTP orchestration can be made quality-preserving, but current verifier costs
  erase the expected speed win.

That does not mean every possible optimization is exhausted. It means the next
likely full-quality gain is not another simple env combination, and it is not
the already-tested naive routed-pack idea. The remaining work is kernel-level:
exact Q8 projection reads, routed MoE gate/up/down, and shared kernels used by
both no-MTP decode and the MTP verifier. A new attempt should preserve the
qwarp-friendly weight streams that made the current kernels faster than the
row-major/block-paired pack experiments.

The fork scan reinforces this: public forks do not currently show a small,
obvious, quality-preserving env toggle missing from this branch. The one major
new CUDA direction is Entrpi's MMQ/layer-graph line, which is a porting project,
not a matrix row.

## Latest Decode-Window Profile

A delayed Nsight Systems pass (`--delay=22 --duration=6`) on 2026-05-23 captured
mostly steady graph decode. The committed artifact is
`tuning/gx10_matrix_results/prototype_20260523_nsys_graph_decode_window/cuda_gpu_kern_sum.txt`.

Top rows: routed MoE gate/up 18.7%, HC-expand SoA 12.4%, `attn_q_b` 12.3%,
attention-output-A SoA 12.1%, MoE down 8.8%, shared gate/up 6.7%, attention
5.5%, f16 pair 5.3%, output head 4.5%.

Matrix implication: MoE down should stay closed unless a genuinely new
structural idea appears. The simple down variants are already represented in
the rejected rows above, while the current frontier is Q8 projection traffic and
routed gate/up scheduling.
