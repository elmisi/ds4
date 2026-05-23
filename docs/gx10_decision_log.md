# GX10 Decision Log

This file is the compact operating summary for the GX10 / GB10 speed branch.
`docs/road_to_20_ts.md` remains the full chronological record and contains the
commands, artifacts, and reasoning needed to reconstruct each choice. This file
answers the practical question: what should be kept, what should be left behind,
and what is still worth exploring before creating a clean upstream PR branch.

## Branch Strategy

- Keep `gx10-cuda-graph-decode` as the research branch. It may contain negative
  probes and diagnostic flags because their history is useful.
- Once a path is good enough for upstream, create a fresh branch from current
  `main` / `upstream/main` and cherry-pick or reimplement only the winning pieces.
- Do not open a PR from the research branch as-is.
- For an upstream PR, prefer small reviewable slices with one clear speed result
  and one clear correctness story.

## Current Baselines

| Configuration | Result | Note |
| --- | ---: | --- |
| Updated upstream/main worktree | ~14.6 t/s | Fresh main after upstream updates and native `ds4-agent`. |
| Current branch plain `ds4-agent` | ~16.3 t/s | Includes local work but not exact-fast wrapper defaults. |
| `ds4-agent-exact-fast` | ~16.7 t/s | Graph decode + default Q8 SoA cache, `ctx=100000`. |

The current production-safe local recommendation is:

```sh
DS4_CUDA_GRAPH_DECODE=1
DS4_CUDA_Q8_SOA_CACHE=1
```

with full K=6 routed MoE, no MTP, no reduced active experts, no Q8/cuBLAS
diagnostic switches, and no half-warp/codegen/metadata-cache probes.

## Promoted Or Worth Keeping

| Area | Status | Why |
| --- | --- | --- |
| CUDA graph decode | Keep | Real decode speed win and part of `exact_fast`. |
| Q8 SoA cache for attention output A/B | Keep | Exact memory-for-speed win; current best full-quality path. |
| `ds4-agent-exact-fast` wrapper | Keep locally | Makes the tested exact-fast runtime easy to launch with `ctx=100000`. |
| `tuning/gx10_matrix.py` | Keep locally | Reproducible matrix runner for speed, coding eval, and `ds4-eval`. |
| `ds4-eval` gates in matrix runner | Keep locally | Caught the IQ2/Q2 codegen regression quickly. |

## Fork Audit Verdict

The 2026-05-24 supplied fork ranking was checked against local refs. It did not
surface a missing no-MTP exact-fast patch:

- Entrpi MMQ/VMM/layer-graphs remain an invasive isolated-port track, not a
  small integration.
- amarrmb PR #121 is already present/equivalent.
- cghart's listed F16/Q8/output ideas overlap with local probes that are already
  neutral or negative.
- ddxxlao partial cache and Dominik FP16 KV are memory/long-context tracks, not
  current 100k-context decode-speed tracks.
- Audreyt's highlighted work is prefill, not generation.

The only newly actionable item is TrevorS's MTP verifier micro-track: GPU argmax
for top1 verifier sites, small-N shared-weight Q8 batch matmul, and a retest of
small-N no-sort MoE only inside MTP. Keep this out of the production exact-fast
path unless MTP is explicitly reopened.

## Diagnostic Only

These may remain in the research branch for reproduction, but they should not be
enabled in the default agent or carried into an upstream PR unless a future
kernel change creates a new reason to retest them.

| Row / Flag | Result |
| --- | --- |
| `soa_b_forced` / `DS4_CUDA_Q8_SOA_ATTN_OUTPUT_B_DECODE=1` | Quality looked okay in repeated coding eval, but slower than default SoA server runs. |
| `soa_qb` / `DS4_CUDA_Q8_SOA_QB=1` | Microbench positive, full decode neutral/negative; 2026-05-23 recheck was 16.00 vs 16.06 t/s control. |
| `soa_qkv` / `DS4_CUDA_Q8_SOA_QKV=1` | Individual microbench positive, fused full route regressed. |
| `soa_shared` / `DS4_CUDA_Q8_SOA_SHARED=1` | Memory-safe and micro-positive, but end-to-end neutral and not byte-identical in stored logprobs. |
| `soa_cache_x` / `DS4_CUDA_Q8_SOA_CACHE_X=1` | Noisy/unstable and slower end-to-end. |
| `attn_b_cublas_min1` / `DS4_CUDA_ATTENTION_OUTPUT_B_CUBLAS_MIN=1` | Did not improve refreshed profile and spent significant extra memory. |
| `output_q8_warp8` / `DS4_CUDA_OUTPUT_Q8_WARP8=1` | Nsys showed output full-logits still using the generic Q8 kernel, but routing it to warp8 was slower: 15.93 vs 16.13 t/s and changes reduction order. |
| `attn_a_hwarp16` / `DS4_CUDA_ATTENTION_OUTPUT_A_HWARP16=1` | Negative speed smoke and changes reduction order. |
| `attn_qb_hwarp16` / `DS4_CUDA_ATTN_Q_B_HWARP16=1` | Negative: 15.92 vs 16.13 t/s same-run control. |
| `attn_qb_soa_hwarp16` / `DS4_CUDA_Q8_SOA_QB=1 DS4_CUDA_ATTN_Q_B_HWARP16=1` | Negative: 16.03 vs 16.13 t/s same-run control. |
| `attn_qb_b32_special` / `DS4_CUDA_ATTN_Q_B_B32_SPECIAL=1` | Exact-order specialization, but still negative: 16.07 vs 16.12 t/s control. |
| `attn_qkv_pair_shape` / `DS4_CUDA_ATTN_QKV_PAIR_SHAPE=1` | Exact-order QKV pair shape specialization raised register use and was slower: 16.07 vs 16.12 t/s control. |
| `attn_b_shape4096` / `DS4_CUDA_ATTENTION_OUTPUT_B_SHAPE4096=1` | Exact-order attention-output-B 4096x4096 shape specialization raised register use and was slower: 15.90 vs 16.12 t/s control. |
| `soa_qkv_no_pair` / `DS4_CUDA_Q8_SOA_QKV=1 DS4_METAL_DISABLE_QKV_PAIR_PROJ=1` | Testing whether QKV SoA works better without the pair projection was negative: 15.70 vs 15.94 t/s control. |
| `soa_hc_expand` / `DS4_CUDA_Q8_SOA_HC_EXPAND=1` | Targeted HC-expand SoA tail probe was negative: 16.01 vs 16.06 t/s same-run control. |
| `hc_expand_nhc4_special` / `DS4_CUDA_HC_EXPAND_NHC4_SPECIAL=1` | Exact-shape HC-expand specialization was negative: 16.11 vs 16.14 t/s control. |
| `hc_expand_no_block_out` / `DS4_CUDA_HC_EXPAND_NO_BLOCK_OUT=1` | Auxiliary `attn_out` store removal was negative: 16.09 vs 16.17 t/s control. |
| `hc_expand_soa_ldg` / `DS4_CUDA_HC_EXPAND_SOA_LDG=1` | Read-only-cache loads for HC-expand Q8 SoA weights raised resources (`REG:63` vs `REG:62`) and regressed: 15.84 vs 16.13 t/s control. |
| `hc_expand_soa_par_hc4` / `DS4_CUDA_HC_EXPAND_SOA_PAR_HC4=1` | Parallelizing the n_hc=4 expansion/store across lanes lowered resources (`REG:51`) but still regressed: 15.91 vs 16.06 t/s control. |
| `shared_gate_up_noaux` / `DS4_CUDA_SHARED_GATE_UP_NOAUX=1` | Separate shared-expert no-aux probe was negative: 16.01 vs 16.15 t/s control. |
| `shared_gate_up_cache_x` / `DS4_CUDA_SHARED_GATE_UP_CACHE_X=1` | ROCm/LDS-inspired shared activation staging for the fused shared gate/up SwiGLU kernel was slower: 15.88 vs 16.00 t/s control. |
| `shared_gate_up_dot2` / `DS4_CUDA_SHARED_GATE_UP_DOT2=1` | Sharing one prequant activation load across the gate/up Q8 dot products lowered the kernel to `REG:61` but was still slower: 15.89 vs 15.97 t/s control. |
| `shared_gate_up_pair_pack` / `DS4_CUDA_SHARED_GATE_UP_PAIR_PACK=1` | Native paired/aligned shared gate/up Q8_0 pack lowered the kernel to `REG:51` but strongly regressed decode: 15.38 vs 15.95 t/s control. |
| `f16_pair_fast_reduce` / `DS4_CUDA_F16_PAIR_FAST_REDUCE=1` | Exact-order final-warp reduction sync reduction for compressor F16 pair was neutral/slower: 16.04 vs 16.07 t/s control. |
| `compressor_pair_off` / `DS4_METAL_DISABLE_COMPRESSOR_PAIR_PROJ=1` | Disabling paired F16 compressor projections was slower: 15.85 vs 16.09 t/s control, so the existing pair kernel remains correct to keep. |
| `moe_down_meta_cache` / `DS4_CUDA_MOE_DOWN_SUM6_META_CACHE=1` | Tiny/noisy +0.6%, below gate. |
| `moe_gate_weight_cache` / `DS4_CUDA_MOE_DECODE_GATE_WEIGHT_CACHE=1` | Negative speed smoke. |
| `moe_span128_template` / `DS4_CUDA_MOE_DECODE_GATE_SPAN128_TEMPLATE=1` | Explicit span<128> template was negative: 16.00 vs 16.16 t/s control. |
| `moe_global_lut` / `DS4_CUDA_MOE_DECODE_GATE_GLOBAL_LUT=1` | Strongly negative: 13.89 vs 16.16 t/s; shared IQ2 LUT copy is better. |
| `moe_gate_maxr48` / `DS4_CUDA_MOE_DECODE_GATE_MAXR48=1` | Register cap was noisy: 16.18 vs 16.09 once, then 16.03 vs 16.10; below gate. |
| `moe_gate_ldg` / `DS4_CUDA_MOE_DECODE_GATE_LDG=1` | Read-only-cache loads for routed gate/up weights were slower: 15.93 vs 16.15 t/s control. |
| `moe_down_ldg` / `DS4_CUDA_MOE_DOWN_SUM6_LDG=1` | Read-only-cache loads for routed down weights were slower: 15.96 vs 16.15 t/s control. |
| `moe_gate_shape2048` / `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1` | Exact and byte-identical; small positive signal: 16.24 vs 16.05 t/s at 128 tokens and 16.22 vs 15.87 t/s at 256 tokens. Keep as a candidate to retest, not promoted alone. |
| `moe_gate_shape2048_conststride` / `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTSTRIDE=1` | Exact and byte-identical; constant DS4 row/expert strides improved the shape probe slightly: 16.23 vs 16.09 at 128 tokens and 16.12 vs 15.80 at 256 tokens, still below the +3% gate. Keep as a minor candidate. |
| `moe_gate_shape2048_dot2` / `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_DOT2=1` | Shared one Q8 activation-block load across the routed gate/up IQ2 dots, but kept the same resources as const-stride (`REG:64`) and was slower: 15.81 vs 16.06 t/s control. |
| `moe_gate_shape2048_constclamp` / `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTCLAMP=1` | Hardcoded DS4 clamp on top of const-stride did not help: 15.97 vs 16.07 t/s control. |
| `moe_gate_shape2048_splitup` / `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_SPLITUP=1` | Gate-first/up-second scheduling kept the same resources as const-stride (`REG:64`) and was slower than control: 16.08 vs 16.14 t/s. |
| `moe_gate_conststride_soa_b_forced` | Const-stride routed gate/up plus forced attention-output-B SoA did not compose: 15.95 vs 16.08 t/s control; `soa_b_forced` was 15.85 in the same run. |
| `moe_gate_prefer_l1` / `DS4_CUDA_MOE_GATE_PREFER_L1=1` | CUDA prefer-L1 cache config was slower: 16.02 vs 16.09 t/s control. Shape2048+L1 did not beat shape2048 alone. |
| `moe_down_shape4096` / `DS4_CUDA_MOE_DOWN_SUM6_SHAPE4096=1` | Shape-specialized down-sum6 was neutral: 16.04 vs 16.05 t/s control. |
| `shared_gate_up_shape2048` / `DS4_CUDA_SHARED_GATE_UP_SHAPE2048=1` | Shared gate/up shape specialization was too small: 16.07 vs 15.98 t/s control; combined with routed gate shape was only 16.03 vs 15.95 at 256 tokens. |
| `attn_a_shape8192` / `DS4_CUDA_ATTENTION_OUTPUT_A_SHAPE8192=1` | Exact-order attention-output-A shape specialization lowered resources after removing unroll (`REG:36`), but 256-token run was slower: 15.95 vs 16.04. |
| `attn_a_cache_x16` / `DS4_CUDA_ATTENTION_OUTPUT_A_CACHE_X16=1` | Full-warp 16-row shared-x attention-output-A probe was slower: 15.90 vs 15.94 and used `REG:64`. |
| `shape_gate_attn_a` | Combining `moe_gate_shape2048` with `attn_a_shape8192` did not compound: 16.16 vs 16.18 for gate shape alone. |
| `indexer_topk_chunk8192` / `DS4_CUDA_TOPK_CHUNK8192=1` | Long-context top-k chunking was negative: 13.36 vs 13.46 t/s at frontier 65536; `uint32_t` 8192 chunk also exceeded GB10 shared memory. |
| `graph_no_presync` / `DS4_CUDA_GRAPH_DECODE_NO_SYNC=1` | Normal decode graph capture without pre-sync was slower: 16.03 vs 16.18 t/s same-run control. |
| `weight_tensor_align2m` / `DS4_CUDA_WEIGHT_TENSOR_ALIGN_MB=2` | Entrpi-inspired 2 MiB tensor-base alignment in the local arena was slower: 15.95 vs 16.04 t/s control. |
| `q8_batch1_cache_x` / `DS4_CUDA_Q8_BATCH1_CACHE_X=1` | Using cached-x warp8 for one-token blocks<=32 Q8 projections was neutral/negative: 16.09 vs 16.13 t/s control. |
| `sample_cache_probs` / `DS4_SAMPLE_CACHE_PROBS=1` | Default sampled CLI output was byte-identical for seed 1, but throughput stayed 18.08 vs 18.08 t/s. |
| `ffn_parallel_shared` / `DS4_CUDA_FFN_PARALLEL_SHARED=1` | Shared gate/up on a second CUDA stream was slower: 15.26 vs 16.14 t/s same-run control. |
| `ffn_shared_first` / `DS4_CUDA_FFN_SHARED_FIRST=1` | Shared gate/up before router/routed MoE failed graph capture with CUDA synchronize-not-permitted. |
| `moe_meta_cache` / both metadata-cache flags | Negative speed smoke. |
| MTP graph/guard/prefix rows | Diagnostic only unless user explicitly accepts the quality/speed tradeoff. |

## Rejected Paths

Do not spend more time on these without a genuinely new design:

| Path | Reason |
| --- | --- |
| Naive native routed packs | Real compute tests were slower despite byte-equivalent pack smokes. |
| Row-major down pack | Bit-exact compute but slower than expert-major stream. |
| Gate/up block-paired pack | Raw read looked interesting, real compute was much slower. |
| Multi-expert shared-xq/LUT K=6 kernel | Slower than current qwarp stream. |
| MoE H16, noaux, pair2, fused-midq, row4, parallel down, span128 template, global LUT | All neutral or negative. |
| MoE read-only-cache load variants | `__ldg` on routed IQ2/Q2 weights made gate/up and down slower. |
| MoE selected/route-weight metadata shared caches | Scalar metadata is not the bottleneck; shared/sync overhead does not pay. |
| IQ2/Q2 dot codegen rewrite | Speed looked +1-2%, but `ds4-eval` smoke regressed from 4/4 to 3/4 and recovered after revert. |
| Full model copy / direct model / no FD cache / arena chunk tuning | Slower, failed, or operationally worse. |
| Q8 block padding/alignment | Reads more bytes and was slower. |
| Shared activation cache for grouped attention-output-A | Exact but slower. |
| Attention-output-A shape specialization and 16-row shared-x shape | Did not beat the generic promoted SoA kernel; no compound gain with MoE shape. |
| Narrow `attn_q_b` half-warp/SoA/specialized kernels | Did not beat the generic warp8 Q8 projection. |
| HC-expand SoA tail, exact-shape specialization, and auxiliary-write removal | Did not beat the current generic/SoA mixed path. |
| Long-context top-k chunk-size-only variants | 8192-row chunks reduced merge width but slowed decode at frontier 65536. |
| Normal decode graph pre-sync removal | Verifier no-pre-sync was already diagnostic; normal decode no-pre-sync was also slower. |
| Local arena tensor-base padding | 2 MiB per-tensor alignment did not improve current native CUDA kernels; leave MMQ/VMM as a separate port track. |
| One-token Q8 cached-x routing for small block projections | Shared staging overhead did not beat the existing batch-warp path. |
| Default sampler duplicate-`expf` avoidance | Exact sampled output, but no measured agent/CLI throughput gain. |
| Shared expert auxiliary-write removal | Same dot-product work dominates; skipping stores did not help. |
| FFN shared/routed scheduling overlap or shared-first reordering | Second-stream overlap was slower; shared-first currently breaks CUDA graph capture. |
| MoE gate/up register cap | `__launch_bounds__(256,5)` is invalid on GB10; `__maxnreg__(48)` is noisy and below gate. |
| Q8/cuBLAS/F16 swaps for attention output | Either slower or too memory-expensive for no speed win. |
| Reduced-K MoE (`K=2/3/...`) | Can cross 20 t/s, but sacrifices full-quality model behavior. |

## Current Gate Policy

A candidate only deserves expensive quality runs if it first clears the cheap
speed gate:

1. Same-run `exact_fast` A/B at 8192 context, 128 generated tokens.
2. At least +3% over same-run `exact_fast`.
3. For exact candidates, deterministic logprob/token comparison.
4. Coding canary with no new failures versus same-run `exact_fast`.
5. `ds4-eval` smoke with no pass-count regression.
6. Long `ds4-eval` only after all cheaper gates pass.

## Upstream PR Extraction Checklist

When a candidate is ready:

1. Create a new branch from current `upstream/main`.
2. Apply only the winning code and minimal support code.
3. Exclude rejected diagnostic flags unless the PR is explicitly a diagnostic
   harness PR.
4. Include a short doc or PR body with:
   - baseline commit and hardware;
   - exact commands;
   - speed A/B;
   - memory impact;
   - correctness gate results.
5. Prefer one behavior change per PR: graph decode, Q8 SoA cache, matrix runner,
   or agent wrapper should not be bundled unless they depend on each other.

## Next Work Worth Considering

The remaining credible full-quality work is not another MoE micro-shape. It
needs either:

- a fresh profile-driven target that shows a non-obvious hotspot after the latest
  branch state; or
- a structural kernel/layout idea that preserves the current qwarp-friendly
  streams while reducing real weight traffic; or
- a clean upstreamable extraction of graph decode + Q8 SoA if the current result
  is already interesting enough.

Fresh profiling on 2026-05-23 reconfirmed the measured target order. A
decode-window Nsight run with `--delay=22 --duration=6` produced:

1. routed MoE gate/up (`moe_gate_up_mid_decode_lut_qwarp32_kernel`);
2. HC-expand SoA, `attn_q_b`, and attention-output-A SoA, each around 12% of
   decode-window kernel time;
3. MoE down at 8.8%, below the Q8 projection frontier and already covered by
   prior row-major/native-pack/meta-cache/LDG/row4/parallel/shape probes;
4. shared gate/up, attention, f16 pair, and output head behind those rows.

The latest pass also rejected the narrow `attn_q_b` half-warp/SoA/specialized
variants, HC-expand exact-shape/no-block-out variants, MoE span/global-LUT/
register-cap/read-only-cache variants, shared expert no-aux writes, and
long-context top-k chunk-size-only variants. Shape-specializing routed gate/up
is the only exact small positive after this pass, but it is still below the
promotion gate by itself. The next credible attempt must either compound that
small win with another exact kernel-level gain or be structural: reduce real
routed-MoE or Q8 projection weight traffic, classify the f16 pair path, or
change scheduling around existing exact kernels without repeating the
row-major/block-paired/native-pack/cache-x/F16/cuBLAS/top-k-chunk experiments
already rejected.
