# Road to 20 t/s on GB10 — work log and findings

This document captures the work, data, and dead-ends from the push to lift
DS4 V4 Flash decode throughput on a DGX Spark (GB10, sm_121) from ~15 t/s to
the 20 t/s target. It is meant as a self-contained handoff so a future
session can pick up without re-discovering anything.

Branch: `gx10-cuda-graph-decode` (pushed to fork `origin`, never to upstream).

For a compact promoted/rejected/diagnostic summary and clean-branch extraction
plan, see `docs/gx10_decision_log.md`. This file remains the chronological
source of truth for commands, artifacts, and rationale.

## TL;DR

- **Current full-quality state**: ~17.8-18.1 t/s avg (8192 ctx, 64-128 gen)
  on GB10 without MTP after the no-MTP third pass. Best recent CLI
  measurements: 18.16 t/s at 64 gen tokens and 18.13 t/s at 128 gen tokens
  with `DS4_CUDA_GRAPH_DECODE=1`. Stable; ~110 GB device-resident with 256k
  ctx server remains the production target.
- **20 t/s has been reached as an opt-in quality-tradeoff mode.** Server +
  graph decode + greedy + `DS4_MOE_ACTIVE_EXPERTS=2` reaches **21.22-21.28
  t/s** on the 64-token prime prompt; adding `DS4_MOE_ACTIVE_EXPERTS_RENORM=1`
  measured **21.41-21.44 t/s**. A later coding smoke found K=3 + renorm also
  holds roughly **20.1-20.3 t/s** in the server decode logs, with much better
  code-format discipline than K=2. The per-layer profile K=6 on layers 0-2 and
  K=3 + renorm elsewhere passed the first 4-task smoke, but a 12-task coding
  eval at **ctx=256000** showed regressions relative to full K=6. These modes
  are therefore diagnostics/quality tradeoffs, not the default path.
- **Phase 4 (CUDA Graph) complete**: capture, dec-kernel infrastructure, and
  `cudaGraphExecUpdate`-based cached exec all working and bit-equivalent to
  direct mode. Does **not** improve wall-clock — see "Why graph capture did
  not help" below.
- **Profile shows decode is memory-bound on MoE weight reads.** The kernel
  initially targeted (`moe_down_sum6_qwarp32`) is only 7 % of GPU time on
  decode. The dominant cost is `matmul_q8_0_preq_warp8_cached_x` (Q/KV/out
  projections, 16 %) tied with `moe_gate_up_mid_decode_lut_qwarp32` (16 %).
- **MTP correctness is now fixed on CUDA.** The old "missing Q4_K" framing was
  incomplete: the actual failure was the CUDA HC pre-norm fused path reading
  MTP HC function tensors as F16 even though the MTP tensors are F32. This
  produced NaNs before attention. The CUDA HC pre-norm path now handles both
  F16 base tensors and F32 MTP tensors correctly.
- **MTP speculative decode has a quality-first candidate, but still does not
  beat baseline.** Router-exact + K2 body with
  `DS4_MTP_BATCH_MARGIN_GUARD=0.25` scored **10/12** on the 12-task coding eval
  at `ctx=256000`, same two persistent failures as no-MTP/full-K, **15.90
  t/s** suite-sum and **16.78 t/s** server decode-only in the warmed run.
  Adding guarded partial-prefix commit,
  `DS4_MTP_CAPTURE_PREFIX1=1 DS4_MTP_CAPTURE_PREFIX1_MIN_MARGIN=2.0`, also
  scored **10/12** in two full-suite runs and improved the first suite-sum to
  **16.42 t/s** by reducing exact replay cost. The second repeat kept the same
  failures but measured **15.16 t/s** because the first task was cold/slow and
  `parse_duration` generated a longer answer. The current best MTP variant adds
  `DS4_MTP_DRAFT2_SKIP_MIN_MARGIN=2.0`, which scored **10/12** twice with the
  same failures and measured **16.87** and **16.82 t/s**. Pure
  `DS4_MTP_CAPTURE_PREFIX1=1` is rejected for now: it was faster (**16.67
  t/s**) but regressed quality to **9/12** with an extra `parse_duration`
  failure.
- **2026-05-21 MTP/output-projection probes did not change the recommendation.**
  A narrow `attn_output_b` batch2 SoA switch passed full coding quality
  (**10/12**, same failures) and reached **16.87 t/s** warmed, effectively tied
  with draft2-skip but not better. A fresh no-MTP graph+SoA warm comparison also
  scored **10/12** at **16.83 t/s** on the same harness, with steadier server
  decode chunks around **17.4-17.9 t/s**. A targeted `attn_output_b`
  F16/cuBLAS decode probe was also neutral/slower in the intrusive stage
  profile, so the remaining gap to 20 is still in exact decode kernel
  throughput, not MTP orchestration or a simple F16-cache switch.
- **The 20 t/s target is low-resident-context server decode, not a nearly full
  256k window.** The production server still allocates `--ctx 256000`, but the
  expected win condition is the beginning of an agent/coding session, followed
  by normal context-length degradation. On the current exact SoA+graph path,
  a prompt-minimal non-thinking server request at `ctx=256000` measured
  **18.09 t/s avg** for 128 generated tokens, and the matching CLI top-only
  path measured **18.06-18.42 t/s**. The remaining full-quality gap to 20 t/s
  is therefore about **10-11%** on low-context decode.
- **For full-quality decode, the closest route is still normal no-MTP decode.**
  The third pass found small default wins in the Q/KV projection path and shared
  expert gate/up path, but also ruled out several tempting "spend more memory
  for speed" ideas. The remaining gap is not a launch or graph issue.
- **`ctx=256k` allocated is not the same as 256k tokens resident.** A 12.5k
  prompt with `--ctx 256000` still measures around **20.28 t/s** in CLI graph
  token-profile mode, but a real 250k-token prompt drops to **9.41 t/s** with
  default SoA. The long-context profile shifts the bottleneck from MoE/Q8-only
  thinking to attention plus indexer: at `pos=250566`, `attention` is
  **31.43 ms/token** and `compressor_indexer` is **27.23 ms/token** over the
  layer stack. Exact 20 t/s at a nearly full 256k context is therefore a
  different and harder problem than 20 t/s at short or medium effective
  context.
- **Fresh upstream baseline is materially lower.** A separate worktree at
  `upstream/main` commit `c9dd949` (`cuda: fix compressed prefill RoPE
  positions`) was built with `make cuda-spark` and measured on the same
  8192/64 `ds4-bench` prompt. It produced **13.58 t/s**, then **13.39 t/s**
  on a repeat with `DS4_CUDA_GRAPH_DECODE=1` in the environment. The current
  branch baseline is therefore not just SoA: the pre-existing CUDA decode work
  already moved the bench to **15.27 t/s** before the new SoA A+B opt-in.
- **Hot-expert cache was measured, not assumed.** A new device-side
  `DS4_MOE_EXPERT_STATS=1` profiler records expert IDs and prev-token overlap.
  On the 12-task coding suite at ctx=256k, a per-layer top64 hot set covered
  only **76.7%** of selected experts when learned globally, while the oracle
  per-request top64 upper bound was **87.0%**. Prev-token overlap averaged
  **1.88/6 experts**. Static or rolling hot caches are not enough unless the
  hot path is roughly 2x faster, which is unlikely if it still reads the same
  quantized bytes.
- **Native routed-pack smoke reached real compute tests and produced useful
  negative results.** Byte-for-byte one-layer gate/up/down packs work under the
  memory cap, and block-level gate/up pairing looked +8.1% in a raw read
  microbench. The real `decode_lut_qwarp32` compute test was bit-exact but
  much slower: paired block-pack was **0.466x** the current separate layout,
  and a K=6 multi-expert shared-xq/LUT kernel was **0.681x**. A down-projection
  row-major compute test was also bit-exact but only **0.859x** the current
  expert-major layout. Do not promote these layouts; the qwarp
  lane-contiguous block stream is more important than simple expert adjacency.
- **Several further exact micro-ideas were ruled out.** `DS4_CUDA_MOE_DOWN_SUM6_PARALLEL=1`
  did not improve the small down kernel; `DS4_CUDA_MOE_DECODE_GATE_SPAN`
  variants around the current 128-row gate/up shape regressed; H16, no-aux, and
  pair2 decode gate/up probes were byte-identical or structurally exact but
  slower; fused gate/up-to-midq removed a launch but `midq` was too small to
  matter; row4 down-sum grouping was byte-identical but slower; and a dense Q8
  aligned pack (`DS4_CUDA_Q8_ALIGN_SMOKE=1`) was exact but slower because
  padding bytes cost more than aligned loads save.
- **One exact memory-for-speed path finally moved a real kernel.** A Q8 SoA
  cache for `attn_output_a/b` (`DS4_CUDA_Q8_SOA_CACHE=1`) stores scales and
  quant bytes separately, keeping the same 34 logical bytes per block while
  letting the kernel use aligned weight loads. The one-layer microbench is
  **1.081x** faster for A and **1.123x** faster for B; the 8192/64
  `ds4-bench` smoke improved from **15.27** to **15.82-15.88 t/s** across
  repeat runs. It costs about **2.86 GiB** for all 43 attention-output-A/B
  tensors, passed a `--ctx 256000` smoke, and `cmp` returned 0 on stored
  coding-prompt logprob JSON. Server graph smoke at 8192/64 reached
  **18.18 t/s avg** on the prime-list prompt. After the later MoE H16/no-aux
  probes were made opt-in only, the same 8192/64 `ds4-bench` control still
  measured **15.84 t/s**, confirming the speed path stayed in the expected
  band. A first 12-task coding eval at `ctx=256000` found **9/12** for SoA
  A+B vs **10/12** for the current no-SoA full-K control, but the follow-up
  isolated the extra `flatten_dict` failure and found greedy run-to-run
  sensitivity even on the no-SoA path. Repeated production-context evals then
  cleared the suspected SoA-specific quality loss: no-SoA, default SoA, and
  B-forced SoA all measured **30/36** on 12 tasks x 3 repeats at
  `ctx=256000`, with the same two persistent failures (`lru_cache`,
  `parse_csv_line`) and `flatten_dict` at **3/3** in every mode. Default SoA
  was fastest in that protocol: server final-decode average **17.91 t/s** vs
  **17.40 t/s** no-SoA and **17.68 t/s** B-forced.
- **`ds4-eval` has now been added as a small extra sanity gate.** The main
  quality gate for the 20 t/s work remains coding-oriented eval plus
  deterministic logprob comparison, because that is the user workload. A quick
  `ds4-eval` smoke on the SoA A+B speed path, `--questions 2 --tokens
  512 --nothink --plain --warm-weights`, passed **2/2**. A full default
  `ds4-eval` run is much longer: it is a 92-question GPQA/SuperGPQA/AIME/COMPSEC
  integration suite with a default 16k token budget per question.
- **Several larger SoA extensions were measured and deliberately not promoted.**
  `attn_q_b` was strong in isolation (**1.167x**) but did not improve the
  full decode (`15.81 t/s`). `attn_q_a/attn_kv` were also strong in isolation
  (**1.250x** and **1.330x**) but the fused pair path regressed to **15.69
  t/s**. SoA+shared-activation-cache was noisy (**15.96**, then **15.65
  t/s**). Shared-expert SoA fit in memory (**3.93 GiB** total SoA cache) and
  passed 256k, but was neutral (**15.83 t/s**) and changed logit JSON at the
  ~1e-6 level, so it remains experimental.
- **Long-context attention shortcuts produced speed but failed the quality
  gate.** Forcing the existing WMMA indexer-score path for one-token decode
  regressed 250k decode (**8.20 t/s**). A one-token heads8 attention route
  improved 250k decode to **9.66 t/s**, and a narrower parallel-dot variant
  reached **10.50 t/s**, but both changed the greedy token on a 12.5k-token
  logprob check at generation step 1 (`" disp"` -> `" hai"`). They remain
  diagnostics, not full-quality candidates. Two exact-shaped follow-ups did
  not open the path either: larger heads8 online row stages were neutral or
  slower, and a pair2 indexer score kernel changed the greedy sequence.
- **Outside-the-box shortcuts were tested.** Server greedy top-only avoids
  full logits readback but is neutral, because exact argmax still has to scan
  the full output head. A direct exact output-head top1 kernel was also slower
  than the existing full-logits Q8 path in server A/B, so it is opt-in only
  (`DS4_CUDA_OUTPUT_TOP1=1`). Reduced active MoE experts is the first shortcut
  that actually crosses 20 t/s in the server path, but the broader coding eval
  says it is not quality-preserving.

## Hardware and model context

- **Box**: DGX Spark, GB10, sm_121 (Blackwell), 128 GB unified LPDDR5X
  (~150-200 GB/s effective for kernel weight reads).
- **Model**: DS4 V4 Flash (`ds4flash.gguf`).
  - `DS4_N_EMBD = 4096`, `DS4_N_HEAD = 64`, `DS4_N_HEAD_DIM = 512`
  - `DS4_N_EXPERT = 256`, `DS4_N_EXPERT_USED = 6`, `DS4_N_EXPERT_SHARED = 1`
  - `DS4_N_HC = 4`, `DS4_N_LORA_Q = 1024`
  - Quantization: routed expert weights are `iq2_xxs` (gate/up) and `q2_K`
    (down). Q/KV/output projections are `q8_0`. HC projections are `f16`.
- **Run config (`dgx-ctl`)**: `--ctx 256000 --kv-disk-dir ~/.ds4-kv
  --kv-disk-space-mb 32768`, env `DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN=1`.
  Note: `/tmp` is ext4 on NVMe on this box (verified via `findmnt /tmp`),
  not tmpfs.

## Phase 4 — CUDA Graph capture for decode

Goal: collapse ~1000-1500 per-token `cudaLaunchKernel` calls into a single
`cudaGraphLaunch`. Profile had suggested ~15 ms/token of host encode
overhead — turned out to be the wrong story (see "Why graph capture did not
help").

### Phase 4a — device-resident decode state (commit `c994f62`)

Added `struct ds4_decode_token_state { token, pos, raw_row, n_raw; }` in a
`__device__` symbol on the CUDA backend (originally `__constant__` — swapped
to `__device__` later for cleaner stream-capture semantics, *not* because
`__constant__` was the actual bug).

API:
- `ds4_gpu_decode_state_set(token, pos, raw_row, n_raw)` —
  `cudaMemcpyToSymbolAsync` from pinned host buffer
  `g_decode_state_host`.
- `ds4_gpu_use_decode_state(int on)` — flips a global `g_use_decode_state`
  that wrappers check.

The host buffer is pinned (`cudaMallocHost`) so the symbol copy can be
re-captured stream-ordered and read fresh values at every launch.

### Phase 4b — `_dec` kernel dispatch (commits `4804f64`, `173fa19`, `8324e40`, `069d3d6`)

Converted the per-token kernels that previously took `pos`/`raw_row` as
value parameters into `_dec` variants reading from `g_decode_state`:

- `embed_token_hc_dec_kernel`
- `head_rms_norm_rope_tail_dec_kernel` (Q path, fused)
- `rope_tail_dec_kernel` (Q, KV, indexer_q, heads inverse callers)
- `kv_rope_fp8_store_raw_dec_kernel` (fused KV path)
- `store_raw_kv_dec_kernel` (unfused KV write fallback)
- `quantize_group_heads_inverse_rope_q8_0_dec_kernel` (attention output
  inverse rope)

Dispatch happens in the existing tensor wrappers via
`if (g_use_decode_state && n_tok == 1)`. Non-decode callers (prefill batch,
fp8 fallbacks) keep using the value-parameter kernels.

**Verified bit-equivalent** to direct mode on short prompt (32 steps) and
on long prompt with indexer rope path active (64 steps).

#### The compressor pos bug (the actual Phase 4b blocker)

Phase 4b sat parked for a while because the wrappers caused logit
divergence. After `printf` instrumentation inside the dec kernel:

```
rope_tail_dec: pos_arg=12 g_decode_state.pos=15
```

Root cause: `ds4_gpu_compressor_update_tensor` (ds4_cuda.cu:7244) calls
`ds4_gpu_rope_tail_tensor` with `pos + 1u - ratio` (i.e. `pos - 3` for
`ratio=4`) — a *derived* position for the compressor block rotation, not
the absolute decode pos. With `g_use_decode_state` on, the dispatch sent
this call to `rope_tail_dec_kernel` which read `g_decode_state.pos` (15)
instead of the caller's 12.

Fix: bypass the wrapper from inside `compressor_update_tensor` and launch
`rope_tail_kernel` directly. All other `rope_tail` decode callers (Q, KV,
indexer_q, heads inverse) do pass the absolute decode pos, so their
dispatch to dec is safe.

**Reusable lesson**: a dispatch wrapper that reads "live state" from a
global is unsound whenever any caller passes a *derived* version of that
state. Either annotate the callers or bypass from those call sites.

### Phase 4c — cache exec via `cudaGraphExecUpdate` (commit `a103cb7`)

Added `ds4_gpu_graph_capture_end_update(handle_inout)`:
- First decode token: capture + `cudaGraphInstantiate`, store handle.
- Subsequent tokens: capture, try `cudaGraphExecUpdate` on the cached exec.
  On topology drift (attention path swap, indexer toggle) the update
  returns failure; transparently fall back to a fresh `cudaGraphInstantiate`
  and swap the exec in place.

The integration in `ds4.c` (`metal_graph_eval_token_raw_swa`) keeps the
handle in a `static __thread` slot across calls.

**Why a single cached graph (without ExecUpdate) is not enough**: attention
kernel args (`n_raw`, `raw_start`, `n_comp`, `comp_mask` pointer) include
*per-layer* values (`n_comp`) that do not fit in a single
`g_decode_state` struct. The choice was either (a) convert all attention
kernels to read per-layer state from device arrays — heavy work — or (b)
re-capture per token and let `ExecUpdate` patch the kernel params on the
cached exec. Option (b) is what's implemented.

## Why graph capture did not help — the real bottleneck

This is the most important finding of the session. Profile with
`DS4_METAL_GRAPH_TOKEN_PROFILE=1` (one decode token at 256k ctx, ~22 prefill
tokens):

**Direct mode (no graph):**
```
encode=13-20 ms   execute=38-42 ms   total=55-58 ms   → ~17.5 t/s
```

**Graph mode (ExecUpdate cached):**
```
encode=1 ms       execute=55 ms      total=56 ms      → ~17.8 t/s
```

`encode` measures wall time of `metal_graph_encode_token_raw_swa` (issuing
kernel launches or recording them into the graph). `execute` is everything
from there to the post-`cudaDeviceSynchronize` boundary, so it includes
`cudaGraphInstantiate` (when fresh) + launch + GPU wait.

**Direct mode encode (13-20 ms) overlaps with GPU work via async stream
semantics.** While the CPU is issuing kernel N+50, the GPU is already
running kernel N. By the time CPU encode finishes, most GPU work is done.
The "wall-clock cost" of the encode is therefore close to zero.

**Graph mode serializes**: encode (capture phase, ~1 ms) → instantiate or
ExecUpdate → launch → GPU work. The GPU work itself is ~55 ms; nothing in
the graph machinery changes that.

Net: **graph capture eliminates a cost that was already being hidden by
async pipelining**.

### 3-run bench (8192 ctx, 64 gen tokens, warm weights)

| Mode | Run 1 | Run 2 | Run 3 | Avg |
|---|---|---|---|---|
| Direct (`./ds4-bench …`) | 15.13 | 15.16 | 15.08 | **15.12** |
| Graph cached (`DS4_CUDA_GRAPH_DECODE=1`) | 15.02 | 14.98 | 14.22 | **14.74** |

ExecUpdate recovers from the per-token `cudaGraphInstantiate` overhead
(without cache the variant was 14.6 t/s), but does not beat direct mode.
The Run 3 dip is likely thermal/clock noise.

### What Phase 4 leaves behind

- The `_dec` kernel infrastructure is still useful: kernels read per-token
  state from device memory instead of via launch params, which removes a
  small amount of per-launch arg-marshaling and is required for any future
  full graph cache work (Phase 5).
- `cudaGraphExecUpdate` works and is the correct mechanism if the GPU
  bottleneck later eases (faster kernels would re-expose host-encode cost).
- No code on the release path is changed: `DS4_CUDA_GRAPH_DECODE=1` is
  opt-in, off by default. Direct mode behaviour and timings are unchanged.

## Decode kernel breakdown (steady state)

`nsys profile --trace=cuda` on `./ds4 --temp 0 -n 64 -p "<long prompt>"`,
filtered to decode-relevant kernels (instance counts ≈ 64 tokens × ~43 MoE
layers = 2709 ± routing variation):

| % GPU | Kernel | Per call | × N layers / token |
|------:|---|---:|---:|
| 16 % | `matmul_q8_0_preq_warp8_cached_x` (Q/KV/output) | 118 µs | ~10 ms |
| 16 % | `moe_gate_up_mid_decode_lut_qwarp32` (MoE gate+up+silu+mul) | 234 µs | ~10 ms |
| 14 % | `matmul_q8_0_hc_expand_preq_warp8` (HC expand) | 103 µs | ~9 ms |
| 11 % | `grouped_q8_0_a_preq_warp8` (output low projection) | 158 µs | ~7 ms |
|  7 % | `moe_down_sum6_qwarp32` (MoE down, top-6 fused) | 110 µs | ~4.7 ms |
|  5 % | `matmul_q8_0_pair_preq_warp8` | 84 µs | ~3.5 ms |
|  4 % | `matmul_f16_pair` (HC) | 45 µs | — |
|  4 % | `attention_decode_mixed` | 63 µs | ~2.7 ms |

Sum of top 8 ≈ 77 % of decode GPU time.

### MoE down is NOT the bottleneck on decode

The earlier framing ("MoE down is 56 % of GPU time") was the **prefill**
batch path, where `moe_down_expert_tile8_row32_kernel` runs at ~3.4 ms per
invocation. On per-token decode the dispatch picks the small fast path
`moe_down_sum6_qwarp32_kernel` (because `n_tokens=1 && n_expert=6` matches
`use_direct_down_sum6`), which is only 7 % of decode GPU time. This kernel
is already well-optimised: single-warp-quarter (8 lanes per row), `dp4a`
inner loop, fused over the 6 top-k experts to write a single output row.

### MoE gate+up is the biggest single MoE cost

`moe_gate_up_mid_decode_lut_qwarp32_kernel` (ds4_cuda.cu:9504) takes
~234 µs per call × 43 MoE layers = ~10 ms per token, 16 % of GPU time.

It already does:
- Per quarter-warp row layout (32 rows × 8 lanes per block)
- IQ2 grid + signs in shared memory (LUT decode)
- xq blocks in shared memory when `xq_blocks <= 16`
- `dp4a` packed-int8 dot accumulators
- Fused gate · silu · up · routing_weight in one kernel

Why it is hard to make faster: it is **memory-bound** by `iq2_xxs` weight
reads. Per layer it reads ~120-240 MB of expert weights (6 experts × 2
matrices × `expert_in_dim * expert_mid_dim / 4` bytes). At an observed
~128-150 GB/s effective bandwidth this is 40-50 % of the LPDDR5X peak —
already in the range cuBLAS achieves on equivalent workloads. Further gains
would require: vectorized async loads via `cuda::pipeline`, better LUT
layout to reduce shared-memory traffic, or moving to a smaller weight
encoding (already iq2 ≈ 2.5 bits/weight).

### Bandwidth-ceiling sanity check

Decode per-token MoE weight reads:
- 6 active experts × 2 matrices (gate+up) × (4096 × `expert_mid_dim` / 4) bytes
- × 43 MoE layers
- ≈ 8-15 GB per token (depending on `expert_mid_dim`)

At ~150 GB/s effective = 53-100 ms per token just for MoE weight motion.
Observed total is 55 ms per token. **MoE is the entire GPU budget** in the
memory sense. Reducing per-kernel time below the bandwidth floor would
require reading less weight per token — i.e. lower-precision weights or
fewer active experts per token.

## What was tried and did not help (or did, but small)

- **`__constant__` → `__device__` for `g_decode_state`**: was an
  *hypothesised* bug fix (turned out to be the wrong hypothesis), but kept
  because `__device__` has unambiguous stream-capture semantics for symbol
  writes and the 16-byte global read per kernel is negligible.
- **CUDA Graph capture per token**: encode 0.8 ms, execute 55 ms (incl.
  `cudaGraphInstantiate`). Total ≈ 56 ms = ~17.8 t/s. ~5 % *slower* than
  direct (the instantiate cost exceeds the encode savings hidden by async).
- **`cudaGraphExecUpdate` caching**: recovers the instantiate cost to be
  ~equal to direct, but does not beat it. Encode (1 ms) was already free in
  direct mode via stream overlap.
- **GPU clock pinning to 3003 MHz**: only ~2463 MHz actually observed
  under sustained load (likely thermal). No measurable change in t/s.
  Confirms the bottleneck is memory bandwidth, not core clock.

## 2026-05-18 second pass — MTP from a clean hypothesis

The previous working assumption was that CUDA MTP could not be useful until
missing quantized tensor support was added. That was stale: Q4_K dispatch is
present, and the actual failure mode was numerical corruption inside the MTP
decode layer.

### Actual MTP bug found

The MTP GGUF stores these HC function tensors as F32:

```
mtp.0.hc_attn_fn.weight  F32  [16384, 24]
mtp.0.hc_ffn_fn.weight   F32  [16384, 24]
mtp.0.hc_head_fn.weight  F32  [16384, 4]
```

The CUDA fused HC pre-norm path assumed the function tensor was F16. That is
valid for the base model, but invalid for MTP. The symptom chain was:

1. MTP `mtp_input_hc` was finite.
2. `attn_norm` became all NaN at the start of the decode layer.
3. Router selection degenerated to `-1,-1,-1,-1,-1,-1`.
4. Final MTP logits were NaN/degenerate, so drafts were token 0 and never
   accepted.

Fix: split the CUDA fused HC pre-norm dispatch by function tensor type. The
base model keeps the original F16 fused path, while MTP uses a new F32 fused
path instead of interpreting the same bytes as half precision.

### MTP after the fix

Diagnostics used:

```sh
DS4_MTP_STATS=1 DS4_MTP_TIMING=1 DS4_MTP_FULL_LOGITS=1 \
DS4_CUDA_MOE_SELECTED_TRACE=1 \
./ds4 --nothink --temp 0 -n 1 --ctx 1024 \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 -p "Hi"
```

Observed result after the fix:

- `attn_norm`, `router_logits`, `routed_out`, `out_hc`, and `logits` are
  finite.
- CUDA MoE selected experts are real expert ids, not `-1`.
- Device top id and host argmax agree when full logits are requested.

On a deterministic acceptance-friendly prompt:

```sh
DS4_MTP_TIMING=1 DS4_MTP_CONF_LOG=1 \
./ds4 --nothink --temp 0 -n 64 --ctx 4096 \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  -p "List the first 50 prime numbers, separated by commas."
```

Observed immediately after the correctness fix:

- Draft probe after warmup: ~22.2 ms.
- Recursive second draft: ~22.4 ms.
- Target verify of 2 drafted positions: ~129-135 ms.
- Acceptance: consistently 2/2 on the predictable prompt.
- End-to-end generation: **12.32 t/s**.

Follow-up verifier work added a K<=2 MoE down path that writes the final
token rows directly for the common `n_expert=6` case. The best default shape
keeps sorted gate/up for K=2, then uses the direct batched down-sum kernel.
The fully direct K=2 gate path is available behind
`DS4_CUDA_MOE_K2_DIRECT_GATE=1`, but measured slightly worse on the prime
prompt.

Second follow-up: the verifier was accidentally taking cuBLAS/F16-cache paths
for Q8 microbatches where the native Q8 kernels are faster on GB10. Two
changes moved the needle:

- A specialized `n_tok=2` Q8 kernel reads each output row's Q8 weights once
  and accumulates both token rows.
- Q8 cuBLAS/F16-cache dispatch is skipped by default for `n_tok <= 2`;
  larger prefill batches can still use cuBLAS. `DS4_CUDA_Q8_CUBLAS_DECODE=1`
  exists only as a diagnostic and measured worse for one-token decode.

Latest observed draft=2 state:

- Target verify of 2 drafted positions: ~103-109 ms.
- Acceptance: still consistently 2/2 on the predictable prompt.
- End-to-end generation: **13.82 t/s** on the prime prompt.

Longer drafts are worse with the current verifier. `--mtp-draft 4` typically
committed only 2 of 4 drafted tokens, paid the same replay cost, and fell to
**6.58 t/s**. `--mtp-draft 3` still commits only 2 of 3 after the Q8 changes
and reaches only **7.62 t/s**.

`DS4_MTP_PREFIX_BATCH=1` tested the more aggressive shape where the next
sampled target token is included in the verifier batch instead of being decoded
first. On the prime prompt it consistently accepted only 2 of 3 proposed rows,
then had to restore and fall back; with the extra MTP prime call it regressed
to ~8.15 t/s. It is not a viable default.

### Why this still misses 20 t/s

MTP draft=2 emits 3 tokens per speculative cycle when both drafts are
accepted: the first full-model token plus two draft tokens. The current cycle
cost is approximately:

| Stage | Time |
|---|---:|
| First target decode | ~59 ms |
| MTP draft probe | ~22 ms |
| MTP recursive draft | ~22 ms |
| Target verify for 2 positions | ~103-109 ms |
| **Total / 3 emitted tokens** | **~206-212 ms = ~14.2-14.6 t/s theoretical** |

To reach 20 t/s, the 3-token cycle must fit inside 150 ms. The current
implementation therefore still needs roughly 55-60 ms removed from the cycle.
That is larger than the remaining obvious verifier micro-optimizations.

Stage profiling confirms this:

- Normal single-token decode at short context: ~58 ms stage sum, matching
  ~17.7-18.0 t/s in end-to-end runs.
- Strict MTP verification positions: ~54-59 ms each.
- Current draft=2 verifier: ~103-109 ms after microbatch Q8 fixes.

Micro-toggles tried on the verifier/MoE path did not open the gap:

| Toggle | Result |
|---|---|
| `DS4_CUDA_MOE_NO_EXPERT_TILES=1` | worse, ~11.67 t/s |
| `DS4_CUDA_MOE_NO_P2=1` | neutral/slightly worse, ~12.04 t/s |
| `DS4_CUDA_MOE_TILE4=1` | worse, ~11.32 t/s |
| gate/down rowspan + atomic down | worse, ~11.11 t/s |
| `DS4_CUDA_MOE_K2_DIRECT_GATE=1` | near-neutral/slightly worse than sorted gate/up + direct down |

### Memory budget note

When MTP is enabled, startup now prepares the CUDA range cache for both the
base model and the MTP model. The MTP file is ~3.55 GiB, which is within the
"spend a few more GiB if it buys speed" budget, and avoids first-use cache
surprises in the server path. It does not by itself buy the 20 t/s target; it
only makes MTP startup behavior deterministic.

Server smoke after the fix:

```sh
DS4_MTP_TIMING=1 ./ds4-server -m ds4flash.gguf --ctx 4096 \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  --host 127.0.0.1 --port 8099 --tokens 16
```

Observed startup:

- Base model CUDA range cache: 80.76 GiB in 19.856s.
- MTP CUDA range cache: 3.55 GiB in 12.643s.
- Context buffers at `--ctx 4096`: 263.46 MiB.
- HTTP `/v1/completions` smoke completed 16 tokens with no OOM or server
  instability.

No-MTP server smoke after the microbatch Q8 changes:

```sh
./ds4-server -m ds4flash.gguf --ctx 8192 \
  --host 127.0.0.1 --port 8101 --tokens 64
```

Observed for an HTTP `/v1/completions` request with 64 generated tokens:

- Context buffers: 333.21 MiB at `--ctx 8192`.
- Generation log: **17.32 t/s avg** in the server path.
- Completion finished normally with no OOM.

## 2026-05-18 third pass — normal decode remains the shortest route

After MTP correctness and verifier work, the hypothesis changed again: rather
than spending more memory on the draft model, look for low-risk wins in the
default server path. The 20 t/s target needs about 50 ms/token; the current
normal decode path is around 55-56 ms/token. That leaves a concrete gap of
roughly 5-6 ms/token.

### Default wins kept

1. **Q/KV pair projection in decode.** When the fused QKV RMS path is active,
   `attn_q_a` and `attn_kv` now use one paired Q8 projection
   (`ds4_gpu_matmul_q8_0_pair_tensor`) from the same `attn_norm` input instead
   of two independent Q8 matmuls. This quantizes the activation once and reads
   the input once. It is on by default and can be disabled with
   `DS4_METAL_DISABLE_QKV_PAIR_PROJ=1`.

   Measured effect: `q_path` dropped from about 9.86 ms to about 9.46 ms over
   43 layers. End-to-end CLI moved from ~17.7-18.0 t/s to ~17.8-18.16 t/s,
   depending on run length and thermal noise.

2. **Shared expert gate/up + SwiGLU fusion.** The shared expert Q8 gate/up
   pair now has a fused path that computes gate, up, and the SwiGLU mid tensor
   in one kernel. It keeps the previous gate/up writes for compatibility, but
   removes the separate elementwise activation pass from the hot path. It is
   on by default and can be disabled with
   `DS4_CUDA_NO_SHARED_GATE_UP_FUSED_SWIGLU=1`.

   Measured effect: `shared_gate_up` dropped from about 4.13 ms to about
   4.00 ms over 43 layers. The end-to-end gain is small but consistent enough
   to keep.

3. **Device top-1 path.** `ds4_gpu_indexer_topk_tensor` now has a dedicated
   top-1 CUDA kernel instead of using the generic one-thread top-k fallback
   when `top_k == 1`. The greedy CLI path also avoids reading full logits back
   for the next token when top tracing is disabled.

   Measured effect: neutral for normal CLI throughput because the server still
   samples from `s->logits`, but it removes a known bad primitive from the
   greedy/top-only path and helps the MTP fallback path.

### Third-pass measurements

Representative 8192-context runs after the default wins:

| Mode | Tokens | Result |
|---|---:|---:|
| Direct CLI, no MTP | 64 | **18.06-18.16 t/s** |
| Direct CLI, no MTP | 128 | **17.81-17.82 t/s** |
| `DS4_CUDA_GRAPH_DECODE=1`, no MTP CLI | 128 | **18.13 t/s** |
| Server, no MTP | 64 | **17.47 t/s** after this pass |
| `DS4_CUDA_GRAPH_DECODE=1` server, no MTP | 64 | **17.81-17.85 t/s** after this pass |

Server smoke command for the current number:

```sh
./ds4-server -m ds4flash.gguf --ctx 8192 \
  --host 127.0.0.1 --port 8101 --tokens 64
```

Observed for an HTTP `/v1/completions` request with 64 generated tokens:
prompt prefill 0.820s, decode **17.47 t/s avg**, finish=length, no OOM.

Graph server smoke:

```sh
DS4_CUDA_GRAPH_DECODE=1 ./ds4-server -m ds4flash.gguf --ctx 8192 \
  --host 127.0.0.1 --port 8102 --tokens 64
```

Observed on the same request: prompt prefill 0.821-0.881s, decode
**17.81-17.85 t/s avg**, finish=length, no OOM.

CUDA graph is still env-gated. The best 128-token number came from graph mode,
but token profiling still shows the same structural shape: graph capture moves
time from host encode to GPU execute rather than removing the GPU bottleneck.
It is now confirmed useful as an optional server mode, but it is not the 20 t/s
solution by itself.

Current stage profile after the default wins, over 43 layers:

| Stage | Total |
|---|---:|
| Routed MoE | ~15.7 ms |
| Attention output projection | ~14.4 ms |
| Q path | ~9.3 ms |
| Shared expert gate/up/down | ~6.3 ms |
| Compressor/indexer | ~3.3 ms |
| Attention kernel | ~2.5 ms |

The remaining budget is concentrated exactly where the bandwidth model says it
should be: routed MoE reads and Q8 projection reads.

### Third-pass dead ends

These were tested specifically because the machine can afford a few more GiB
of resident memory if that buys speed. None did.

| Experiment | Switch | Result |
|---|---|---|
| HC expand via F16 cache + cuBLAS | `DS4_CUDA_HC_EXPAND_F16=1` | worse, ~15.98 t/s at 64 tokens |
| HC expand cached activation | `DS4_CUDA_HC_EXPAND_CACHE_X=1` | no benefit; `attn_output` slightly worse (~14.6 ms) |
| Targeted cuBLAS for `attn_q_b` | `DS4_CUDA_ATTN_Q_B_CUBLAS_DECODE=1` | much worse; `q_path` ~31 ms |
| Quarter-warp Q8 decode kernel | `DS4_CUDA_Q8_QWARP_DECODE=1` | worse; `q_path` ~10.05 ms |
| Q8 cuBLAS/F16 cache for small decode | `DS4_CUDA_Q8_CUBLAS_DECODE=1` | worse; kept diagnostic-only |

Conclusion from this pass: there is no obvious "use more memory, get tensor
cores, reach 20" switch left in the current implementation. The next 5-6
ms/token must come from either reading less weight per token or materially
improving the two dominant custom Q8/IQ2 kernels.

## 2026-05-18 fourth pass — outside-the-box checks

This pass deliberately tested less orthodox ideas:

1. **Server greedy top-only state.** For `temperature=0` without MTP, the
   server does not need the full logits row on the CPU after every token; it
   only needs the next argmax. The session now has a greedy eval path that runs
   the decode, computes top-1 on the GPU, caches that token, and materializes
   full logits lazily only if sampling/logprobs/KV snapshot code needs them.
   It can be disabled with `DS4_SERVER_DISABLE_GREEDY_TOP=1`.

   Result: correct and stable, but almost neutral. Direct server moved from
   17.47 t/s to **17.50 t/s** on the 64-token smoke; graph server measured
   **17.81-17.83 t/s**, effectively the same as the previous 17.85 t/s. This proves
   the CPU logits readback is not the missing 5-6 ms/token. The expensive part
   is computing the full output head on GPU, which exact argmax still requires.

2. **Direct exact output-head top1.** Added an experimental Q8 output projection
   path that computes the exact argmax directly during the vocab matmul instead
   of writing `g->logits` and then running top-k. It is correctness-preserving
   for greedy top-only decode, and it materializes full logits lazily if a later
   API call asks for logprobs or session payload output.

   Result: negative in A/B. With graph server and the same 64-token prime prompt,
   `DS4_CUDA_OUTPUT_TOP1=1` measured **17.77-17.78 t/s** after warmup, while the
   previous full-logits output head measured **17.95-17.98 t/s** in the same
   build. The reason is practical rather than theoretical: the existing Q8
   output matmul is already highly tuned, and the direct top1 variant adds a
   second reduction stage without reducing the dominant weight read. The code is
   retained behind `DS4_CUDA_OUTPUT_TOP1=1` for future kernel work, but it is not
   a default.

3. **Reduced active routed experts.** Added experimental
   `DS4_MOE_ACTIVE_EXPERTS=N` / `DS4_CUDA_MOE_ACTIVE_EXPERTS=N` for one-token
   decode, with optional `DS4_MOE_ACTIVE_EXPERTS_RENORM=1` to renormalize the
   top-N router weights. The idea was to read fewer MoE expert weights per
   token, accepting a quality/speed trade-off if the speed crossed 20 t/s.

   First result: K=5 measured **16.09 t/s** with renorm and **16.08 t/s**
   without renorm. K=4 measured **15.81 t/s**. That showed the naive K<6 path
   was not enough.

   Follow-up: added dedicated one-token direct down-sum kernels for K=5/4/3/2,
   so reduced-K decode no longer falls through the generic dynamic down-sum
   loop. The new measurements:

   | Mode | Measurement |
   |---|---:|
   | `ds4-bench`, K=6 full model, same build | 15.55 t/s |
   | `ds4-bench`, K=5 | 16.07 t/s |
   | `ds4-bench`, K=4 | 16.65 t/s |
   | `ds4-bench`, K=3 | 17.62 t/s |
   | `ds4-bench`, K=2 | 18.28 t/s |
   | `ds4-bench`, K=2 + `DS4_CUDA_MOE_K2_DIRECT_GATE=1` | 18.37 t/s |
   | server graph+greedy, K=2 | **21.22-21.28 t/s** |
   | server graph+greedy, K=2 + renorm | **21.41-21.44 t/s** |
   | server graph+greedy, K=2 + renorm + direct-gate | 21.35 t/s |
   | server graph+greedy, K=3 + renorm, coding eval prompts | **20.09-20.26 t/s** |
   | server graph+greedy, K=4 + renorm, coding eval prompts | 18.83-19.05 t/s |

   The K=2 server recipe that crossed the target:

   ```sh
   DS4_CUDA_GRAPH_DECODE=1 \
   DS4_MOE_ACTIVE_EXPERTS=2 \
   DS4_MOE_ACTIVE_EXPERTS_RENORM=1 \
   ./ds4-server -m ds4flash.gguf --ctx 8192 \
     --host 127.0.0.1 --port 8106 --tokens 64
   ```

   This used the same 80.76 GiB CUDA model cache and 333.21 MiB context buffers
   at 8192 ctx; no OOM or server instability was observed in the two-pass smoke.
   It is a quality-changing mode: every decode layer keeps only the top-2
   routed experts instead of the model's top-6. It should not replace the
   full-quality default without an eval pass.

### Coding eval smoke for reduced-K

The first coding-oriented eval used `tuning/coding_eval.py`: four deterministic
Python implementation prompts (`merge_intervals`, `top_k_frequent`, `LRUCache`,
`parse_duration`) sent through the OpenAI-compatible server at `temperature=0`,
then executed against local unit tests. This is intentionally small, but it
tests the failure modes that matter for coding agents: syntactically complete
code, following "return code only", and passing edge-case tests.

| Mode | Unit tests | Client token rate | Server decode logs | Notes |
|---|---:|---:|---:|---|
| K=6 full model | 3/4 | 15.50-17.20 t/s | ~17.6-17.8 t/s | Failed `LRUCache` with a logic bug. |
| K=2 + renorm | 2/4 | 19.65-20.39 t/s | ~20.7-21.1 t/s | Two tasks hit 700-token cap; one malformed code fence, one incomplete rewrite. |
| K=3 + renorm | 3/4 | 18.56-19.53 t/s | ~20.1-20.3 t/s | No format collapse; failed `parse_duration` with a logic bug. |
| K=4 + renorm | 3/4 | 17.31-18.37 t/s | ~18.8-19.1 t/s | Better behaved, but under the 20 t/s target. |
| K=6 on layers 0-2, K=3 + renorm elsewhere | 4/4 | 18.49-19.28 t/s | ~20.0 t/s | First coding smoke that hits the speed target and passes all four tasks. |

Commands used:

```sh
DS4_CUDA_GRAPH_DECODE=1 DS4_MOE_ACTIVE_EXPERTS=3 \
DS4_MOE_ACTIVE_EXPERTS_RENORM=1 \
./ds4-server -m ds4flash.gguf --ctx 8192 \
  --host 127.0.0.1 --port 8106 --tokens 900

python3 tuning/coding_eval.py --label k3_renorm --max-tokens 700
```

Takeaway: the "cheat" that works is reading far fewer routed expert weights.
K=2 is fast enough but not coding-safe in this smoke: it rambles, self-corrects,
and can leave malformed or truncated code. K=3 is the interesting point: it
still crosses 20 t/s in the server decode logs and is qualitatively much closer
to the full model on these coding tasks. K=4 recovers a little more behavior but
falls below target. The engineering question is now whether K=3 is acceptable
for real use, or whether an adaptive K policy can use K=2 only when router
confidence is high and fall back to K=3/K4/K6 elsewhere.

### Adaptive shadow and per-layer K profile

The first adaptive attempt is deliberately "shadow only":

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_MOE_ADAPTIVE_SHADOW=1 \
DS4_MOE_ADAPTIVE_K3_MASS=0.62 \
DS4_MOE_ADAPTIVE_K4_MASS=0.78 \
./ds4-server -m ds4flash.gguf --ctx 8192 --host 127.0.0.1 --port 8110 --tokens 900
```

It records, on device, what K a router-mass policy would choose after top-6
routing and before any reduced-K renorm. It does not change generation. The
initial 0.90/0.96 thresholds were wrong for coding prompts: top-3 mass averaged
only about 0.64-0.66, so the policy chose K=6 almost everywhere. With thresholds
0.62/0.78, it picked K3 for roughly 63-74% of layer/token decisions, but the
same low-confidence layers kept recurring: especially layers 0, 1, and 2.

That led to a simpler production-compatible policy: static per-layer K. This
keeps CUDA Graph topology stable, unlike token-by-token host-selected K values.
The new env map is:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_MOE_ACTIVE_EXPERTS=3 \
DS4_MOE_ACTIVE_EXPERTS_RENORM=1 \
DS4_MOE_ACTIVE_EXPERTS_LAYERS=0-2:6 \
./ds4-server -m ds4flash.gguf --ctx 8192 --host 127.0.0.1 --port 8110 --tokens 900
```

Coding eval result saved as `tuning/coding_eval_results/layer0_2_k6_k3.json`:
**4/4 passed**. Server decode logs stayed at about **19.96-20.10 t/s** across
the four tasks. The measured run also had shadow logging enabled; the production
command above omits it, so this should be a lower-bound speed measurement. This
is the first result in this log that meets the 20 t/s goal and improves the
small coding smoke over both full K=6 and uniform K=3.

### Long-context reality check at ~250k resident tokens

This section is a stress test, not the primary 20 t/s target. The clarified
target is **~20 t/s at the start of a session with `--ctx 256000` allocated**;
normal degradation as resident context grows is acceptable. The 250k frontier
is still useful because it exposes which ideas only work by changing attention
or indexer numerics, but it should not be used as the pass/fail bar for the
20 t/s effort.

The earlier `ctx=256000` coding and server tests allocate the production
context size, but most prompts in that suite do not fill it. A separate
frontier test used a truncated `promessi_sposi.txt` prompt with **250558**
raw tokens (`/tmp/ds4-promessi-790k.txt`) and measured decode at a **250000**
token frontier:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
DS4_METAL_GRAPH_TOKEN_PROFILE=1 \
./ds4-bench --cuda -m ds4flash.gguf \
  --prompt-file /tmp/ds4-promessi-790k.txt \
  --ctx-start 250000 --ctx-max 250000 --ctx-alloc 256000 \
  --step-incr 1 --gen-tokens 16 \
  --csv /tmp/ds4-bench-soa-graph-ctx250k.csv
```

Result:

```text
ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
250000,250000,233.33,16,9.41,3464990668
```

The per-token graph profile averaged about **105.88 ms/token** at positions
250000-250015. By contrast, the same 12.5k-token prompt at `--ctx 256000`
measured **20.28 t/s** in CLI generation reporting and ~65.7 ms average
graph-token total. The allocation itself is not the problem: the real number
of resident KV/compressed rows is.

The non-graph stage profile at `pos=250566` is compatible with the graph token
time and shows the new bottleneck:

| Decode stage | Stack over 43 layers |
|---|---:|
| `attention` | **31.427 ms** |
| `compressor_indexer` | **27.232 ms** |
| `routed_moe` | **16.081 ms** |
| `attn_output` | **14.066 ms** |
| `q_path` | **9.269 ms** |
| shared expert gate/up/down | **6.391 ms** |

Inside the ratio-4 indexer at `comp=62641`:

| Indexer substage | Stack over 21 ratio-4 layers |
|---|---:|
| `decode_score` | **17.289 ms** |
| `decode_topk` | **4.170 ms** |
| `decode_attention` | **4.160 ms** |

The prefill indexer profile also scales linearly with compressed rows:
`score` grew from ~2.8 ms/chunk/layer at `pos=2048` to ~191.5
ms/chunk/layer at `pos=247808`; `topk` grew from ~0.5 to ~51.6 ms. This
does not directly set decode speed, but it confirms that long-context indexer
cost is fundamentally proportional to compressed-row count.

Three one-token long-context attention/indexer probes were measured:

| Probe | Switch | 250k gen t/s | Quality/logprob result |
|---|---|---:|---|
| Force one-token indexer score through the existing WMMA batch path | `DS4_CUDA_NO_INDEXER_DIRECT_ONE=1` | **8.20** | Slower; not pursued. |
| Route one-token decode attention through the heads8 online kernels | `DS4_CUDA_DECODE_HEADS8_ATTENTION=1` | **9.66** | Changed the greedy token at step 1 on the 12.5k logprob check (`" disp"` -> `" hai"`). |
| Parallelize only the one-token attention dot, keeping the existing score-buffer softmax/value pass | `DS4_CUDA_DECODE_PARALLEL_ATTENTION_DOT=1` | **10.50** | Also changed the greedy token at step 1 (`" disp"` -> `" hai"`). |
| Increase the online heads8 row stage size from 4 to 8 rows | `DS4_CUDA_DECODE_HEADS8_ROWS=8` | **9.47** | Same checksum at 250k, but only +0.06 t/s; a 64k proxy showed it slower than default. |
| Pair two compressed rows per one-token indexer-score block | `DS4_CUDA_INDEXER_DIRECT_ONE_PAIR2=1` | not benchmarked | Failed the 12.5k logprob gate; greedy diverged from step 1. |

The dot-parallel flag is the best speed probe so far for long context
(~**+11.6%** over the 9.41 t/s control), but it is not quality-preserving
under the current bar because changing the dot summation order is enough to
flip close logits. It should remain a diagnostic only. The default quality path
therefore stays exact SoA; getting near 20 t/s at a genuinely full 256k context
needs either an exact attention/indexer redesign that preserves numeric
behavior, or a verifier/fallback scheme that can prove fast-path agreement
before committing tokens.

Because a full 250k resident-context benchmark costs roughly 18 minutes of
prefill on this machine, row-stage tuning was also checked on a shorter
resident-context proxy. A `/tmp/ds4-promessi-205k.txt` prompt contains 65482
raw tokens and was measured at `ctx-start=64000`, enough to exercise the
long-context online attention fallback:

| Online heads8 row stage | 64k gen t/s | Avg graph-token total | Checksum |
|---|---:|---:|---:|
| default 4 rows | **13.55** | **73.451 ms** | 904891788 |
| 8 rows | **13.43** | 74.088 ms | 904891788 |
| 16 rows | **13.36** | 74.506 ms | 904891788 |

The row-stage idea is therefore behavior-stable but not a performance lever.
The pair2 indexer idea was more aggressive and looked exact-shaped on paper,
but in practice changed the selected tokens (`48076,11062,...` became
`48076,41995,...`), so it is rejected for quality-sensitive coding use.

After the target clarification, the low-context exact baseline was remeasured
with the production allocation:

| Mode | Context allocation | Resident prompt | Tokens | Decode result |
|---|---:|---:|---:|---:|
| CLI top-only, SoA+graph | 256000 | 25 tokens | 256 | **18.06 t/s** |
| CLI top-only, SoA+graph + graph-token profile | 256000 | 25 tokens | 128 | **18.42 t/s** reported, ~18.28 t/s by token totals |
| Server `/v1/completions`, `deepseek-chat`, SoA+graph | 256000 | 25 tokens | 128 | **18.09 t/s avg** |
| `ds4-bench`, full-logits greedy | 256000 | 2048 tokens | 128 | **16.47 t/s** |
| `ds4-bench`, full-logits greedy | 256000 | 8192 tokens | 128 | **15.99 t/s** |

The 20 t/s work should optimize the first three rows. `ds4-bench` remains
useful as a stable primitive benchmark, but it reads full logits every token and
is stricter than the top-only/server target path.

Follow-up low-context checks after the target clarification:

| Probe | Result |
|---|---:|
| CLI SoA+graph repeat, 128 tokens | **18.47 t/s** |
| Force `attn_output_b` through generic SoA decode | **18.45 t/s** |
| Add `attn_q_b` SoA cache | **18.38 t/s** |
| Add `attn_q_a/attn_kv` SoA cache | **18.37 t/s** |
| Disable decode MoE LUT gate | **15.98 t/s** |
| Server SoA+graph repeat, 128 tokens | **17.93 t/s avg** |
| Server SoA+graph + direct output top1 | **18.08 t/s avg** |

These keep the target honest: the remaining gap is not in the already-tested
Q8 SoA extension flags, and direct output top1 is at best a small/noisy server
win, not the missing 10%.

Hardware state during a 256-token low-context run was also sampled. The GPU was
in P0 and about 91-95% utilized, but SM clocks stayed around **2457 MHz** while
`nvidia-smi -q` reports a **3003 MHz** max clock. Power draw was about **43 W**
and GPU temperature about **54-55 C**. A later privileged clock-lock attempt
accepted `sudo nvidia-smi -lgc 3003,3003`, but a real decode run still stayed
at **2476 MHz** under load and measured **17.92 t/s**, effectively unchanged.
On this GB10/driver stack, `nvidia-smi` clock lock alone is therefore not an
available route to the missing 10%.

### Extended coding eval at ctx=256k

The 4-task smoke was too small. A 12-task eval (`tuning/coding_eval_extended.py`)
was run through the server at `--ctx 256000`, `--tokens 1000`, `temperature=0`.
It adds common coding-agent tasks: grouping anagrams, flattening nested dicts,
sliding windows, bracket validation, CSV parsing, topological sort, edit
distance, and binary-search bounds.

| Mode | ctx | Unit tests | Server decode logs | Failures |
|---|---:|---:|---:|---|
| Full K=6 | 8192 | 10/12 | ~17.5 t/s | `lru_cache`, `parse_csv_line` |
| K=6 on layers 0-2, K=3 + renorm elsewhere | 8192 | 10/12 | ~20.0 t/s | `merge_intervals`, `parse_csv_line` |
| Full K=6 | 256000 | 10/12 | ~17.5 t/s | `lru_cache`, `parse_csv_line` |
| K=6 on layers 0-2, K=3 + renorm elsewhere | 256000 | 10/12 | ~20.0 t/s | `parse_duration`, `parse_csv_line` |
| Current no-SoA full K=6 control | 256000 | 10/12 | ~17.1-17.5 t/s | `lru_cache`, `parse_csv_line` |
| SoA A+B full K=6 (`DS4_CUDA_Q8_SOA_CACHE=1`) | 256000 | 9/12 single sample | ~17.8-17.9 t/s | `lru_cache`, `flatten_dict`, `parse_csv_line` |
| Current no-SoA full K=6, repeated | 256000 | 30/36 | 17.40 t/s final-decode avg | `lru_cache` 0/3, `parse_csv_line` 0/3 |
| SoA default full K=6, repeated | 256000 | 30/36 | 17.91 t/s final-decode avg | `lru_cache` 0/3, `parse_csv_line` 0/3 |
| SoA B-forced full K=6, repeated | 256000 | 30/36 | 17.68 t/s final-decode avg | `lru_cache` 0/3, `parse_csv_line` 0/3 |

Takeaway: same pass count is not enough. The layer-profile mode regressed
tasks that full K=6 passed (`merge_intervals` at 8k, `parse_duration` at
256k). It also passed `lru_cache` where full K=6 failed, which means the
reduced-K mode changes behavior rather than being a harmless acceleration.
With the quality constraint, this mode should stay opt-in only.

Fresh SoA controls on 2026-05-19 first looked like a second quality warning:
the SoA A+B path still passed the short byte-identical logprob check and gave a
real speed gain, but the 12-task coding eval at `ctx=256000` dropped from
**10/12** to **9/12**. The new failure was `flatten_dict`: no-SoA + graph
generated a harness-passing solution, while SoA generated a valid-looking
normal-Python solution that used `type(...)` inside an error message; the eval
harness exposes a restricted builtin set and therefore raised `NameError`.

A first direct logprob dump on that prompt showed the first selected token
divergence at generation step 23:

```text
no-SoA: selected ' must'   logit=42.7325974 logprob=-0.585161805
SoA:    selected ' cannot' logit=42.6642227 logprob=-0.668570459
```

The competing logits were close and swapped order (`' cannot'` was second for
no-SoA at logit 42.502327; `' must'` was second for SoA at logit 42.6130142).
The textual fork was only an error string (`"Separator must..."` vs
`"Separator cannot..."`).

The follow-up changed the interpretation. A fresh no-SoA run after rebuild also
selected `' cannot'` at the same step, so the earlier no-SoA/SoA diff was not a
clean SoA-only delta. Two no-SoA runs with `DS4_CUDA_FORCE_ORDERED_F16_MATMUL=1`
still differed (`"Separator must..."` vs `"Separator cannot..."`), and two
no-SoA `--quality` runs also diverged later in the same prompt. The prompt sits
on a close-logit boundary where greedy CUDA decode is not bit-stable enough for
a single-answer verdict.

Targeted server repeats at `ctx=256000` therefore used pass rate instead of
single-run text equality. On the same `flatten_dict` task, no-SoA passed **8/8**
with client-observed throughput **15.30-16.20 t/s** and server decode logs
around **17.0-17.1 t/s**. SoA A+B passed **8/8** with client-observed
throughput **15.93-16.51 t/s** and server decode logs around
**17.35-17.47 t/s**. At that stage, SoA was not cleared as full-quality by the
short `cmp=0`, but it was also not convicted by the single 9/12 sample. The
correct gate became repeated coding eval at production context.

That full repeated gate was then run:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
./ds4-server -m ds4flash.gguf --ctx 256000 \
  --host 127.0.0.1 --port 8130 --tokens 1000

python3 tuning/coding_eval_extended.py \
  --base-url http://127.0.0.1:8130 \
  --label no_soa_graph_repeat3_ctx256k_20260519 \
  --out-dir /tmp/ds4-repeat-quality \
  --max-tokens 1000 \
  --repeat 3
```

And repeated for default SoA (`DS4_CUDA_Q8_SOA_CACHE=1`) and B-forced SoA
(`DS4_CUDA_Q8_SOA_CACHE=1 DS4_CUDA_Q8_SOA_ATTN_OUTPUT_B_DECODE=1`). Results:

| Mode | Passes | Client t/s avg | Server final-decode avg | Per-task failures |
|---|---:|---:|---:|---|
| no-SoA graph | 30/36 | 16.06 | 17.40 | `lru_cache` 0/3, `parse_csv_line` 0/3 |
| SoA default graph | 30/36 | 16.51 | 17.91 | `lru_cache` 0/3, `parse_csv_line` 0/3 |
| SoA B-forced graph | 30/36 | 16.31 | 17.68 | `lru_cache` 0/3, `parse_csv_line` 0/3 |

All other tasks were **3/3** in all three modes, including `flatten_dict`.
This is the strongest quality result so far for default SoA: it improves
throughput in the production-context coding harness without a measured pass-rate
regression. It is still not enough to reach 20 t/s, but it is no longer merely
a risky speed-only experiment.

The eval harness now supports that gate directly:

```sh
python3 tuning/coding_eval_extended.py \
  --base-url http://127.0.0.1:8128 \
  --label soa_ab_flatten_repeat_ctx256k \
  --out-dir /tmp/ds4-soa-coding-eval \
  --max-tokens 1000 \
  --only flatten_dict \
  --repeat 8
```

`--only` accepts comma-separated task ids and can be passed multiple times.
Omit it for full-suite repeats; for example `--repeat 3` runs 36 coding
requests and reports both global pass count and per-task pass totals in JSON.

### Expert locality probe for no-sacrifice cache ideas

To test the "execute all 6 experts faster" idea, `DS4_MOE_EXPERT_STATS=1`
records selected routed expert IDs on device after top-6 routing. It adds no
host synchronization during decode and reports at request end. Optional
`DS4_MOE_EXPERT_STATS_DUMP=/tmp/file.csv` writes per-request
`seq,layer,expert,count` rows. `tuning/moe_expert_stats.py` summarizes the CSV
and server log.

Run used:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_MOE_EXPERT_STATS=1 \
DS4_MOE_EXPERT_STATS_DUMP=/tmp/ds4-moe-temporal-stats.csv \
./ds4-server -m ds4flash.gguf --ctx 256000 \
  --host 127.0.0.1 --port 8121 --tokens 1000

python3 tuning/coding_eval_extended.py \
  --base-url http://127.0.0.1:8121 \
  --label ext_k6_temporal_ctx256k \
  --out-dir /tmp/ds4-moe-temporal-eval \
  --max-tokens 1000

python3 tuning/moe_expert_stats.py \
  --csv /tmp/ds4-moe-temporal-stats.csv \
  --log /tmp/ds4-moe-temporal-server.log
```

Quality remained the same as the full-K baseline: **10/12** passed, failures
`lru_cache` and `parse_csv_line`. The profiler covered ~2970 generated tokens
across 12 coding requests.

Static per-layer hot-set coverage, learned over the whole coding run:

| Hot experts/layer | Selected-expert coverage |
|---:|---:|
| 4 | 20.1% |
| 8 | 31.0% |
| 16 | 44.4% |
| 32 | 59.5% |
| 64 | 76.7% |
| 96 | 86.6% |
| 128 | 92.8% |

Even an oracle per-request hot set is limited:

| Hot experts/layer | Mean | Min | Max |
|---:|---:|---:|---:|
| 16 | 53.0% | 47.0% | 60.9% |
| 32 | 70.3% | 63.6% | 78.1% |
| 64 | 87.0% | 80.3% | 92.8% |

Worst aggregate layers remain the early router layers:

| Hot set | Weakest layers |
|---|---|
| top16 | L01 17.3%, L00 17.9%, L02 19.6%, L37 29.8% |
| top32 | L01 30.0%, L00 31.1%, L02 32.9%, L35 46.5% |
| top64 | L00 50.4%, L01 50.9%, L02 51.7%, L35 67.0% |

Temporal locality is also weak. Overlap between the current top-6 and the
previous token's top-6 in the same layer averaged **1.88/6** experts, or
**31.3%** selected-expert hit rate:

| Overlap count | Share of layer/token transitions |
|---:|---:|
| 0 | 21.3% |
| 1 | 22.1% |
| 2 | 22.7% |
| 3 | 19.2% |
| 4 | 11.0% |
| 5 | 3.5% |
| 6 | 0.3% |

The memory picture matters. In this GGUF, routed expert tensors are about
**72.56 GiB** of the **80.76 GiB** model file:

| Routed tensor group | Size |
|---|---:|
| `ffn_gate_exps` | 22.17 GiB |
| `ffn_up_exps` | 22.17 GiB |
| `ffn_down_exps` | 28.22 GiB |

A duplicate all-expert hot cache is impossible inside the 110 GiB steady-state
budget. A static top64/layer cache would still miss about one quarter of expert
weight reads and would need an implausibly large hot-path speedup to close the
full 5-6 ms/token gap by itself. The only cache-shaped idea that still looks
worth exploring is **not** a hot subset: it is replacing the original routed
tensor cache with an all-expert native layout, so memory stays roughly flat
instead of duplicating 72 GiB.

## Open avenues for the next session

In rough order of effort vs likely payoff:

### 1. Native routed-expert pack, replacing the GGUF layout (open, but not naive)

Hot-subset caching does not have enough locality. The remaining no-sacrifice
MoE idea is to repack **all** routed expert tensors into a CUDA-native layout
at startup and stop caching the original routed GGUF spans. This preserves all
six experts and all quantized values, but changes memory layout so the decode
kernels can read more coalesced, prearranged blocks.

Important constraint: do not duplicate the routed tensors. They are 72.56 GiB,
so the implementation must either:

- skip startup CUDA range-cache preload for `ffn_gate_exps`, `ffn_up_exps`,
  and `ffn_down_exps` when native-pack mode is enabled; then build native packs
  from the CPU mmap, or
- build packs first and teach routed MoE kernels to use pack pointers instead
  of `cuda_model_range_ptr()` for those tensors.

This is a real kernel/data-layout project, not an env-toggle experiment. To
reach 20 t/s from ~17.6 t/s by improving routed MoE alone, the routed path
needs to save about **6.8 ms/token**, roughly **43%** of the current routed MoE
budget. With 100% routed coverage that means the native layout must make the
routed path about **1.8x faster**. This is ambitious, but it attacks the actual
bottleneck without changing model behavior.

#### Upstream/fork reconnaissance for this direction

Before writing the native pack, upstream and the public fork graph were checked
to avoid duplicating existing work. GitHub reports 881 forks of `antirez/ds4`;
875 public forks were accessible through the API, with 1187 branch heads and
316 unique head commits. No branch found implements the exact path proposed
here: a CUDA-native, all-routed-expert pack that replaces the original routed
GGUF ranges while preserving all six experts and all quantized values.

Closest branches found:

| Area | Branches | What it tells us |
|---|---|---|
| Upstream | `antirez/ds4` `main`, `rocm`, `responses-api` | No native routed pack. CUDA routed MoE still reads model-range-backed GGUF spans. |
| Expert sharding / streaming | `mirkodandrea/ds4` `moe-expert-sharding`, `speculative-shared-expert`; `Haimrich/ds4` `stream_experts` | Useful as tensor-enumeration and residency thinking, but moves experts to CPU/TCP or streaming paths. This solves memory, not GB10 decode speed. |
| Partial CUDA cache / CPU-GPU hybrid | `ddxxlao/ds4` `codex/cuda-partial-weight-cache`, `codex/cpu-gpu-hybrid-inference` | Good small-VRAM strategy. It intentionally avoids full residency, so it is the wrong trade-off for a stable 110 GiB GB10 server. |
| Q4 cache / batching | `ngc-shj/ds4` `perf/q4-only`, `perf/batched-decode-poc` | Strong confirmation that decode is weight-bandwidth-bound and layout/cache can move tok/s materially. Not exact for us because it changes dense-weight numeric format and does not solve routed IQ2/Q2 packing. |
| MoE microkernels | `berschmitt/ds4` `codex/moe-decode-h16-lut`, `codex/moe-decode-gate-pair2`; `amarrmb/ds4` `cuda-moe-down-tile8-rowspan` | Useful kernel shapes and A/B toggles, but they operate inside the existing GGUF-oriented layout. |
| Q2/MoE primitive plans | `cghart/ds4` `cuda-gb10-q2-foundation` | Useful correctness-harness and primitive-test ideas, not a production replacement cache. |
| Other adjacent work | `adis-b/ds4-64gb` sparse residency/sub-2bit, tensor-parallel branches, KV-fp16 branches | Not the no-quality-loss 20 t/s path for this machine. |

Strategic result: this route is open territory. The useful things to borrow
are the split/enumeration discipline from the sharding forks, the
layout-is-the-lever lesson from `ngc-shj`, and the decode MoE microkernel
experiments from `berschmitt`/`amarrmb`. The design here should stay stricter:
all experts, exact quantized values, replacement residency rather than duplicate
residency, and a numerical equivalence gate before enabling any server path.

Immediate implementation plan:

1. Add a CUDA routed-pack metadata/budget pass behind an env flag. This must
   report gate/up/down bytes, raw replacement floor, duplicate-cache cost, and
   compatibility of the layer tensor shapes.
2. Add a pack descriptor that can eventually hold per-layer device pointers,
   row/expert strides, and tensor types without changing execution yet.
3. Pack one tensor class for one layer into a scratch native layout and compare
   byte/row reads against the existing GGUF layout.
4. Add a microbenchmark for selected top-6 expert access in current layout vs
   packed layout.
5. Add a compute-equivalence microbenchmark before routing inference through
   any pack. Raw read bandwidth alone is not predictive enough.
6. Only after equivalence and compute speed data, route `ds4_gpu_routed_moe_*`
   through the pack with fallback to the current path.

Phase 0 has started with:

```sh
DS4_CUDA_ROUTED_PACK_PLAN=1 \
DS4_CUDA_ROUTED_PACK_BUDGET_GB=110 \
./ds4-server -m ds4flash.gguf --ctx 256000 --host 127.0.0.1 --port 8121 --tokens 16
```

This is diagnostic-only. It does not allocate a native pack and does not change
inference. The expected report is a sanity gate for the next patch: all 43
layers must have uniform routed tensor shapes/types, the raw routed floor should
remain about 72.56 GiB, and the duplicate-cache cost should exceed the 110 GiB
server budget. If that report changes, stop before writing pack kernels.

Observed on the GB10 with `DS4_CUDA_DIRECT_MODEL=1` and `--ctx 256000`:

```text
layers=43 compatible=43 incompatible=0
gate=iq2_xxs 22.17 GiB, up=iq2_xxs 22.17 GiB, down=q2_k 28.22 GiB, total=72.56 GiB
gate_row=1056, gate_expert=2162688, down_row=672, down_expert=2752512, experts=256
tensor_total=80.76 GiB, non_routed=8.20 GiB
replacement=80.76 GiB, duplicate=153.32 GiB, budget=110.00 GiB
replacement_headroom=29.24 GiB, duplicate_headroom=-43.32 GiB
```

The first pack smoke is also available:

```sh
DS4_CUDA_DIRECT_MODEL=1 \
DS4_CUDA_ROUTED_PACK_SMOKE=1 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 -n 1 -p "test"
```

Controls:

| Env | Default | Meaning |
|---|---:|---|
| `DS4_CUDA_ROUTED_PACK_SMOKE_LAYER` | `0` | Layer to pack. |
| `DS4_CUDA_ROUTED_PACK_SMOKE_TENSOR` | `gate` | One of `gate`, `up`, `down`. |
| `DS4_CUDA_ROUTED_PACK_SMOKE_LAYOUT` | `row-major` | `row-major` packs as row -> expert -> row bytes; `expert-major` preserves the GGUF order. |
| `DS4_CUDA_ROUTED_PACK_SMOKE_LIMIT_MB` | `768` | Hard cap for this one temporary device allocation. |
| `DS4_CUDA_ROUTED_PACK_SMOKE_CHECK_MB` | `16` | Chunk size for readback verification. |

Observed at ctx=256k with direct-model weights:

| Tensor | Temporary device allocation | Layout | Result |
|---|---:|---|---|
| `blk.0.ffn_gate_exps` | 528 MiB | row-major | Byte-for-byte verified; hash `e0a0aee60b83c1b8`. |
| `blk.0.ffn_down_exps` | 672 MiB | row-major | Byte-for-byte verified; hash `799b7a84a2290241`. |

This is the first concrete native-layout step. It deliberately allocates only
one tensor and frees it immediately. The row-major layout is the one worth
benchmarking next because the six selected experts for the same output row live
close together (`expert * row_bytes`) instead of megabytes apart
(`expert * expert_bytes`).

The same smoke can run a read-only layout microbench:

```sh
DS4_CUDA_DIRECT_MODEL=1 \
DS4_CUDA_ROUTED_PACK_SMOKE=1 \
DS4_CUDA_ROUTED_PACK_SMOKE_BENCH=1 \
DS4_CUDA_ROUTED_PACK_SMOKE_BENCH_ITERS=200 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 -n 1 -p "test"
```

It allocates two temporary copies of the selected tensor: current
expert-major and row-major. The combined allocation is capped by
`DS4_CUDA_ROUTED_PACK_SMOKE_BENCH_LIMIT_MB` (default 1536 MiB).

Observed:

| Tensor | Expert-major read | Row-major read | Interpretation |
|---|---:|---:|---|
| `blk.0.ffn_gate_exps` | 756.8 GiB/s | 737.9 GiB/s | Row-major is not automatically faster for simple fused top-6 reads. |
| `blk.0.ffn_down_exps` | 750.5 GiB/s | 626.4 GiB/s | Down row-major is materially worse in this naive read kernel. |

This is an important negative result. Plain row-major packing alone is not the
win. The next pack candidate should target **gate/up pairing** or a kernel shape
that actually removes work: current gate and up are separate 22 GiB tensors
read with the same activation row. A useful native pack may need to co-locate
`gate(row, expert)` and `up(row, expert)`, not merely reorder experts within one
tensor.

The next smoke tests that gate/up idea directly:

```sh
DS4_CUDA_DIRECT_MODEL=1 \
DS4_CUDA_ROUTED_PACK_PAIR_SMOKE=1 \
DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_BENCH=1 \
DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_BENCH_ITERS=400 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 -n 1 -p "test"
```

Controls:

| Env | Default | Meaning |
|---|---:|---|
| `DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_LAYER` | `0` | Layer to pack. |
| `DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_LAYOUT` | `expert-major-pair` | `expert-major-pair`, `row-major-pair`, or `expert-major-block-pair`. |
| `DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_LIMIT_MB` | `1152` | Hard cap for the temporary paired pack. |
| `DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_BENCH_LIMIT_MB` | `2304` | Hard cap for paired pack + separate gate/up comparison tensors. |
| `DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_COMPUTE` | unset | Also run an exact synthetic gate/up/mid compute comparison. Requires `expert-major-block-pair`. |
| `DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_COMPUTE_LIMIT_MB` | `2304` | Hard cap for paired pack + separate gate/up comparison tensors. |
| `DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_COMPUTE_ITERS` | `200` | Iterations for the compute comparison. |

Observed at ctx=256k:

| Gate/up pair layout | Pack size | Bench result |
|---|---:|---|
| `row-major-pair` (`row -> expert -> gate,up`) | 1056 MiB | Worse than separate tensors: 282.8 GiB/s vs 409.2 GiB/s in the read kernel. This fights the row-parallel access pattern. |
| `expert-major-pair` (`expert -> row -> gate row,up row`) | 1056 MiB | Neutral: 435.3 GiB/s vs 436.2 GiB/s. Co-locating whole rows alone does not move the needle. |
| `expert-major-block-pair` (`expert -> row -> block -> gate block,up block`) | 1056 MiB | Positive in the block-ordered microbench: 140.8 GiB/s vs 130.3 GiB/s with 400 iterations, about **+8.1%**. |

The block-pair result was the first positive layout signal in a raw read
microbench, but the real compute smoke refuted it as an inference kernel
candidate:

```sh
DS4_CUDA_DIRECT_MODEL=1 \
DS4_CUDA_ROUTED_PACK_PAIR_SMOKE=1 \
DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_LAYOUT=expert-major-block-pair \
DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_COMPUTE=1 \
DS4_CUDA_ROUTED_PACK_PAIR_SMOKE_COMPUTE_ITERS=200 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 -n 1 -p "test"
```

Observed:

| Compute path | Output diff vs current separate LUT path | Time, 200 iters | Effective read BW | Speedup |
|---|---:|---:|---:|---:|
| Current separate `decode_lut_qwarp32` | reference | 30.235 ms | 159.9 GiB/s | 1.000x |
| Paired block-pack LUT kernel | gate/up/mid all 0 | 64.894 ms | 74.5 GiB/s | 0.466x |
| Separate-layout multi-expert row40 LUT kernel | gate/up/mid all 0 | 44.400 ms | 108.9 GiB/s | 0.681x |

Takeaway: the byte layout is exact, but block-level gate/up interleaving breaks
the qwarp memory pattern. In the real dot-product, the eight lanes want adjacent
IQ2 blocks for the same tensor (`66, 66, 66...` stride). The block-pair layout
makes each lane's gate stream step by 132 bytes and then read up from the
interleaved half, which loses coalescing. The multi-expert K=6 experiment also
failed: sharing one xq/LUT load across all selected experts reduces block-level
parallelism more than it saves. Both remain useful diagnostics, but neither
should be promoted to the server path.

The next native-pack suspicion was the routed down projection, because the
current down kernel sums six selected experts for the same output row. A
row-major layout could in theory place those six rows close together:

```sh
DS4_CUDA_DIRECT_MODEL=1 \
DS4_CUDA_ROUTED_PACK_SMOKE=1 \
DS4_CUDA_ROUTED_PACK_SMOKE_TENSOR=down \
DS4_CUDA_ROUTED_PACK_SMOKE_LAYOUT=row-major \
DS4_CUDA_ROUTED_PACK_SMOKE_COMPUTE=1 \
DS4_CUDA_ROUTED_PACK_SMOKE_COMPUTE_ITERS=200 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 --nothink --temp 0 -n 1 -p "test"
```

Observed:

| Down compute path | Output diff vs current | Time, 200 iters | Effective read BW | Speedup |
|---|---:|---:|---:|---:|
| Current expert-major `sum6` | reference | 6.558 ms | 469.1 GiB/s | 1.000x |
| Row-major `sum6` | 0 | 7.636 ms | 402.9 GiB/s | **0.859x** |

This closes the naive down-layout branch too. Co-locating the selected experts
by output row is not enough; the current expert-major stream keeps the qwarp
block reads more efficient.

This changes the native-pack hypothesis. A replacement pack is still the only
quality-preserving memory idea that fits the 110 GiB budget, but it cannot be a
naive row-major down pack or block-interleaved gate/up pack. Any future pack
must preserve the qwarp lane-contiguous block stream while removing actual
weight traffic or launch work.

Additional exact probes after that result:

| Probe | Switch | Result |
|---|---|---|
| Parallelize the six down experts inside one block | `DS4_CUDA_MOE_DOWN_SUM6_PARALLEL=1` | Exact, but no win. Steady per-layer down stayed about **0.047 ms** vs **0.045 ms** for the current serial-in-qwarp sum6 kernel. The existing down kernel is already small enough that extra block structure does not pay. |
| Change decode gate/up row span | `DS4_CUDA_MOE_DECODE_GATE_SPAN=64/256/512` | Negative. The current 128-row block shape is the useful balance. 256 and 512 reduce parallelism too much; 64 adds noise/overhead. |
| Align dense `q8_0` blocks by repacking 34-byte GGUF blocks to 36/40 bytes | `DS4_CUDA_Q8_ALIGN_SMOKE=1` | Exact, but slower on `blk.0.attn_output_b`: stride 36 was **0.904x**, stride 40 was **0.861x**. Half-misaligned `int32` loads are not the dominant Q8 issue; reading extra padding bytes costs more than alignment helps. |
| Cache HC-expand activations in shared memory | `DS4_CUDA_HC_EXPAND_CACHE_SMOKE=1` | Exact, but slower. `blk.0.attn_output_b` HC-expand was **0.983x** cached-x vs plain at `--ctx 256000`; the activation cache copy costs more than it saves. |
| Cache grouped attention-output-A activations in shared memory | `DS4_CUDA_ATTENTION_OUTPUT_A_CACHE_SMOKE=1` | Exact, but slower. `grouped_q8_0_a_preq_warp8_cached_x` was **0.907x** vs the plain grouped kernel. |
| Split Q8 scales and quant bytes into a SoA duplicate cache | `DS4_CUDA_Q8_SOA_CACHE=1` | Exact and positive for the short `attn_output_a/b` checks. `blk.0.attn_output_a` microbench: **1.081x** (219.35 -> 237.10 GiB/s effective). `blk.0.attn_output_b` microbench: **1.123x** (204.17 -> 229.26 GiB/s effective). End-to-end 8192/64 graph `ds4-bench`: **15.27 -> 15.82-15.88 t/s**. Server graph smoke 8192/64: **18.18 t/s avg**. Extra memory for all layers: ~**2.86 GiB**. `cmp` returned 0 on a short coding-prompt logprob JSON. The first extended 12-task coding eval at `ctx=256000` was **9/12** vs **10/12** for the current no-SoA control, but targeted `flatten_dict` repeats later passed **8/8** on both paths and the full repeated suite measured **30/36** for both no-SoA and default SoA. Default SoA is the best current exact speed path, though still short of 20 t/s. |
| Extend SoA to `attn_q_b` | `DS4_CUDA_Q8_SOA_QB=1` with `DS4_CUDA_Q8_SOA_CACHE=1` | Microbench looked good: **1.167x** (224.33 -> 261.74 GiB/s effective), but 8192/64 graph decode was **15.81 t/s**, not better than A+B. Kept experimental; default cache limit stays 4 GiB unless this flag or `DS4_CUDA_Q8_SOA_ALL=1` is set. |
| Extend SoA to the `attn_q_a/attn_kv` pair | `DS4_CUDA_Q8_SOA_QKV=1` with `DS4_CUDA_Q8_SOA_CACHE=1` | Individual microbenches were strong: `q_a` **1.250x**, `kv` **1.330x**. The real fused pair route regressed to **15.69 t/s**, so the flag remains experimental and off by default. |
| Add shared-activation reuse to SoA kernels | `DS4_CUDA_Q8_SOA_CACHE_X=1` | Exact in the tested kernels, but not stable end-to-end. Dense `attn_output_b` smoke fell from the plain-SoA **1.123x** to **1.057x**. Full 8192/64 runs were **15.96 t/s** then **15.65 t/s**. Not promoted. |
| Extend SoA to shared expert Q8 tensors | `DS4_CUDA_Q8_SOA_SHARED=1` with `DS4_CUDA_Q8_SOA_CACHE=1` | All shared tensors fit with A+B: total SoA cache **3.93 GiB** and `--ctx 256000` passed. Microbench: shared gate **1.150x**, shared down **1.331x**. End-to-end was neutral (**15.83 t/s**) and the stored logprob JSON differed at ~1e-6 while selected tokens stayed the same. Not a quality-preserving default. |
| Use 16 lanes for decode MoE gate/up LUT | `DS4_CUDA_MOE_DECODE_GATE_H16=1` | Negative. The reduction was arranged to preserve the current 8-lane summation order, but 8192/64 graph + SoA A+B measured **15.75 t/s**. A short MoE profile showed `gateup` unchanged (**57.31 -> 57.41 ms** over 43 layers with profiling enabled). |
| Specialize decode MoE gate/up for no auxiliary gate/up writes | `DS4_CUDA_MOE_DECODE_GATE_NOAUX=1` | Negative. The normal server path does not need `gate_out/up_out`, but a no-aux kernel did not lower the profiled `gateup` time and regressed 8192/64 graph + SoA A+B to **15.73 t/s**. Kept opt-in only. After restoring the default kernel, graph + SoA A+B measured **15.84 t/s** and the coding-prompt logprob JSON still compared byte-identical (`cmp=0`). |
| Pair two decode MoE gate/up slots in one block | `DS4_CUDA_MOE_DECODE_GATE_PAIR2=1` | Negative. This keeps all 6 experts and preserves each qwarp's 8-lane summation order while sharing the per-block `xq`/LUT load across two slots. It was byte-identical on the coding logprob JSON (`cmp=0`), but decode-profile A/B was slightly worse (`gateup` **10.24 -> 10.59 ms** over 43 layers on the filtered `tokens=1` profile) and 8192/64 graph + SoA A+B regressed to **15.65 t/s**. |
| Fuse decode gate/up directly into `midq` | `DS4_CUDA_MOE_DECODE_FUSED_MIDQ=1` | Negative. This keeps all 6 experts and preserves the per-row qwarp sum plus the same Q8_K block quantization. The short coding logprob JSON was byte-identical (`cmp=0`), but 8192/64 graph + SoA measured **15.57 t/s** vs **15.67 t/s** for the adjacent default control. MoE profile showed why: separate `midq` costs only **0.0052 ms/layer**, and fusing it moved enough work into `gateup` that total MoE regressed (**1.5432 -> 1.5485 ms/layer** in the 86-profile sample). |
| Group four output rows per qwarp in decode down sum6 | `DS4_CUDA_MOE_DOWN_SUM6_ROW4=1` | Negative. This is byte-identical on the short coding logprob JSON (`cmp=0`) and keeps each row's slot/block accumulation order, but the down profile did not improve: `down` **0.7206 -> 0.7243 ms/layer**, total MoE **1.5432 -> 1.5549 ms/layer**. Fewer CTAs and repeated selected/midq reuse do not compensate for the larger per-qwarp work. |

The Q8 align smoke command:

```sh
DS4_CUDA_DIRECT_MODEL=1 \
DS4_CUDA_Q8_ALIGN_SMOKE=1 \
DS4_CUDA_Q8_ALIGN_SMOKE_ITERS=400 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 -n 1 -p "test"
```

Observed stride 36:

```text
original read=13.28 GiB time=62.438 ms effective=212.71 GiB/s
stride=36 read=14.06 GiB time=69.099 ms effective=203.51 GiB/s speedup=0.9036x diff=0
```

Observed stride 40:

```text
original read=9.96 GiB time=46.729 ms effective=213.16 GiB/s
stride=40 read=11.72 GiB time=54.291 ms effective=215.85 GiB/s speedup=0.8607x diff=0
```

Net: the "obvious" full-quality micro-optimizations around padding, row spans,
and shared activation caches are ruled out.
The useful distinction is now clear: padding/alignment and shared activation
caches are not enough, but changing Q8 weight layout without increasing logical
bytes can move the needle. The SoA cache is the first exact positive result in
this memory-for-speed family. It is not enough alone to reach 20 t/s, but it
opens the next concrete path: extend SoA selectively to other hot Q8 kernels
only after one-layer microbench proof, and keep the duplicate-cache budget under
the 110 GiB target.

SoA commands and observations:

```sh
DS4_CUDA_Q8_SOA_SMOKE=1 \
DS4_CUDA_Q8_SOA_SMOKE_ITERS=800 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 --nothink --temp 0 -n 1 -p test
```

Use `DS4_CUDA_Q8_SOA_SMOKE_TENSOR=a` to run the same smoke on
`attn_output_a`; the default is `attn_output_b`. Additional selectors used in
the later pass: `q_b`, `q_a`, `kv`, `shared_gate`, `shared_up`, and
`shared_down`.

Observed for B:

```text
original in=8192 out=4096 blocks=256 read=26.56 GiB time=130.100 ms effective=204.17 GiB/s
soa      in=8192 out=4096 blocks=256 read=26.56 GiB time=115.860 ms effective=229.26 GiB/s speedup=1.1229x diff=0
```

Other SoA one-layer results:

| Tensor | Shape | Speedup | Notes |
|---|---|---:|---|
| `attn_output_a` | in=8192 out=4096 blocks=256 | **1.0809x** | Exact; promoted with B. |
| `attn_q_b` | in=1024 out=32768 blocks=32 | **1.1668x** | Exact in isolation; no end-to-end win. |
| `attn_q_a` | in=4096 out=1024 blocks=128 | **1.2497x** | Exact in isolation; fused pair route regressed. |
| `attn_kv` | in=4096 out=512 blocks=128 | **1.3304x** | Exact in isolation; fused pair route regressed. |
| `ffn_gate_shexp` | in=4096 out=2048 blocks=128 | **1.1495x** | Exact in isolation; shared path not byte-identical end-to-end. |
| `ffn_down_shexp` | in=2048 out=4096 blocks=64 | **1.3309x** | Exact in isolation; shared path neutral end-to-end. |

Promoted opt-in path:

```sh
DS4_CUDA_Q8_SOA_CACHE=1 ./ds4 --cuda -m ds4flash.gguf --ctx 256000 --nothink --temp 0 -n 1 -p test
```

This passed after fixing the preload path so `DS4_CUDA_Q8_SOA_CACHE=1` no
longer accidentally enables the existing Q8->F16 preload. The SoA cache is
prebuilt during startup for `attn_output_a/b` tensors, so CUDA Graph capture does
not see allocation or pack kernels inside decode.

The default SoA budget is **4096 MiB**. `DS4_CUDA_Q8_SOA_QB=1` or
`DS4_CUDA_Q8_SOA_ALL=1` raises the default budget to **6144 MiB** because
`attn_q_b` pushes the duplicate cache past the A+B-only envelope. Use
`DS4_CUDA_Q8_SOA_CACHE_MB=<MiB>` for explicit experiments.

Experimental flags:

| Flag | Status |
|---|---|
| `DS4_CUDA_Q8_SOA_QB=1` | Caches/routes `attn_q_b`; micro-positive, full decode neutral. |
| `DS4_CUDA_Q8_SOA_QKV=1` | Caches/routes `attn_q_a/attn_kv`; pair route regressed. |
| `DS4_CUDA_Q8_SOA_CACHE_X=1` | Uses shared activation cache in SoA kernels; unstable end-to-end. |
| `DS4_CUDA_Q8_SOA_SHARED=1` | Caches/routes shared expert Q8 tensors; memory-safe but not byte-identical on logprob JSON and neutral in throughput. |
| `DS4_CUDA_Q8_SOA_NO_ATTN_OUTPUT_A=1` | Diagnostic isolation switch: keep SoA cache enabled but do not route/cache attention-output A. |
| `DS4_CUDA_Q8_SOA_NO_ATTN_OUTPUT_B=1` | Diagnostic isolation switch: keep SoA cache enabled but do not route/cache attention-output B. |
| `DS4_CUDA_Q8_SOA_ATTN_OUTPUT_B_DECODE=1` | Diagnostic switch: force `attn_output_b` through the generic SoA decode route. Default behavior is unchanged without this flag. |

A/B isolation after adding those switches:

| Mode | 8192/64 graph `ds4-bench` | Read |
|---|---:|---|
| A+B with B forced (`DS4_CUDA_Q8_SOA_ATTN_OUTPUT_B_DECODE=1`) | **15.90 t/s** | Not rejected; may be useful, but needs repeat/server confirmation. |
| A only (`DS4_CUDA_Q8_SOA_NO_ATTN_OUTPUT_B=1`) | **15.60 t/s** | Unexpectedly below the historical SoA band; do not use one run to trim B cache. |
| B only (`DS4_CUDA_Q8_SOA_NO_ATTN_OUTPUT_A=1 DS4_CUDA_Q8_SOA_ATTN_OUTPUT_B_DECODE=1`) | **14.66 t/s** | Negative. B generic route alone cannot replace the A path. |
| Default SoA control after the isolation runs | **15.39 t/s** | This control was low versus earlier **15.82-15.88 t/s** repeats, so the isolation series is noisy. |

The later repeat/server quality gate confirmed that B-forced is not a quality
problem in this suite (**30/36**, same failures as no-SoA and default SoA), but
it was slower than default SoA in the full repeated server protocol
(**17.68 t/s** vs **17.91 t/s** server final-decode avg). Keep B-forced as a
diagnostic or future kernel-design hook; default SoA remains the better route.

Deterministic quality check:

```sh
./ds4 --cuda -m ds4flash.gguf --ctx 4096 --nothink --temp 0 -n 8 \
  --dump-logprobs /tmp/ds4-logprobs-base.json --logprobs-top-k 5 \
  -p "Write a tiny C function that adds two integers."

DS4_CUDA_Q8_SOA_CACHE=1 ./ds4 --cuda -m ds4flash.gguf --ctx 4096 \
  --nothink --temp 0 -n 8 \
  --dump-logprobs /tmp/ds4-logprobs-soa.json --logprobs-top-k 5 \
  -p "Write a tiny C function that adds two integers."

cmp -s /tmp/ds4-logprobs-base.json /tmp/ds4-logprobs-soa.json
```

`cmp` returned 0 for SoA A+B: same greedy tokens and same stored logprob JSON
for this coding prompt. This short check was later shown to be insufficient:
the extended 12-task coding eval at `ctx=256000` found an extra
`flatten_dict` failure in one SoA sample. The subsequent repeat isolated that
prompt and found no-SoA run-to-run sensitivity as well, so the short `cmp=0`
should be read as a narrow numerical smoke, while the real acceptance gate must
be repeated production-context coding eval.

Upstream and `ds4-eval` checkpoint:

```sh
git worktree add /tmp/ds4-upstream-main upstream/main
(cd /tmp/ds4-upstream-main && make cuda-spark)

(cd /tmp/ds4-upstream-main && ./ds4-bench --cuda \
  -m /home/alessandro/projects/ds4/ds4flash.gguf \
  --prompt-file /home/alessandro/projects/ds4/speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 8257 \
  --gen-tokens 64 --warm-weights)
```

`upstream/main` was `c9dd949` (`cuda: fix compressed prefill RoPE positions`).
The upstream bench produced **13.58 t/s**. A repeat with
`DS4_CUDA_GRAPH_DECODE=1` in the environment produced **13.39 t/s** after an
intermediate transient `cudaSetDevice` OOM; no `ds4` processes remained and
`nvidia-smi` showed the GPU free afterwards. This confirms the branch's
pre-SoA CUDA decode work is already a large part of the gain: **13.4-13.6 t/s**
upstream vs **15.27 t/s** current-branch no-SoA vs **15.82-15.88 t/s** with
SoA A+B. A later control run after the MoE H16/no-aux probes were made opt-in
only produced **15.84 t/s** on the same 8192/64 graph + SoA A+B bench:

```text
8192,8192,219.89,64,15.84,136750476
```

Quick `ds4-eval` smoke on the SoA A+B speed path:

```sh
DS4_CUDA_Q8_SOA_CACHE=1 ./ds4-eval --cuda -m ds4flash.gguf \
  --questions 2 --tokens 512 --hard-limit-reply-budget 128 \
  --soft-limit-reply-budget 256 --nothink --plain --warm-weights \
  --trace /tmp/ds4-eval-soa-ab-2q.txt
```

Result: **2/2 passed** (`GPQA Diamond/recNu3MXkvWUzHZr9` answer B and
`SuperGPQA/001b51d76b4d422988f2c11f104a2c6c` answer C). This is a smoke, not a
replacement for the coding eval: the full default `ds4-eval` suite has 92 hard
questions and a default 16k token budget per question.

### 2. Reduced-K modes remain diagnostic/opt-in only

The fastest coding candidate is still K=6 on layers 0-2 and K=3 + renorm on
all other layers:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_MOE_ACTIVE_EXPERTS=3 \
DS4_MOE_ACTIVE_EXPERTS_RENORM=1 \
DS4_MOE_ACTIVE_EXPERTS_LAYERS=0-2:6 \
./ds4-server -m ds4flash.gguf --ctx 8192 --host 127.0.0.1 --port 8106 --tokens 900
```

Uniform K=3 + renorm remains the simpler baseline:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_MOE_ACTIVE_EXPERTS=3 \
DS4_MOE_ACTIVE_EXPERTS_RENORM=1 \
./ds4-server -m ds4flash.gguf --ctx 8192 --host 127.0.0.1 --port 8106 --tokens 900
```

The faster but degraded turbo mode is K=2 + renorm:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_MOE_ACTIVE_EXPERTS=2 \
DS4_MOE_ACTIVE_EXPERTS_RENORM=1 \
./ds4-server -m ds4flash.gguf --ctx 8192 --host 127.0.0.1 --port 8106 --tokens 64
```

The 12-task ctx=256k eval above means these should not be the default for
coding-agent use. A safe use would require a deterministic verifier and clean
fallback to full K=6; otherwise faster bad code can poison the context and cost
more retries.

### 3. Full-quality normal decode: remove ~5-6 ms/token

The best measured path is now no-MTP decode at ~17.8-18.1 t/s. Hitting 20 t/s
without changing model behavior still requires a token time around 50 ms, down
from the current ~55-56 ms. Decode stage profiling after the third pass shows
the remaining per-token cost
concentrated in:

| Stage | Total over 43 layers |
|---|---:|
| Routed MoE | ~15.7-16.0 ms |
| Attention output projection | ~14.4-14.6 ms |
| Q path | ~9.3-9.5 ms |
| Shared expert gate/up/down | ~6.3 ms |

The next realistic target is another 5-6 ms from these four areas. The most
promising work items are one-token MoE gate/up and attention-output Q8 kernels;
cuBLAS/F16 decode (`DS4_CUDA_Q8_CUBLAS_DECODE=1`) was tested and was worse
(~15.26 t/s), targeted `attn_q_b` cuBLAS was much worse, and HC expand F16
also regressed. The win is not simply "use more F16 cache".

### 4. True batched target verifier for MTP (large change, currently not enough)

DS4 V4 ships with an MTP (multi-token prediction) head. CUDA MTP is now
numerically usable, and the draft head is cheap enough (~22 ms/draft), but the
current speculative verifier is the wrong shape for GB10: it verifies drafted
positions like normal target decodes and rereads the same target weights per
position.

The needed shape is:

1. Draft model generates K candidate tokens cheaply (the MTP head is much
   smaller than the full model).
2. Full model verifies all K in a *single* forward pass (the only
   per-token cost is the attention update; MoE etc. are batched).
3. Accept the prefix where draft and full agree; discard the rest.

For this repo that likely means adding a target-model verify path where
`n_tokens > 1` is first-class across the decode layer, with special attention
to the routed MoE kernels. The win only appears if the batched kernels share
or coalesce target weight reads. If they still reread all expert weights per
position, the verifier remains bandwidth-bound and cannot hit 20.

Caveats: requires careful integration with the KV cache (need to roll back
on rejection), tool-call exact-replay (DSML emission semantics), and the
existing single-graph-worker server serialization. This is several days of
work. After the Q8 microbatch fixes, however, MTP draft=2 still reaches only
~13.8 t/s, so this is no longer the closest route to 20 on GB10 unless the MTP
draft head itself also becomes much cheaper.

### 5. Attention path matmul optimization (medium payoff)

`matmul_q8_0_preq_warp8*` is used for Q-proj, KV-proj, attention-output, and
small downstream matmuls. The useful full-quality result here is now the SoA
duplicate cache for `attn_output_a/b`; it buys only a few percent total because
the remaining bottleneck is still routed MoE. The follow-up variants were
measured:

- `attn_q_b` SoA: strong one-layer result, no full-decode win.
- `attn_q_a/attn_kv` SoA: strong one-layer result, fused pair regressed.
- SoA shared activation cache: noisy/unstable end-to-end.
- shared expert SoA: memory-safe, but neutral and not byte-identical in stored
  logprob JSON.

The next Q8 work should therefore not be another broad "apply SoA everywhere"
pass. It needs a targeted kernel redesign with an end-to-end gate, or it should
move back to the routed MoE layout where the remaining wall time is larger.

### 6. Reduce HC expand cost (`matmul_q8_0_hc_expand_preq_warp8`, medium risk)

The HC expand kernel runs N_HC=4 times per layer for HC partition mixing.
If the four expansions can be folded into a single grouped GEMM (or run on
tensor cores), there's potential. Worth a look but the kernel already has
a `_warp8` micro-optimization, and both the F16-cache/cuBLAS path and the
cached-activation variant regressed in the third pass. Treat this as a deeper
kernel redesign, not an env-toggle candidate.

### 7. Tune kernel launch shapes for full-GPU occupancy

Spot-check with nsys metrics: `sm__cycles_active.avg.pct_of_peak_sustained`,
`l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`. If active SM
fraction is <80 % during the dominant kernels, there is room. If it is
already >90 %, the kernel is achieving its memory-bound ceiling and the
only further win is structural (specdec, fewer reads per token).

### 8. Accept the full-quality GB10 ceiling

The MoE weight-read ceiling math above suggests ~18 t/s is roughly the
hardware-imposed limit for this model on this box without speculative
decoding or a larger kernel redesign. If specdec remains too invasive for the
project's appetite, the honest framing is: the current architecture is stable
around 18 t/s on GB10 for DS4 V4 Flash, and 20 t/s needs a real reduction in
per-token weight traffic or a major improvement in the dominant custom kernels.

## 2026-05-20 web literature pass

The external literature broadly agrees with the measurements in this file:
large wins come from reducing memory traffic per committed token, batching
target verification, or changing the model/training recipe. There is no public
paper that suggests a simple lossless env-toggle path from ~18 t/s to 20 t/s
for single-user, low-context, all-weights-resident GB10 decode.

Primary DeepSeek sources:

- DeepSeek-V2/V3 technical reports: V2 introduced the efficient-inference
  combination of MLA and DeepSeekMoE, with 21B active parameters and a 93.3%
  KV-cache reduction versus DeepSeek 67B; V3 keeps MLA/DeepSeekMoE and adds
  MTP as an inference-acceleration foundation.
  Sources: https://arxiv.org/abs/2405.04434 and
  https://arxiv.org/abs/2412.19437
- DeepSeekMoE explains why the routed experts are not disposable: the model
  relies on fine-grained routed experts plus shared experts for specialization.
  This supports the current quality rule: K<6 is a draft/diagnostic mode unless
  a full-K verifier commits the final token.
  Source: https://arxiv.org/abs/2401.06066
- DeepSeek-V4 is especially relevant to `ds4flash.gguf`: V4-Flash is reported
  as a 284B total / 13B active model, with 1M context, CSA/HCA attention,
  routed expert FP4 QAT, and infrastructure notes about a single fused MoE
  kernel that overlaps computation, communication, and memory access.
  Source: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf
- DeepSeek-V4 also says routed expert weights use FP4 QAT at deployment. That
  is not a drop-in replacement for our GGUF IQ2/Q2/Q8 path: without the model
  being trained/adapted to that format, further quantization is a quality risk.

Kernel/system sources:

- FlashMLA focuses on high-performance MLA/DSA kernels and reports very high
  H800 bandwidth/TFLOP numbers, but it targets SM90/SM100 style kernels and
  DeepSeek's dense/sparse MLA path, not our current custom GGUF Q kernels.
  Source: https://github.com/deepseek-ai/FlashMLA
- DeepGEMM is the strongest public signal about where DeepSeek spends kernel
  effort now: FP8/FP4 GEMMs, fused MoE/Mega-MoE, indexer scoring, and small
  hand-tuned CUDA/CuTe/CUTLASS-style kernels. It is conceptually useful, but
  its primitives do not directly match our IQ2/Q2/Q8 matrix-vector decode.
  Source: https://github.com/deepseek-ai/DeepGEMM
- PyTorch's persistent grouped-GEMM MoE work, MegaBlocks, Tutel, and MoE-Gen
  all reinforce the same idea: group many independent expert GEMMs, keep CTAs
  persistent, and batch module work. Those wins are real, but they mainly need
  many tokens, many experts, or multi-GPU execution. Single-token decode on one
  GB10 cannot harvest the full paper speedups unless we introduce a batched
  verifier or concurrent server batching.
  Sources: https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/,
  https://arxiv.org/abs/2211.15841, https://arxiv.org/abs/2206.03382,
  https://arxiv.org/abs/2503.09716
- KTransformers and PowerInfer are useful counterexamples. They exploit
  CPU/GPU hybrid placement, hot/cold neuron locality, and expert deferral for
  machines that cannot keep everything hot on GPU. Our GB10 target already has
  the model resident in ~110 GiB; moving routed work to CPU is likely the wrong
  direction for low-latency decode, unless used only as a separate small-VRAM
  mode.
  Sources: https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
  and https://arxiv.org/abs/2312.12456

Speculative / multi-token sources:

- Lookahead decoding claims exact acceleration without an auxiliary model by
  trading more parallel work per step for fewer sequential steps.
  Source: https://arxiv.org/abs/2402.02057
- Medusa and EAGLE show the more general verified-decoding path: propose
  multiple future tokens cheaply, verify them in parallel with the target
  model, and preserve the target distribution/quality when the verifier is
  exact.
  Sources: https://arxiv.org/abs/2401.10774 and
  https://arxiv.org/abs/2401.15077
- This is the one literature-backed route that can beat the per-token memory
  wall without changing final quality. Our current MTP result is bad because
  the target verifier is effectively shaped like repeated normal decode. A
  proper verifier must run K candidate positions in one target pass and share
  weight traffic across the batch.

Quality-sensitive interpretation for this project:

1. Do not spend more time on naive native layouts. We tested row-major down,
   block-paired gate/up, qwarp pair2, no-aux gate, fused midq, row4 down, Q8
   padding, and shared activation caches. The papers point to fused/grouped
   kernels, but our negative results show that preserving qwarp block access is
   more important than visually co-locating experts in memory.
2. The strongest exact local improvement remains Q8 SoA for attention-output
   A/B: it costs only a few GiB, passes repeated coding eval at ctx=256k, and
   improves server decode, but it is a few-percent win, not the missing 10%.
3. Reduced-K should be reframed as a draft model, not a final model. K=3/K=2
   gives the kind of speed the target needs, but coding quality changes. If a
   batched full-K verifier accepts most reduced-K tokens, we can keep full-K
   output quality and recover speed. If acceptance is low, reject the route.
4. MTP should be tested the same way: not "does draft=2 work with the current
   verifier", but "can one batched target pass verify 2-4 positions faster than
   repeated full decode while preserving logits/token acceptance".
5. Long-context sparse attention papers (NSA/DSA/V3.2/V4 CSA/HCA) are not a
   bolt-on exact optimization for this checkpoint. They explain why future
   DeepSeek models handle 1M contexts cheaply, but retrofitting them to this
   GGUF would change the model. For our clarified goal, low resident context
   first-token speed remains the priority.

Recommended next experiment:

1. Build a target verifier microbench outside the server: given a prefix and
   2-4 proposed tokens, run a single batched target forward and compare its
   accepted prefix and logits against repeated full K=6 one-token decode.
2. Run it with two proposers: the built-in MTP head and reduced-K K=3+renorm.
3. Measure both speed and acceptance on the 12-task coding eval prompts at
   `--ctx 256000`, low resident context. The useful metric is committed
   full-quality tokens/s, not draft tokens/s.
4. Only if the verifier microbench is positive, integrate it into the server
   with KV rollback and graph-safe fixed shapes. Otherwise, accept that exact
   single-token decode on this GB10 is currently an ~18 t/s class path.

### First follow-up: MTP strict vs fast batch verifier

The next measurement went straight at the quality/speed fork in the existing
MTP implementation, using a real coding prompt at `--ctx 256000`:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
DS4_MTP_STRICT=1 \
DS4_MTP_TIMING=1 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 \
  --nothink --temp 0 -n 64 \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  -p 'Scrivi una funzione Python is_prime(n) robusta e concisa.'
```

Control without MTP on the same prompt:

```text
generation: 18.49 t/s
```

Strict MTP output compared byte-identical to that control (`cmp=0`). That is
the quality-preserving path. Its hot verifier timings, however, stayed around
**109-128 ms** for the two verified positions, with hot MTP drafts around
**4 ms**. The first MTP draft still has a large one-time warmup cost
(~7.8-10.0 s in these CLI runs), but even ignoring that startup cost, strict
MTP does not beat normal decode because the verifier is effectively two exact
decode positions interleaved rather than one weight-sharing batch.

The fast batch verifier was then forced:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
DS4_MTP_BATCH_VERIFY=1 \
DS4_MTP_TIMING=1 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 --quality \
  --nothink --temp 0 -n 64 \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2 \
  -p 'Scrivi una funzione Python is_prime(n) robusta e concisa.'
```

This is the attractive speed shape: hot two-position verify was about
**80-88 ms**, and the end-to-end run reported **17.50 t/s** even with MTP
overheads. But it diverged from the full-K baseline early:

```diff
-    """Verifica se un numero intero positivo è primo."""
+    """Restituisce True se n è un numero primo, False altrimenti."""
```

The same divergence happened without `--quality`, so the mismatch is not just
from optional fast fusions inside the batch layer.

A narrower attempt batched only the two output heads after the strict decode2
layer pass:

```sh
DS4_MTP_STRICT=1 DS4_MTP_DECODE2_BATCH_HEAD=1 ...
```

After fixing the helper to use the supplied HC rows all the way through the HC
weighted-sum step, this path compared byte-identical to the no-MTP baseline
(`cmp=0`). It did **not** improve speed: the strict verifier stayed in the same
~**109-120 ms** hot band, with occasional slower rows. That localizes the real
problem away from the final output head. The missing speed is inside the target
layer pass itself: attention/MoE/Q8 work for two proposed positions is still
being paid almost like two exact decodes.

Code safety change from this pass: strict MTP now ignores the old
`DS4_MTP_BATCH_VERIFY=1` override and prints a warning. The non-exact verifier
can still be forced for diagnostics only with:

```sh
DS4_MTP_UNSAFE_BATCH_VERIFY=1
```

Current conclusion: the correct route is still "make the batched target
verifier exact", not "use the existing fast batch verifier". The fast verifier
has the right speed envelope for 20 t/s, but it is not acceptable for coding
quality until its layer/output numerics match strict decode.

### Second follow-up: exact-islands inside the batch verifier

The next pass kept the same goal, but treated the fast batch verifier as a
layer-major scaffold and added small opt-in "exact islands" where the batch
path was numerically different from strict decode. All flags below are
diagnostic/experimental; strict mode still ignores `DS4_MTP_BATCH_VERIFY=1`
unless `DS4_MTP_UNSAFE_BATCH_VERIFY=1` is set.

New diagnostics:

```sh
DS4_MTP_VERIFY_SHADOW=1
DS4_MTP_LAYER_SHADOW=1
DS4_MTP_LAYER_SHADOW_EPS=0.000001
```

`DS4_MTP_VERIFY_SHADOW=1` runs the batch verifier against the same frontier,
restores the frontier, then runs the strict exact verifier and reports top
agreement, max/rms logit deltas, and top1-top2 margins. `DS4_MTP_LAYER_SHADOW`
compares the two-row HC state after each target layer.

Localization results on the `is_prime` coding prompt at `--ctx 256000`:

- Plain fast batch drifted immediately: first layer-shadow diff at layer 0,
  row 0 (`max=0.0012635`), with worst final-layer row diff above 15.
- Exact HC pre/norm, exact attention output tail, exact shared tail, and exact
  routed MoE moved the first layer-shadow diff to layer 2 row 1, the first
  ratio-4 compressed/indexer layer.
- Forcing token-per-token attention with `DS4_MTP_BATCH_ATTENTION_EXACT=1`
  moved the first diff to layer 4 row 1.
- Dump comparison then showed the layer-4 cause: batch compressor projections
  used two separate F16 matmuls while strict decode used the paired F16
  compressor kernel. The raw compressor projections differed by up to
  `5.87e-05` / `1.27e-04`, enough to amplify through attention and FFN.
- Adding `DS4_MTP_BATCH_COMPRESS_PROJ_EXACT=1` made the layer-shadow pass show
  no HC diff above `1e-6` on the same prompt.

The current conservative batch-verifier island set is:

```sh
DS4_MTP_BATCH_HC_PRE_EXACT=1
DS4_MTP_BATCH_ATTENTION_EXACT=1
DS4_MTP_BATCH_ATTN_TAIL_EXACT=1
DS4_MTP_BATCH_COMPRESS_PROJ_EXACT=1
DS4_MTP_BATCH_ROUTED_EXACT=1
DS4_MTP_BATCH_SHARED_TAIL_EXACT=1
```

With that full set, shadow logits are not byte-identical, but the observed
errors are small compared to the greedy margins in the smoke:

```text
row max deltas: up to about 0.021 in the sampled cycles
top margins:    about 3.57 to 13.85 in the same cycles
argmax:         matched exact for row0 and row1 in the sampled cycles
```

The hot unsafe-batch verifier timing with the conservative island set is around
**97-100 ms** for two verified target positions, plus ~4 ms MTP draft time.
That is near a **20 tok/s** verifier envelope when both drafts are accepted and
startup graph capture is amortized. In a CLI run the first graph capture still
dominates the reported end-to-end number, so the useful measurement is the hot
`mtp timing micro` line, not the first cold cycle.

Coding-oriented server smoke at production context:

```sh
DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 \
DS4_MTP_STRICT=1 DS4_MTP_BATCH_VERIFY=1 DS4_MTP_UNSAFE_BATCH_VERIFY=1 \
DS4_MTP_BATCH_HC_PRE_EXACT=1 \
DS4_MTP_BATCH_ATTENTION_EXACT=1 \
DS4_MTP_BATCH_ATTN_TAIL_EXACT=1 \
DS4_MTP_BATCH_COMPRESS_PROJ_EXACT=1 \
DS4_MTP_BATCH_ROUTED_EXACT=1 \
DS4_MTP_BATCH_SHARED_TAIL_EXACT=1 \
./ds4-server --cuda -m ds4flash.gguf --ctx 256000 --port 8106 \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2

python3 tuning/coding_eval_extended.py \
  --base-url http://127.0.0.1:8106 \
  --label mtp_exact_islands_256k_r1 \
  --max-tokens 700 \
  --out-dir /tmp/ds4-mtp-exact-islands-eval
```

Result: **10/12**, matching the no-MTP baseline run at the same `ctx=256000`
and token cap. The two failed tasks (`lru_cache`, `parse_csv_line`) produced
byte-identical outputs in candidate and baseline. Across all 12 tasks, 9
responses were byte-identical; the 3 different responses still passed. This is
a good quality smoke, not a proof: the mode remains opt-in and unsafe until a
guarded exact fallback exists.

Speed reality check: the same server coding harness did **not** show a net
speed win over no-MTP baseline. Candidate responses were mostly ~14-16 tok/s,
while no-MTP baseline was ~15-17 tok/s. The reason is workload-dependent MTP
acceptance plus the conservative exact-islands. Removing key islands is not
acceptable: without `DS4_MTP_BATCH_ATTENTION_EXACT`, sampled logit deltas grew
to **~3.36**; without routed/shared exact islands, deltas reached **~4.06**.
Those variants can still match argmax on easy prompts, but the error scale is
too close to realistic coding margins.

Current next step: build an adaptive verifier that uses the fast batch result
only when its top margins are comfortably above the empirically bounded
batch-vs-exact error, and falls back to exact verification otherwise. Until
that guard exists, the conservative island set is the best quality-preserving
research candidate, while no-MTP graph+SoA remains the production-safe default.

### Third follow-up: margin-guarded adaptive verifier

An opt-in guard now exists for the unsafe batch verifier:

```sh
DS4_MTP_BATCH_MARGIN_GUARD=4
```

The guard is active only in strict MTP mode with forced unsafe batch verify.
For `draft=2` it reads the two batch logits rows, computes top1-top2 margins,
and accepts both draft rows only if both margins are above the configured
threshold and row0 still verifies draft1. If the guard rejects the batch result,
the session restores the saved frontier and replays exactly one target token.
That fallback preserves the exact target stream at the cost of losing the
second-token speculative win for that cycle.

On the conservative exact-island set above, `DS4_MTP_BATCH_MARGIN_GUARD=4`
behaved as intended on the `is_prime` prompt:

```text
row0=5.59463 row1=3.57612 threshold=4 decision=exact-prefix1
row0=17.7085 row1=10.3931 threshold=4 decision=batch
row0=17.5382 row1=11.2001 threshold=4 decision=batch
row0=18.4857 row1=7.48682 threshold=4 decision=batch
```

Hot batch cycles that passed the guard remained around **96-99 ms** verify
time plus ~4 ms draft time. Rejected cycles fell back to an exact one-token
replay and took ~166 ms in the sampled hot path.

Important negative result: margin by itself is not enough to rescue the plain
fast batch verifier. A shadow run without the exact-islands showed rows such as:

```text
row0 max=1.96396 rms=0.312271 margin batch=9.84183 exact=9.10697
row1 max=3.65453 rms=0.700331 margin batch=5.92597 exact=6.94428
```

So a simple high-margin rule can still see large logit drift on the fast-pure
path. The guard is useful only after the exact-islands have bounded the
batch-vs-exact error into the ~0.02 range observed above. The next improvement
should therefore be either:

1. reduce the cost of one of the conservative exact-islands without increasing
   the observed error bound, or
2. add a stronger device-side confidence/error proxy than top margin alone.

### Fourth follow-up: shrinking the exact-island set

The conservative island set was profiled by layer stage at `--ctx 256000`,
looking only at hot two-token verifier passes (`tokens=2`, `pos>=20`). The
largest measured buckets were:

```text
ffn routed_moe          total=102.421 ms count=129 avg=0.794 ms
attn output_proj        total=85.467 ms  count=129 avg=0.663 ms
attn q_path             total=34.791 ms  count=129 avg=0.270 ms
attn indexer_setup      total=34.108 ms  count=63  avg=0.541 ms
ffn shared_exact_tail   total=33.702 ms  count=129 avg=0.261 ms
attn attention          total=17.457 ms  count=129 avg=0.135 ms
attn compressor         total=14.507 ms  count=123 avg=0.118 ms
```

This suggested that the explicit exact tails might be defensive scaffolding
rather than necessary islands. Two removals were tested on the same `is_prime`
coding prompt:

- Without `DS4_MTP_BATCH_ATTN_TAIL_EXACT`, but keeping HC pre, exact attention,
  exact compressor projection, routed exact, and shared exact: layer shadow
  still found no HC diff above `1e-6`, sampled shadow-logit deltas stayed at
  the same ~`0.021` scale, and hot unsafe full-accept verifier timing improved
  to about **91.85 ms** average.
- Without both `DS4_MTP_BATCH_ATTN_TAIL_EXACT` and
  `DS4_MTP_BATCH_SHARED_TAIL_EXACT`, keeping only HC pre, exact attention,
  exact compressor projection, and routed exact: layer shadow still found no
  HC diff above `1e-6`, sampled shadow-logit deltas stayed at the same scale,
  and hot unsafe full-accept verifier timing improved to about **89.19 ms**
  average verify time, **93.99 ms** average including draft.

The important negative control was removing routed exact as well. With only HC
pre, exact attention, and exact compressor projection, the sampled logit drift
became large again:

```text
start=26 row0 max=1.76828 rms=0.301379
start=26 row1 max=4.05968 rms=0.794885
start=28 row0 max=0.480094 row1 max=1.36066
start=31 row0 max=0.481647 row1 max=1.33232
```

Conclusion: the current minimum exact-island candidate is HC pre, exact
attention, exact compressor projection, and routed exact. The attention tail
and shared tail flags are now considered redundant unless a broader eval proves
otherwise. This reduced candidate still needs the full `ctx=256000` coding eval
before it can replace the conservative island set as the main research mode.

### Fifth follow-up: split exact router from routed MoE body

The next target was the remaining expensive `DS4_MTP_BATCH_ROUTED_EXACT`
island. That flag was too coarse: it forced both the router projection and the
routed MoE body to run token-per-token. A new diagnostic split was added:

```sh
DS4_MTP_BATCH_ROUTER_EXACT=1
```

This keeps the router logits token-exact while allowing the routed body to use
the two-token CUDA path. The first negative control was `DS4_CUDA_MOE_K2_DIRECT_GATE=1`
without exact router and without routed exact. It did not work: although the
sampled argmax still matched exact, layer-shadow reported worst HC diffs above
5 and shadow-logit deltas reached **2.73334**. This is not quality-safe.

The useful variant is:

```sh
DS4_CUDA_MOE_K2_DIRECT_GATE=1
DS4_MTP_BATCH_ROUTER_EXACT=1
```

with `DS4_MTP_BATCH_ROUTED_EXACT` unset. On the `top_k_frequent` server shadow
run at `ctx=256000`, this restored the good numerical envelope:

```text
layer shadow: no HC diff above 1e-6
shadow cycles: 33
arg mismatches: 0
max logit delta: 0.0175514
min verified margins: row0=1.10484 row1=0.445637
```

Coding eval results at `ctx=256000`:

| Candidate | Guard | Result | Notes |
|---|---:|---:|---|
| Reduced routed-exact islands | `4` | **10/12** | Quality matched baseline, but server speed was poor on several tasks. |
| Reduced routed-exact islands | `1` | **10/12** | Same two failures as baseline; better than guard 4. |
| Reduced routed-exact islands | none | **10/12** | Fastest routed-exact diagnostic, but more output variants. |
| Router-exact + K2 body | none | **9/12** | Extra `flatten_dict` failure (`type` not available in sandbox); reject for quality. |
| Router-exact + K2 body | `1` | **10/12** | Same two persistent failures as baseline; quality-safe but not fastest. |
| Router-exact + K2 body | `0.5` | **10/12** | Same two persistent failures; `flatten_dict` recovered vs no-guard. |
| Router-exact + K2 body | `0.25` | **10/12** | Same two persistent failures; best warmed guard sweep so far. |
| Router-exact + K2 body | `0.1` | **10/12** | Same two persistent failures, but no clear speed win over `0.25`. |
| Router-exact + K2 body + GPU top2, keeping full logits | `0.25` | **10/12** | Same two persistent failures; safe but no speed win yet. |
| Router-exact + K2 body + GPU top2, no full logits | `0.25` | **9/12**, then **10/12** | Conflicting repeats; now requires an explicit unsafe flag. |

The warmed full-suite sums for router-exact + K2 body were:

| Guard | Result | Generated tokens | Suite-sum t/s | Server decode-only t/s | Failures |
|---:|---:|---:|---:|---:|---|
| `1` | **10/12** | 2671 | **15.42** | not extracted | `lru_cache`, `parse_csv_line` |
| `0.5` | **10/12** | 2606 | **15.73** | **16.63** | `lru_cache`, `parse_csv_line` |
| `0.25` | **10/12** | 2689 | **15.90** | **16.78** | `lru_cache`, `parse_csv_line` |
| `0.1` | **10/12** | 2790 | **15.82** | **16.68** | `lru_cache`, `parse_csv_line` |
| `0.25` + safe GPU top2 | **10/12** | 2718 | **14.67** | cold run | `lru_cache`, `parse_csv_line` |
| `0.25` + GPU top2 + keep-logits diagnostic | **10/12** | 2678 | **15.18** | cold run | `lru_cache`, `parse_csv_line` |
| `0.25` + top2 compare diagnostic | **10/12** | 2658 | **15.03** | cold run | `lru_cache`, `parse_csv_line` |
| `0.25` + unsafe GPU top2/no-logits r1 | **9/12** | 2703 | **15.94** | not extracted | `lru_cache`, `flatten_dict`, `parse_csv_line` |
| `0.25` + unsafe GPU top2/no-logits r2 | **10/12** | 2728 | **14.94** | cold run | `lru_cache`, `parse_csv_line` |
| none | **9/12** | 2660 | **15.05** | not extracted | `lru_cache`, `flatten_dict`, `parse_csv_line` |

`guard=0.25` is now the best quality-first MTP research candidate. It is faster
than guard1 on the warmed full-suite measurement and keeps the failure set
identical to the no-MTP/full-K baseline. Server chunks still reach the **19-21
t/s** band on some short tasks (`sliding_window`, `parse_duration`,
`binary_search_bounds`), but the full coding aggregate remains below the no-MTP
baseline. This is progress on accept policy, not the finish line.

A post-patch repeat with the default CPU-logits margin guard at `guard=0.25`
again scored **10/12** and kept `flatten_dict` green. Its suite-sum was only
**14.82 t/s** because the first request paid a cold slow path; it is useful as a
quality confirmation, not as the speed reference.

The CUDA top2 margin guard has been split into safe and unsafe modes. A
side-by-side diagnostic (`DS4_MTP_BATCH_MARGIN_COMPARE_TOP2=1`) computed CPU
top2 from full logits and GPU top2 in the same verifier. The `flatten_dict`
canary logged 37 guard cycles with bit-identical ids, margins, and decisions;
the full suite logged **zero** CPU/GPU top2 mismatches and scored **10/12**.
Using GPU top2 for the decision while still keeping the full logits also scored
**10/12**. Therefore the CUDA top2 kernel itself is not the quality problem.

The non-promoted case is specifically the old no-logits variant: using GPU top2
and skipping the full logits readback dropped one full-suite run to **9/12**
with the same extra `flatten_dict` failure as no-guard, even though standalone
`flatten_dict` canaries passed. A repeat with the explicit unsafe flag later
scored **10/12**, so the correct conclusion is not a deterministic failure; it
is a workload/order-sensitive mode that is not reliable enough to become the
default. After this finding, `DS4_MTP_BATCH_MARGIN_GPU_TOP2=1` keeps full logits
by default; the old behavior requires
`DS4_MTP_BATCH_MARGIN_GPU_TOP2_NO_LOGITS_UNSAFE=1`.

Reproduction command for the current candidate:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
DS4_CUDA_MOE_K2_DIRECT_GATE=1 \
DS4_MTP_STRICT=1 \
DS4_MTP_BATCH_VERIFY=1 \
DS4_MTP_UNSAFE_BATCH_VERIFY=1 \
DS4_MTP_BATCH_MARGIN_GUARD=0.25 \
DS4_MTP_BATCH_HC_PRE_EXACT=1 \
DS4_MTP_BATCH_ATTENTION_EXACT=1 \
DS4_MTP_BATCH_COMPRESS_PROJ_EXACT=1 \
DS4_MTP_BATCH_ROUTER_EXACT=1 \
./ds4-server --cuda -m ds4flash.gguf --ctx 256000 --port 8119 \
  --mtp gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf --mtp-draft 2

python3 tuning/coding_eval_extended.py \
  --base-url http://127.0.0.1:8119 \
  --label mtp_routerexact_k2body_guard025_default_256k_r2 \
  --max-tokens 700 \
  --out-dir /tmp/ds4-mtp-exact-islands-eval
```

Relevant artifacts:

```text
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard1_256k_r1.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard05_256k_r1.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard025_256k_r1.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard025_default_256k_r2.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard01_256k_r1.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard025_top2compare_256k_r1.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard025_gputop2_keeplogits_256k_r1.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard025_gputop2_safe_256k_r1.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard025_gputop2_256k_r1.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_guard025_gputop2_nologits_unsafe_256k_r2.json
/tmp/ds4-mtp-exact-islands-eval/mtp_routerexact_k2body_noguard_256k_r1.json
/tmp/ds4-routerexact-k2body-guard025-server.log
/tmp/ds4-routerexact-k2body-guard025-default-server.log
/tmp/ds4-routerexact-k2body-guard025-gputop2-server.log
/tmp/ds4-routerexact-k2body-guard025-top2compare-canary-server.log
/tmp/ds4-routerexact-k2body-guard025-top2compare-full-server.log
/tmp/ds4-routerexact-k2body-guard025-gputop2-keeplogits-server.log
/tmp/ds4-routerexact-k2body-guard025-gputop2-safe-server.log
/tmp/ds4-routerexact-k2body-guard025-gputop2-nologits-unsafe-r2-server.log
/tmp/ds4-routerexact-k2body-shadow.log
/tmp/ds4-k2direct-server-shadow.log
```

Do not confuse the two router-exact variants:

- `guard0.25` is the current quality-first MTP candidate: **10/12**, failures
  `lru_cache, parse_csv_line`, 2689 generated tokens, suite-sum **15.90 t/s**
  and server decode-only **16.78 t/s** in the warmed run.
- no guard is rejected for now: **9/12**, failures
  `lru_cache, flatten_dict, parse_csv_line`, 2660 generated tokens, suite-sum
  **15.05 t/s**.
- `DS4_MTP_BATCH_MARGIN_GPU_TOP2=1` is safe by default after the split: it keeps
  full logits and scored **10/12**. It is not a speed win yet.
- `DS4_MTP_BATCH_MARGIN_GPU_TOP2_NO_LOGITS_UNSAFE=1` is not promoted: one full
  run regressed to **9/12** with the no-guard `flatten_dict` failure, while a
  repeat returned to **10/12**. Treat it as unstable until repeat statistics say
  otherwise.

Next useful guard work: either run repeat statistics on the unsafe no-logits
variant to quantify instability, or skip that micro-optimization and attack the
larger overhead: exact replay frequency and the cost of guard-rejected cycles.

### 2026-05-20 continuation — MTP aggregate stats and guarded prefix1 commit

The next hypothesis was that the router-exact+K2+guard0.25 path was spending
too much time on exact replay after partial MTP accepts. Per-token
`DS4_MTP_TIMING=1` is too noisy for long coding evals, so the runtime now has a
request-level aggregate report:

```sh
DS4_MTP_AGG_STATS=1
```

When enabled in the server, it prints one report after each request and resets
the counters. The report includes speculative calls, first-draft misses,
drafted/committed extra tokens, margin guard checks, path counts, and average
cycle/draft/snapshot/verify/replay times. This is diagnostic-only and does not
change generation.

Baseline aggregate profile for router-exact+K2+`guard0.25` on the 12-task
coding eval at `ctx=256000`:

```text
label: mtp_agg_guard025_12task_r1
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2862
suite-sum: 15.88 t/s
calls: 1130
cycles: 957
drafted extra: 1913
committed extra: 1737
extra/call: 1.537
emitted/call: 2.537
extra-token accept: 90.8%
avg cycle: 105.787 ms
avg draft: 4.119 ms
avg verify: 90.228 ms
avg replay: 10.796 ms
paths:
  micro-full: 781 cycles, 95.0 ms avg, 0 replay
  micro-exact-replay1: 176 cycles, 153.9 ms avg, 58.7 ms replay
```

This confirmed the bottleneck shape: full accepts are already close to the
best MTP envelope, but partial accepts pay an extra one-token exact replay of
about **59 ms**. That replay is frequent enough to cost about **10.8 ms per
MTP cycle** across the suite.

Pure prefix1 capture was tested next:

```sh
DS4_MTP_CAPTURE_PREFIX1=1
```

It removed almost all replay cost, but it is not acceptable as-is:

```text
label: mtp_agg_guard025_capture_prefix1_12task_r1
quality: 9/12, failures lru_cache, parse_duration, parse_csv_line
tokens: 2805
suite-sum: 16.67 t/s
avg cycle: 96.045 ms
avg replay: 0.101 ms
paths:
  micro-full: 761 cycles, 95.9 ms avg
  micro-prefix1: 175 cycles, 96.8 ms avg
```

The extra `parse_duration` failure means that the batch row-0 frontier is not
safe enough to use unconditionally as the committed one-token state, even
though the accepted token prefix is still verified.

The compromise implemented in this pass is margin-gated prefix1 capture:

```sh
DS4_MTP_CAPTURE_PREFIX1=1
DS4_MTP_CAPTURE_PREFIX1_MIN_MARGIN=2.0
```

For partial accepts, the verifier already has row-0 logits because the
`guard0.25` path keeps full logits. The new guard uses prefix1 capture only
when row-0 top1-top2 margin is at least the configured threshold; otherwise it
restores the snapshot and uses the exact replay path. Result:

```text
label: mtp_agg_guard025_capture_prefix1_m2_12task_r1
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2891
suite-sum: 16.39-16.42 t/s
calls: 1145
cycles: 963
drafted extra: 1922
committed extra: 1752
extra/call: 1.530
emitted/call: 2.530
extra-token accept: 91.16%
avg cycle: 100.142 ms
avg draft: 4.107 ms
avg verify: 91.015 ms
avg replay: 4.353 ms
paths:
  micro-full: 793 cycles, 95.7 ms avg, 0 replay
  micro-prefix1: 100 cycles, 96.6 ms avg, 0.5 ms replay bookkeeping
  micro-exact-replay1: 70 cycles, 155.2 ms avg, 59.1 ms replay
```

The repeat run kept the exact same quality shape:

```text
label: mtp_agg_guard025_capture_prefix1_m2_12task_r2
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2938
suite-sum: 15.16-15.20 t/s
calls: 1175
cycles: 980
drafted extra: 1958
committed extra: 1768
extra/call: 1.505
emitted/call: 2.505
extra-token accept: 90.3%
avg cycle: 101.693 ms
avg draft: 4.121 ms
avg verify: 91.324 ms
avg replay: 5.568 ms
paths:
  micro-full: 790 cycles, 96.1 ms avg, 0 replay
  micro-prefix1: 99 cycles, 96.4 ms avg, 0.5 ms replay bookkeeping
  micro-exact-replay1: 91 cycles, 155.6 ms avg, 59.3 ms replay
```

Per-task repeat details:

```text
merge_intervals PASS 243 tokens, 9.44 t/s
top_k_frequent PASS 102 tokens, 14.86 t/s
lru_cache FAIL 177 tokens, 17.13 t/s
parse_duration PASS 417 tokens, 15.19 t/s
group_anagrams PASS 78 tokens, 14.55 t/s
flatten_dict PASS 165 tokens, 16.15 t/s
sliding_window PASS 70 tokens, 15.60 t/s
valid_brackets PASS 82 tokens, 15.77 t/s
parse_csv_line FAIL 900 tokens, 16.08 t/s
topological_sort PASS 265 tokens, 16.60 t/s
edit_distance PASS 173 tokens, 16.77 t/s
binary_search_bounds PASS 266 tokens, 17.25 t/s
```

Interpretation: `MIN_MARGIN=2.0` is now a repeatable quality-preserving MTP
candidate over two full suites, but the speed win is not yet enough. The
first repeat was about **3.4%** faster than the same aggregate baseline
(**16.39-16.42 t/s** vs **15.88 t/s**). The second repeat was dragged down by
a cold first task and longer `parse_duration` output, while its warmed chunks
still landed mostly in the **15.6-17.3 t/s** band. The key unchanged fact is
that verifier time stayed at **~91 ms** in both repeats; prefix1 only attacks
the partial replay tail.

A more aggressive `DS4_MTP_CAPTURE_PREFIX1_MIN_MARGIN=1.0` was canaried on
`lru_cache, parse_duration, flatten_dict, parse_csv_line`. It kept the expected
2/4 pass shape, but `parse_duration` generated 448 tokens instead of 363 in
the `2.0` canary and canary wall time was worse. It is not worth a full suite
until repeat data says the longer output was just run variance.

Relevant artifacts:

```text
/tmp/ds4-mtp-agg-eval/mtp_agg_guard025_12task_r1.json
/tmp/ds4-mtp-agg-eval/mtp_agg_guard025_capture_prefix1_12task_r1.json
/tmp/ds4-mtp-agg-eval/mtp_agg_guard025_capture_prefix1_m2_canary.json
/tmp/ds4-mtp-agg-eval/mtp_agg_guard025_capture_prefix1_m2_12task_r1.json
/tmp/ds4-mtp-agg-eval/mtp_agg_guard025_capture_prefix1_m2_12task_r2.json
/tmp/ds4-mtp-agg-eval/mtp_agg_guard025_capture_prefix1_m1_canary.json
/tmp/ds4-mtp-agg-guard025-server.log
/tmp/ds4-mtp-agg-guard025-capture-prefix1-server.log
/tmp/ds4-mtp-agg-guard025-capture-prefix1-m2-server.log
/tmp/ds4-mtp-agg-guard025-capture-prefix1-m2-r2-server.log
/tmp/ds4-mtp-agg-guard025-capture-prefix1-m1-server.log
```

Current MTP research recommendation: keep `guard0.25` as the quality anchor and
keep `MIN_MARGIN=2.0` as the partial-prefix candidate. Do not promote pure
prefix1 capture. The next speed target is still the **90-91 ms verifier**
itself; replay is now partly controlled.

### 2026-05-20 continuation — verifier profiling after prefix1

After prefix1 reduced the partial-replay tail, the next question was whether
the remaining verifier cost was output-head/readback, margin computation, or
the 43 layer pass itself. A low-intrusion verifier profiler was added:

```sh
DS4_MTP_VERIFY_PROFILE=1
```

It prints one line per target verifier call:

```text
ds4: mtp verify profile start=... tokens=... top2=... logits=...
  upload=... layers=... head_topk=... readback=... total=... ms
```

This profiler uses the command-buffer boundaries the verifier already has; it
does not add the heavy per-layer synchronization used by the stage profiler.
On `valid_brackets` with router-exact+K2+`guard0.25`+prefix1 `MIN_MARGIN=2.0`
at `ctx=256000`, the hot `tokens=2` verifier calls looked like:

```text
upload:    usually 0.02-0.8 ms, occasional ~2 ms
layers:    82-88 ms
head_topk: ~2.5 ms
readback:  ~0.04 ms after warmup
total:     85-90 ms
```

The important result: full-logit readback for the CPU margin guard is not the
20 t/s blocker. GPU-top2/no-logits may still be cleaner, but it cannot remove
the large cost; the layer pass dominates.

A short intrusive layer-stage profile was then run with:

```sh
DS4_METAL_LAYER_STAGE_PROFILE=1
```

The canary was `top_k_frequent` capped at 32 generated tokens, and only
`tokens=2` rows were aggregated so prompt prefill was excluded. This profiler
synchronizes at every stage, so use the percentages and stage order, not the
absolute t/s:

```text
stage lines: 7777
verifier cycles: 11
stage-sum total: 996.578 ms
per-cycle stage sums:
  first cold cycle: 110.805 ms
  later cycles: mostly 87-90 ms

part totals:
  attn: 570.432 ms, 57.2%
  ffn:  426.146 ms, 42.8%

top stages:
  ffn.routed_moe:      301.318 ms, 30.2%
  attn.output_proj:    230.385 ms, 23.1%
  attn.q_path:         114.830 ms, 11.5%
  attn.attention:       81.117 ms,  8.1%
  attn.indexer_setup:   57.493 ms,  5.8%
  ffn.shared_gate_up:   54.346 ms,  5.5%
  attn.compressor:      49.797 ms,  5.0%
  ffn.shared_down:      28.050 ms,  2.8%
```

This confirms the older non-MTP decode profile in the new verifier context:
the useful targets are still routed MoE and attention output projection. The
logit head and readback are secondary.

`DS4_CUDA_MOE_PROFILE=1` was then used without CUDA graph decode because CUDA
events inside decode graph capture fail. On the same short verifier-shaped
canary, `tokens=2` routed-MoE rows split as:

```text
default K2 direct gate, no graph, tokens=2 rows=688:
  gateup: 356.699 ms, 70.6%
  down:   138.799 ms, 27.5%
  xq/sort/midq/sum: ~1.9% combined
  warm last-8 verifier cycles: 27.151 ms routed-MoE total
```

The existing no-auxiliary-write variant was retested specifically for
`tokens=2`:

```sh
DS4_CUDA_MOE_DECODE_GATE_NOAUX=1
```

Its aggregate over all profiled rows looked better because it removed a few
outliers:

```text
noaux, no graph, tokens=2 rows=688:
  gateup: 296.016 ms, 67.8%
  down:   132.325 ms, 30.3%
  warm last-8 verifier cycles: 27.111 ms routed-MoE total
```

But the warm steady-state MoE total was effectively unchanged. A production-like
graph canary with noaux and the full MTP m2 environment scored **4/4** on
`top_k_frequent, parse_duration, flatten_dict, valid_brackets`, but did not
show a real speed win:

```text
label: mtp_noaux_m2_canary_4task
quality: 4/4
tokens: 716
suite-sum including cold first task: 12.70 t/s
top_k_frequent: PASS 102 tokens, 5.44 t/s  (cold/capture path)
parse_duration: PASS 396 tokens, 16.65 t/s
flatten_dict:   PASS 136 tokens, 16.07 t/s
valid_brackets: PASS 82 tokens, 15.58 t/s
hot verifier averages: ~87.9-91.0 ms depending on task
```

Conclusion: `DS4_CUDA_MOE_DECODE_GATE_NOAUX=1` remains diagnostic-only. It is
not the missing 10-11%.

The attention-output finding led to one new opt-in kernel experiment:

```sh
DS4_CUDA_Q8_SOA_BATCH2=1
```

Rationale: for `n_tok=2`, `attn_output_b` used the generic batch2 cached-x Q8
kernel over interleaved GGUF blocks, even when the SoA cache for
`attn_output_b` was already resident. The new kernel keeps the same batch2
dot-product shape but reads scales and quant bytes from the SoA duplicate
cache. It is opt-in only.

The first production-like canary rejected it:

```text
label: mtp_soa_batch2_m2_canary_4task
quality: 3/4
failure: parse_duration (missed expected ValueError for "0s")
tokens: 684
suite-sum including cold first task: 12.37 t/s
top_k_frequent: PASS 102 tokens, 5.29 t/s  (cold/capture path)
parse_duration: FAIL 335 tokens, 16.30 t/s
flatten_dict:   PASS 165 tokens, 16.59 t/s
valid_brackets: PASS 82 tokens, 15.64 t/s
hot verifier averages: still ~87-91 ms
```

This is a useful negative result. The batch2 SoA path is not promotable without
shadow/logit-drift work, and even the canary speed did not justify that risk.

Relevant artifacts:

```text
/tmp/ds4-mtp-profile-eval/mtp_verify_profile_valid_brackets.json
/tmp/ds4-mtp-layer-stage-profile-server.log
/tmp/ds4-mtp-moe-profile-nograph-server.log
/tmp/ds4-mtp-moe-profile-noaux-nograph-server.log
/tmp/ds4-mtp-profile-eval/mtp_noaux_m2_canary_4task.json
/tmp/ds4-mtp-noaux-canary-server.log
/tmp/ds4-mtp-profile-eval/mtp_soa_batch2_m2_canary_4task.json
/tmp/ds4-mtp-soa-batch2-canary-server.log
```

Updated recommendation: keep the `MIN_MARGIN=2.0` prefix1 candidate as the
best quality-preserving MTP line, but do not spend more time on CPU margin
readback, noaux, or batch2 SoA as primary routes. The remaining verifier cost
is in the layer body itself: routed-MoE gate/up+down, attention output
projection, Q path, and sparse attention/indexer setup.

### 2026-05-20 continuation — quality-preserving draft2 skip

The stage profiles show that making the N=2 verifier much faster requires a
large structural rewrite. A smaller but more promising quality-preserving idea
is to avoid entering that verifier when the second recursive MTP draft is
weak.

New opt-in:

```sh
DS4_MTP_DRAFT2_SKIP_MIN_MARGIN=2.0
```

In strict MTP, after the base target token has verified `drafts[0]`, recursive
MTP proposes `drafts[1]`. With this flag, the runtime asks MTP for the top-2
margin for that second draft. If the margin is below the threshold, it skips
the N=2 target verifier and instead exact-decodes `drafts[0]` only. That costs
about one normal exact token, but avoids the bad partial path:

```text
old partial exact-replay path: ~150-160 ms
new margin-skip path:          ~61-62 ms
```

This does not accept an unverified target token: `drafts[0]` has already matched
the base target logits, and the skipped `drafts[1]` is simply not emitted.

First 4-task canary with router-exact+K2+guard0.25+prefix1 m2+draft2 skip:

```text
label: mtp_draft2skip_m2_canary_4task
quality: 4/4
tokens: 722
suite-sum including cold first task: 13.49 t/s
top_k_frequent: PASS 102 tokens, 5.96 t/s  (cold/capture path)
parse_duration: PASS 405 tokens, 17.23 t/s
flatten_dict:   PASS 133 tokens, 17.06 t/s
valid_brackets: PASS 82 tokens, 16.47 t/s
```

Aggregate shape from the canary:

```text
margin-skip cycles: ~61-62 ms each
parse_duration hot chunks: 16.6-19.5 t/s, avg 17.85 t/s
flatten_dict hot avg: 18.32 t/s server decode-only
valid_brackets hot avg: 18.85 t/s server decode-only
```

The full 12-task eval was then run twice at `ctx=256000`.

```text
label: mtp_draft2skip_m2_12task_r1
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2803
suite-sum: 16.87 t/s
calls: 1155
cycles: 979
drafted extra: 1958
committed extra: 1650
extra/call: 1.429
emitted/call: 2.429
extra-token accept: 84.3%
avg cycle: 87.851 ms
avg draft: 4.143 ms
avg verify: 82.130 ms
avg replay: 1.040 ms
paths:
  margin-skip: 259 cycles, 62.2 ms avg
  micro-full: 671 cycles, 95.7 ms avg
  micro-prefix1: 32 cycles, 95.8 ms avg
  micro-exact-replay1: 17 cycles, 156.4 ms avg
```

Per-task r1:

```text
merge_intervals PASS 242 tokens, 17.30 t/s
top_k_frequent PASS 102 tokens, 16.11 t/s
lru_cache FAIL 177 tokens, 17.07 t/s
parse_duration PASS 312 tokens, 17.17 t/s
group_anagrams PASS 78 tokens, 14.79 t/s
flatten_dict PASS 136 tokens, 16.90 t/s
sliding_window PASS 70 tokens, 15.46 t/s
valid_brackets PASS 82 tokens, 16.45 t/s
parse_csv_line FAIL 900 tokens, 17.01 t/s
topological_sort PASS 265 tokens, 16.76 t/s
edit_distance PASS 173 tokens, 16.83 t/s
binary_search_bounds PASS 266 tokens, 17.55 t/s
```

Repeat:

```text
label: mtp_draft2skip_m2_12task_r2
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2879
suite-sum: 16.82 t/s
calls: 1210
cycles: 1002
drafted extra: 2003
committed extra: 1673
extra/call: 1.383
emitted/call: 2.383
extra-token accept: 83.5%
avg cycle: 87.156 ms
avg draft: 4.140 ms
avg verify: 81.640 ms
avg replay: 0.846 ms
paths:
  margin-skip: 280 cycles, 62.2 ms avg
  micro-full: 672 cycles, 95.6 ms avg
  micro-prefix1: 36 cycles, 96.0 ms avg
  micro-exact-replay1: 14 cycles, 157.0 ms avg
```

Per-task r2:

```text
merge_intervals PASS 242 tokens, 17.21 t/s
top_k_frequent PASS 102 tokens, 16.08 t/s
lru_cache FAIL 177 tokens, 16.64 t/s
parse_duration PASS 389 tokens, 17.18 t/s
group_anagrams PASS 78 tokens, 15.21 t/s
flatten_dict PASS 135 tokens, 16.91 t/s
sliding_window PASS 70 tokens, 15.50 t/s
valid_brackets PASS 82 tokens, 15.70 t/s
parse_csv_line FAIL 900 tokens, 16.98 t/s
topological_sort PASS 265 tokens, 16.91 t/s
edit_distance PASS 173 tokens, 16.78 t/s
binary_search_bounds PASS 266 tokens, 17.48 t/s
```

Interpretation: draft2 skip trades some emitted/call rate for much lower cycle
cost. The full-accept path is unchanged at ~95-96 ms, but many would-be weak
second-draft cycles are diverted to an exact one-token path at ~62 ms instead
of falling into partial replay. This is the best quality-first MTP result so
far: same 10/12 quality over two full suites, with stable suite-sum around
**16.8-16.9 t/s** and many warmed decode chunks in the **18-19 t/s** band.

Relevant artifacts:

```text
/tmp/ds4-mtp-profile-eval/mtp_draft2skip_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_draft2skip_m2_12task_r1.json
/tmp/ds4-mtp-profile-eval/mtp_draft2skip_m2_12task_r2.json
/tmp/ds4-mtp-draft2skip-m2-canary-server.log
```

Updated MTP recommendation: if testing MTP, use draft2 skip with the guarded
prefix1 candidate. It is still below the production no-MTP path, but it is the
first MTP variant that improves quality-preserving full-suite speed repeatably
after the exact-islands work.

Threshold tuning canaries were run on
`top_k_frequent, parse_duration, flatten_dict, valid_brackets`:

```text
threshold 1.0:
  quality: 4/4
  tokens: 715
  suite-sum including cold first task: 13.43 t/s
  top_k_frequent 102 tokens, 6.03 t/s
  parse_duration 401 tokens, 17.10 t/s
  flatten_dict 130 tokens, 17.03 t/s
  valid_brackets 82 tokens, 16.09 t/s

threshold 2.0:
  quality: 4/4
  tokens: 722
  suite-sum including cold first task: 13.49 t/s
  top_k_frequent 102 tokens, 5.96 t/s
  parse_duration 405 tokens, 17.23 t/s
  flatten_dict 133 tokens, 17.06 t/s
  valid_brackets 82 tokens, 16.47 t/s

threshold 3.0:
  quality: 4/4
  tokens: 719
  suite-sum including cold first task: 12.84 t/s
  top_k_frequent 102 tokens, 5.21 t/s
  parse_duration 405 tokens, 17.30 t/s
  flatten_dict 130 tokens, 16.61 t/s
  valid_brackets 82 tokens, 16.25 t/s
```

`1.0` skips too little and keeps more expensive verifier/replay work.
`3.0` skips too much and loses too many full accepts. The current best default
for this experiment remains **2.0**.

Additional artifacts:

```text
/tmp/ds4-mtp-profile-eval/mtp_draft2skip_m1_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_draft2skip_m3_canary_4task.json
/tmp/ds4-mtp-draft2skip-m1-canary-server.log
/tmp/ds4-mtp-draft2skip-m3-canary-server.log
```

### 2026-05-20 continuation — prefix batch retest

The next attempt revisited `DS4_MTP_PREFIX_BATCH`, but from a stricter
quality-first angle. The old prefix-batch idea tried to verify the already
sampled next target token plus two MTP drafts before running the normal base
decode. It regressed because it often accepted only 2 of 3 proposed rows, then
had to restore and fall back.

New opt-in diagnostics:

```sh
DS4_MTP_PREFIX_BATCH=1
DS4_MTP_PREFIX_BATCH_MAX_TOKENS=2
DS4_MTP_PREFIX_BATCH_DRAFT2_MIN_MARGIN=2.0
DS4_MTP_PREFIX_BATCH_MIN_MARGIN=3.0
```

`DS4_MTP_PREFIX_BATCH_MAX_TOKENS=2` limits the prefix attempt to the already
sampled target token plus one MTP draft. That uses the same guarded N=2 verifier
as the current best MTP line, and can commit prefix-1 with captured frontiers.

`DS4_MTP_PREFIX_BATCH_DRAFT2_MIN_MARGIN` gates the prefix attempt before the
target verifier: if the MTP proposal for the next token has low top-2 margin,
the runtime falls back to the normal draft2-skip path. This avoids spending a
~96 ms prefix verifier just to emit one token.

`DS4_MTP_PREFIX_BATCH_MIN_MARGIN` is the deeper experiment: allow the original
3-token prefix batch only when all recursive MTP drafts clear a margin gate, and
also apply the target batch margin guard to every output row.

Canary results on
`top_k_frequent, parse_duration, flatten_dict, valid_brackets` at `ctx=256000`:

```text
plain prefix2, no draft gate:
  quality: 4/4
  tokens: 1023
  suite-sum including cold first task: 13.37 t/s
  top_k_frequent 102 tokens, 5.62 t/s
  parse_duration 704 tokens, 15.74 t/s
  flatten_dict 135 tokens, 16.06 t/s
  valid_brackets 82 tokens, 15.55 t/s

prefix2, MTP draft margin gate 2.0:
  quality: 4/4
  tokens: 691
  suite-sum including cold first task: 13.47 t/s
  top_k_frequent 102 tokens, 6.33 t/s
  parse_duration 372 tokens, 17.33 t/s
  flatten_dict 135 tokens, 15.83 t/s
  valid_brackets 82 tokens, 15.80 t/s

prefix2, MTP draft margin gate 3.0:
  quality: 4/4
  tokens: 720
  suite-sum including cold first task: 13.29 t/s
  top_k_frequent 102 tokens, 5.92 t/s
  parse_duration 404 tokens, 17.00 t/s
  flatten_dict 132 tokens, 16.53 t/s
  valid_brackets 82 tokens, 15.80 t/s

same-build draft2-skip rerun, no prefix:
  quality: 4/4
  tokens: 784
  suite-sum including cold first task: 13.55 t/s
  top_k_frequent 102 tokens, 5.91 t/s
  parse_duration 435 tokens, 16.80 t/s
  flatten_dict 165 tokens, 17.03 t/s
  valid_brackets 82 tokens, 16.28 t/s

prefix3, all-draft MTP margin gate 3.0:
  quality: 4/4
  tokens: 714
  suite-sum including cold first task: 11.69 t/s
  top_k_frequent 102 tokens, 5.01 t/s
  parse_duration 404 tokens, 15.29 t/s
  flatten_dict 126 tokens, 14.71 t/s
  valid_brackets 82 tokens, 14.29 t/s
```

Interpretation:

- Prefix2 is not a clear win. Its successful `prefix-full` cycles emit two
  tokens in about **98-100 ms**, which is near 20 t/s locally, but partial
  prefix cycles cost about **95-97 ms for one token**. The MTP draft-margin gate
  reduces those bad partials but also removes useful prefix-full opportunities.
- Prefix3 is worse. The rare `prefix-full` cycle that survived the strict gates
  measured about **169 ms total**, with **156 ms** in the target verifier. Three
  emitted tokens at that cost are only ~17.7 t/s before request overhead.
- The same-build no-prefix draft2-skip rerun remained the best canary aggregate
  and was stronger on `flatten_dict` and `valid_brackets`.

Decision: keep the prefix-batch changes as opt-in diagnostics only. Do not add
them to the recommended MTP environment. The useful lesson is that skipping the
base decode is not enough if the replacement verifier has to process fewer
extra tokens or enters a partial path. The current best speed/quality line
remains draft2-skip m2 without prefix batch.

Artifacts:

```text
/tmp/ds4-mtp-profile-eval/mtp_prefix2_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_prefix2_gate2_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_prefix2_gate3_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_draft2skip_m2_rerun_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_prefix3_gate3_m2_canary_4task.json
/tmp/ds4-mtp-prefix2-gated-canary-server.log
/tmp/ds4-mtp-prefix2-gate3-canary-server.log
/tmp/ds4-mtp-prefix3-gate3-canary-server.log
```

### 2026-05-20 continuation — CUDA Graph verifier experiment

Another low-risk idea was to CUDA-graph the MTP target verifier. This should
not change accepted tokens: it only changes how the existing N=2 verifier
kernels are submitted.

New opt-in:

```sh
DS4_CUDA_GRAPH_VERIFY=1
```

First attempt: enabling it with `DS4_MTP_CAPTURE_PREFIX1=1` failed immediately.
The prefix1 verifier path copies captured compressor/indexer frontiers while
the CUDA stream is in capture mode, and CUDA rejects those tensor copies:

```text
CUDA tensor copy failed: operation not permitted when stream is capturing
```

The code now refuses graph-verify capture when prefix1 capture is active, and
adds a CUDA capture abort helper so a failed capture does not leave the stream
poisoned.

Follow-up: CUDA `ds4_gpu_tensor_copy()` was changed to use
`cudaMemcpyAsync(..., cudaStreamPerThread)` for device-to-device copies while a
graph capture is active. That makes the prefix1 frontier copies capturable, so
`DS4_CUDA_GRAPH_VERIFY=1` can now run with `DS4_MTP_CAPTURE_PREFIX1=1`.

The compatible test therefore ran with graph verifier enabled but without
`DS4_MTP_CAPTURE_PREFIX1`:

```sh
DS4_CUDA_GRAPH_DECODE=1
DS4_CUDA_GRAPH_VERIFY=1
DS4_CUDA_Q8_SOA_CACHE=1
DS4_CUDA_MOE_K2_DIRECT_GATE=1
DS4_MTP_STRICT=1
DS4_MTP_BATCH_VERIFY=1
DS4_MTP_UNSAFE_BATCH_VERIFY=1
DS4_MTP_BATCH_MARGIN_GUARD=0.25
DS4_MTP_BATCH_HC_PRE_EXACT=1
DS4_MTP_BATCH_ATTENTION_EXACT=1
DS4_MTP_BATCH_COMPRESS_PROJ_EXACT=1
DS4_MTP_BATCH_ROUTER_EXACT=1
DS4_MTP_DRAFT2_SKIP_MIN_MARGIN=2.0
```

Canary result:

```text
label: mtp_graphverify_noprefix1_m2_canary_4task
quality: 4/4
tokens: 626
suite-sum including cold first task: 13.50 t/s
top_k_frequent: PASS 102 tokens, 6.55 t/s
parse_duration: PASS 307 tokens, 17.18 t/s
flatten_dict:   PASS 135 tokens, 16.94 t/s
valid_brackets: PASS 82 tokens, 16.47 t/s
```

The server-side decode chunks looked better than the eval elapsed figure on
some tasks:

```text
parse_duration avg decode: 18.00 t/s
flatten_dict avg decode:   18.16 t/s
valid_brackets avg decode: 18.85 t/s
```

Full 12-task eval at `ctx=256000`:

```text
label: mtp_graphverify_noprefix1_m2_12task_r1
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2861
suite-sum: 16.84 t/s
```

Per-task:

```text
merge_intervals PASS 242 tokens, 16.98 t/s
top_k_frequent PASS 102 tokens, 15.89 t/s
lru_cache FAIL 177 tokens, 17.12 t/s
parse_duration PASS 370 tokens, 17.58 t/s
group_anagrams PASS 78 tokens, 14.71 t/s
flatten_dict PASS 136 tokens, 16.54 t/s
sliding_window PASS 70 tokens, 15.48 t/s
valid_brackets PASS 82 tokens, 16.18 t/s
parse_csv_line FAIL 900 tokens, 16.98 t/s
topological_sort PASS 265 tokens, 16.79 t/s
edit_distance PASS 173 tokens, 16.46 t/s
binary_search_bounds PASS 266 tokens, 17.29 t/s
```

Interpretation: CUDA Graph verifier is quality-neutral in this shape, and it
slightly improves some hot chunks, but it does not beat the best
draft2-skip m2 full-suite repeats (**16.87-16.90 t/s**). Disabling
`capture_prefix1` to make the verifier capturable gives back roughly the same
gain that graph launch reduction can recover.

Decision: keep `DS4_CUDA_GRAPH_VERIFY=1` diagnostic-only for now. It becomes
interesting again only if prefix1 frontier capture can be made graph-capturable
or avoided.

After the async D2D copy change, the prefix1-compatible graph-verifier canary
was run:

```text
label: mtp_graphverify_prefix1_m2_canary_4task
quality: 4/4
tokens: 606
suite-sum including cold first task: 13.12 t/s
top_k_frequent: PASS 102 tokens, 5.23 t/s
parse_duration: PASS 287 tokens, 16.98 t/s
flatten_dict:   PASS 135 tokens, 15.98 t/s
valid_brackets: PASS 82 tokens, 16.23 t/s
```

This proves the capture compatibility issue is fixed, but it is still not a
speed win. Hot `micro-full` verifier cycles remained around **95-102 ms**,
similar to or worse than the no-graph draft2-skip path.

One more verifier-graph variant removed the pre-capture stream synchronize for
the verifier graph only. The verifier prompt rows are already uploaded before
capture starts, so this is safe for this path and avoids paying an extra
sync during graph creation/update. Normal decode graph capture still keeps its
pre-sync.

New helper:

```c
ds4_gpu_graph_capture_begin_no_sync()
```

Prefix1 + graph verifier + no pre-sync canary:

```text
label: mtp_graphverify_prefix1_nosync_m2_canary_4task
quality: 4/4
tokens: 770
suite-sum including cold first task: 13.65 t/s
top_k_frequent: PASS 102 tokens, 6.11 t/s
parse_duration: PASS 420 tokens, 17.10 t/s
flatten_dict:   PASS 166 tokens, 16.81 t/s
valid_brackets: PASS 82 tokens, 15.55 t/s
```

Server-side hot chunks improved on the long tasks but not enough:

```text
parse_duration avg decode: 17.65 t/s
flatten_dict avg decode:   17.78 t/s
valid_brackets avg decode: 18.40 t/s
```

Aggregate cycle timings still show the real cost in the verifier layer body,
not graph submission:

```text
parse_duration micro-full: 95.880 ms total, 91.049 ms verify
flatten_dict micro-full:   95.558 ms total, 90.735 ms verify
valid_brackets micro-full: 95.041 ms total, 90.226 ms verify
```

The same no-pre-sync graph verifier was also tested without prefix1 capture,
to isolate graph submission from the prefix replay path:

```text
label: mtp_graphverify_noprefix1_nosync_m2_canary_4task
quality: 4/4
tokens: 724
suite-sum including cold first task: 13.65 t/s
top_k_frequent: PASS 102 tokens, 6.10 t/s
parse_duration: PASS 405 tokens, 17.57 t/s
flatten_dict:   PASS 135 tokens, 16.51 t/s
valid_brackets: PASS 82 tokens, 16.15 t/s
```

Against the earlier no-prefix graph-verifier canary this is only a small net
change (**13.65 t/s** vs **13.50 t/s**) and loses speed on three of the four
task-level eval timings. The server hot path still spends about **88-95 ms**
per full verifier cycle:

```text
parse_duration micro-full: 94.401 ms total, 89.567 ms verify
flatten_dict micro-full:   94.268 ms total, 89.349 ms verify
valid_brackets micro-full: 92.832 ms total, 88.009 ms verify
```

Updated decision: `DS4_CUDA_GRAPH_VERIFY=1` remains diagnostic-only even after
making prefix1 capture compatible and removing the verifier pre-sync. The next
speed work should attack the verifier layer body itself.

### 2026-05-21 continuation — attention output A F16/cuBLAS probe

Since stage profiling attributed a large fraction of verifier time to
`attn.output_proj`, the next narrow probe was to force the attention output A
projection to use the existing expanded-F16 cuBLAS path even for the N=2
verifier:

```sh
DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN=2
```

This spends extra device memory on-demand for `attn_output_a` expanded weights
but should leave the rest of the candidate unchanged.

Canary:

```text
label: mtp_attnouta_cublas2_m2_canary_4task
quality: 4/4
tokens: 716
suite-sum including cold first task: 13.39 t/s
top_k_frequent: PASS 102 tokens, 5.93 t/s
parse_duration: PASS 401 tokens, 17.31 t/s
flatten_dict:   PASS 131 tokens, 16.26 t/s
valid_brackets: PASS 82 tokens, 16.21 t/s
```

The hot verifier cycle did not improve:

```text
parse_duration micro-full: 96.783 ms total, 91.914 ms verify
flatten_dict micro-full:   95.508 ms total, 90.669 ms verify
valid_brackets micro-full: 94.099 ms total, 89.255 ms verify
```

Decision: do not include `DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN=2` in the
recommendation. It is quality-neutral on the canary, but it is slower than the
same-build draft2-skip canary and adds memory pressure.

Artifacts:

```text
/tmp/ds4-mtp-profile-eval/mtp_attnouta_cublas2_m2_canary_4task.json
/tmp/ds4-mtp-attnouta-cublas2-m2-canary-server.log
```

### 2026-05-21 continuation — attention output B SoA batch2 probe

The earlier `DS4_CUDA_Q8_SOA_BATCH2=1` probe enabled SoA layout for every
generic Q8 N=2 matmul and failed a 4-task coding canary. A narrower variant was
added to target only the large attention output B projection:

```sh
DS4_CUDA_Q8_SOA_BATCH2_ATTN_OUTPUT_B=1
```

Implementation detail: the old global `DS4_CUDA_Q8_SOA_BATCH2=1` behavior is
unchanged. The new env only allows the batch2 SoA kernel when the Q8 matmul
label is `attn_output_b` / `attention_output_b`.

Canary:

```text
label: mtp_attnoutb_soa_batch2_m2_canary_4task
quality: 4/4
tokens: 702
suite-sum including cold first task: 13.65 t/s
top_k_frequent: PASS 102 tokens, 6.16 t/s
parse_duration: PASS 383 tokens, 17.54 t/s
flatten_dict:   PASS 135 tokens, 16.81 t/s
valid_brackets: PASS 82 tokens, 16.35 t/s
```

Hot server chunks:

```text
parse_duration avg decode: 18.06 t/s
flatten_dict avg decode:   18.03 t/s
valid_brackets avg decode: 18.84 t/s
```

Hot verifier cycles improved slightly in the shortest task but remained close
to the draft2-skip path overall:

```text
parse_duration micro-full: 94.245 ms total, 89.425 ms verify
flatten_dict micro-full:   93.969 ms total, 89.128 ms verify
valid_brackets micro-full: 91.906 ms total, 87.103 ms verify
```

Decision: this is the first narrow output-projection probe worth escalating.
Run a full 12-task `ctx=256000` eval before adding it to the recommendation.

Full 12-task eval, fresh server / cold first task:

```text
label: mtp_attnoutb_soa_batch2_m2_12task_r1
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2895
suite-sum: 15.91 t/s
```

Per-task:

```text
merge_intervals PASS 242 tokens, 9.68 t/s
top_k_frequent PASS 102 tokens, 15.27 t/s
lru_cache FAIL 177 tokens, 17.09 t/s
parse_duration PASS 405 tokens, 17.37 t/s
group_anagrams PASS 78 tokens, 14.78 t/s
flatten_dict PASS 135 tokens, 16.95 t/s
sliding_window PASS 70 tokens, 15.24 t/s
valid_brackets PASS 82 tokens, 16.37 t/s
parse_csv_line FAIL 900 tokens, 17.08 t/s
topological_sort PASS 265 tokens, 16.99 t/s
edit_distance PASS 173 tokens, 16.85 t/s
binary_search_bounds PASS 266 tokens, 17.49 t/s
```

Interpretation: quality held, but the fresh-server first task is not a fair
speed comparison with the warmed draft2-skip repeats. A warmed repeat was run
on the same server after a one-task warm-up.

Warm repeat:

```text
label: mtp_attnoutb_soa_batch2_m2_12task_r2_warm
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2773
suite-sum: 16.87 t/s
```

Per-task:

```text
merge_intervals PASS 242 tokens, 17.07 t/s
top_k_frequent PASS 102 tokens, 14.93 t/s
lru_cache FAIL 177 tokens, 17.17 t/s
parse_duration PASS 287 tokens, 17.35 t/s
group_anagrams PASS 78 tokens, 14.84 t/s
flatten_dict PASS 131 tokens, 16.59 t/s
sliding_window PASS 70 tokens, 15.47 t/s
valid_brackets PASS 82 tokens, 16.39 t/s
parse_csv_line FAIL 900 tokens, 17.05 t/s
topological_sort PASS 265 tokens, 16.94 t/s
edit_distance PASS 173 tokens, 16.89 t/s
binary_search_bounds PASS 266 tokens, 17.52 t/s
```

Decision: keep `DS4_CUDA_Q8_SOA_BATCH2_ATTN_OUTPUT_B=1` as an opt-in probe,
not a recommendation. It is quality-neutral across the full suite and reaches
the same speed band as draft2-skip, but it does not beat the best warmed
draft2-skip repeat (**16.87 t/s** vs **16.90 t/s**).

Artifacts:

```text
/tmp/ds4-mtp-profile-eval/mtp_attnoutb_soa_batch2_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_attnoutb_soa_batch2_m2_12task_r1.json
/tmp/ds4-mtp-profile-eval/mtp_attnoutb_soa_batch2_m2_12task_r2_warm.json
/tmp/ds4-mtp-profile-eval/mtp_attnoutb_soa_batch2_m2_warmup_merge.json
/tmp/ds4-mtp-attnoutb-soa-batch2-m2-canary-server.log
/tmp/ds4-mtp-attnoutb-soa-batch2-m2-12task-r1-server.log
/tmp/ds4-mtp-attnoutb-soa-batch2-m2-12task-r2-warm-server.log
```

### 2026-05-21 continuation — exact decode refresh and attention output B F16/cuBLAS probe

After the anti-loop reread, the no-MTP exact decode profile was refreshed at
the production allocation (`--ctx 256000`). `DS4_CUDA_MOE_PROFILE=1` cannot be
combined with CUDA graph decode because CUDA events inside graph capture fail
with `operation not permitted when stream is capturing`, so this profile was
run without graph and should be read as a relative stage map, not as the server
throughput number.

Command:

```sh
DS4_CUDA_Q8_SOA_CACHE=1 \
DS4_METAL_DECODE_STAGE_PROFILE=1 \
DS4_CUDA_MOE_PROFILE=1 \
./ds4 --cuda -m ds4flash.gguf --ctx 256000 --nothink --temp 0 -n 4 \
  -p "Write a Python function merge_intervals(intervals) that merges overlapping intervals."
```

Hot generated positions:

```text
pos=26 total=56.816 ms, 17.60 t/s
pos=27 total=56.220 ms, 17.79 t/s
pos=28 total=54.468 ms, 18.36 t/s
```

Aggregated over 129 layer-stage rows (3 generated tokens):

```text
routed_moe         47.979 ms total, 15.993 ms/token
attn_output        41.728 ms total, 13.909 ms/token
q_path             27.125 ms total,  9.042 ms/token
shared_gate_up     11.923 ms total,  3.974 ms/token
compressor_indexer 10.819 ms total,  3.606 ms/token
attention           8.581 ms total,  2.860 ms/token
shared_down         6.801 ms total,  2.267 ms/token
```

The MoE event split remained:

```text
tokens=1 calls=129
gateup=30.678 ms, down=14.664 ms, total=46.709 ms
per generated token: gateup=10.226 ms, down=4.888 ms, routed total=15.570 ms
```

This reconfirms that the shortest exact path is still the same two kernels:
routed MoE gate/up and attention-output projection. To isolate the latter
without turning on the already-rejected global Q8/cuBLAS mode, a new narrow
opt-in switch was added:

```sh
DS4_CUDA_ATTENTION_OUTPUT_B_CUBLAS_MIN=N
```

When set, only Q8 matmuls labelled `attn_output_b` /
`attention_output_b` with `n_tok >= N` may use the existing Q8-to-F16 cache
plus cuBLAS path. Default behavior is unchanged.

The `N=1` probe at `ctx=256000` was negative:

```text
DS4_CUDA_Q8_SOA_CACHE=1
DS4_CUDA_ATTENTION_OUTPUT_B_CUBLAS_MIN=1
DS4_METAL_DECODE_STAGE_PROFILE=1
DS4_CUDA_MOE_PROFILE=1
DS4_CUDA_WEIGHT_CACHE_VERBOSE=1

pos=26 total=57.423 ms, 17.41 t/s
pos=27 total=56.048 ms, 17.84 t/s
pos=28 total=55.114 ms, 18.14 t/s

attn_output 41.902 ms total vs 41.728 ms control
routed_moe  47.571 ms total vs 47.979 ms control
MoE total   46.301 ms total vs 46.709 ms control
```

The small MoE movement is noise/indirect scheduling; the target stage did not
improve. The run also filled about **10.58 GiB** of Q8 F16 cache during the
short prefill/decode process, so this is the wrong memory-for-speed trade.

Decision: keep `DS4_CUDA_ATTENTION_OUTPUT_B_CUBLAS_MIN` as diagnostic-only and
do not run a full server/coding eval. It is not a quality problem; it fails the
speed gate before quality testing is worth the cost.

### 2026-05-21 continuation — no-MTP warm coding comparison

After the MTP output-projection probe tied but did not beat draft2-skip, the
current production-safe path was re-measured with the same 12 coding tasks and
`ctx=256000`:

```sh
DS4_CUDA_GRAPH_DECODE=1
DS4_CUDA_Q8_SOA_CACHE=1
```

A one-task warm-up was run first to remove first-request graph/cache effects.

Warm full-suite result:

```text
label: nomtp_graph_soa_12task_warm_r1
quality: 10/12, failures lru_cache, parse_csv_line
tokens: 2938
suite-sum: 16.83 t/s
```

Per-task:

```text
merge_intervals PASS 243 tokens, 17.17 t/s
top_k_frequent PASS 102 tokens, 15.54 t/s
lru_cache FAIL 177 tokens, 16.50 t/s
parse_duration PASS 446 tokens, 17.31 t/s
group_anagrams PASS 78 tokens, 15.26 t/s
flatten_dict PASS 136 tokens, 16.62 t/s
sliding_window PASS 70 tokens, 14.76 t/s
valid_brackets PASS 82 tokens, 15.77 t/s
parse_csv_line FAIL 900 tokens, 17.16 t/s
topological_sort PASS 265 tokens, 16.82 t/s
edit_distance PASS 173 tokens, 16.76 t/s
binary_search_bounds PASS 266 tokens, 17.01 t/s
```

Server decode chunks were steadier than MTP, mostly **17.4-17.9 t/s**. The eval
suite-sum is nevertheless in the same band as the best quality-first MTP runs
because request/prompt overhead and answer length still matter. This confirms
the current situation: MTP is no longer a clear quality regression, but it also
does not create the missing 10-11% jump to 20 t/s. The next useful work should
target exact decode kernels shared by both no-MTP and the MTP verifier.

Artifacts:

```text
/tmp/ds4-mtp-profile-eval/nomtp_graph_soa_warmup_merge.json
/tmp/ds4-mtp-profile-eval/nomtp_graph_soa_12task_warm_r1.json
/tmp/ds4-nomtp-graph-soa-12task-warm-server.log
```

Artifacts:

```text
/tmp/ds4-mtp-profile-eval/mtp_graphverify_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_graphverify_noprefix1_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_graphverify_noprefix1_m2_12task_r1.json
/tmp/ds4-mtp-profile-eval/mtp_graphverify_noprefix1_nosync_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_graphverify_prefix1_m2_canary_4task.json
/tmp/ds4-mtp-profile-eval/mtp_graphverify_prefix1_nosync_m2_canary_4task.json
/tmp/ds4-mtp-graphverify-m2-canary-server.log
/tmp/ds4-mtp-graphverify-noprefix1-m2-canary-server.log
/tmp/ds4-mtp-graphverify-noprefix1-nosync-m2-canary-server.log
/tmp/ds4-mtp-graphverify-prefix1-m2-canary-server.log
/tmp/ds4-mtp-graphverify-prefix1-nosync-m2-canary-server.log
```

### Compact checkpoint for continuation

Current production-safe default remains **no-MTP graph decode + Q8 SoA cache**.
It is the quality-preserving path used as the baseline for coding work.

### 2026-05-21 consolidation — exact-fast server build

The consolidated "fast but no quality tradeoff" build is the normal CUDA server
with only these acceleration switches:

```sh
DS4_CUDA_GRAPH_DECODE=1
DS4_CUDA_Q8_SOA_CACHE=1
```

This means:

- no MTP;
- full six routed experts;
- no active-expert reduction or renormalization;
- no verifier shortcuts;
- no diagnostic F16/cuBLAS Q8 switches;
- no broad/experimental SoA extensions beyond the default attention-output
  A/B cache.

A wrapper was added at repo root:

```sh
./ds4-server-exact-fast
```

With no arguments it starts:

```sh
./ds4-server --cuda -m ds4flash.gguf --ctx 256000 \
  --host 0.0.0.0 --port 8000 --tokens 900
```

The defaults can be overridden with env (`DS4_MODEL`, `DS4_CTX`, `DS4_HOST`,
`DS4_PORT`, `DS4_TOKENS`) or by passing explicit `ds4-server` args. The wrapper
also unsets known experiment/tradeoff env vars (`DS4_MTP_*`,
`DS4_MOE_ACTIVE_EXPERTS*`, Q8/cuBLAS diagnostics, broad SoA probes, MoE micro
probes, profiler envs) before exporting the exact-fast pair above.

Build:

```text
make cuda-spark
result: ds4, ds4-server, ds4-bench, ds4-eval all up to date
```

Smoke:

```text
DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 ./ds4 --cuda \
  -m ds4flash.gguf --ctx 256000 --nothink --temp 0 -n 8 \
  -p "Write a Python function add(a, b)."

result: generation 20.64 t/s on the short generated snippet
```

`ds4-eval` smoke:

```sh
DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 ./ds4-eval --cuda \
  -m ds4flash.gguf --questions 2 --tokens 512 \
  --hard-limit-reply-budget 128 --soft-limit-reply-budget 256 \
  --nothink --plain --trace /tmp/ds4-eval-graph-soa-2q-repeat.txt
```

Result: **2/2 passed**:

```text
GPQA Diamond/recNu3MXkvWUzHZr9: B expected B
SuperGPQA/001b51d76b4d422988f2c11f104a2c6c: C expected C
```

Control with graph disabled and SoA kept on also passed **2/2**. One earlier
run with `--warm-weights` produced **1/2** by answering `G` on the SuperGPQA
item, so `--warm-weights` is not part of the wrapper default and should not be
used as a quality gate unless repeated. The current trial build to hand-test is
therefore `./ds4-server-exact-fast` without `--warm-weights`.

HTTP wall check on the running consolidated server:

```text
server: ./ds4-server --cuda -m ds4flash.gguf --ctx 256000 --host 0.0.0.0 --port 8000 --tokens 900
env:    DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1
prompt: "Write a Python function merge_intervals(intervals)."

run 1, 128 completion tokens: 25.09 s wall  (cold request; graph/cache overhead)
run 2, 128 completion tokens:  7.43 s wall, 17.22 tok/s wall-clock
run 3, 256 completion tokens: 14.72 s wall, 17.39 tok/s wall-clock
```

Metric correction after log review: the HTTP wall number above is an end-to-end
smoke, not the canonical decode throughput. It includes request handling,
prompt prefill, and response overhead. The server log separates those phases:

```text
prompt start
prompt done 0.xxxs              # prompt prefill; afterwards the prompt is in KV cache
gen=... decoding chunk=... avg=... t/s
gen=... finish=... ...s         # total request time since prompt start
```

For the 20 t/s target, compare the `decoding avg=... t/s` line. Do not compare
`completion_tokens / HTTP wall` and do not infer steady decode speed from the
`finish=...` request duration. On the previous no-MTP graph+SoA server log,
steady decode chunks were in the **17.4-17.9 t/s** band even when `finish`
included an additional **0.5-0.9 s** of prefill. A first real coding request can
still look like **~15-16 t/s** from a client if it includes graph/cache warmup,
a longer prompt prefill, chat/request overhead, or non-empty resident context.

Closing decision for this research pass:

- The quality-preserving server build to keep is `./ds4-server-exact-fast`.
  It is exact decode with graph decode and the default Q8 SoA cache, no MTP,
  no active-expert reduction, and no verifier shortcuts.
- The no-quality-loss target of **20 t/s decode** was not reached on this line.
  The honest steady server-decode band is **~17.4-17.9 t/s** on warm no-MTP
  graph+SoA logs, with client-observed end-to-end numbers often lower because
  they include prefill and request overhead.
- The MTP/verifier path produced useful diagnostics but is not the current
  production answer: all quality-preserving variants converged around the same
  **16.8-16.9 t/s** full-suite band or lower once measured on real coding
  tasks at `ctx=256000`.
- Do not spend more time on MTP orchestration knobs as the next step. The
  remaining path to 20 t/s without quality loss would need a real kernel-level
  improvement in exact decode, especially Q8 projection reads, routed MoE
  gate/up/down, HC expand, or a native layout that improves memory access
  without changing the executed experts or logits.
- Future measurements must report both phases separately:
  `prompt done` for prefill and `decoding avg` for decode. `finish` and HTTP
  wall time remain useful only as end-to-end latency checks.

Current quality-first MTP research candidate after the aggregate-stats pass:

```sh
DS4_CUDA_GRAPH_DECODE=1
DS4_CUDA_Q8_SOA_CACHE=1
DS4_CUDA_MOE_K2_DIRECT_GATE=1
DS4_MTP_STRICT=1
DS4_MTP_BATCH_VERIFY=1
DS4_MTP_UNSAFE_BATCH_VERIFY=1
DS4_MTP_BATCH_MARGIN_GUARD=0.25
DS4_MTP_BATCH_HC_PRE_EXACT=1
DS4_MTP_BATCH_ATTENTION_EXACT=1
DS4_MTP_BATCH_COMPRESS_PROJ_EXACT=1
DS4_MTP_BATCH_ROUTER_EXACT=1
DS4_MTP_CAPTURE_PREFIX1=1
DS4_MTP_CAPTURE_PREFIX1_MIN_MARGIN=2.0
DS4_MTP_DRAFT2_SKIP_MIN_MARGIN=2.0
```

Do not include `DS4_MTP_PREFIX_BATCH` in the current recommendation. Prefix2
and prefix3 are now measured as diagnostic-only paths.

Do not include `DS4_CUDA_GRAPH_VERIFY` in the current recommendation either.
It is now technically valid both without prefix1 capture and with prefix1
capture after async D2D copies plus verifier no-pre-sync capture, but neither
combination beat draft2-skip m2 full-suite repeats.

Quality facts to preserve:

- Layer shadow with the conservative island set: no HC diff above `1e-6`.
- Layer shadow with the reduced island set above: no HC diff above `1e-6` on
  the sampled coding prompt.
- Layer shadow with router-exact + K2 body: no HC diff above `1e-6` on the
  sampled `top_k_frequent` server prompt.
- Shadow logits with the conservative, reduced, and router-exact+K2 sets:
  argmax matched in sampled cycles; max logit drift stayed around `0.018-0.021`,
  with top margins usually much larger.
- 12-task coding eval at `ctx=256000` for the conservative set: candidate and
  no-MTP baseline both scored **10/12**; the two failures were byte-identical.
- 12-task coding eval at `ctx=256000` for reduced routed-exact guard1 and
  router-exact+K2 guards `1`, `0.5`, `0.25`, and `0.1` all scored **10/12**;
  the persistent failures remain `lru_cache` and `parse_csv_line`.
- Router-exact+K2 with safe GPU-top2 margin guard also scored **10/12** when
  it kept full logits; CPU/GPU top2 compare logged zero mismatches in the full
  suite.
- Router-exact+K2 with `guard0.25` plus
  `DS4_MTP_CAPTURE_PREFIX1_MIN_MARGIN=2.0` scored **10/12** in two full
  suites, with the same persistent `lru_cache` and `parse_csv_line` failures.
  Pure prefix1 capture scored **9/12** and must stay rejected.
- Adding `DS4_MTP_DRAFT2_SKIP_MIN_MARGIN=2.0` also scored **10/12** in two
  full suites, with the same persistent failures. This is quality-preserving
  because skipped second drafts are not emitted.
- `DS4_MTP_PREFIX_BATCH_MAX_TOKENS=2` and the prefix MTP margin gates scored
  **4/4** on canaries, but did not beat the no-prefix draft2-skip canary.
- `DS4_MTP_PREFIX_BATCH_MIN_MARGIN=3.0` for the original 3-token prefix shape
  also scored **4/4** on the canary, but was much slower and is rejected.
- `DS4_CUDA_GRAPH_VERIFY=1` without prefix1 capture scored **10/12** on the
  full suite with the same persistent failures, but was not faster than the
  best draft2-skip m2 repeats.
- `DS4_CUDA_GRAPH_VERIFY=1` with prefix1 capture is now technically valid after
  switching CUDA D2D tensor copies to async copies while capture is active, but
  it still did not beat draft2-skip m2.
- `DS4_CUDA_GRAPH_VERIFY=1` with prefix1 capture and verifier no-pre-sync
  capture scored **4/4** on the canary, but still did not beat draft2-skip m2.
- `DS4_CUDA_GRAPH_VERIFY=1` without prefix1 capture and with verifier
  no-pre-sync capture also scored **4/4** on the canary, but the gain over the
  previous no-prefix graph-verifier canary was only **13.65 t/s** vs
  **13.50 t/s**, so it is not worth a full 12-task run.
- `DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN=2` scored **4/4** on the canary, but
  was slower than the draft2-skip canary and kept hot full verifier cycles near
  **94-97 ms**.
- `DS4_CUDA_Q8_SOA_BATCH2_ATTN_OUTPUT_B=1` scored **4/4** on the canary and was
  slightly faster than the same-build draft2-skip canary (**13.65 t/s** vs
  **13.55 t/s**). It then scored **10/12** on two full evals with the same
  persistent failures; the warmed repeat measured **16.87 t/s**, effectively
  tied with but not better than draft2-skip.
- `DS4_CUDA_Q8_SOA_BATCH2=1` is rejected for now: first 4-task canary regressed
  `parse_duration` from pass to fail.
- Plain fast batch verifier is not acceptable: top margins can be high while
  logit drift is already in the `1-4` range.
- Removing routed exact without exact router is not acceptable: sampled row
  drift reached **2.73334** and layer-shadow worst diffs exceeded 5.
- Router-exact+K2 without guard is also not acceptable yet: it introduced an
  extra `flatten_dict` coding failure.
- Router-exact+K2 with GPU-top2 and no full logits is also not acceptable yet:
  it produced conflicting full-suite repeats, including one extra
  `flatten_dict` coding failure.

Speed facts to improve:

- Conservative guarded hot full-accept cycles are around **100 ms** for two
  verified target rows plus ~4 ms draft time.
- Reduced hot full-accept cycles are around **89 ms** verify time, about
  **94 ms** including draft, before server-workload acceptance effects.
- Router-exact+K2 guard0.25 improves real hot coding chunks into the **19-21
  t/s** band on some short tasks, but the full warmed suite sum is still
  **15.90 t/s** and server decode-only **16.78 t/s**, below the no-MTP baseline
  aggregate.
- Margin-gated prefix1 capture reduces average replay cost from **10.8 ms** to
  **4.35-5.57 ms** per MTP cycle and improved the first same-protocol run from
  **15.88 t/s** to **16.39-16.42 t/s**, with the same 10/12 task pass rate.
- Draft2 skip reduces the average cycle to **87.2-87.9 ms** and replay to
  **0.85-1.04 ms** by routing weak second drafts to an exact one-token path.
  Full-suite speed improved to **16.82-16.87 t/s** with the same 10/12 quality.
- Narrow `attn_output_b` SoA batch2 for the MTP verifier tied the draft2-skip
  speed band (**16.87 t/s** warmed) but did not beat it.
- Targeted `attn_output_b` F16/cuBLAS decode
  (`DS4_CUDA_ATTENTION_OUTPUT_B_CUBLAS_MIN=1`) did not improve the refreshed
  exact profile: `attn_output` was **41.902 ms** over 129 rows vs **41.728 ms**
  for the control, while the run filled about **10.58 GiB** of Q8 F16 cache.
- The warmed no-MTP graph+SoA coding comparison scored **10/12** at
  **16.83 t/s** suite-sum, with steadier server decode chunks around
  **17.4-17.9 t/s**.
- Verifier profiling shows the hot cost is **layers (~82-88 ms)**, not
  output-head/topk (~2.5 ms) or full-logit readback (~0.04 ms warm).
- CUDA Graph verifier no-pre-sync confirms the same bottleneck: hot
  `micro-full` verifier cycles still spend about **90-91 ms** inside verify
  even when graph capture itself is valid.
- Intrusive stage profiling of `tokens=2` verifier calls attributes about
  **30%** of stage time to routed MoE and **23%** to attention output
  projection. Within routed MoE, CUDA event profiling attributes about **70%**
  of that time to gate/up and **27%** to down.
- `DS4_CUDA_MOE_DECODE_GATE_NOAUX=1` stayed effectively neutral in warm
  verifier cycles and remains diagnostic-only.
- Guard-rejected cycles replay one exact token and were around **166 ms** in
  the sampled hot path.
- Real server coding throughput did not beat no-MTP baseline because the
  exact-islands and guard fallback eat the speculative win.

Anti-loop checkpoint:

- Stop spending cycles on MTP orchestration knobs unless a kernel-level change
  first lowers the hot `micro-full` verifier body. Prefix batching, graphing the
  verifier, pre-sync removal, partial-prefix replay tuning, draft2 skip, and
  narrow verifier-only output-projection tweaks have all converged to the same
  **16.8-16.9 t/s** full-suite band.
- The MTP line is now a quality-preserving research candidate, not the shortest
  route to 20 t/s. Keep it as a regression harness for exact kernel work, but
  do not promote new MTP env combinations based only on 4-task canaries.
- The next useful work must target exact decode kernels shared by no-MTP and
  the MTP verifier: Q8 projection reads that are not simple F16/cuBLAS swaps,
  routed MoE gate/up, HC expand, or a native weight layout that preserves qwarp
  lane-contiguous access. Do not remove an exact island based only on argmax
  agreement; require shadow logit deltas to remain small relative to top
  margins and re-run the coding eval at `ctx=256000`.

## 2026-05-22 continuation - rebase baseline, matrix, and fork scan

Branch state:

- Local branch `gx10-cuda-graph-decode` was rebased onto `origin/main` /
  `upstream/main` at `8d57664`.
- Current branch head during this pass: `dca0d8c`.
- A separate upstream baseline worktree was created at
  `/home/alessandro/projects/ds4-main-baseline`, also at `8d57664`.
- `ds4-server` and `ds4-agent` were built in that baseline worktree.

Manual agent baseline after the rebase:

| Build | Agent throughput | Interpretation |
| --- | ---: | --- |
| `origin/main` worktree at `8d57664` | 14.6 t/s | upstream-after-rebase baseline, including native `ds4-agent` |
| current branch, plain `ds4-agent` | 16.3 t/s | most GX10 CUDA optimizations are compiled in |
| current branch, `ds4-agent-exact-fast` | 16.7 t/s | current branch plus graph decode and default Q8 SoA cache |

Relative to the updated upstream worktree, the current branch is about
**+11.6%** for plain `ds4-agent` and **+14.4%** for `ds4-agent-exact-fast`.
This is the right comparison; comparing `ds4-agent` vs `ds4-agent-exact-fast`
inside the same optimized branch only measures the final graph+SoA env delta.

New local artifacts:

- `ds4-agent-exact-fast`: quality-preserving agent launcher mirroring
  `ds4-server-exact-fast`, with `DS4_CUDA_GRAPH_DECODE=1` and
  `DS4_CUDA_Q8_SOA_CACHE=1`; default context is now `100000` to match
  native `ds4-agent`.
- `docs/gx10_test_matrix.md`: executable matrix, row definitions, promotion
  rules, and systematic benchmark protocol.
- `docs/gx10_fork_recon.md`: fork reconnaissance and external idea shortlist.
- `docs/gx10_action_plan.md`: phased plan for measurement, exact-safe sweeps,
  tradeoff rows, and next kernel work.
- `tuning/gx10_matrix.py`: sanitized-env runner for row-level `bench`, `server`,
  `eval`, plus `bench-suite`, `eval-suite`, and `summary`.

Systematic benchmark commands established:

```sh
python3 tuning/gx10_matrix.py list
python3 tuning/gx10_matrix.py bench-suite core \
  --ctx-alloc 100000 --ctx-start 8192 --ctx-max 8192 --gen-tokens 128
python3 tuning/gx10_matrix.py summary
python3 tuning/gx10_matrix.py eval exact_fast --ctx 100000 --canary --repeat 1
```

Expanded exact sweep, only after the core pass is recorded:

```sh
python3 tuning/gx10_matrix.py bench-suite exact \
  --ctx-alloc 100000 --ctx-start 2048 --ctx-max 65536 \
  --gen-tokens 128
```

Promotion rule from this point forward:

1. Speed smoke must beat `exact_fast` by at least 1-2%.
2. Candidate must pass the coding canary with no new failures.
3. Candidate must pass repeated coding gate (`repeat=3`) before promotion.
4. Every executed benchmark must be appended to this file with command,
   context, artifact path, speed, quality result, decision, and next action.

Fork reconnaissance summary:

- GitHub returned 930 public forks; 66 were prefiltered by recent push,
  stars/forks, or non-`main` default branch.
- Relevant branch refs were fetched under `refs/remotes/scan/*` only.
- No external branch revealed a small quality-preserving env toggle missing
  from this branch.
- Most public work clusters into Apple M5/Metal prefill, ROCm/HIP backend work,
  MTP/speculative proof work, memory/distribution work, and steering/agent
  features.

Entrpi branch check:

- Source: `Entrpi/ds4:mmq-step-A-full-layer-graphs`.
- Interesting ideas: vendored llama.cpp `cuda/mmq` / `mmvq`, per-layer CUDA
  graph replay, VMM weight arena, and MTP proof harness.
- GB10 numbers committed in that branch do **not** exceed our current
  exact-fast path:

| Entrpi artifact | GB10 generation result |
| --- | ---: |
| `speed-bench/gb10_spark.csv`, `ctx=2048` | 14.15 t/s |
| `speed-bench/gb10_spark.csv`, `ctx=8192` | 13.74 t/s |
| `speed-bench/gb10_spark.csv`, `ctx=65536` | 11.70 t/s |
| `speed-bench/gb10_spark.csv`, sweep mean | 12.80 t/s |
| README short prompt headline | 15.81 t/s |
| README long prompt headline | 13.56 t/s |
| MTP microbench prime prompt, optimized MTP | 16.21 t/s |
| `gb10_exact_mtp.csv`, sweep mean | 12.62 t/s |

Decision: mine Entrpi for design ideas only. Do not integrate it into this
branch now. If revisited, do it as a dedicated port branch with the same
logprob/coding gates as `exact_fast`.

Action plan:

1. Run and record the `core` matrix at `ctx_alloc=100000`, `ctx=8192`,
   `gen_tokens=128`.
2. Run the expanded `exact` sweep only if core results are stable.
3. Reject existing env rows that fail to beat `exact_fast` by 1-2%.
4. Keep reduced-K and MTP rows as research/tradeoff rows only.
5. Start new kernel work only after current measurements are in this document.
   Priorities remain Q8 projection reads and full-K routed MoE gate/up/down.

### 2026-05-22 continuation - core and exact matrix results

Core matrix command:

```sh
python3 tuning/gx10_matrix.py bench-suite core \
  --ctx-alloc 100000 --ctx-start 8192 --ctx-max 8192 --gen-tokens 128
```

Core artifacts were copied to:

```text
tuning/gx10_matrix_results/core_20260522_1537/
```

Core result at `ctx_alloc=100000`, `ctx=8192`, `gen_tokens=128`:

| Row | Gen t/s | Decision |
| --- | ---: | --- |
| `plain` | 15.87 | branch baseline |
| `graph` | 15.72 | graph-only is not the speed source |
| `soa` | 16.07 | SoA is the useful exact-safe component |
| `exact_fast` | 16.05 | operational baseline; effectively tied with SoA in this CLI bench |

Expanded exact sweep command:

```sh
python3 tuning/gx10_matrix.py bench-suite exact \
  --ctx-alloc 100000 --ctx-start 2048 --ctx-max 65536 \
  --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/exact_20260522_1540 \
  --summary tuning/gx10_matrix_results/exact_20260522_1540/summary.csv \
  --markdown tuning/gx10_matrix_results/exact_20260522_1540/summary.md
```

The sweep produced 32 context points per row, from 2048 to 65536 in 2048-token
steps. Main artifact:

```text
tuning/gx10_matrix_results/exact_20260522_1540/summary.md
```

Exact sweep summary, sorted by mean generation throughput. Deltas are relative
to `exact_fast`.

| Row | Mean t/s | @8192 t/s | @65536 t/s | Mean delta |
| --- | ---: | ---: | ---: | ---: |
| `soa_b_forced` | 15.046 | 16.17 | 13.66 | +0.98% |
| `soa_shared` | 14.964 | 16.04 | 13.60 | +0.42% |
| `soa_qb` | 14.908 | 16.05 | 13.54 | +0.05% |
| `moe_h16` | 14.907 | 16.07 | 13.53 | +0.04% |
| `exact_fast` | 14.901 | 16.06 | 13.53 | baseline |
| `soa_qkv` | 14.869 | 16.04 | 13.47 | -0.21% |
| `moe_noaux` | 14.861 | 16.01 | 13.49 | -0.26% |
| `soa` | 14.856 | 15.98 | 13.51 | -0.30% |
| `attn_b_cublas_min1` | 14.853 | 16.00 | 13.48 | -0.32% |
| `output_top1` | 14.850 | 16.00 | 13.50 | -0.34% |
| `soa_cache_x` | 14.817 | 15.97 | 13.45 | -0.56% |
| `moe_fused_midq` | 14.746 | 15.87 | 13.43 | -1.04% |
| `moe_pair2` | 14.656 | 15.76 | 13.33 | -1.64% |
| `plain` | 14.653 | 15.78 | 13.32 | -1.66% |
| `graph` | 14.602 | 15.73 | 13.26 | -2.00% |

Decision:

- `soa_b_forced` is the only exact row that beat `exact_fast` across the sweep,
  but the gain is small: **+0.98% mean**, **+0.68% at 8192**, and **+0.96% at
  65536**. This is below the 1-2% promotion threshold. Do not promote it yet.
  If we want to chase this small default improvement, first repeat only
  `exact_fast` vs `soa_b_forced`, then run the canary coding gate if the repeat
  still clears at least 1%.
- `soa_shared` is not worth promoting: it is only **+0.42% mean**, slightly
  below `exact_fast` at 8192, and already had stored-logprob drift risk.
- `soa_qb`, `moe_h16`, and other near-baseline rows are noise-level results,
  not candidates.
- `graph` alone, `output_top1`, cuBLAS attention-output-B, broad SoA variants,
  and the existing MoE micro-toggles are confirmed neutral or negative in this
  post-rebase matrix.
- The matrix did not reveal an env-row path to full-quality 20 t/s.

Next action:

1. Optionally repeat `exact_fast` vs `soa_b_forced` if we want to pursue a
   sub-1% wrapper/default change.
2. Do not restart the broad native-pack work as originally phrased here. The
   roadmap already contains byte-verified routed packs plus real compute tests,
   and the naive row-major / block-paired layouts were slower once used by the
   actual dot-product kernels.
3. The remaining implementation work must be narrower: preserve the current
   qwarp lane-contiguous weight stream while reducing real memory traffic or
   doing a targeted attention-output/Q8 projection kernel redesign that is not
   another SoA/cuBLAS/cache-X toggle.

### 2026-05-22 continuation - proposal recheck and residency probes

The two proposed next directions were rechecked against the earlier roadmap.
This changed the recommendation:

- **Native routed-expert pack, as a broad item, was already explored.** The log
  already contains `DS4_CUDA_ROUTED_PACK_SMOKE`, row-major gate/down packs,
  gate/up block-paired packs, raw-read benches, and compute-equivalence
  benches. The important negative result is the real compute path: block-paired
  gate/up was byte-exact but **0.466x** the current separate layout, a K=6
  multi-expert shared-xq/LUT kernel was **0.681x**, and row-major down compute
  was **0.859x**. The lesson is not "pack experts"; it is "do not break the
  current qwarp block stream."
- **Targeted Q8 projection was also mostly explored at the toggle/layout level.**
  The branch already tested output top1, Q8 cuBLAS/F16, attention-output-B
  cuBLAS, activation cache-X, Q8 aligned padding, SoA A/B, SoA QB, SoA QKV,
  SoA shared, and SoA batch2 probes. The only promoted exact path remains
  default attention-output A/B SoA. A future Q8 projection attempt must be a
  new kernel shape, not another broad cache toggle.

One genuinely under-covered family remained: **model/weight residency and CUDA
allocation placement**. These probes were run at `ctx_alloc=100000`,
`ctx=8192`, `gen_tokens=128`, same prompt as the exact matrix.

Full model copy probe:

```sh
DS4_CUDA_COPY_MODEL=1 \
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
./ds4-bench --cuda -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 \
  --gen-tokens 128 \
  --csv tuning/gx10_matrix_results/model_residency_20260522/copy_model_exact_fast_bench.csv
```

Result:

- Startup copied **80.76 GiB** of model image to device memory and took
  **457.539s**.
- Benchmark result: **15.75 t/s**, prefill **223.08 t/s**.
- Adjacent/default exact-fast controls were around **16.05-16.06 t/s** on the
  same primitive bench.

Decision: reject. A single device-owned full-image base is slower at steady
decode and has unacceptable startup cost. `DS4_CUDA_COPY_MODEL_CHUNKED` was not
run after this because it changes upload mechanics, not the steady-state
device-owned pointer layout that already failed the speed gate.

Arena placement probe:

```sh
DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=<chunk> \
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
./ds4-bench --cuda -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 \
  --gen-tokens 128 \
  --csv tuning/gx10_matrix_results/arena_20260522/<chunk>_exact_fast_bench.csv
```

Results:

| Weight arena chunk | Gen t/s | Result |
| --- | ---: | --- |
| default | 16.06 | control |
| 1024 MiB | n/a | failed startup cache at tensor span 116 with OOM |
| 4096 MiB | 15.80 | slower |
| 8192 MiB | 15.88 | slower |

Decision: keep the default arena chunk. The default placement is already better
than larger arenas, and smaller arenas can fail the complete 80.76 GiB cache.

Direct/no-FD residency probes:

| Probe | Result | Decision |
| --- | --- | --- |
| `DS4_CUDA_DIRECT_MODEL=1` with graph+SoA | no usable CSV; after a long run, decode failed during CUDA graph capture with `attention_output_low_q8_rope` reporting a previous capture error | reject for exact-fast graph path |
| `DS4_CUDA_NO_FD_CACHE=1` with graph+SoA | no CSV; did not finish within a 300s timeout while preparing/running the same 8192/128 bench | reject; worse operationally before any speed gate |

Overall decision from the recheck:

- The earlier "native routed-expert pack" proposal was too broad and repeated
  work already documented in this file.
- The residency/allocation probes close another tempting outside-the-box family:
  full model copy, arena chunk tuning, direct model pointers, and non-FD cache
  placement do not move toward 20 t/s.
- The remaining credible full-quality path is new kernel work with a stricter
  design target: keep exact logits, keep full K=6, keep memory under control,
  preserve the qwarp-friendly streams that existing kernels depend on, and cut
  actual per-token weight traffic or per-row work in the hot routed-MoE and
  attention-output/Q8 projection kernels.

### 2026-05-22 continuation - quality gates and timebox

The next full-quality push will use `ds4-eval` as an explicit capability gate
in addition to deterministic logprob checks and the coding eval. `ds4-eval` is
not a leaderboard score, but it is useful here because it exercises GPQA
Diamond, audited SuperGPQA, AIME2025, and COMPSEC prompts through the same local
inference path.

Runner support was added to `tuning/gx10_matrix.py`:

```sh
python3 tuning/gx10_matrix.py ds-eval exact_fast \
  --questions 4 --tokens 1024 --nothink --seed 1

python3 tuning/gx10_matrix.py ds-eval-suite exact_fast <candidate-row> \
  --questions 12 --tokens 4096 --think --seed 1

python3 tuning/gx10_matrix.py ds-eval-suite exact_fast <candidate-row> \
  --questions 92 --tokens 16000 --think --seed 1
```

Promotion gates from this point:

1. Speed smoke: candidate must beat `exact_fast` by at least **+3%** on the
   8192/128 primitive bench before expensive quality runs.
2. Deterministic exactness: for kernels intended to be exact, logprob/token
   output must not diverge.
3. Coding canary: no new failures versus same-run `exact_fast`.
4. `ds4-eval` smoke: 12 questions, thinking mode, no pass-count regression
   versus same-run `exact_fast`.
5. Long gate: full 92-question `ds4-eval` only after the candidate passes all
   earlier gates; no pass-count regression versus same-run `exact_fast`.

Best-effort stop condition:

- Try at most two targeted kernel prototypes, or about one short working day of
  implementation/profiling.
- Stop earlier if neither prototype clears the +3% speed smoke.
- Park any faster prototype that fails deterministic exactness, coding canary,
  or `ds4-eval` smoke.
- Do not spend the long 92-question `ds4-eval` run on rows that have not already
  passed the cheaper filters.

Baseline smoke executed:

```sh
python3 tuning/gx10_matrix.py ds-eval exact_fast \
  --questions 4 --tokens 1024 --nothink --seed 1 --timeout-sec 900
```

Result:

- Artifact: `tuning/gx10_matrix_results/exact_fast_ds4_eval.txt`
- Context auto-sized to **1337 tokens**.
- Result: **4/4 passed**.
- Runtime: under one minute after the model cache startup.

### 2026-05-22 continuation - exact MoE metadata-cache prototypes

After rechecking the roadmap, the next exact kernel attempt deliberately avoided
the already-failed broad routed-pack, H16/noaux/pair2/fused-midq, row4 down, and
Q8 toggle families. The tested idea was narrower: keep the current
qwarp-friendly weight stream and only remove redundant tiny metadata loads.

Two opt-in diagnostic flags were added:

| Row | Env | Intent |
| --- | --- | --- |
| `moe_down_meta_cache` | `DS4_CUDA_MOE_DOWN_SUM6_META_CACHE=1` | Load the six selected experts once into shared memory in the exact K=6 down-sum kernel. |
| `moe_gate_weight_cache` | `DS4_CUDA_MOE_DECODE_GATE_WEIGHT_CACHE=1` | Load the route weight once into shared memory per gate/up CTA instead of reloading it for every output row. |
| `moe_meta_cache` | both flags | Combined diagnostic row. |

Build and smoke:

```sh
make -j$(nproc) cuda-spark

python3 tuning/gx10_matrix.py bench-suite exact_fast moe_down_meta_cache \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260522_moe_meta \
  --summary tuning/gx10_matrix_results/prototype_20260522_moe_meta/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260522_moe_meta/summary.md \
  --stop-on-fail

python3 tuning/gx10_matrix.py bench-suite exact_fast moe_gate_weight_cache moe_meta_cache \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260522_moe_weight_cache \
  --summary tuning/gx10_matrix_results/prototype_20260522_moe_weight_cache/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260522_moe_weight_cache/summary.md \
  --stop-on-fail
```

Results:

| Row | 8192/128 gen t/s | Same-run control | Decision |
| --- | ---: | ---: | --- |
| `moe_down_meta_cache` | **16.14** | 16.05 | Tiny/noisy +0.6%; below +3% speed gate. Do not promote. |
| `moe_gate_weight_cache` | **15.93** | 16.12 | Negative. Do not promote. |
| `moe_meta_cache` | **15.99** | 16.12 | Negative. Do not promote. |

Takeaway: this exact metadata-cache family is closed for now. The existing MoE
kernels are not bottlenecked by selected-expert or route-weight scalar loads;
the extra shared memory/synchronization is neutral to negative. The flags remain
as opt-in diagnostics so the result is reproducible, but they should not be
included in `ds4-agent-exact-fast` or any quality-gate suite unless the kernel
body changes substantially.

### 2026-05-22 continuation - IQ2/Q2 dot codegen probe

The next non-layout probe targeted the real gate/up dot body instead of metadata:
`dev_dot_iq2_xxs_q8_K_block_lut` was temporarily rewritten to force inlining,
unroll the fixed 8-step block loop, and replace the local `w[8]` array with
explicit scalar temporaries. A matching force-inline/unroll cleanup was then
tried for the Q2 down dot.

This was exact in intent and did not change model weights, active experts,
kernel layout, or accumulation order at source level. It was still treated as a
quality-risk candidate because CUDA codegen/register allocation can move enough
floating-point detail to change generation.

Speed smoke looked tempting:

| Build | Artifact | 8192/128 gen t/s |
| --- | --- | ---: |
| IQ2 codegen run 1 | `prototype_20260522_iq2_codegen/exact_fast_bench.csv` | **16.28** |
| IQ2 codegen run 2 | `prototype_20260522_iq2_codegen/exact_fast_repeat2_bench.csv` | **16.44** |
| IQ2+Q2 codegen | `prototype_20260522_iq2_q2_codegen/exact_fast_bench.csv` | **16.28** |

But the cheap `ds4-eval` gate caught a regression:

```sh
python3 tuning/gx10_matrix.py ds-eval exact_fast \
  --questions 4 --tokens 1024 --nothink --seed 1 --timeout-sec 900 \
  --label exact_fast_codegen_smoke \
  --out-dir tuning/gx10_matrix_results/prototype_20260522_iq2_q2_codegen
```

Result: **3/4 passed**. The SuperGPQA grass-pellet case changed from the prior
correct `C` answer to `G`.

The codegen changes were then removed and the same smoke was repeated:

```sh
python3 tuning/gx10_matrix.py ds-eval exact_fast \
  --questions 4 --tokens 1024 --nothink --seed 1 --timeout-sec 900 \
  --label exact_fast_after_codegen_revert_smoke \
  --out-dir tuning/gx10_matrix_results/prototype_20260522_iq2_q2_codegen
```

Result after revert: **4/4 passed**. A post-revert 8192/128 speed smoke measured
**15.96 t/s**, consistent with the previous exact-fast noise band.

Decision: reject the IQ2/Q2 codegen rewrite despite the apparent +1-2% speed
signal. It is a useful negative result: source-level "same order" is not enough
for this target; any codegen-level dot change needs deterministic logprob/coding
checks immediately, before chasing speed.

### 2026-05-22 continuation - attention-output-A half-warp probe

Because the MoE history already rules out many full-K micro-shapes, the next
probe deliberately avoided routed MoE. The candidate targeted
`attn_output_a`, whose promoted path is the SoA Q8 grouped projection. The idea
was to use 16 lanes per low-row instead of 32 lanes, doubling rows per CTA and
reducing CTA count while keeping the same SoA weight layout.

Implemented diagnostic row:

| Row | Env | Note |
| --- | --- | --- |
| `attn_a_hwarp16` | `DS4_CUDA_ATTENTION_OUTPUT_A_HWARP16=1` | Uses a half-warp grouped `attn_output_a` SoA kernel. |

This is not source-order identical to the default full-warp kernel: each row's
dot-product blocks are partitioned across 16 lanes instead of 32, so the
floating-point reduction order changes. For that reason the speed gate had to be
clearly positive before spending any quality budget.

Smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast attn_a_hwarp16 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260522_attn_a_hwarp16 \
  --summary tuning/gx10_matrix_results/prototype_20260522_attn_a_hwarp16/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260522_attn_a_hwarp16/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.16** | control |
| `attn_a_hwarp16` | **16.08** | negative; no quality gate |

Decision: reject the half-warp attention-output-A variant. It is a useful
non-MoE negative: reducing CTA count is not enough, and changing the reduction
shape is not justified without a speed win.

### 2026-05-23 continuation - corrected exact-fast profile and HC-expand SoA probe

After the upstream refresh and branch rebase, the current exact-fast path was
profiled again before opening a new kernel direction.

Built-in stage/MoE profile command:

```sh
DS4_CUDA_Q8_SOA_CACHE=1 \
DS4_METAL_DECODE_STAGE_PROFILE=1 \
DS4_CUDA_MOE_PROFILE=1 \
./ds4 --cuda -m ds4flash.gguf --ctx 100000 --nothink --temp 0 -n 4 \
  -p "Write a Python function merge_intervals(intervals) that merges overlapping intervals."
```

`DS4_CUDA_GRAPH_DECODE=1` is intentionally absent here because the MoE profiler
uses CUDA events and cannot run inside graph capture.

Fresh decode stage totals:

| Stage | Total | Per generated token |
| --- | ---: | ---: |
| `routed_moe` | 48.056 ms | 16.019 ms |
| `attn_output` | 41.359 ms | 13.786 ms |
| `q_path` | 27.290 ms | 9.097 ms |
| `shared_gate_up` | 11.766 ms | 3.922 ms |
| `compressor_indexer` | 10.797 ms | 3.599 ms |
| `attention` | 8.380 ms | 2.793 ms |
| `shared_down` | 6.671 ms | 2.224 ms |

Filtered decode-only MoE split (`tokens=1` lines only):

```text
calls=129 pairs=774
xq=0.610 sort=0.130 gateup=30.937 midq=0.541 down=14.534 sum=0.129 total=46.766 ms
per generated token: gateup=10.312 ms, down=4.845 ms, total=15.589 ms
```

This matches the earlier profile: routed MoE gate/up, attention output, and
`attn_q_b`/Q path are still the real decode budget.

A corrected Nsight Systems decode-node trace was then captured:

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_CUDA_Q8_SOA_CACHE=1 \
nsys profile --trace=cuda --cuda-graph-trace=node \
  --output=tuning/gx10_matrix_results/profile_20260523_exact_fast/exact_fast_graph_soa_node_decode \
  --force-overwrite=true \
  ./ds4 --cuda -m ds4flash.gguf --ctx 100000 --nothink --temp 0 -n 64 \
    -p "Write a Python function merge_intervals(intervals) that merges overlapping intervals."
```

Important tooling note: Nsight's default CUDA graph trace granularity is
`graph`; that hides decode child kernels and mostly shows preload/prefill work.
Use `--cuda-graph-trace=node` for kernel-level decode attribution.

Top exact-fast decode-node kernels:

| Kernel | Time share | Total | Instances | Median |
| --- | ---: | ---: | ---: | ---: |
| `moe_gate_up_mid_decode_lut_qwarp32_kernel` | 16.1% | 633.659 ms | 2709 | 231.680 us |
| `matmul_q8_0_hc_expand_preq_warp8_soa_kernel` | 10.8% | 423.387 ms | 2709 | 155.712 us |
| `matmul_q8_0_preq_batch_warp8_kernel` | 10.8% | 422.425 ms | 2709 | 155.808 us |
| `grouped_q8_0_a_preq_warp8_soa_kernel` | 10.5% | 411.996 ms | 2709 | 151.488 us |
| `moe_down_sum6_qwarp32_kernel` | 7.5% | 296.216 ms | 2709 | 108.768 us |
| `matmul_q8_0_pair_swiglu_preq_warp8_kernel` | 5.7% | 224.497 ms | 2709 | 82.592 us |
| `attention_decode_mixed_kernel` | 4.4% | 171.723 ms | 2709 | 57.152 us |
| `matmul_q8_0_hc_expand_preq_warp8_kernel` | 3.0% | 119.195 ms | 2709 | 43.552 us |

The un-SoA `hc_expand` tail suggested one narrow, not-yet-isolated probe:
cache and route only HC-expand Q8 tensors beyond the default attention-output
A/B cache, without enabling broad `DS4_CUDA_Q8_SOA_SHARED=1`.

Implemented diagnostic row:

| Row | Env | Note |
| --- | --- | --- |
| `soa_hc_expand` | `DS4_CUDA_Q8_SOA_HC_EXPAND=1` | Allows SoA cache/routing for `*_hc_expand`; preloads `ffn_down_shexp` so graph capture does not allocate at decode time. |

The first implementation allowed the runtime label `shared_down_hc_expand` but
not the preload label `ffn_down_shexp`, causing CUDA graph capture to fail with
`operation failed due to a previous error during capture`. The flag now also
permits the original shared-down preload label.

Speed smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast soa_hc_expand \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_soa_hc_expand_retry \
  --summary tuning/gx10_matrix_results/prototype_20260523_soa_hc_expand_retry/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_soa_hc_expand_retry/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.06** | control |
| `soa_hc_expand` | **16.01** | negative; no quality gate |

Decision: reject `soa_hc_expand` for promotion. The remaining interleaved
HC-expand tail is too small, and the extra cache/routing does not improve
end-to-end decode. Keep the flag as a diagnostic in the research branch only.

Because the corrected node trace also showed `attn_q_b` as the largest
remaining non-SoA Q8 projection, the existing `soa_qb` row was rechecked once
on the current post-upstream branch rather than assumed from old data:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast soa_qb \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/recheck_20260523_soa_qb \
  --summary tuning/gx10_matrix_results/recheck_20260523_soa_qb/summary.csv \
  --markdown tuning/gx10_matrix_results/recheck_20260523_soa_qb/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.06** | control |
| `soa_qb` | **16.00** | negative; confirms prior neutral/negative full-decode result |

Decision: do not keep chasing `attn_q_b` through the existing SoA toggle. Any
future `attn_q_b` attempt must be a new kernel shape with a clear speed gate,
not another cache-selection rerun.

### 2026-05-23 continuation - non-repeating hot-kernel probes

The next pass stayed on the corrected decode-node profile and tested only
paths that were not already covered by the earlier MoE pack/H16/no-aux,
Q8 alignment, broad SoA, F16/cuBLAS, and cache-x experiments.

#### `attn_q_b` kernel-shape variants

`attn_q_b` remained visible as `matmul_q8_0_preq_batch_warp8_kernel` in the
node trace. Three narrow variants were tried:

| Row | Env | Intent |
| --- | --- | --- |
| `attn_qb_hwarp16` | `DS4_CUDA_ATTN_Q_B_HWARP16=1` | half-warp CTA shape for the interleaved Q8 path |
| `attn_qb_soa_hwarp16` | `DS4_CUDA_Q8_SOA_QB=1 DS4_CUDA_ATTN_Q_B_HWARP16=1` | same shape on a SoA Q/B cache |
| `attn_qb_b32_special` | `DS4_CUDA_ATTN_Q_B_B32_SPECIAL=1` | exact-order specialization for the decode shape `in=1024,out=32768,blocks=32` |

Speed smokes:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast attn_qb_hwarp16 attn_qb_soa_hwarp16 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_attn_qb_hwarp16 \
  --summary tuning/gx10_matrix_results/prototype_20260523_attn_qb_hwarp16/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_attn_qb_hwarp16/summary.md \
  --stop-on-fail

python3 tuning/gx10_matrix.py bench-suite exact_fast attn_qb_b32_special \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_attn_qb_b32_special \
  --summary tuning/gx10_matrix_results/prototype_20260523_attn_qb_b32_special/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_attn_qb_b32_special/summary.md \
  --stop-on-fail
```

Results:

| Row | Same-run control | 8192/128 gen t/s | Decision |
| --- | ---: | ---: | --- |
| `attn_qb_hwarp16` | 16.13 | **15.92** | negative; no quality gate |
| `attn_qb_soa_hwarp16` | 16.13 | **16.03** | negative; no quality gate |
| `attn_qb_b32_special` | 16.12 | **16.07** | negative; no quality gate |

Decision: reject this family for promotion. The half-warp variants also change
reduction shape; the exact-order `blocks=32` specialization avoided that issue
but still did not beat the generic warp8 kernel.

#### HC-expand exact-shape specialization

The SoA HC-expand tail probe was negative, so the next HC-expand attempt avoided
extra residency and instead specialized the exact decode shape where
`n_hc == 4` and `out_dim == n_embd`.

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast hc_expand_nhc4_special \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_hc_expand_nhc4_special \
  --summary tuning/gx10_matrix_results/prototype_20260523_hc_expand_nhc4_special/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_hc_expand_nhc4_special/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.14** | control |
| `hc_expand_nhc4_special` | **16.11** | negative; no quality gate |

Decision: reject. The specialized store/indexing path did not move end-to-end
decode enough to justify carrying it forward.

#### MoE LUT/template variants

Because the largest decode-node kernel remained
`moe_gate_up_mid_decode_lut_qwarp32_kernel`, two small variants were tested
without repeating the already-failed native packs, H16, MoE no-aux, pair2,
fused-midq, metadata-cache, row4, or parallel-down paths:

| Row | Env | Intent |
| --- | --- | --- |
| `moe_span128_template` | `DS4_CUDA_MOE_DECODE_GATE_SPAN128_TEMPLATE=1` | force an explicit `span<128>` template instead of relying on the existing branch |
| `moe_global_lut` | `DS4_CUDA_MOE_DECODE_GATE_GLOBAL_LUT=1` | avoid per-CTA shared IQ2 LUT copies and read the constant/global LUTs directly |

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast moe_span128_template moe_global_lut \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_lut_micro \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_lut_micro/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_lut_micro/summary.md \
  --stop-on-fail
```

Results:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.16** | control |
| `moe_span128_template` | **16.00** | negative; no quality gate |
| `moe_global_lut` | **13.89** | strongly negative; no quality gate |

Decision: reject. The explicit span template did not help, and keeping the IQ2
LUTs in shared memory is clearly better than reading the global tables in the
hot MoE gate/up loop.

#### Shared expert no-aux write probe

The earlier no-aux result applied to routed MoE gate/up. A separate
shared-expert probe was still worth one cheap smoke because the normal shared
path consumes only `shared_mid`; `shared_gate` and `shared_up` are mainly
debug-visible outputs. The flag keeps the same dot products and SwiGLU formula
but skips writing the auxiliary gate/up tensors:

| Row | Env | Intent |
| --- | --- | --- |
| `shared_gate_up_noaux` | `DS4_CUDA_SHARED_GATE_UP_NOAUX=1` | compute only `shared_mid` in the fused shared expert gate/up kernel |

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast shared_gate_up_noaux \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_noaux \
  --summary tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_noaux/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_noaux/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.15** | control |
| `shared_gate_up_noaux` | **16.01** | negative; no quality gate |

Decision: reject. Skipping two small float stores per row does not offset the
same dot-product work and does not help full decode.

#### HC-expand auxiliary `block_out` write removal

The fused attention-output-B HC-expand path writes both `after_attn_hc` and the
intermediate `attn_out` vector. In the normal no-directional-steering path,
`attn_out` is only debug-visible after the fused kernel. A diagnostic
`DS4_CUDA_HC_EXPAND_NO_BLOCK_OUT=1` variant was added for the no-add, `n_hc=4`
shape so the kernel writes only `after_attn_hc`.

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast hc_expand_no_block_out \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_hc_expand_no_block_out \
  --summary tuning/gx10_matrix_results/prototype_20260523_hc_expand_no_block_out/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_hc_expand_no_block_out/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.17** | control |
| `hc_expand_no_block_out` | **16.09** | negative; no quality gate |

Decision: reject. Removing the auxiliary vector store does not matter while the
dot-product and weight stream dominate.

#### Routed MoE gate/up register-pressure probe

The dominant routed MoE gate/up kernel uses 64 registers/thread. A first
attempt to force `__launch_bounds__(256,5)` was invalid on the GB10 target:
ptxas reported the requested threads per SM out of range and ignored the
constraint. That is recorded here so it is not repeated.

A second diagnostic variant used `__maxnreg__(48)`:

| Row | Env | Resource delta |
| --- | --- | --- |
| `moe_gate_maxr48` | `DS4_CUDA_MOE_DECODE_GATE_MAXR48=1` | `REG:64 STACK:0` -> `REG:48 STACK:16` for `moe_gate_up_mid_decode_lut_qwarp32` |

Speed smokes:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast moe_gate_maxr48 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_gate_maxr48 \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_gate_maxr48/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_gate_maxr48/summary.md \
  --stop-on-fail

python3 tuning/gx10_matrix.py bench-suite exact_fast moe_gate_maxr48 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_gate_maxr48_r2 \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_gate_maxr48_r2/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_gate_maxr48_r2/summary.md \
  --stop-on-fail
```

Results:

| Run | `exact_fast` | `moe_gate_maxr48` | Decision |
| --- | ---: | ---: | --- |
| r1 | 16.09 | **16.18** | small positive, below gate |
| r2 | 16.10 | **16.03** | negative |

Decision: reject for promotion. Register capping is not a stable speed win and
introduces stack traffic; it does not deserve quality runs.

#### Long-context indexer top-k chunking

Because the MoE/Q8/MTP histories already rule out the obvious micro-shapes, the
next exact probe targeted the ratio-4 compressor/indexer top-k path at a long
frontier. This is not expected to close the whole 20 t/s gap, but it is a real
decode component at `ctx=100000` and had not yet been tested as a chunk-size
A/B.

The default chunked top-k path sorts 4096 score rows per chunk, then merges
`n_chunks * 512` candidates. A diagnostic
`DS4_CUDA_TOPK_CHUNK8192=1` variant was added to use 8192-row chunks when
`n_comp < 65535`, reducing the number of chunks and final candidates for the
100k/250k compressed-context regime. The first direct `uint32_t` 8192 variant
did not compile on GB10:

```text
ptxas error: uses too much shared data (0x10000 bytes, 0xc000 max)
```

The tested variant therefore stores chunk-local candidate indices as `uint16_t`
in shared memory and writes the usual `uint32_t` candidate scratch output.

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast indexer_topk_chunk8192 \
  --ctx-start 65536 --ctx-max 65536 --ctx-alloc 100000 --gen-tokens 64 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_indexer_topk_chunk8192 \
  --summary tuning/gx10_matrix_results/prototype_20260523_indexer_topk_chunk8192/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_indexer_topk_chunk8192/summary.md \
  --stop-on-fail
```

Result:

| Row | 65536/64 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **13.46** | 337.40 | control |
| `indexer_topk_chunk8192` | **13.36** | 333.03 | negative; no quality gate |

Decision: reject for promotion. The larger chunk reduces merge width but makes
the per-chunk sort heavier enough to lose decode throughput. Do not continue
with chunk-size-only top-k variants unless a later indexer redesign changes the
sort/merge algorithm.

#### Shared FFN scheduling probes

The next exact-safe probe targeted ordering and overlap inside the decode FFN
section. This was deliberately different from the earlier shared-expert
aux-write and MoE-kernel shape probes: the math kernels stayed full K=6 and
quality-preserving, while only the scheduling around shared gate/up and routed
MoE changed.

Two diagnostic rows were added:

- `DS4_CUDA_FFN_PARALLEL_SHARED=1`: launches shared gate/up/SwiGLU on a second
  non-blocking CUDA stream after `ffn_norm`, runs router/routed MoE on the main
  stream, then waits before shared-down.
- `DS4_CUDA_FFN_SHARED_FIRST=1`: runs shared gate/up/SwiGLU immediately after
  `ffn_norm`, before router/routed MoE, to test whether cache/wavefront locality
  improves without using a second stream.

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast ffn_parallel_shared \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_ffn_parallel_shared \
  --summary tuning/gx10_matrix_results/prototype_20260523_ffn_parallel_shared/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_ffn_parallel_shared/summary.md \
  --stop-on-fail

python3 tuning/gx10_matrix.py bench-suite exact_fast ffn_shared_first \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_ffn_shared_first \
  --summary tuning/gx10_matrix_results/prototype_20260523_ffn_shared_first/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_ffn_shared_first/summary.md \
  --stop-on-fail
```

Results:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` control for parallel run | **16.14** | 393.98 | control |
| `ffn_parallel_shared` | **15.26** | 391.04 | negative |
| `exact_fast` control for shared-first run | **16.14** | 391.39 | control |
| `ffn_shared_first` | failed | n/a | graph-capture unsafe |

The shared-first row failed during decode graph capture with:

```text
ds4: CUDA synchronize failed: operation not permitted when stream is capturing
ds4: Metal synchronize after graph eval failure also failed
ds4-bench: decode at frontier 8192 failed: cuda decode failed
```

Decision: reject both rows. The second stream does not expose useful overlap on
GB10; the added event and memory scheduling pressure loses roughly 5.5% versus
the same-run control. The shared-first ordering is not graph-capture safe in the
current fused path. Do not spend more time on FFN ordering/overlap unless a
future kernel redesign removes the capture-time synchronization or changes the
shared/routed dependency graph.

#### Normal decode graph no-pre-sync

The MTP verifier history had already tested `ds4_gpu_graph_capture_begin_no_sync`
for verifier graph capture, but the normal decode graph path still used
`ds4_gpu_graph_capture_begin()` and therefore paid a pre-capture
`cudaDeviceSynchronize()`. Since normal graph decode synchronizes at the end of
each token before logits readback, a narrow exact-safe probe added
`DS4_CUDA_GRAPH_DECODE_NO_SYNC=1` to use the no-pre-sync capture begin only for
normal decode graph capture.

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast graph_no_presync \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_graph_no_presync \
  --summary tuning/gx10_matrix_results/prototype_20260523_graph_no_presync/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_graph_no_presync/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **16.18** | 392.54 | control |
| `graph_no_presync` | **16.03** | 391.36 | negative; no quality gate |

Decision: reject for promotion. The pre-capture sync is not the limiting cost
for normal decode graph mode on this path, and removing it does not expose
hidden overlap.

#### Weight tensor 2 MiB alignment

Entrpi's CUDA/MMQ branch notes that per-tensor 2 MiB virtual placement can
matter for its MMQ/VMM path, but that full implementation is invasive and has a
documented reduction-order caveat. As a small exact-fast-compatible probe, the
local CUDA fd-cache arena gained `DS4_CUDA_WEIGHT_TENSOR_ALIGN_MB=2`, which
keeps the existing allocator and bytes but aligns each cached tensor base within
the device arena to 2 MiB.

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast weight_tensor_align2m \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_weight_tensor_align2m \
  --summary tuning/gx10_matrix_results/prototype_20260523_weight_tensor_align2m/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_weight_tensor_align2m/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **16.04** | 392.32 | control |
| `weight_tensor_align2m` | **15.95** | 391.36 | negative; no quality gate |

Decision: reject for promotion. The simplified placement idea does not transfer
into the current native warp8/qwarp kernels. Do not continue weight placement
padding inside this allocator; a real MMQ/VMM evaluation should be isolated on a
dedicated port branch.

#### Q8 batch1 cached-x projection path

`attn_q_b` remains visible in the refreshed profile as
`matmul_q8_0_preq_batch_warp8_kernel` for the `n_tok=1, blocks=32` Q8 decode
shape. A narrow exact-order probe added `DS4_CUDA_Q8_BATCH1_CACHE_X=1` to route
that shape through the existing warp8 `cached_x` kernel, copying the one-token
`xq`/scale blocks into shared memory once per CTA before the eight output-row
warps consume them.

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast q8_batch1_cache_x \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_q8_batch1_cache_x \
  --summary tuning/gx10_matrix_results/prototype_20260523_q8_batch1_cache_x/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_q8_batch1_cache_x/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **16.13** | 393.16 | control |
| `q8_batch1_cache_x` | **16.09** | 390.52 | negative; no quality gate |

Decision: reject for promotion. The extra shared-memory staging does not pay
for the one-token small-block projection shape, matching the broader lesson from
the previous cache-x experiments.

#### Default sampler probability cache

The user's agent measurements include default sampled decoding, not only the
greedy `ds4-bench` path. The default sampler uses `temperature=1`,
`top_p=1`, `min_p=0.05`, scans the full vocabulary to compute probabilities,
and previously recomputed `expf` during the final sampling pass. A CPU-side
diagnostic flag, `DS4_SAMPLE_CACHE_PROBS=1`, now stores the first pass
probabilities in a thread-local buffer and reuses them for the random draw.

This is exact for a fixed seed because it samples from the same stored
single-precision probabilities the first pass computed.

```sh
python3 tuning/gx10_matrix.py run exact_fast bash -lc './ds4 --cuda \
  -m ds4flash.gguf --ctx 100000 --tokens 128 --seed 1 \
  --temp 1 --top-p 1 --min-p 0.05 --nothink \
  -p "Write a long technical explanation of CUDA optimization strategies in 20 paragraphs." \
  > tuning/gx10_matrix_results/prototype_20260523_sample_cache_probs/exact_fast.out \
  2> tuning/gx10_matrix_results/prototype_20260523_sample_cache_probs/exact_fast.log'

python3 tuning/gx10_matrix.py run sample_cache_probs bash -lc './ds4 --cuda \
  -m ds4flash.gguf --ctx 100000 --tokens 128 --seed 1 \
  --temp 1 --top-p 1 --min-p 0.05 --nothink \
  -p "Write a long technical explanation of CUDA optimization strategies in 20 paragraphs." \
  > tuning/gx10_matrix_results/prototype_20260523_sample_cache_probs/sample_cache_probs.out \
  2> tuning/gx10_matrix_results/prototype_20260523_sample_cache_probs/sample_cache_probs.log'
```

Result:

| Row | CLI sampled generation t/s | Output compare | Decision |
| --- | ---: | --- | --- |
| `exact_fast` | **18.08** | control | control |
| `sample_cache_probs` | **18.08** | `cmp=0` | neutral; no promotion |

Decision: leave diagnostic-only. The default agent sampling CPU path is not the
current throughput limiter.

#### MoE read-only-cache load probes

The next exact idea did not repeat the native pack or row-span history. It kept
the current qwarp-friendly memory stream and changed only model-weight load
opcodes in the hot routed MoE dot products:

| Row | Env delta | Purpose |
| --- | --- | --- |
| `moe_gate_ldg` | `DS4_CUDA_MOE_DECODE_GATE_LDG=1` | use `__ldg` for routed gate/up IQ2 weight fields |
| `moe_down_ldg` | `DS4_CUDA_MOE_DOWN_SUM6_LDG=1` | use `__ldg` for routed down Q2 weight fields |
| `moe_ldg_weights` | both flags | combined read-only-cache probe |

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast moe_gate_ldg moe_down_ldg moe_ldg_weights \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_ldg_weights \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_ldg_weights/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_ldg_weights/summary.md \
  --stop-on-fail
```

Result:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **16.15** | 394.01 | control |
| `moe_gate_ldg` | **15.93** | 391.37 | negative |
| `moe_down_ldg` | **15.96** | 389.57 | negative |
| `moe_ldg_weights` | **15.86** | 389.23 | negative |

Decision: reject read-only-cache load variants for promotion. On GB10 these
loads do not beat the normal model-cache path for the current kernels.

#### DS4 shape-specialized MoE kernels

The shape-specialization probe was different from H16/noaux/span/pair2. It kept
the same lane assignment, the same qwarp reduction order, and the same model
layout, but removed runtime genericity for the actual DS4 decode shapes:

- routed gate/up: `n_tokens=1`, K=6, `xq_blocks=16`, `expert_mid_dim=2048`;
- routed down: K=6, `midq_blocks=8`, `out_dim=4096`;
- shared gate/up: Q8 `4096 -> 2048`, 128 Q8_0 blocks.

Rows added:

| Row | Env delta |
| --- | --- |
| `moe_gate_shape2048` | `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1` |
| `moe_down_shape4096` | `DS4_CUDA_MOE_DOWN_SUM6_SHAPE4096=1` |
| `moe_shape_special` | both routed MoE shape flags |
| `shared_gate_up_shape2048` | `DS4_CUDA_SHARED_GATE_UP_SHAPE2048=1` |
| `shape_gate_shared` | routed gate shape + shared gate shape |

First 128-token sweep:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast moe_gate_shape2048 moe_down_shape4096 moe_shape_special \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_shape_special \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_shape_special/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_shape_special/summary.md \
  --stop-on-fail
```

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **16.05** | 393.71 | control |
| `moe_gate_shape2048` | **16.24** | 390.59 | small positive |
| `moe_down_shape4096` | **16.04** | 389.60 | neutral |
| `moe_shape_special` | **16.18** | 388.56 | below gate |

Longer gate-only recheck:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast moe_gate_shape2048 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 256 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape2048_256 \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape2048_256/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape2048_256/summary.md \
  --stop-on-fail
```

| Row | 8192/256 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **15.87** | 390.17 | control |
| `moe_gate_shape2048` | **16.22** | 389.30 | small positive, +2.2% |

Bit-equivalence check:

```sh
env DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 ./ds4 --cuda \
  -m ds4flash.gguf --ctx 100000 \
  --dump-logprobs tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape2048_quality/exact_fast.json \
  --logprobs-top-k 5 --temp 0 -n 32 --nothink \
  -p 'Quanti anni ha la repubblica italiana?'

env DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 \
  DS4_CUDA_MOE_DECODE_GATE_SHAPE2048=1 ./ds4 --cuda \
  -m ds4flash.gguf --ctx 100000 \
  --dump-logprobs tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape2048_quality/moe_gate_shape2048.json \
  --logprobs-top-k 5 --temp 0 -n 32 --nothink \
  -p 'Quanti anni ha la repubblica italiana?'
```

`cmp` returned **0** for both JSON logprobs and generated stdout.

Shared-shape check:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast shared_gate_up_shape2048 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_shape2048 \
  --summary tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_shape2048/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_shape2048/summary.md \
  --stop-on-fail
```

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **15.98** | 392.00 | control |
| `shared_gate_up_shape2048` | **16.07** | 389.26 | too small |

Combined routed-gate + shared-gate shape recheck:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast shape_gate_shared \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 256 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_shape_gate_shared_256 \
  --summary tuning/gx10_matrix_results/prototype_20260523_shape_gate_shared_256/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_shape_gate_shared_256/summary.md \
  --stop-on-fail
```

| Row | 8192/256 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **15.95** | 390.08 | control |
| `shape_gate_shared` | **16.03** | 388.06 | below gate |

Decision: keep `moe_gate_shape2048` as an exact minor candidate because it is
byte-identical and repeated positive, but do not promote it alone. It is below
the +3% cheap gate and does not move the branch meaningfully toward 20 t/s by
itself. Reject the down-shape, shared-shape, combined shape, and `__ldg` rows
for promotion.

Net effect of this pass: the plausible narrow variants around `attn_q_b`,
HC-expand, MoE LUT handling, shared expert aux writes, MoE register pressure,
long-context top-k chunk sizing, FFN shared/routed scheduling, and normal
decode graph pre-sync, plus the small Entrpi-inspired tensor-base alignment
probe, batch1 Q8 cached-x routing, default sampling CPU probability reuse,
read-only-cache MoE loads, and DS4 shape-specialized MoE/shared kernels are now
covered. Only routed gate/up shape specialization produced a repeatable exact
positive, and it remains a small candidate rather than a promoted default.

## Reproducing the measurements

### Direct vs cached graph bench

```sh
make cuda-spark
for i in 1 2 3; do
  ./ds4-bench --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 8192 --ctx-max 8192 --gen-tokens 64 --warm-weights
done
for i in 1 2 3; do
  DS4_CUDA_GRAPH_DECODE=1 ./ds4-bench --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 8192 --ctx-max 8192 --gen-tokens 64 --warm-weights
done
```

### K=2 server turbo smoke

```sh
DS4_CUDA_GRAPH_DECODE=1 \
DS4_MOE_ACTIVE_EXPERTS=2 \
DS4_MOE_ACTIVE_EXPERTS_RENORM=1 \
./ds4-server -m ds4flash.gguf --ctx 8192 \
  --host 127.0.0.1 --port 8106 --tokens 64

curl -sS http://127.0.0.1:8106/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"List the first 50 prime numbers, separated by commas.","max_tokens":64,"temperature":0}'
```

Expected server log on GB10: **21.4 t/s avg** for the 64-token decode chunk,
with no OOM at 8192 ctx.

### Per-token encode/execute profile

```sh
DS4_METAL_GRAPH_TOKEN_PROFILE=1 ./ds4 --temp 0 -n 8 -p "Test." 2>&1 \
  | grep "metal graph token"
DS4_CUDA_GRAPH_DECODE=1 DS4_METAL_GRAPH_TOKEN_PROFILE=1 ./ds4 --temp 0 -n 8 \
  -p "Test." 2>&1 | grep "metal graph token"
```

### Kernel breakdown via nsys

```sh
nsys profile --trace=cuda --output=/tmp/decode_prof --force-overwrite=true \
  ./ds4 --temp 0 -n 64 -p "<long prompt>"
nsys stats --report cuda_gpu_kern_sum /tmp/decode_prof.nsys-rep | head -30
```

### Bit-equivalence check (any time you touch a `_dec` kernel)

```sh
PROMPT="Quanti anni ha la repubblica italiana?"
./ds4 --dump-logprobs /tmp/direct.json --logprobs-top-k 5 --temp 0 -n 32 \
  -p "$PROMPT" > /dev/null 2>&1
DS4_CUDA_GRAPH_DECODE=1 ./ds4 --dump-logprobs /tmp/graph.json \
  --logprobs-top-k 5 --temp 0 -n 32 -p "$PROMPT" > /dev/null 2>&1
md5sum /tmp/direct.json /tmp/graph.json
# expect: identical md5s
```

Also run a longer prompt that exercises the indexer rope path (any prompt
that pushes the model past the early raw-cache phase, e.g. the OOP-vs-FP
one used during Phase 4b validation).

### 2026-05-23 continuation - post-push cache-policy and attention-A shape probes

After commit `8df127f` was pushed to `origin/gx10-cuda-graph-decode`, the next
round deliberately avoided repeating the already-rejected MoE pack/LDG/maxreg/
cache-x families. The first small probe tested whether the hot routed MoE
gate/up kernel was sensitive to CUDA's L1/shared cache split:

| Row | Env | Note |
| --- | --- | --- |
| `moe_gate_prefer_l1` | `DS4_CUDA_MOE_GATE_PREFER_L1=1` | `cudaFuncSetCacheConfig(..., cudaFuncCachePreferL1)` for routed gate/up kernels |
| `moe_gate_shape2048_l1` | shape2048 + prefer-L1 | tests whether the only positive MoE shape signal compounds with L1 preference |

Smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast moe_gate_prefer_l1 \
  moe_gate_shape2048 moe_gate_shape2048_l1 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_gate_l1 \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_gate_l1/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_gate_l1/summary.md
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.09** | control |
| `moe_gate_prefer_l1` | **16.02** | negative |
| `moe_gate_shape2048` | **16.22** | small positive |
| `moe_gate_shape2048_l1` | **16.21** | no improvement over shape2048 |

Resource usage from `cuobjdump --dump-resource-usage ds4_cuda.o` explains why
the shape win is small: `moe_gate_up_mid_decode_lut_qwarp32_shape2048_kernel`
uses `REG:63` vs `REG:64` for the generic kernel and fewer constant bytes
(`CONSTANT[0]:420` vs `452`), but this does not open a new occupancy tier. The
previous `maxr48` variant still spills (`STACK:16`), and prefer-L1 does not help.

The next exact probe targeted `attn_output_a`, not MoE. It specialized the
promoted SoA grouped Q8 kernel for the DS4 decode shape:

- `group_dim=8192`;
- `rank=512`;
- `n_groups=8`;
- `blocks=256`;
- one-token decode, full-warp row reduction preserved.

Rows:

| Row | Env | Note |
| --- | --- | --- |
| `attn_a_shape8192` | `DS4_CUDA_ATTENTION_OUTPUT_A_SHAPE8192=1` | DS4-shape attention-output-A SoA kernel, same reduction order |
| `attn_a_cache_x16` | `DS4_CUDA_ATTENTION_OUTPUT_A_CACHE_X16=1` | 16 rows per CTA, shared activation cache, still one full warp per row |
| `shape_gate_attn_a` | `moe_gate_shape2048 + attn_a_shape8192` | compound test across two different hot kernels |

The first `attn_a_shape8192` implementation unrolled the 8 block-steps and was
worse: `REG:64` vs `REG:55` for the generic SoA kernel, and 16.08 vs 16.10 t/s.
Removing the unroll dropped the specialized kernel to `REG:36`, but speed stayed
noise-level:

| Run | Row | 8192 gen t/s | Control | Decision |
| --- | --- | ---: | ---: | --- |
| 128-token retry | `attn_a_shape8192` | **16.04** | 15.91 | small/noisy positive |
| 256-token compound | `attn_a_shape8192` | **15.95** | 16.04 | negative |
| 256-token compound | `moe_gate_shape2048` | **16.18** | 16.04 | small positive |
| 256-token compound | `shape_gate_attn_a` | **16.16** | 16.04 | below gate-only shape |
| 128-token cache-x16 | `attn_a_cache_x16` | **15.90** | 15.94 | negative |

`attn_a_cache_x16` also had a bad resource signal (`REG:64`), worse than the
generic cached-x SoA kernel (`REG:44`) and the no-unroll shape kernel (`REG:36`).

Decision: reject prefer-L1, attention-A shape, and attention-A cache-x16 as
promotion candidates. Keep `moe_gate_shape2048` as the only exact small positive
from this pass, still below the +3% speed gate and not enough by itself.

### 2026-05-23 continuation - MoE gate/up const-stride and const-clamp probes

The next pass kept MoE down closed and stayed inside the only remaining small
positive MoE route: routed gate/up shape specialization. The external/local
scan suggested shape-specific address generation is one of the few transferable
ideas from MMQ-style CUDA paths without changing the DS4 weight layout. This
probe therefore did not repack weights or change expert order; it only replaced
dynamic row/expert byte strides with DS4 constants when the runtime shape
matches exactly:

- `n_tokens=1`;
- `n_expert=6`;
- `xq_blocks=16`;
- `expert_mid_dim=2048`;
- `gate_row_bytes=1056`;
- `gate_expert_bytes=2162688`;
- no auxiliary gate/up writes.

Rows:

| Row | Env | Note |
| --- | --- | --- |
| `moe_gate_shape2048_conststride` | `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTSTRIDE=1` | shape2048 routed gate/up with constant DS4 row/expert strides |
| `moe_gate_shape2048_constclamp` | `DS4_CUDA_MOE_DECODE_GATE_SHAPE2048_CONSTCLAMP=1` | const-stride plus hardcoded DS4 SwiGLU clamp `10.0f` guarded by `clamp == 10.0f` |

Resource usage:

| Kernel | REG | STACK | SHARED | CONSTANT[0] |
| --- | ---: | ---: | ---: | ---: |
| generic `moe_gate_up_mid_decode_lut_qwarp32_kernel` | 64 | 0 | 6848 | 452 |
| `shape2048` | 63 | 0 | 6848 | 420 |
| `shape2048_conststride` | 64 | 0 | 6848 | 404 |
| `shape2048_constclamp` | 64 | 0 | 6848 | 400 |

128-token smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast \
  moe_gate_shape2048 moe_gate_shape2048_conststride \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape_conststride \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape_conststride/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape_conststride/summary.md
```

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.09** | control |
| `moe_gate_shape2048` | **16.12** | small positive |
| `moe_gate_shape2048_conststride` | **16.23** | best in this smoke, still below gate |

256-token recheck:

| Row | 8192/256 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **15.80** | control |
| `moe_gate_shape2048` | **16.05** | small positive |
| `moe_gate_shape2048_conststride` | **16.12** | small positive, about +2.0%, still below +3% gate |

Bit-equivalence check for `moe_gate_shape2048_conststride` against
`exact_fast` on the standard 32-token logprob prompt returned
`json_cmp=0` and `out_cmp=0`.

Const-clamp did not help:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.07** | control |
| `moe_gate_shape2048_conststride` | **16.10** | small positive/noisy |
| `moe_gate_shape2048_constclamp` | **15.97** | negative |

Decision: keep `moe_gate_shape2048_conststride` as an exact minor candidate
and as a possible component for a later clean rebuild, but do not promote it
alone. Reject `moe_gate_shape2048_constclamp`; hardcoding the clamp did not
recover registers and was slower.

### 2026-05-23 continuation - split gate/up scheduling, stage profile, and combo checks

Before trying more MoE ideas, the branch was rechecked against the roadmap and
the fork scan. MoE down remains closed for now: shape4096, metadata/cache,
LDG, row/parallel/down-pack and native-pack lines were already neutral or
negative. The only open MoE signal is still routed gate/up shape specialization.

An extra non-graph stage profile was captured because
`DS4_METAL_DECODE_STAGE_PROFILE=1` cannot run inside CUDA graph capture
(`CUDA end commands failed: operation not permitted when stream is capturing`).
The fallback run used the current exact path without graph capture:

```sh
DS4_CUDA_Q8_SOA_CACHE=1 DS4_METAL_DECODE_STAGE_PROFILE=1 \
  ./ds4 --temp 0 -n 128 -p "Ecco una funzione" \
  > tuning/gx10_matrix_results/prototype_20260523_stage_profile_current/stdout.txt \
  2> tuning/gx10_matrix_results/prototype_20260523_stage_profile_current/stage.log
```

Aggregated stage time over 129 one-token decode layer samples:

| Stage | Total ms | Mean ms |
| --- | ---: | ---: |
| `routed_moe` | 46.133 | 0.357620 |
| `attn_output` | 41.920 | 0.324961 |
| `q_path` | 27.220 | 0.211008 |
| `shared_gate_up` | 11.879 | 0.092085 |
| `compressor_indexer` | 10.016 | 0.077643 |
| `attention` | 8.625 | 0.066860 |
| `shared_down` | 6.736 | 0.052217 |
| `router` | 3.573 | 0.027698 |

This profile is not throughput-comparable to graph decode, but it confirms the
same priority order: routed MoE, attention output, then Q path. MoE down is not
the next target unless a later graph-safe profile contradicts this.

The next exact probe split the shape2048 const-stride gate/up kernel so each
row computes gate first and up second, rather than keeping both accumulators
live together. The intent was to see whether scheduling pressure, not the
weight stream itself, was hiding a small win while preserving the same
per-path reduction order.

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast \
  moe_gate_shape2048_conststride moe_gate_shape2048_splitup \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape_splitup \
  --summary tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape_splitup/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_moe_gate_shape_splitup/summary.md
```

Resource usage did not improve: the split-up kernel still used `REG:64`,
`STACK:0`, `SHARED:6848`, `CONSTANT[0]:404`, the same as const-stride.

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.14** | control |
| `moe_gate_shape2048_conststride` | **16.00** | noisy below control |
| `moe_gate_shape2048_splitup` | **16.08** | negative; no resource win |

Decision: reject `moe_gate_shape2048_splitup`. Do not spend more time on
gate/up scheduling permutations unless a new profiler result shows a different
resource bottleneck.

One compound check tested whether the only small MoE gate/up candidate composes
with the previously revisited `soa_b_forced` attention-output-B path:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast soa_b_forced \
  moe_gate_shape2048_conststride moe_gate_conststride_soa_b_forced \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_conststride_soa_b_forced \
  --summary tuning/gx10_matrix_results/prototype_20260523_conststride_soa_b_forced/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_conststride_soa_b_forced/summary.md
```

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.08** | control |
| `moe_gate_shape2048_conststride` | **16.11** | small/noisy positive |
| `soa_b_forced` | **15.85** | negative in this run |
| `moe_gate_conststride_soa_b_forced` | **15.95** | negative combo |

Decision: reject the combo. `soa_b_forced` does not compose with routed
gate/up const-stride in this state.

External scan note: `cghart/ds4:cuda-gb10-q2-foundation` was fetched and read
as an idea source. Its older CUDA matvec/prequant branch points at persistent
Q8 scratch, prequantized matvecs, F16 reductions, and launch-depth/attention
experiments, but most of those concepts overlap with the current branch's
tmp-prequant, Q8 SoA, and graph decode work. No direct import was identified.
Keep it as a reference when looking for structural Q8 projection ideas, not as
an integration target.

The next pass moved away from MoE entirely and targeted `q_path` /
`attn_output`, because those were the two largest non-MoE buckets in the
profile. The fork scan also clarified that one cghart idea named output
Q8 warp-rows is already equivalent to this branch's current `warp8` Q8 decode
kernel, so it was not repeated.

First, `attn_q_a/attn_kv` pair projection was shape-specialized for the DS4
decode dimensions (`4096 -> 1024` and `4096 -> 512`). This is not the old
SoA-QKV route; it preserves the current AoS weight stream and only removes
dynamic shape/stride checks from the already-promoted pair kernel.

Resource usage:

| Kernel | REG | STACK | SHARED | CONSTANT[0] |
| --- | ---: | ---: | ---: | ---: |
| generic `matmul_q8_0_pair_preq_warp8_kernel` | 62 | 0 | 0 | 436 |
| `matmul_q8_0_pair_preq_warp8_qkv_shape_kernel` | 63 | 0 | 0 | 404 |

Smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast attn_qkv_pair_shape \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_attn_qkv_pair_shape \
  --summary tuning/gx10_matrix_results/prototype_20260523_attn_qkv_pair_shape/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_attn_qkv_pair_shape/summary.md
```

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.12** | control |
| `attn_qkv_pair_shape` | **16.07** | negative |

Decision: reject the QKV pair shape specialization. It raised register use and
did not improve throughput.

Second, `attn_output_b` was shape-specialized for the default AoS Q8 path
(`4096 x 4096`, 128 Q8 blocks). This is distinct from the earlier
`soa_b_forced` and cuBLAS/F16 probes.

Resource usage:

| Kernel | REG | STACK | SHARED | CONSTANT[0] |
| --- | ---: | ---: | ---: | ---: |
| generic `matmul_q8_0_preq_warp8_kernel` | 62 | 0 | 0 | 412 |
| `matmul_q8_0_attn_output_b_shape4096_kernel` | 63 | 0 | 0 | 388 |

Smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast attn_b_shape4096 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_attn_b_shape4096 \
  --summary tuning/gx10_matrix_results/prototype_20260523_attn_b_shape4096/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_attn_b_shape4096/summary.md
```

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.12** | control |
| `attn_b_shape4096` | **15.90** | negative |

Decision: reject attention-output-B shape specialization. The generic warp8
kernel remains better.

Finally, the old observation that `attn_q_a` and `attn_kv` SoA were individually
micro-positive but regressed inside the fused pair route was checked directly by
disabling the pair projection while enabling QKV SoA:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast soa_qkv soa_qkv_no_pair \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_soa_qkv_no_pair \
  --summary tuning/gx10_matrix_results/prototype_20260523_soa_qkv_no_pair/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_soa_qkv_no_pair/summary.md
```

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **15.94** | control |
| `soa_qkv` | **16.01** | noisy/small, still below gate and historically negative |
| `soa_qkv_no_pair` | **15.70** | negative |

Decision: reject `soa_qkv_no_pair`. The pair projection remains worth keeping;
the SoA-QKV line is still diagnostic only and not a promotion candidate.

The micro-shape probes above made it clear that more static Q8 dimensions are
not enough. A short Nsight Systems profile was captured on the graph path to
rank actual kernel families rather than relying only on manual stage buckets:

```sh
DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 \
  nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --force-overwrite=true \
  --output tuning/gx10_matrix_results/prototype_20260523_nsys_graph_exact/decode_graph_exact \
  ./ds4 --cuda -m ds4flash.gguf --ctx 8192 -n 32 --temp 0 \
  -p "Ecco una funzione"

nsys stats --report cuda_gpu_kern_sum \
  tuning/gx10_matrix_results/prototype_20260523_nsys_graph_exact/decode_graph_exact.nsys-rep \
  > tuning/gx10_matrix_results/prototype_20260523_nsys_graph_exact/cuda_gpu_kern_sum.txt
```

The full `.nsys-rep` / `.sqlite` files were not committed because they are
generated and large; the committed artifact is the kernel summary text.

Top graph-path kernel families from `cuda_gpu_kern_sum.txt`:

| Kernel family | Time % | Instances | Avg |
| --- | ---: | ---: | ---: |
| `moe_gate_up_mid_decode_lut_qwarp32_kernel` | 14.9 | 1333 | 235.378 us |
| `matmul_q8_0_hc_expand_preq_warp8_soa_kernel` | 10.0 | 1333 | 157.406 us |
| `matmul_q8_0_preq_batch_warp8_kernel` (`attn_q_b`) | 9.8 | 1333 | 155.578 us |
| `grouped_q8_0_a_preq_warp8_soa_kernel` | 9.7 | 1333 | 153.431 us |
| `moe_down_sum6_qwarp32_kernel` | 6.9 | 1333 | 109.840 us |
| `matmul_q8_0_pair_swiglu_preq_warp8_kernel` | 5.3 | 1333 | 83.982 us |
| output full-logits `matmul_q8_0_preq_kernel` | 3.7 | 32 | 2.462 ms |
| `attention_decode_mixed_kernel` | 3.7 | 1333 | 58.151 us |

This profile explained why the old cghart "output Q8 warp rows" idea was worth
rechecking: the full-logits output head still used the generic one-row-per-CTA
Q8 kernel. A new diagnostic row routed only the output-head-sized
`4096 -> vocab` full-logits matmul through the existing `warp8` Q8 kernel:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast output_q8_warp8 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_output_q8_warp8 \
  --summary tuning/gx10_matrix_results/prototype_20260523_output_q8_warp8/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_output_q8_warp8/summary.md
```

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.13** | control |
| `output_q8_warp8` | **15.93** | negative |

Decision: reject `output_q8_warp8`. Even though the output full-logits kernel
is visible in the profile, replacing it with the warp8 reducer worsened
end-to-end throughput and also changes reduction order, so it does not deserve
a quality gate.

### Decode-window Nsight profile

The previous Nsight pass still included some startup/prefill-side work. A
follow-up run delayed collection until the decode loop so the top rows reflect
steady graph replay more closely:

```sh
DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 \
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --delay=22 --duration=6 --force-overwrite=true \
  --output tuning/gx10_matrix_results/prototype_20260523_nsys_graph_decode_window/decode_window \
  ./ds4 --cuda -m ds4flash.gguf --ctx 8192 -n 128 --temp 0 \
  -p "Ecco una funzione"

nsys stats --report cuda_gpu_kern_sum \
  tuning/gx10_matrix_results/prototype_20260523_nsys_graph_decode_window/decode_window.nsys-rep \
  > tuning/gx10_matrix_results/prototype_20260523_nsys_graph_decode_window/cuda_gpu_kern_sum.txt
```

The run printed `generation: 17.12 t/s`. As before, only the text summary is
kept in git; `.nsys-rep` and `.sqlite` are generated artifacts.

Top decode-window kernel families:

| Kernel family | Time % | Instances | Avg |
| --- | ---: | ---: | ---: |
| `moe_gate_up_mid_decode_lut_qwarp32_kernel` | 18.7 | 4345 | 234.587 us |
| `matmul_q8_0_hc_expand_preq_warp8_soa_kernel` | 12.4 | 4345 | 156.239 us |
| `matmul_q8_0_preq_batch_warp8_kernel` (`attn_q_b`) | 12.3 | 4345 | 155.170 us |
| `grouped_q8_0_a_preq_warp8_soa_kernel` | 12.1 | 4345 | 152.388 us |
| `moe_down_sum6_qwarp32_kernel` | 8.8 | 4345 | 110.218 us |
| `matmul_q8_0_pair_swiglu_preq_warp8_kernel` | 6.7 | 4345 | 84.352 us |
| `attention_decode_mixed_kernel` | 5.5 | 4345 | 69.454 us |
| `matmul_f16_pair_kernel` | 5.3 | 6262 | 45.878 us |
| output full-logits `matmul_q8_0_preq_kernel` | 4.5 | 101 | 2.418 ms |
| `matmul_q8_0_hc_expand_preq_warp8_kernel` | 3.5 | 4344 | 44.450 us |
| `matmul_q8_0_pair_preq_warp8_kernel` | 2.8 | 4345 | 35.016 us |

Interpretation:

- MoE down is not the next target. It is only the fifth row in the true decode
  window, and the branch has already tested row-major/native pack, meta-cache,
  `__ldg`, row4, parallel, exact shape, and reduced-K direct variants.
- The first four rows are the real frontier: routed MoE gate/up, HC-expand SoA,
  `attn_q_b`, and attention-output-A SoA. Their simple shape/cache/register
  probes are already documented as neutral or negative, so future work needs a
  structural idea rather than another dimension-specialized wrapper.
- The output head is visible but still a bad target for now: the warp8 output
  route was slower and changed reduction order before it even reached a quality
  gate.
- The f16 pair path is now worth classifying, but not blindly optimizing: the
  instance count does not match a single per-layer decode row, so first identify
  which compressor/HC calls produce it before adding kernels.

### Fork-inspired LDS/cache probes after decode-window profile

The next scan looked at recent CUDA/ROCm-oriented fork work and similar
projects for ideas that were not just another MoE-down attempt. Two useful
themes were worth testing locally:

- Entrpi's MMQ/layer-graph branch had rejected a one-token Q8 pair lift because
  it quantized activations twice; our current Q8 pair kernels already share the
  prequant activation, so there was no direct MMQ import without porting that
  whole backend.
- ROCm branches repeatedly found wins from staging duplicated per-block inputs
  in LDS/shared memory. The CUDA branch had already rejected broad Q8 cache-x
  variants, but had not isolated the fused shared-expert gate/up SwiGLU kernel
  nor the compressor F16 pair reduction.

The shared gate/up probe added
`DS4_CUDA_SHARED_GATE_UP_CACHE_X=1`, staging the already-prequantized Q8
activation and scales once per CTA before the existing 8 warp rows consume
them. This is exact-order and keeps the same resource signal as the existing
kernel (`REG:62`, same as baseline):

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast shared_gate_up_cache_x \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_cache_x \
  --summary tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_cache_x/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_cache_x/summary.md
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.00** | control |
| `shared_gate_up_cache_x` | **15.88** | negative |

Decision: reject. The shared staging overhead does not pay for this kernel,
matching the earlier pattern from HC-expand and attention-output cache-x probes.

The F16 pair profile row was then classified. The 6262 `matmul_f16_pair_kernel`
instances in the decode-window profile are exactly:

- 41 attention-compressor paired projections per generated token
  (all compressed layers 2..42);
- plus 21 indexer-compressor paired projections on ratio-4 layers.

A narrow exact-order reduction probe added `DS4_CUDA_F16_PAIR_FAST_REDUCE=1`.
It preserves the current 256-thread partial-sum order, but replaces the final
warp-local block barriers with `__syncwarp()`. Resource usage was not worse
(`REG:15` vs `REG:16`, same 2048 B shared memory):

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast f16_pair_fast_reduce \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_f16_pair_fast_reduce \
  --summary tuning/gx10_matrix_results/prototype_20260523_f16_pair_fast_reduce/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_f16_pair_fast_reduce/summary.md
```

Result:

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.07** | control |
| `f16_pair_fast_reduce` | **16.04** | neutral/negative |

Finally, a control row disabled paired compressor projections entirely to make
sure the current pair route was still worth keeping:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast compressor_pair_off \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_compressor_pair_off \
  --summary tuning/gx10_matrix_results/prototype_20260523_compressor_pair_off/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_compressor_pair_off/summary.md
```

| Row | 8192/128 gen t/s | Decision |
| --- | ---: | --- |
| `exact_fast` | **16.09** | control |
| `compressor_pair_off` | **15.85** | keep paired F16 compressor path |

Decision: reject both new probes, but keep the diagnostic flags and summaries.
The paired F16 compressor kernel is already the right path; the remaining
headroom is not in its final barrier pattern.

### 2026-05-23 continuation - shared gate/up dot2 probe

After re-checking the roadmap, MoE down stayed closed: the decode-window
profile puts it behind the Q8 projection frontier, and the branch already has
row-major/native pack, metadata-cache, `__ldg`, row4, parallel, shape4096, and
reduced-K direct attempts recorded.

The next exact probe targeted the fused shared-expert gate/up SwiGLU Q8 kernel
without repeating the failed shared-memory cache-x idea. The baseline computes
the gate and up dot products separately against the same prequantized
activation block, so `DS4_CUDA_SHARED_GATE_UP_DOT2=1` added:

- `dot_i8_block2_shared_x`, which loads the aligned activation vector once and
  applies it to the two unaligned Q8_0 weight blocks;
- `matmul_q8_0_pair_swiglu_preq_warp8_dot2_kernel`, preserving the per-dot
  DP4A loop and accumulation order;
- a matrix row, `shared_gate_up_dot2`, to keep the probe opt-in.

Resource usage was slightly better than the baseline:

| Kernel | Registers |
| --- | ---: |
| `matmul_q8_0_pair_swiglu_preq_warp8_dot2_kernel` | 61 |
| `matmul_q8_0_pair_swiglu_preq_warp8_kernel` | 62 |
| `matmul_q8_0_pair_swiglu_preq_warp8_cached_x_kernel` | 62 |

Smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast shared_gate_up_dot2 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_dot2 \
  --summary tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_dot2/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_dot2/summary.md
```

Result:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **15.97** | 393.09 | control |
| `shared_gate_up_dot2` | **15.89** | 389.92 | negative |

Decision: reject for promotion. The single-load dot2 form saves a register but
does not improve end-to-end decode; the likely cost is instruction scheduling
and independent memory latency hiding inside the existing two-dot loop. Keep the
flag and artifact as a negative probe so this exact idea is not repeated. The
next promising local directions are not MoE down or shared activation staging;
they are either a real native paired Q8 layout/tensor-core port branch, or
another measured frontier kernel with a structural change.

### 2026-05-23 continuation - shared gate/up native paired pack

The next "outside the wrapper" attempt tried the smallest native layout that
could be memory-safe: only the shared expert Q8_0 gate/up weights were
duplicated into a paired, aligned layout. This is **not** the routed MoE native
pack already rejected earlier; it targets the much smaller shared expert path.

Implementation:

- `DS4_CUDA_SHARED_GATE_UP_PAIR_PACK=1`;
- startup preload through `ds4_gpu_cache_q8_f16_range`, keyed by matching
  `ffn_gate_shexp` / `ffn_up_shexp` tensor labels, so graph capture does not
  allocate;
- paired pack format per Q8 block:
  `gate_scale(2), up_scale(2), gate_q[32], up_q[32]` = 68 bytes;
- `matmul_q8_0_pair_swiglu_preq_warp8_pack68_kernel`, using aligned Q8 weight
  loads plus one shared activation load for both dots.

The resource signal looked promising:

| Kernel | Registers |
| --- | ---: |
| `matmul_q8_0_pair_swiglu_preq_warp8_pack68_kernel` | 51 |
| `matmul_q8_0_pair_swiglu_preq_warp8_dot2_kernel` | 61 |
| `matmul_q8_0_pair_swiglu_preq_warp8_kernel` | 62 |

A first scratch version used 72-byte stride padding and was already slower
(`15.55` vs `16.07` t/s control), so the measured artifact below uses the final
68-byte compact format.

Smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast shared_gate_up_pair_pack \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_pair_pack \
  --summary tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_pair_pack/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_shared_gate_up_pair_pack/summary.md
```

Result:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **15.95** | 393.04 | control |
| `shared_gate_up_pair_pack` | **15.38** | 388.87 | negative |

Decision: reject. The lower register count did not translate to speed; the
native paired layout likely worsens cache-line behavior or independent memory
latency hiding enough to swamp the instruction/register savings. This also
weakens the case for a similar small paired pack on Q8 gate/up-style projections
unless a profiler points to a different access pattern.

### 2026-05-23 continuation - HC-expand SoA read-only-load probe

After closing shared gate/up cache, dot2, and pair-pack attempts, the next probe
returned to a different decode-window frontier: HC-expand Q8 SoA. This was not
the earlier MoE `__ldg` experiment; it targeted only the already-cached SoA Q8
weights used by `matmul_q8_0_hc_expand_preq_warp8_soa_kernel`.

Implementation:

- `DS4_CUDA_HC_EXPAND_SOA_LDG=1`;
- `dot_i8_block_weight_aligned_ldg`, using `__ldg` for aligned Q8 SoA weight
  words;
- `matmul_q8_0_hc_expand_preq_warp8_soa_ldg_kernel`, also loading the SoA
  half scale through `__ldg`;
- exact store and HC expansion order unchanged.

Resource usage was worse before runtime:

| Kernel | Registers |
| --- | ---: |
| `matmul_q8_0_hc_expand_preq_warp8_soa_ldg_kernel` | 63 |
| `matmul_q8_0_hc_expand_preq_warp8_soa_kernel` | 62 |
| `matmul_q8_0_hc_expand_preq_warp8_soa_cached_x_kernel` | 50 |

Smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast hc_expand_soa_ldg \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_hc_expand_soa_ldg \
  --summary tuning/gx10_matrix_results/prototype_20260523_hc_expand_soa_ldg/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_hc_expand_soa_ldg/summary.md
```

Result:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **16.13** | 393.27 | control |
| `hc_expand_soa_ldg` | **15.84** | 389.48 | negative |

Decision: reject. Read-only-cache loads are now negative on both routed MoE
IQ2/Q2 weights and HC-expand Q8 SoA weights. Do not repeat `__ldg` as a generic
weight-load fix unless a later architecture-specific profile contradicts this.

### 2026-05-23 continuation - HC-expand lane-parallel HC4 store probe

One remaining HC-expand question was whether the fused kernel was losing time
after the Q8 dot, where lane 0 serially writes all four HC outputs. The new
`DS4_CUDA_HC_EXPAND_SOA_PAR_HC4=1` probe keeps the exact Q8 SoA dot and warp
reduction, then broadcasts the row sum and lets lanes 0..3 compute the four
`n_hc=4` outputs independently. For each HC lane the arithmetic order is the
same as the previous serial helper:

```text
block_v * post[h] + comb[h] * r0 + comb[h+4] * r1
                  + comb[h+8] * r2 + comb[h+12] * r3
```

Resource usage was again encouraging but misleading:

| Kernel | Registers |
| --- | ---: |
| `matmul_q8_0_hc_expand_preq_warp8_soa_par_hc4_kernel` | 51 |
| `matmul_q8_0_hc_expand_preq_warp8_soa_kernel` | 62 |

Smoke:

```sh
python3 tuning/gx10_matrix.py bench-suite exact_fast hc_expand_soa_par_hc4 \
  --ctx-start 8192 --ctx-max 8192 --ctx-alloc 100000 --gen-tokens 128 \
  --out-dir tuning/gx10_matrix_results/prototype_20260523_hc_expand_soa_par_hc4 \
  --summary tuning/gx10_matrix_results/prototype_20260523_hc_expand_soa_par_hc4/summary.csv \
  --markdown tuning/gx10_matrix_results/prototype_20260523_hc_expand_soa_par_hc4/summary.md
```

Result:

| Row | 8192/128 gen t/s | Prefill t/s | Decision |
| --- | ---: | ---: | --- |
| `exact_fast` | **16.06** | 391.77 | control |
| `hc_expand_soa_par_hc4` | **15.91** | 389.54 | negative |

Decision: reject. The HC expansion/store tail is not the limiting part of this
kernel; the Q8 dot and memory behavior dominate. Together with `hc_expand_soa_ldg`,
`hc_expand_nhc4_special`, `hc_expand_no_block_out`, and prior cache-x probes,
HC-expand should stay closed until a tensor-core/native-layout rewrite is on
the table.

### 2026-05-23 continuation - isolated MMQ/MMVQ port branch

The next structural path is now being worked in an isolated worktree instead of
on the main tuning branch:

- worktree: `/home/alessandro/projects/ds4-mmqv-port`;
- branch: `gx10-mmqv-port`;
- base: current `gx10-cuda-graph-decode` after commit `7fee9f1`;
- source reference: `Entrpi/ds4:mmq-step-A-full-layer-graphs`;
- scope for the first checkpoint: import only `cuda/mmq/`, build it beside the
  current backend, and keep runtime routing unchanged.

This deliberately excludes Entrpi's layer graphs, MoE graphs, VMM arena, MTP
proof harness, and dispatcher policy. The imported source is useful as a
kernel/library scaffold, but the branch's public GB10 CSV remains below this
branch's current exact-fast band, so this is not being treated as a drop-in win.

Checkpoint build:

```sh
make -j$(nproc) cuda-spark
```

Result: success. `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent`
now link with the MMQ/MMVQ objects, but no Q8 runtime path calls them yet.

Decision: keep as an inactive scaffold and commit separately. The next gate is
an explicit opt-in single-token Q8 projection route, measured against same-run
`exact_fast` before any larger dispatcher or graph-capture changes.

## Branch / commit map

```
gx10-cuda-graph-decode  (this branch)
 ├ a103cb7  Phase 4c — cudaGraphExecUpdate-based exec cache
 ├ 069d3d6  attention output inverse rope → dec dispatch
 ├ 8324e40  kv_rope_fp8_store_raw + store_raw_kv → dec
 ├ 173fa19  rope_tail/head_rms dec dispatch + compressor pos bug fix
 ├ a06c538  (park: rope_tail dec / head_rms dec — superseded by 173fa19)
 ├ 4804f64  embed_token_hc → dec (Phase 4b start)
 ├ c994f62  Phase 4a — device-resident per-token decode state
 ├ 3c140c1  Phase 3 — route decode through graph capture when env set
 ├ 8838e62  Phase 2 — enable graph capture (per-thread stream + smoke test)
 ├ 8a9bf89  Phase 2 scaffolding — graph API surface
 ├ 7ef65b8 / a6bbdc8  Upstream correctness fixes
 ├ 20a4f2c  Drop dead g_cuda_sm_minor after sm_major Blackwell gate
 ├ 017e5c0  Revert PR #158 HMM path (lost ~13 % gen on GB10)
 └ 79b5b68 / 9f4ba78  Notes for the upstream import pass and #158 decision
```

All commits pushed to `origin/gx10-cuda-graph-decode` (the user's fork at
`elmisi/ds4`). Never push to the upstream remote.
