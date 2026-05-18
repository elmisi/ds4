# Road to 20 t/s on GB10 — work log and findings

This document captures the work, data, and dead-ends from the push to lift
DS4 V4 Flash decode throughput on a DGX Spark (GB10, sm_121) from ~15 t/s to
the 20 t/s target. It is meant as a self-contained handoff so a future
session can pick up without re-discovering anything.

Branch: `gx10-cuda-graph-decode` (pushed to fork `origin`, never to upstream).

## TL;DR

- **Current state**: ~15.1 t/s avg (8192 ctx, 64 gen) on GB10. Stable, no
  regressions; 110 GB device-resident with 256k ctx server.
- **Phase 4 (CUDA Graph) complete**: capture, dec-kernel infrastructure, and
  `cudaGraphExecUpdate`-based cached exec all working and bit-equivalent to
  direct mode. Does **not** improve wall-clock — see "Why graph capture did
  not help" below.
- **Profile shows decode is memory-bound on MoE weight reads.** The kernel
  initially targeted (`moe_down_sum6_qwarp32`) is only 7 % of GPU time on
  decode. The dominant cost is `matmul_q8_0_preq_warp8_cached_x` (Q/KV/out
  projections, 16 %) tied with `moe_gate_up_mid_decode_lut_qwarp32` (16 %).
- **The 20 t/s target cannot be reached with single-kernel work.** Needs
  either many small wins distributed across the pipeline, or speculative
  decoding via the model's MTP draft head (architecturally large change,
  potentially 2-3× win), or accepting the GB10 ceiling for this model.

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

## Open avenues for the next session

In rough order of effort vs likely payoff for hitting ≥20 t/s:

### 1. Speculative decoding via MTP (highest payoff, largest change)

DS4 V4 ships with an MTP (multi-token prediction) head. The release runtime
already has scaffolding (`ds4_mtp_weights`,
`metal_graph_eval_mtp_draft_from_hc`) but does not appear to enable
speculative *decoding* — only optional draft emission. The standard pattern:

1. Draft model generates K candidate tokens cheaply (the MTP head is much
   smaller than the full model).
2. Full model verifies all K in a *single* forward pass (the only
   per-token cost is the attention update; MoE etc. are batched).
3. Accept the prefix where draft and full agree; discard the rest.

On memory-bound decode (which this is), accept rates of 60-80 % yield 1.8-
2.5× effective throughput. Reaching 20 t/s would require ~33 % effective
gain — well within typical MTP speedup envelopes.

Caveats: requires careful integration with the KV cache (need to roll back
on rejection), tool-call exact-replay (DSML emission semantics), and the
existing single-graph-worker server serialization. This is several days of
work but is the single highest-leverage path.

### 2. Attention path matmul optimization (medium payoff)

`matmul_q8_0_preq_warp8_cached_x_kernel` (16 % of GPU) is used for all of
Q-proj, KV-proj, attn-output, and small downstream matmuls. If a cuBLAS
LtMatmul invocation with q8 → fp16 fused dequant outperforms the custom
kernel for the typical decode sizes (`m=1, n in {1024, 4096, 7168}, k in
{4096, 7168}`), substituting would be a low-risk swap.

Worth a microbenchmark: take the actual launch shapes from `nsys` and time
both kernels on a synthetic harness, then either swap globally or add an
env gate. Expected upside: 10-30 % on these matmuls = ~1.5-3 % total
decode wall-clock. Modest by itself.

### 3. Reduce HC expand cost (`matmul_q8_0_hc_expand_preq_warp8`, 14 %)

The HC expand kernel runs N_HC=4 times per layer for HC partition mixing.
If the four expansions can be folded into a single grouped GEMM (or run on
tensor cores), there's potential. Worth a look but the kernel already has
a `_warp8` micro-optimization, so the easy wins may be taken.

### 4. Tune kernel launch shapes for full-GPU occupancy

Spot-check with nsys metrics: `sm__cycles_active.avg.pct_of_peak_sustained`,
`l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`. If active SM
fraction is <80 % during the dominant kernels, there is room. If it is
already >90 %, the kernel is achieving its memory-bound ceiling and the
only further win is structural (specdec, fewer reads per token).

### 5. Accept the GB10 ceiling

The MoE weight-read ceiling math above suggests ~17-18 t/s is roughly the
hardware-imposed limit for this model on this box without speculative
decoding. If specdec is too invasive for the project's appetite, the
honest framing is: 15 t/s is GB10 reality for DS4 V4 Flash with the
current architecture; the work here delivered stable parity with a cleaner
graph infrastructure.

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
