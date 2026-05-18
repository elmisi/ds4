# Road to 20 t/s on GB10 — work log and findings

This document captures the work, data, and dead-ends from the push to lift
DS4 V4 Flash decode throughput on a DGX Spark (GB10, sm_121) from ~15 t/s to
the 20 t/s target. It is meant as a self-contained handoff so a future
session can pick up without re-discovering anything.

Branch: `gx10-cuda-graph-decode` (pushed to fork `origin`, never to upstream).

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
  code-format discipline than K=2. The best current coding smoke is a per-layer
  profile: K=6 on layers 0-2, K=3 + renorm everywhere else. It passed **4/4**
  coding tasks while holding **~20.0 t/s** in server decode logs. These modes
  are not equivalent to the full model: they use fewer routed experts than
  top-6 in most decode layers.
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
- **MTP speculative decode still does not beat baseline.** With draft=2 the
  verifier is now down to ~103-109 ms after Q8 microbatch fixes, and MTP lands
  around 13.8 t/s despite good acceptance. The remaining cost is structural:
  one normal target decode, two MTP draft-head calls, and a target verifier.
- **For full-quality decode, the closest route is still normal no-MTP decode.**
  The third pass found small default wins in the Q/KV projection path and shared
  expert gate/up path, but also ruled out several tempting "spend more memory
  for speed" ideas. The remaining full-quality gap to 20 t/s is about 5-6
  ms/token, not a launch or graph issue.
- **Outside-the-box shortcuts were tested.** Server greedy top-only avoids
  full logits readback but is neutral, because exact argmax still has to scan
  the full output head. A direct exact output-head top1 kernel was also slower
  than the existing full-logits Q8 path in server A/B, so it is opt-in only
  (`DS4_CUDA_OUTPUT_TOP1=1`). Reduced active MoE experts is the first shortcut
  that actually crosses 20 t/s in the server path. K=2 is fastest but visibly
  degraded on coding; the current practical candidate is the layer-profiled
  K6/K3 mode.

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

## Open avenues for the next session

In rough order of effort vs likely payoff:

### 1. Validate layer-profiled reduced-K coding mode (highest immediate payoff)

The best current coding candidate is K=6 on layers 0-2 and K=3 + renorm on all
other layers:

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

The next step is not more micro-optimization; it is broader quality
characterization. Run real repo-edit prompts and longer coding-agent
conversations against K=6 full-quality, uniform K=3 renorm, and the 0-2:6 layer
profile. If the layer profile holds up, it is the simplest practical 20 t/s
server mode. The next adaptive step should be more per-layer profiles or a
device-side dynamic-K kernel; token-by-token host-selected K would destabilize
CUDA Graph topology.

### 2. Full-quality normal decode: remove ~5-6 ms/token

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

### 3. True batched target verifier for MTP (large change, currently not enough)

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

### 4. Attention path matmul optimization (medium payoff)

`matmul_q8_0_preq_warp8_cached_x_kernel` is used for all of Q-proj, KV-proj,
attn-output, and small downstream matmuls. The Q/KV front of this path already
has the low-risk paired projection win. cuBLAS/F16-cache variants were tested
for small decode and for `attn_q_b` and were worse, so the remaining likely
payoff is in the custom kernels themselves, especially attention-output.

Worth a microbenchmark: take the actual launch shapes from `nsys` and time
both kernels on a synthetic harness, then either swap globally or add an
env gate. Expected upside: 10-30 % on these matmuls = ~1.5-3 % total
decode wall-clock. Modest by itself.

### 5. Reduce HC expand cost (`matmul_q8_0_hc_expand_preq_warp8`, medium risk)

The HC expand kernel runs N_HC=4 times per layer for HC partition mixing.
If the four expansions can be folded into a single grouped GEMM (or run on
tensor cores), there's potential. Worth a look but the kernel already has
a `_warp8` micro-optimization, and both the F16-cache/cuBLAS path and the
cached-activation variant regressed in the third pass. Treat this as a deeper
kernel redesign, not an env-toggle candidate.

### 6. Tune kernel launch shapes for full-GPU occupancy

Spot-check with nsys metrics: `sm__cycles_active.avg.pct_of_peak_sustained`,
`l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second`. If active SM
fraction is <80 % during the dominant kernels, there is room. If it is
already >90 %, the kernel is achieving its memory-bound ceiling and the
only further win is structural (specdec, fewer reads per token).

### 7. Accept the full-quality GB10 ceiling

The MoE weight-read ceiling math above suggests ~18 t/s is roughly the
hardware-imposed limit for this model on this box without speculative
decoding or a larger kernel redesign. If specdec remains too invasive for the
project's appetite, the honest framing is: the current architecture is stable
around 18 t/s on GB10 for DS4 V4 Flash, and 20 t/s needs a real reduction in
per-token weight traffic or a major improvement in the dominant custom kernels.

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
