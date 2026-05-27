# GX10 agent stepwise matrix - 2026-05-26

## Scope

Worktree: `/home/alessandro/projects/ds4-stepwise`

Branch: `gx10-agent-stepwise` at `0ee3c82 cuda: add GX10 opt-in decode fast paths`

Baseline: `origin/main` at `ad0209f Fix PRO routed MoE expert mapping`

Model: `/home/alessandro/projects/ds4/ds4flash.gguf`

Command shape:

```sh
./ds4-agent --cuda --ctx 32768 --chdir /home/alessandro/projects/ds4-stepwise \
  -m /home/alessandro/projects/ds4/ds4flash.gguf \
  --non-interactive \
  -p 'mi fai la lista delle province italiane che cominciano per la lettera "B" ?' \
  -n 256 --trace <run>/trace.log
```

`trace_tps` is computed from generated-token timestamps in the trace file. Memory is sampled while the agent process runs. Raw per-run output is under `tuning/agent_stepwise_runs/results/`, with the aggregate CSV at `tuning/agent_stepwise_runs/results/summary.csv`.

## Main conclusions

1. The useful agent decode speedup is not explained by the three runtime switches alone.
2. The main working improvement is the compiled-in fused HC pre path introduced by `81a1a3b cuda: fuse GB10 decode HC and RoPE paths`.
3. Disabling that path with `DS4_CUDA_NO_FUSED_HC_PRE=1` drops the current graph+indexer case from about `15.52 t/s` to `14.62 t/s`.
4. `DS4_CUDA_GRAPH_DECODE=1` adds a small decode gain in this harness, roughly `+0.25 t/s` on the current branch.
5. `DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1` is neutral for this decode prompt. It may still matter for prefill/indexer-heavy cases, but it is not a measured decode win here.
6. `DS4_CUDA_Q8_SOA_CACHE=1` is not acceptable for the final agent path right now:
   - without graph it runs, but uses about `+1.43 GiB` CUDA memory and does not improve decode throughput;
   - with graph it crashes before generation with a CUDA capture error.
7. The old PR branch `origin/gx10-graph-soa-indexer-topk` reproduces the bad agent behavior: about `13.8 t/s` with graph and a crash with all three switches. That branch is not representative of the useful stepwise stack.

## Recommended final candidate

For an agent-focused clean branch, the strongest candidate is:

- keep the compiled GB10 decode tuning through `81a1a3b`, especially fused HC pre;
- optionally keep `DS4_CUDA_GRAPH_DECODE=1` as an opt-in small win;
- do not claim `DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1` as a decode win from this test;
- do not include or recommend `DS4_CUDA_Q8_SOA_CACHE=1` until the graph interaction is fixed and the memory/throughput tradeoff is positive.

The performance claim should be phrased as compiled-in GB10 CUDA decode tuning plus optional graph decode, not as the three-switch set.

## Current branch variants

| Variant | Return | Tokens | trace_tps | Peak CUDA GiB | Peak used GiB | Env |
|---|---:|---:|---:|---:|---:|---|
| `0ee3c82_no_env` | 0 | 256 | 15.291 | 103.20 | 108.94 | `{}` |
| `0ee3c82_graph` | 0 | 256 | 15.550 | 103.20 | 109.04 | `DS4_CUDA_GRAPH_DECODE=1` |
| `0ee3c82_indexer_no_graph` | 0 | 256 | 15.280 | 103.20 | 108.98 | `DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1` |
| `0ee3c82_graph_indexer` | 0 | 256 | 15.523 | 103.20 | 109.00 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1` |
| `0ee3c82_soa_no_graph` | 0 | 194 | 15.379 | 104.63 | 110.43 | `DS4_CUDA_Q8_SOA_CACHE=1` |
| `0ee3c82_soa_indexer_no_graph` | 0 | 256 | 15.328 | 104.63 | 110.48 | `DS4_CUDA_Q8_SOA_CACHE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1` |
| `0ee3c82_graph_soa` | -11 | 0 | 0.000 | 103.20 | 108.93 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1` |
| `0ee3c82_graph_soa_indexer` | -11 | 0 | 0.000 | 103.20 | 108.98 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1` |
| `0ee3c82_graph_indexer_no_hcpre` | 0 | 256 | 14.624 | 103.20 | 109.06 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1 DS4_CUDA_NO_FUSED_HC_PRE=1` |
| `0ee3c82_graph_indexer_no_qnorm` | 0 | 216 | 15.514 | 103.20 | 109.04 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1 DS4_CUDA_NO_Q_NORM_ROPE_FUSED=1` |
| `0ee3c82_graph_indexer_no_kvrope` | 0 | 256 | 15.577 | 103.20 | 108.97 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1 DS4_CUDA_NO_KV_ROPE_STORE_FUSED=1` |
| `0ee3c82_graph_indexer_no_attnrope` | 0 | 256 | 15.625 | 103.20 | 109.02 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1 DS4_CUDA_NO_ATTN_OUTPUT_ROPE_LOW_FUSED=1` |
| `0ee3c82_graph_indexer_down_tile8_rowspan` | 0 | 256 | 15.601 | 103.20 | 109.01 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1 DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN=1` |
| `0ee3c82_graph_no_q8_cache_x` | 0 | 197 | 15.541 | 103.20 | 108.99 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_NO_Q8_CACHE_X=1` |

## Historical checkpoints

| Variant | Return | Tokens | trace_tps | Peak CUDA GiB | Peak used GiB | Env |
|---|---:|---:|---:|---:|---:|---|
| `origin_main_no_env` | 0 | 256 | 13.800 | 103.20 | 109.39 | `{}` |
| `01662bc_tune_f16_decode_no_env` | 0 | 256 | 14.322 | 103.20 | 109.52 | `{}` |
| `5ee0970_q8_cache_x_no_env` | 0 | 256 | 14.502 | 103.20 | 109.36 | `{}` |
| `81a1a3b_fused_hc_rope_no_env` | 0 | 256 | 15.423 | 103.20 | 109.41 | `{}` |
| `81a1a3b_fused_hc_rope_no_hcpre` | 0 | 256 | 14.545 | 103.20 | 109.04 | `DS4_CUDA_NO_FUSED_HC_PRE=1` |
| `81a1a3b_fused_hc_rope_no_qnorm` | 0 | 256 | 15.376 | 103.20 | 108.95 | `DS4_CUDA_NO_Q_NORM_ROPE_FUSED=1` |
| `81a1a3b_fused_hc_rope_no_kvrope` | 0 | 256 | 15.352 | 103.20 | 108.96 | `DS4_CUDA_NO_KV_ROPE_STORE_FUSED=1` |
| `81a1a3b_fused_hc_rope_no_attnrope` | 0 | 256 | 15.342 | 103.20 | 108.95 | `DS4_CUDA_NO_ATTN_OUTPUT_ROPE_LOW_FUSED=1` |
| `bcc390e_skip_ordered_no_env` | 0 | 256 | 15.400 | 103.20 | 109.26 | `{}` |
| `dbb6cd7_moe_tile8_no_env` | 0 | 256 | 15.391 | 103.20 | 109.17 | `{}` |
| `dbb6cd7_moe_tile8_enabled` | 0 | 256 | 15.399 | 103.20 | 108.96 | `DS4_CUDA_MOE_DOWN_TILE8_ROWSPAN=1` |
| `cfd44c3_graph_update_no_env` | 0 | 256 | 15.218 | 103.20 | 109.24 | `{}` |
| `cfd44c3_graph_update_graph` | 0 | 256 | 15.469 | 103.20 | 109.00 | `DS4_CUDA_GRAPH_DECODE=1` |

## Old PR branch check

| Variant | Return | Tokens | trace_tps | Peak CUDA GiB | Peak used GiB | Env |
|---|---:|---:|---:|---:|---:|---|
| `old_pr_no_env` | 0 | 256 | 13.597 | 103.20 | 109.26 | `{}` |
| `old_pr_graph` | 0 | 256 | 13.806 | 103.20 | 109.04 | `DS4_CUDA_GRAPH_DECODE=1` |
| `old_pr_graph_soa_indexer` | -11 | 0 | 0.000 | 103.20 | 109.02 | `DS4_CUDA_GRAPH_DECODE=1 DS4_CUDA_Q8_SOA_CACHE=1 DS4_CUDA_INDEXER_SCORE_TOPK_FUSED=1` |

The crash signature for graph plus SoA is:

```text
ds4: CUDA attention_output_low_q8_rope launch failed: operation failed due to a previous error during capture
ds4: CUDA synchronize failed: operation not permitted when stream is capturing
ds4: Metal synchronize after graph eval failure also failed
ds4-agent: cuda decode failed
```

The old PR branch shows the same class of failure, with `attention_output_low_q8` instead of `attention_output_low_q8_rope`.

## What to do next

1. Build the final candidate branch from `origin/main` around the compiled GB10 F16/HC decode tuning first.
2. Reintroduce graph decode only after the compiled path is proven stable in agent testing.
3. Leave SoA out of the final branch unless it is redesigned to be capture-safe and shows a real decode or memory win.
4. Treat indexer fused as a separate prefill/indexer optimization, not part of the agent decode speed claim.
