# CUDA SSD ordered-read A/B — 2026-10-26

## Result

On the active V4.1 Flash 1017 Q2 service profile, the default ordered selected-
expert reads completed a 30,475-token prefill faster than the previous read
order. The same 76-token assignment-ledger answer was returned byte for byte
in all four valid runs, and all 16 expected assignments were correct.

| Selected-read mode | Prefill times | Mean | Mean prefill rate |
| --- | ---: | ---: | ---: |
| Ordered (default) | 110.478 s, 108.637 s | 109.558 s | 278.16 tok/s |
| Previous order (`DS4_CUDA_DISABLE_SSD_SORT_SELECTED_READS=1`) | 117.613 s, 117.518 s | 117.566 s | 259.22 tok/s |

Mean prefill time fell 6.81% in this sample. The complete request wall times
were 119.023 s and 116.965 s with ordered reads, and 125.799 s and 125.812 s
with the previous order.

## Method

- Runtime: `feature/external-engram-gguf` at `24b7b7ab`, based on upstream
  `ee7e8405`; native CUDA `sm_121a` build.
- Model: `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` on NVMe, with its
  matching external Engram GGUF on the Lexar.
- Service profile: CUDA SSD streaming, 80 GiB expert-cache budget, 262144
  context, Engram readers 8, top-k batch enabled, stage-sync and decode graphs
  disabled.
- Prompt: `tests/long_context_story_prompt.txt`, sent through
  `/v1/chat/completions` with temperature 0, `reasoning_effort=none`, and a
  200-token completion limit. The rendered fixture was split into its system
  and user messages. Every valid run reported 30475 prompt tokens, 76 output
  tokens, and `cached_tokens=0`.
- Modes ran in counterbalanced order: ordered, previous, previous, ordered.
  The two previous-order runs and the second ordered run used distinct
  temporary KV directories. The first ordered run used the service's normal
  KV directory and also reported no cached prompt tokens.
- The server journal supplied the prefill durations. Each response matched
  exactly:

  ```text
  Bob=34
  Alice=52
  Clara=71
  Diego=93
  Elena=16
  Felix=88
  Greta=47
  Hugo=29
  Iris=64
  Jonas=12
  Kira=81
  Leo=39
  Marta=76
  Nadia=23
  Owen=58
  Priya=97
  ```

## Limits

This is two runs per mode on one prompt and one model. The OS file cache warmed
across the sequence; the mode order was reversed for the second pair, but the
test was not a cold-storage benchmark. It checks generated answer equality and
the fixture's expected facts, not full-vocabulary logit equality. Keep the
result as a promising local observation, not a universal or promotion-quality
performance claim.
