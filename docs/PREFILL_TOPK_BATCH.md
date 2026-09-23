# Exact causal batched top-k: GB10 V4.1 experiment

2026-09-23, `feature/external-engram-gguf`, implementation2452f795.
Not deployed; `DS4_CUDA_V41_TOPK_BATCH=1` is explicitly opt-in.

## Change

The V4.1 causal top-k loop previously dispatched one row at a time. The
experimental path reuses the streaming exact top512 selection kernel, with
physical stride separate from each row's visible bound. One block per row;
no future entries read. Same packed score/index order for non-NaN values.
Visible NaNs trigger the original path: integer-bit scan, a4-byte flag read,
and stream synchronization. This safety cost is included in every benchmark.
No additional score matrix is allocated. Single GPU,32..65535rows,width>8192.
Option0/unset and existing top-k disable flags retain the original path.

## Correctness evidence

- Expanded causal oracle PASS through131071 visible entries, ratios1/2,
  row counts1/2/31/32/33/127/128/129, ties, NaNs, both infinities, signed zero,
  attractive/NaN future entries, and output sentinel.
- Complete GPU numeric suite PASS (`prefill-topk-full-unit-v1`).
- Initial NaN ordering failure retained in `prefill-topk-unit-on-v1`;
  corrected version PASS in `prefill-topk-unit-on-v2`.
- Model gates: all256 float32 vocabulary vectors (129280values each) identical
  between OFF and ON, and OFF matches the pre-change native reference.
  SHA256 `9dee9a5d2cc7d4a254716e50e07cd995e58b2c885fdea7fcdd36b36eb47a8298`.
- GPU memory sanitizer and near256K real-output quality gate: pending.

## Clean model timing

65536 populated tokens;262144 allocated context;256teacher-forced decode.
Cache argument72GB, pool8, decode graphs0, stage-sync OFF, no profiling/dumps.
Internal expert/model GGUF, alternate Engram GGUF on Lexar.
No cache drops or OS/desktop/service changes. Native sm_121a.

| Pair/order | OFF prefill tok/s | ON prefill tok/s | OFF decode | ON decode |
|---|---:|---:|---:|---:|
|1 ON/OFF|364.48|405.03|9.51|9.55|
|2 OFF/ON|364.13|404.56|9.54|9.53|

Both pairs clean, process swap0 throughout. Means364.305->404.795tok/s,
**+11.114% prefill**. Decode9.525->9.54tok/s (effectively unchanged).
Whole-job wall time~210.11->192.13s. These are two counterbalanced pairs,
not a confidence interval or a claim about populated256K performance.

Reproduce with `speed-bench/prefill_stage_suite.py --toggle
DS4_CUDA_V41_TOPK_BATCH`, supplying the runner/model/Engram paths and reference.
Raw prefix `prefill-topk64k-v1` under
`/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw`.
Each run has source provenance, command, resource samples and summaries.
Binary SHA256 `e47291805525ecf8bf8a203f1643760fa29e9b721a35b934277e2f0db9cdda18`.

Stage-sync is a separate unpromoted experiment; do not add its +6.38% estimate
to this result or assume their benefits combine. See `PREFILL_STAGE_SYNC.md`.
