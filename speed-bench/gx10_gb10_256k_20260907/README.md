# GB10 256K validation and promotion — 2026-09-07

## Decision and provenance

Promote the validated rebased DGX candidate, retaining target-only decoding
as the general 256K default. Do not replace the DGX branch with plain main.

| Variant | Commit | Description |
|---|---|---|
| A | `9dfa943257594f380dbceb1f83de2643c0d4da87` | Previous DGX source, rebuilt in isolation |
| B | `b6af0adf8ca97c89145c9f9c15be70c9fd6c4507` | New origin/main, matching upstream at validation |
| C | `73bbdffca5726770d31227b9da8258018cc01103` | All 37 DGX commits rebased onto B |

Hardware: one NVIDIA GB10, aarch64, 121 GiB physical RAM, driver 595.84,
CUDA toolkit 13.0.88. `CUDA_ARCH=sm_121` maps to compute_121a/sm_121a and MXF4.
The model is Flash 0731 Q2, not GLM:

```text
DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
SHA256 ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0
DeepSeek-V4-Flash-DSpark-support-0731.gguf
SHA256 7e319924541db3f7a163ed7e11d7532a70d48228ab59d36cb81e1d4511885360
```

Rebase conflict resolutions retained both Spark detection paths, upstream
Metal prefill preparation and the DGX non-Apple full-layer threshold,
the union of test cleanup targets, and the reorganized upstream README.
No inference-source edits were introduced after validation. Promotion moves
the branch to the already-tested history without adding a merge commit.
The following documentation commit does not alter inference code.

## Ordinary decoding: three alternating repetitions

Order A/B/C, C/B/A, B/C/A. Identical teacher-forced continuations, 128 tokens
per frontier. Context allocation 262144; actual frontiers shown below.

| Actual context | A decode t/s | B decode t/s | C decode t/s | C/A | C/B |
|---|---:|---:|---:|---:|---:|
| 250000 | 12.54 | 12.31 | 12.78 | +1.91% | +3.82% |
| 254096 | 12.48 | 12.27 | 12.72 | +1.92% | +3.67% |
| 258192 | 12.42 | 12.23 | 12.67 | +2.01% | +3.60% |

Initial prefill medians: A 758.00, B 759.51, C 757.72 t/s; append around
593 t/s on all three. Overall benchmark medians are about 399 seconds:
the decode gain is not an equivalent end-to-end improvement.
See [medians](benchmark-medians.csv) and all nine `bench-*-r*.csv` files.
The last CSV cache-size zero means no snapshot at that frontier, not zero KV.

Build and representative benchmark command, from each exact checkout:

```sh
make -j8 cuda CUDA_ARCH=sm_121
DS4_CUDA_DECODE_GRAPHS=0 DS4_BENCH_FORCE_SNAPSHOT=1 ./ds4-bench \
  --cuda -m /path/to/the/model.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-alloc 262144 --ctx-start 250000 --ctx-max 258192 \
  --step-incr 4096 --gen-tokens 128 --teacher-forced-decode \
  --prefill-chunk 4096 --dump-frontier-logits-dir /path/to/run/logits
```

The runner cleared inherited DS4 overrides, then set decode graphs off,
forced benchmark snapshots and DSpark statistics. CSV header emission is
not TTFT; `gen_first_ms` measures the first decode evaluation, not cold startup.

## Quality and correctness gate

- All 12 A/C full-logit comparisons are identical: nine benchmark frontiers
  and three replay stages. [Full comparison metrics](logit-comparisons.json).
- At 254000 tokens, snapshot save/restore and replay of 128 fixed tokens
  are byte-identical within each variant. Prefix reuse and append to 258096
  succeed. A/C also have identical logprobs and full replay logits.
- Main differs after decode/append; mean NLL on this single continuation is
  A/C 1.870867 versus B 1.914225. This is not a general quality ranking.
- All six A/B/C recall runs, temperatures 0 and 1, return 16/16 distributed
  facts with exact formatting at 254932 actual tokens. [Scores](recall-scores.json).
- Coding requirements were distributed at the start/middle/end of a 254536
  token archive. All six ordinary functions and five speculative/diagnostic
  functions compiled with UBSan and passed 26354 cases each, including reversed
  bounds and INT_MIN/INT_MAX. Ordinary greedy outputs are identical across
  A/B/C but add unwanted Markdown. A/C sampled thinking output also matches;
  B has different reasoning but the same final function.
- Long-context, official logprob/local golden, frontend, CUDA kernel,
  snapshot/session and sampling distribution gates passed. Short fixtures
  are functional gates only, not 256K performance evidence.

The `dgx-benchmark-gate` requires quality evidence before promotion. These
task checks do not guarantee identical quality on every possible workload.

## DSpark and sampling temperature

Single-run screening, not repeated performance qualification. Same target
controls, top-p 1, min-p 0.05 and seed 12345. Sampled code uses thinking high
and a 2048-token budget; greedy code uses no thinking and 512 tokens. Prose
uses no thinking and 1024 tokens, at 254424 actual context tokens.

| Workload/mode | Target decode t/s | DSpark decode t/s | Target total s | DSpark total s |
|---|---:|---:|---:|---:|
| Code greedy t0 | 12.59 | 17.42 | 373.00 | 415.97 |
| Code exact t1 | 12.55 | 13.55 | 384.58 | 448.91 |
| Code opportunistic t1 | 12.55 | 17.69 | 384.58 | 415.57 |
| Prose greedy t0 | 12.60 | 12.25 | 391.22 | 419.18 |
| Prose exact t1 | 12.43 | 11.91 | 398.12 | 425.21 |
| Prose opportunistic t1 | 12.43 | 12.16 | 398.12 | 420.76 |

All five prose outputs are complete Italian scenes (253–284 words), without
obvious loops or repeated eight-word sequences. Minor imperfections were
also found in the ordinary target. This is descriptive inspection, not a
statistical literary-quality score. Speculative outputs differ, so total
times compare completed tasks, not equal-length teacher-forced sequences.

Temperature zero verifies greedy drafts, but batched numerical reductions
can change the continuation. At positive temperature, default opportunistic
acceptance changes the ordinary sampling distribution. `--mtp-exact-sampling`
preserves that distribution algorithmically, not seed-for-seed text identity;
the GB10 seed-batch optimization is disabled in exact mode. `--quality` and
`--dspark-strict` disable speculative acceptance, not make it fast and identical.

Code exact accepted 97/107 proposed tokens but had 159 no-draft cycles out
of 186: conditional acceptance rate alone is misleading. Prose accepted only
11/9/17 draft tokens in 413/436/472 cycles. No verifier errors occurred.
Warm-prefix coding may benefit from faster decode, but a persistent agent
workload was not qualified here. DSpark does not accelerate cold prefill.

## Memory and thermal checks

With chunk4096, DSpark code minimum usable RAM was about 1.42–1.64 GiB.
Code exact recorded 4821 MiB swap-in / 3348 MiB swap-out in system counters;
these are global, not per-process attribution. Ordinary prose greedy had
39 MiB in / 0.035 MiB out. Full VM telemetry began after the first DSpark
greedy run had already started; its earlier swap activity is not quantified.

Matched temperature1/thinking diagnostic at the same 254536-token context:

| Chunk512 measure | Target-only | DSpark exact |
|---|---:|---:|
| Prefill t/s | 558.68 | 558.77 |
| Decode t/s | 12.51 | 13.94 |
| Total s | 495.52 | 501.50 |
| Minimum usable GiB | 10.58 | 6.35 |
| Swap-in / out MiB | 17.29 / 0.22 | 3.98 / 0.00 |

The smaller chunk avoids swap pressure, but does not beat ordinary chunk4096
cold completion. Output lengths differ. [All output metrics](output-metrics.csv).

After the main benchmark, 5-second GPU monitoring observed a maximum 89 C,
24/1649 samples with SW thermal slowdown active and none with HW thermal
slowdown active. End-of-benchmark spot readings were 68–73 C; they do not
prove absence of prefill throttling. No power, clock or cooling changes.
Usable RAM means MemAvailable minus CmaFree while the model is loaded.

## Operational boundary and retained evidence

The agent auto-compacts at 85% (about 223K of 262144); a tool loop can compact
after its first round near 256K. These engine tests retain the actual stated
frontiers; they are not benchmarks of a compacted agent loop.

Agent/server and the other CUDA executables are rebuilt in the operational
worktree during promotion. Services were inactive and are left inactive;
this update does not start a server, enable DSpark, or change the model link.
The previous branch tip is retained in a local backup reference.

Promotion checks passed: forced CUDA rebuild of all five executables without
warnings, agent/server help, both frontend suites, and a fresh real-model
`ds4_test --long-context` functional smoke (30474 tokens, not a new 256K
performance measurement). Host `.text` sections match the candidate across
all five executables. CUDA SASS from the rebuilt agent matches the candidate
after normalizing 25 build-specific anonymous-namespace identifiers; raw
fatbinary/rodata hashes differ, so whole-binary identity is not claimed.

Detailed prompts, scripts, full logits, stdout/stderr, build and frontend logs,
per-process command/commit/environment, code tests and telemetry remain in
the original local `ds4-256k-results` experiment archive. This directory
publishes compact, portable summaries and raw ordinary benchmark CSVs.
The final experiment audit covered 22 inspected responses and 76 successful
structured invocations (including help/frontend/kernel checks, not 76 model
benchmarks), plus standalone tests.

An initial recall invocation incorrectly used `--raw` on a rendered-chat
fixture; it was interrupted without an answer and excluded. CUDA smoke test
linkage on A/C needed a test-only image object overlay. The initial combined
A gate invocation refused a second engine; vector gates passed in separate
processes. None of these harness adjustments modified measured inference code.
