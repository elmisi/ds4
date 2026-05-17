# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DwarfStar 4 (`ds4`) is a self-contained native inference engine **dedicated to DeepSeek V4 Flash**. It is intentionally not a generic GGUF runner. See `README.md` for motivations and the full feature list, `AGENT.md` for project rules, and `CONTRIBUTING.md` for the regression-test protocol expected on PRs.

## Build

The Makefile selects backends by platform. On Linux, plain `make` prints help instead of guessing a CUDA arch.

```sh
make                  # macOS Metal: ds4, ds4-server, ds4-bench, ds4-eval
make cuda-spark       # Linux CUDA, DGX Spark / GB10 (no explicit -arch on purpose)
make cuda-generic     # Linux CUDA, local GPU (CUDA_ARCH=native)
make cuda CUDA_ARCH=sm_120   # explicit arch
make cpu              # CPU-only diagnostics build (defines DS4_NO_GPU)
make clean
```

The CPU build reuses the same sources via `-DDS4_NO_GPU`. CPU is **reference/debug only**: do not benchmark it, and do not run large CPU inference on macOS (kernel VM bug — has crashed the kernel).

## Tests

```sh
make test                         # builds and runs ./ds4_test (== --all)
./ds4_test --server               # API/render/streaming/tool/KV-disk bookkeeping
./ds4_test --logprob-vectors      # token bytes + top-logprobs vs official vectors
./ds4_test --long-context         # fact-recall regression on long prose prompt
./ds4_test --tool-call-quality    # DSML emission, fast and exact paths
./ds4_test --metal-kernels        # Metal kernel numeric checks (macOS)
make cuda-regression              # Linux CUDA only: builds + runs tests/cuda_long_context_smoke
```

Override inputs when the default `ds4flash.gguf` symlink isn't what you want:

```sh
DS4_TEST_MODEL=/path/to/model.gguf ./ds4_test --logprob-vectors
DS4_TEST_VECTOR_FILE=/path/to/official.vec ./ds4_test --logprob-vectors
DS4_TEST_LONG_PROMPT=/path/to/prompt.txt ./ds4_test --long-context
```

Speed regressions use `./ds4-bench` with a fixed prompt (`speed-bench/promessi_sposi.txt`) over a context sweep — see `CONTRIBUTING.md` for the exact before/after methodology required on backend changes.

## Architecture

The engine has a deliberately small file surface; most logic lives in a handful of C translation units.

- **`ds4.c` + `ds4.h`** — model loading (mmap-backed GGUF, do not eagerly copy), tokenizer, CPU reference kernels, graph scheduling and dispatch into Metal/CUDA, sessions, and the disk-cache **payload** serialization (the `DSV4` blob inside `.kv` files).
- **`ds4_gpu.h`** — backend-agnostic GPU surface. Both Metal and CUDA implement it. Code outside the backend files should not know tensor internals.
- **`ds4_metal.m` + `metal/*.metal`** — Objective-C runtime and compute kernels. Only place Objective-C is allowed.
- **`ds4_cuda.cu` + `ds4_iq2_tables_cuda.inc`** — CUDA backend. The GB10/sm_121 decode path is selected at runtime for one-token F16 matmuls (faster on DGX Spark while batched prefill stays on cuBLAS). `DS4_CUDA_FORCE_ORDERED_F16_MATMUL=1` reverts to the older ordered kernels for A/B testing.
- **`ds4_cli.c` + `linenoise.{c,h}`** — interactive REPL, multi-turn transcript, `/think` / `/nothink` / `/ctx` / `/read` commands.
- **`ds4_server.c` + `rax.{c,h}`** — OpenAI/Anthropic/Responses-compatible HTTP server. Request parsing runs in client threads; **inference is serialized through one graph worker** (no multi-request batching). Owns the disk KV cache policy and the tool-call exact-replay map.
- **`ds4_bench.c`** — instantaneous prefill/generation throughput at context frontiers (not whole-run averages); writes CSV. Use this for speed regressions.
- **`ds4_eval.c`** — TUI/plain capability harness over an embedded 75-item mix (GPQA Diamond / SuperGPQA / AIME 2025). Not a leaderboard — a hard regression suite.
- **`tests/`** — `ds4_test.c` runner plus vector/prompt data; `cuda_long_context_smoke.c` for the CUDA build.
- **`gguf-tools/`** — offline tooling for GGUF generation, imatrix, and quality scoring. Separate build (`make -C gguf-tools quality-score`).
- **`dir-steering/`**, **`speed-bench/`**, **`docs/`** — steering vectors, bench CSVs/plots, and tuning notes (notably `docs/GX10_CUDA_TUNING.md`).

### Server-side concepts worth knowing before editing

1. **Single live KV checkpoint in RAM.** Stateless clients resend the whole conversation; the server first does an exact token-prefix check, then falls back to rendered-bytes-vs-decoded-checkpoint comparison so a longer version of the same prompt reuses the prefix. Only one session is live in memory — disk cache is the resume mechanism across sessions and restarts.
2. **Disk KV cache format.** `<sha1>.kv` keyed by SHA1 of the rendered byte prefix. The file is a fixed `KVC` header + rendered text + `DSV4` payload (tokens, next-token logits, raw/compressed/indexer KV rows) + optional `KTM` tool-id map. Plain `read`/`write` I/O — explicitly **not** mmap so restores don't pile more VM mappings on top of the model. Snapshots are written at four moments: `cold` (post-prefill, suffix trimmed + aligned), `continued` (next absolute aligned frontier, ~every 10k tokens), `evict` (before another session replaces the live one), `shutdown`.
3. **Tool-call replay.** The model emits DSML; clients send back normalized JSON. The primary path is **exact replay**: each tool call gets an unguessable API ID mapped to the exact sampled DSML block (bounded RAM map, persistable inside `.kv` files). **Canonicalization is only the fallback** — used if exact replay is missing or `--disable-exact-dsml-tool-replay`. After a tool turn the server diffs live tokens vs. the next rendered prompt and either rewrites the live checkpoint or falls back to an older disk snapshot, replaying only the suffix.
4. **Sampling split during tool calls.** When the model is emitting DSML *syntax* (tags, parameter headers, JSON punctuation, closers) sampling is forced to `temperature=0`. Argument *payloads* (`string=true` bodies, JSON string values) use the request's normal sampling. Do not collapse these into one mode — deterministic decoding on long file bodies produces repeated text.
5. **Thinking modes are distinct.** Non-thinking / thinking / Think Max are not the same path. `reasoning_effort=max` only upgrades to Think Max when the context is large enough; OpenAI `xhigh` stays in normal thinking. `deepseek-chat` is the non-thinking alias.

## Project rules (from AGENT.md — do not violate)

- **No C++.** This is a C codebase; Objective-C only inside `ds4_metal.m`.
- **No permanent semantic variants behind flags.** Diagnostic switches are fine *only* to validate the one release path; they must not become parallel maintained code.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep public APIs narrow: CLI/server code must not poke tensor internals — go through `ds4.h` / `ds4_gpu.h`.
- Comments belong **beside the implementation**, explaining *why* a shape/ordering/cache boundary/memory choice exists. Prefer this over separate design docs.
- Correctness before speed: do not keep a faster path with unexplained attention, KV, or logits drift. The CPU path is the reference oracle.
- Don't run multiple huge model processes concurrently — the instance lock is intentional.

## Debugging quick reference

```sh
./ds4 --dump-tokens -p "..."                                      # tokenize and exit; shows DSML specials
./ds4 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."   # greedy + top-k alternatives
./ds4-server --trace /tmp/ds4-trace.txt ...                       # full session: prompts, cache decisions, tool events
```

For any model-output bug report, the trace file is the artifact to capture.
