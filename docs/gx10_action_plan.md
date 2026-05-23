# GX10 Action Plan

This plan keeps the speed work scoped after the upstream rebase and fork scan.
The target remains quality-preserving GB10 decode toward 20 t/s. The accepted
daily-driver configuration is still `exact_fast` at `ctx=100000`.

## Ground Rules

- Keep `docs/road_to_20_ts.md` updated after every benchmark or code change.
- Treat `exact_fast` as the production baseline unless a row passes both speed
  and quality gates.
- Do not integrate external fork code just because it exists; mine ideas first.
- Do not promote reduced-K or MTP rows without an explicit quality decision.
- Prefer small, reversible experiment branches for invasive kernel work.
- For the next full-quality push, use a strict best-effort timebox: at most two
  targeted kernel prototypes, or about one short working day of implementation
  and profiling. Stop earlier if neither prototype clears the speed smoke.

## Quality Gates

Use `ds4-eval` in addition to deterministic logprob checks and coding eval. It
is not a leaderboard score; it is a capability regression gate for this local
inference path.

Fast sanity gate after any candidate build:

```sh
python3 tuning/gx10_matrix.py ds-eval exact_fast \
  --questions 4 --tokens 1024 --nothink --seed 1
python3 tuning/gx10_matrix.py ds-eval <candidate-row> \
  --questions 4 --tokens 1024 --nothink --seed 1
```

Capability smoke before spending time on a full run:

```sh
python3 tuning/gx10_matrix.py ds-eval-suite exact_fast <candidate-row> \
  --questions 12 --tokens 4096 --think --seed 1
```

Long gate only for a candidate that has already passed speed, logprob, coding
canary, and the 12-question `ds4-eval` smoke:

```sh
python3 tuning/gx10_matrix.py ds-eval-suite exact_fast <candidate-row> \
  --questions 92 --tokens 16000 --think --seed 1
```

Accept a candidate only if it keeps coding eval parity and does not regress the
`ds4-eval` pass count versus the same-run `exact_fast` baseline. If the kernel
is meant to be numerically exact, any deterministic logprob/token divergence is
enough to park it before long eval.

## Phase 1: Stabilize Measurement

Goal: make every future claim comparable.

1. Run the core matrix at `ctx_alloc=100000`, `ctx_start=8192`,
   `ctx_max=8192`, `gen_tokens=128`.
2. Run the same core matrix once against the `origin/main` worktree if we need a
   fresh branch-vs-main percentage.
3. Store summaries in `tuning/gx10_matrix_results/`.
4. Append the result table and decision to `docs/road_to_20_ts.md`.

Command:

```sh
python3 tuning/gx10_matrix.py bench-suite core \
  --ctx-alloc 100000 --ctx-start 8192 --ctx-max 8192 --gen-tokens 128
```

## Phase 2: Exact-Safe Sweep

Goal: prove whether any existing env row still has unexplored value.

1. Run the `exact` group over `2048..65536`.
2. Reject rows that do not beat `exact_fast` by at least 1-2%.
3. For rows that beat it, run canary coding eval.
4. For rows that pass canary, run `repeat=3` full coding eval.

Command:

```sh
python3 tuning/gx10_matrix.py bench-suite exact \
  --ctx-alloc 100000 --ctx-start 2048 --ctx-max 65536 \
  --gen-tokens 128
```

## Phase 3: Tradeoff Rows As Research Only

Goal: quantify speed/quality tradeoffs without confusing them with exact decode.

Rows:

- `k3_renorm`
- `k2_renorm`
- `k6_0_2_k3_renorm`
- `mtp_quality`
- `mtp_attn_b_soa_batch2`

These rows can be useful for understanding where 20 t/s lives, but they do not
replace `exact_fast` unless the user explicitly accepts the quality tradeoff and
the coding gate is clean.

## Phase 4: Next Kernel Work

Only start this after Phase 1 and Phase 2 are current in the road log.

The previous broad "native routed pack" and Q8 projection toggle space has
already been tested. The remaining exact-safe work should be narrower:

1. Q8 projection kernels that preserve exact output ordering but reduce real
   per-row work or weight traffic. Do not retry broad SoA/cuBLAS/cache-X flags.
2. Routed MoE gate/up and down paths that preserve full K=6 and the current
   qwarp-friendly weight stream. Do not retry naive row-major or block-paired
   packs.
3. Avoid FFN shared/routed scheduling-only probes unless the graph-capture
   dependency changes; second-stream overlap was slower and shared-first broke
   decode capture.
4. Entrpi MMQ/layer-graph ideas only as a dedicated port branch, not mixed into
   the current exact-fast branch.

Stop conditions:

- no candidate reaches at least +3% over `exact_fast` in the 8192/128 speed
  smoke;
- a candidate changes deterministic logprobs/tokens when it is intended to be
  exact;
- a candidate passes speed but fails coding canary or `ds4-eval` smoke;
- the two-prototype / one-working-day timebox expires without a promoted
  candidate.

## Decision Log Format

Append entries to `docs/road_to_20_ts.md` like this:

```md
### YYYY-MM-DD continuation - short title

- Branch/commit:
- Command(s):
- Context/token settings:
- Result artifacts:
- Speed result:
- Quality result:
- Decision:
- Next action:
```

This keeps the historical road document useful as the single source of truth.
