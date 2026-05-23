# GX10 Fork Reconnaissance

Date: 2026-05-22; updated 2026-05-24 with supplied decode-throughput fork
claims.

Scope:

- base repository: `antirez/ds4`
- upstream baseline: `upstream/main` at `8d57664`
- current branch: `gx10-cuda-graph-decode`; original scan at `dca0d8c`,
  supplied-claim audit at `7fee9f1`
- public fork scan: 930 forks returned by GitHub, 66 prefiltered as interesting
  by recent push, stars/forks, or non-`main` default branch
- local refs fetched under `refs/remotes/scan/*` only; no working branch changes

Commands used:

```sh
gh api --paginate 'repos/antirez/ds4/forks?per_page=100&sort=newest'
git fetch upstream '+refs/heads/*:refs/remotes/upstream/*'
git fetch --no-tags https://github.com/<fork>.git \
  +refs/heads/<branch>:refs/remotes/scan/<name>
```

## Upstream State

| Ref | Commit | Status |
| --- | --- | --- |
| `upstream/main` | `8d57664` | current main; already merged into this branch |
| `upstream/rocm` | `7a751eb` | 1 commit ahead of an older main, 85 commits behind current main; AMD backend only |
| `upstream/responses-api` | `acb40bf` | 0 ahead, 74 behind current main; obsolete branch |

The relevant upstream changes have already been accounted for by the rebase onto
`8d57664`. Future matrix baselines should compare against the separate
`/home/alessandro/projects/ds4-main-baseline` worktree, not the old pre-rebase
state.

## Shortlist

| Fork / branch | Ahead vs `upstream/main` | Why it matters | Applicability to GB10 CUDA branch |
| --- | ---: | --- | --- |
| `Entrpi/ds4:mmq-step-A-full-layer-graphs` | 211 | `cuda/mmq` vendor import, MMVQ decode path, per-layer CUDA graph replay, VMM arena, MTP proof harness | strongest external CUDA lead, but invasive and not env-compatible |
| `Entrpi/ds4:pr-prep-2026-05-18` | 132 | docs/bench harness for the same MMQ/VMM/MTP direction | use as handoff reference if porting Entrpi work |
| `audreyt/ds4:main` | 143 | Metal 4 / M5 Tensor prefill, direction steering, KV/server fixes | mostly Apple/M5; steering and KV fixes may be portable |
| `ejpir/ds4-hip:main` | 108 | ROCm/HIP backend, prequant decode, ngram speculative drafter, launch-overhead work | AMD-specific; useful conceptual reference for prequant/scratch decisions |
| `cchuter/ds4:feat/q8-k-routed-experts` | 23 | Q8_K/Q8_0 routed expert support on Metal/CPU | model-format/Metal work, not immediate GB10 speed |
| `NimbleMarkets/ds4:nm-shared` | 16 | logging/error-boundary refactors, CI CUDA config | infrastructure only |
| `reffdev/ds4:fused-matmul-mtp` | 7 | fused small-batch MTP verifier, prefix-K commit | already aligned with the MTP research history; mine for verifier design |
| `spmurrayzzz/ds4:main` | 8 | CUDA decode fast-path kernels, device top1, split-sync avoidance | early source/inspiration for our CUDA line; not patch-identical after rebase |
| `Swival/ds4-m5:m5opts` | 42 | M5 Metal Tensor prefill experiments | Apple/M5 only |
| `Chida82/ds4:steeringMoE_GPT` | 2 | expert steering support and sweep tooling | feature/quality axis, not decode speed |
| `mirkodandrea/ds4:moe-expert-sharding` | 26 | expert sharding, remote expert server, selective weight loading | memory/distribution route, not single-machine exact speed |
| `ldclabs/ds4:main` | 3 | Rust inference engine | separate implementation |
| `rcarmo/go-ds4:main` | 99 | Go implementation with CUDA/Metal experiments | separate implementation; concept mining only |

## Entrpi CUDA/MMQ Line

This is the main item not already represented in `docs/road_to_20_ts.md`.

Notable pieces:

- vendored llama.cpp `cuda/mmq` and `mmvq` kernels;
- Q8_0 dispatcher with `mmq` default and `warp8`/cuBLAS fallbacks;
- `DS4_CUDA_LAYER_GRAPHS` default-on per-layer decode graph replay;
- `DS4_CUDA_MOE_GRAPHS` routed-MoE graph replay;
- in-process CUDA VMM weight arena and separate weight-server proof harness;
- MTP exact decode2/decode3 proof tooling and comparison scripts;
- branch-specific docs under `misc/cuda-env-vars.md` and `misc/cuda-mtp/README.md`.

Important caveat: this branch documents reduction-order drift under VMM/MMQ and
routes the MTP verifier back to legacy kernels by default. It is not a simple
drop-in exact-safe toggle. If we evaluate it, do it on a dedicated port branch
with the same coding/logprob gates used for `exact_fast`.

Public benchmark caveat: the branch's own `speed-bench/gb10_spark.csv` reports
**13.74 t/s** at `ctx=8192, gen=128` and **11.70 t/s** at
`ctx=65536, gen=128`. Those numbers are below the current branch's exact-fast
band, so this line is still a source of ideas, not a direct performance
shortcut.

## MTP Forks

`reffdev/ds4:fused-matmul-mtp` and Entrpi's MTP commits overlap with the MTP
history in `docs/road_to_20_ts.md`: prefix commit, small-batch verifier fusion,
top2/certificate proof ideas and proof harnesses. The current branch already
has a quality-first MTP matrix row, but these forks suggest a separate MTP
port/review track if we decide to revisit speculation.

Recommendation: do not spend more time on MTP orchestration until either:

- a verifier kernel change reduces exact replay cost, or
- an imported proof harness shows byte-identical output and a real speed win on
  the 12-task coding eval.

## Exact-Speed Implications

The fork scan did not reveal a small missing env combination for exact GB10
decode. Public work clusters into:

- Apple M5/Metal prefill work;
- ROCm/HIP backend work;
- MTP/speculative proof work;
- memory/distribution work;
- feature work such as steering or agent tooling;
- one large CUDA/MMQ/layer-graph port candidate.

Therefore the immediate matrix on this branch should stay focused on the
existing exact-safe rows. The next non-trivial speed branch should be one of:

1. Port/evaluate Entrpi's CUDA MMQ/layer-graph line in isolation.
2. Cherry-pick small correctness/infrastructure fixes such as the Audreyt KV
   cache guard if they reproduce.
3. Continue original kernel-level exact work on Q8 projection reads and routed
   MoE, using fork results only as design references.

## 2026-05-23 External Project Check

A follow-up search around llama.cpp/ggml CUDA, DeepEP, DeepGEMM, Tutel, and
similar MoE inference kernels did not expose a small exact-safe DS4 patch that
we can drop into this branch. The useful signal is architectural:

- ggml/llama.cpp CUDA keeps a split between DP4A vector kernels and MMQ/MMA
  tensor-core kernels; that validates an isolated MMQ/MMVQ port track, not
  another local flag around the current warp8 kernels.
- DeepEP/DeepGEMM/Tutel style work is mostly expert dispatch, grouped GEMM, FP8
  or multi-GPU routing. Those ideas are high-throughput serving directions, but
  they do not directly solve this single-token, Q8_0/IQ2 exact decode path.
- The remaining local work should therefore either be a real native-layout or
  tensor-core port branch, or narrow exact probes on the four measured frontier
  kernels. MoE down remains closed unless a profile changes.

## 2026-05-24 Supplied Decode-Claim Audit

The later user-supplied ranking was checked against local refs and upstream PR
metadata. It did not reveal a missing no-MTP exact-fast patch, but it did add a
bounded MTP-only idea set from TrevorS.

| Source | Claim checked | Current assessment |
| --- | --- | --- |
| `Entrpi/ds4:mmq-step-A-full-layer-graphs` / PR #187 | MMQ/MMVQ, VMM, layer graphs, MTP proof harness | Already classified as an invasive port. Public GB10 CSV is below our exact-fast band; local Entrpi-inspired MMQ/VMM/alignment probes were negative. |
| `amarrmb/ds4:thor-sm110-f16-dispatch` / PR #121 | skip ordered F16 path on Blackwell | Already present/equivalent in this branch through `cuda_skip_ordered_f16_matmul()`. |
| `cghart/ds4:main` | F16 vec8, CTA-parallel Q8 GEMV, output fusion | Concepts already mined; comparable local Q8/HC/output/F16 probes are documented as neutral or negative. |
| `TrevorS/ds4:mtp-beats-plain*` | CUDA graphs, fused kernels, small-N MTP verifier kernels | Graph/fused-Q pieces overlap with local work. Still-new candidates are MTP-only: GPU argmax for top1 sites and small-N shared-weight Q8 batch fallback. |
| `ddxxlao/ds4:codex/cuda-partial-weight-cache` / PR #153 | partial weight cache for limited VRAM | Useful memory fallback, wrong default trade-off for full-resident GB10 decode. |
| `DominikBucko/ds4:cuda-fp16-kv-cache` / PR #191 | FP16 KV and long-context GH200 work | Future memory/long-context track, not a current 100k-context decode-speed lever. |
| `audreyt/ds4` `a464840` line | WMMA indexer/top-k prefill optimization | Prefill-oriented; no generation-throughput signal. |

Actionable follow-up if MTP is reopened:

1. Port a dedicated `ds4_gpu_argmax_tensor()` and replace the current
   `indexer_topk(top_k=1)` MTP top-id sites.
2. Probe TrevorS's `n_tok=2..4` shared-weight Q8 batch kernel only inside the
   MTP verifier path.
3. Retest the small-N no-sort routed-MoE path only as an MTP batch-verifier row,
   because local `DS4_CUDA_MOE_K2_DIRECT_GATE=1` history already covers the
   same idea class for K=2.
