# GB10 upstream/rebase validation, 2026-09-14

This run compares upstream `main` at `9139e2ae`, the previous
`dgx-performance` tip `b17c1fda`, and the rebased candidate `dfae9e91` on one
DGX Spark / GB10. The final local branch tip is `ba07537d`; its only additional
change links two CUDA smoke tests with `ds4_image.o` and does not alter the
benchmarked inference code.

All measurements use the Flash 0731 Q2 model (SHA-256
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`),
the same populated 250086-token prompt, a 262144-token allocation, 128 generated
tokens, greedy decoding, and `DS4_CUDA_DECODE_GRAPHS=0`.

| Candidate | Runs | Prefill tok/s | Decode tok/s | Steady decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| upstream `9139e2ae` | 1 | 736.40 | 12.20 | 12.23 |
| old DGX `b17c1fda` | 2 | 755.18 | 12.61 | 12.64 |
| rebased DGX `dfae9e91` | 2 | 737.39 | 12.58 | 12.61 |

The rebase keeps about 3.1% more decode throughput than the new upstream run,
but loses about 2.4% prefill throughput relative to the old DGX tip. Its decode
throughput is within 0.3% of the old tip. All five frontier-logit dumps are
byte-identical (SHA-256
`bf06afd61f33781fe1af68697abb2677db2fcb8a8eca738e4c4df9c605decf9f`).

The rebased candidate also passed:

- real 254932-token recall, 16/16 exact facts, with output byte-identical to the
  previously validated result;
- long-context, logprob-vector and local-golden-vector tests;
- CUDA long-context regression, fused-decode, split-K, Q8 scratch and DSpark
  parity smoke tests;
- server, agent and web-recovery frontend tests.

Peak runtime planning was 86.48 GiB. The benchmark process used about 1.8 GiB
RSS, the GPU allocation was about 108477 MiB, and no process swap was observed.
The compiler emits an upstream warning around the Qwen n-gram `pread` buffer;
this Flash path did not exercise it.

Raw CSVs are stored beside this report. Full logs, timings, logits, recall output
and build logs are retained in
`/home/alessandro/projects/ds4-benchmark-results-20260914/` on the benchmark
host.
