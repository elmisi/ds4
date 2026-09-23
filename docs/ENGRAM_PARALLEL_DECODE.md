# Experimental CUDA Engram decode readers

Latest validation: [native GB10 results](ENGRAM_NATIVE_GB10.md), including the
CUDA build-architecture correction and 254K real-output recall. The initial
measurements below used stale sm_75 objects; use the native report for current
performance claims. Resume state: [ENGRAM_RESUME.md](ENGRAM_RESUME.md).

`DS4_CUDA_ENGRAM_READERS=N` enables a persistent session-owned pool of 1–16
readers. Unset or `0` retains the existing serial path. Start with `8` for
experiments; this is not a validated default. The option applies to CUDA
single-token evaluation, with either internal or alternate Engram storage.

After hashing the current token, the CPU submits both tables' 48 rows. Workers
use the existing `ds4_engram_read` conversion and disjoint output slots. Table 0
jobs have priority. The caller proceeds with GPU work and waits for each table
immediately before uploading its rows at layers 1 and 14. No future-token
prediction or numerical change is involved. Prefill retains its existing
batch reader and prefetch implementation.

The pool copies IDs and table descriptors but does not own the file descriptors.
Every submitted batch is drained on success and error before the next token or
graph teardown. Teardown joins workers before closing tables. Worker failures
are propagated to the decoding caller. A failed partially evaluated token
invalidates the graph. Images retain the existing masked Engram behavior.

Each session owns at most 16 threads and 48 row IDs; there is no model-sized
cache or allocation. Configure the environment before starting the process;
changing reader count during a session is not supported.

## Validation

`make test-engram` exercises pool sizes 1, 2, 4, 8 and 16, repeated submissions,
copied input IDs, exact row conversion, table-specific failures, recovery after
failed reads and teardown with work outstanding. Address/undefined sanitizers
can run the same test. ThreadSanitizer availability depends on the host runtime.

For GPU numerical gates, `DS4_BENCH_DECODE_LOGITS_FILE=/absolute/path/decode.f32`
dumps every target-only evaluation's full vocabulary vector as native F32,
without a header. Use exactly one frontier per invocation. Compare equal-sized
files byte for byte between serial and parallel teacher-forced runs. This checks
the changed decode path, unlike the pre-existing prefill-frontier dump.
Dump work is outside per-token timing but inside total generation wall time;
use separate runs without dumps for performance measurements. Do not use this
diagnostic with speculative decoding.

Performance comparisons use the same binary, GGUF, populated prompt, allocated
context, expert-cache budget and GPU settings. Keep decode graphs off on GB10.
Use 65,536 populated tokens and 262,144 allocated context for this experiment,
and report it as such, not as a fully populated 256K test. Alternate run order
and retain all raw outputs; filesystem cache state is uncontrolled.

## GB10 / Lexar SL500 results, 2026-09-23

Source: `feature/external-engram-gguf`, base `459942ff`, plus the patch captured
in each long run's `source.patch`. All GPU runs used the same benchmark binary:
`93f68177b629ad9a05d8584c6858197ebc034cf973919c8ff180958ac0f855b9`.
The internal primary model and the Lexar alternate GGUF were unchanged.

| Run | Readers | Decode dump | Prefill tok/s | Decode tok/s | Steady tok/s |
| --- | ---: | --- | ---: | ---: | ---: |
| gate-off | 0 | 256 full vectors | 98.36 | 8.32 | 8.38 |
| gate-on8 | 8 | 256 full vectors | 98.97 | 9.24 | 9.31 |
| timing-on8 | 8 | none | 98.02 | 9.18 | 9.23 |
| timing-off | 0 | none | 98.40 | 8.62 | 8.67 |

The clean timing pair gives **+6.50% decode** and **+6.46% steady decode**.
Prefill differs by -0.39%, with its algorithm unchanged. Overall run wall times
are nearly equal (700.37 vs 700.53 seconds) because prefill dominates this test.
The diagnostic pair also favors the pool (+11.06% decode), but its dump work is
included in generation wall time; it is corroborating evidence, not the primary
throughput result. This is one clean timing pair, not a statistical guarantee.
No comparison against an optimized internal-only configuration was performed.

Both long diagnostic files contain 256 x 129280 F32 values (132382720 bytes),
all finite and bit-identical. SHA-256:
`220c454212be697b11fd8b5de51776d8e874ddc1113d32858e1608b3b022e1a3`.
An earlier 4096-token prompt / 32-token decode gate was also bit-identical:
`6a05f2a786641178c4be673bf3659b6f1390471d1a5577ea6a439ccbb48395a0`.

All six inference runs completed with zero process swap. Minimum system
available memory across the long suite was about 18.93 GiB. The timing run
showed 12 threads during decode (four baseline threads plus eight readers).
No USB reset/disconnection or I/O error was found in the inspected kernel log.

Validation completed:

- `make test-engram`: pass, including pool sizes 1/2/4/8/16 and failure recovery.
- AddressSanitizer + UndefinedBehaviorSanitizer: pass.
- ThreadSanitizer: pass with `setarch aarch64 -R /tmp/ds4-engram-pool-tsan`.
  The normal ASLR invocation could not initialize (`unexpected memory mapping`);
  disabling ASLR for the test process resolved it. No global ASLR setting changed.
- `make -j2 CUDA_ARCH=sm_121 ds4 ds4-agent ds4-server ds4-bench ds4-eval`: built.
- `make test-deepseek41-gguf`: pass; existing fixture `mincore` pointer-signedness
  warning remains unrelated to this change.

Raw run directories are under
`/home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/raw/`:
`engram-pool-gate-{off,on8}-4k` and
`engram-pool-64k-v1-{gate-off,gate-on8,timing-on8,timing-off}`.
The long runs contain CSV, resources, logs, full source patch, binary/source
provenance, quota checkpoints, and (for the gates) full logits. The second gate
also contains `comparison.json`.

Reproduce the long suite with a new, unused prefix:

```sh
python3 speed-bench/engram_pool_suite.py \
  --runner /home/alessandro/projects/ds4-ds41/speed-bench/dgx_ds41/run.py \
  --model /home/alessandro/projects/ds4/gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --engram-model /mnt/ds4-models/models/DeepSeek-V4.1-Flash-Q2.gguf \
  --prefix engram-pool-64k-new
```

The implementation remains opt-in via `DS4_CUDA_ENGRAM_READERS=8` in the newly
built worktree executables. Existing aliases and services were not deployed or
restarted. The runner's quota floor remains 60%; final observed quota was 79%.
