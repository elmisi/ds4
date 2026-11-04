# GB10 `sm_121` quality gate — 2026-08-08

All runs use the 0731 Flash GGUF and deterministic sampling (`--temp 0`).

| Run | Purpose | Result |
| --- | --- | --- |
| `mmq_sm121.stdout` / `.stderr` | Reproduce the default CUDA decode-graph behavior. | **Rejected**: D2R prefill ran, but generation stopped after `Bob=34`; capture reported repeated legacy-stream failures. |
| `mmq_off_control.stdout` / `.stderr` | MMQ-off long-context correctness control. | All 16 assignment lines correct. |
| `mmq_sm121_no_decode_graph.stdout` / `.stderr` | MMQ/D2R with decode capture explicitly disabled. | Byte-for-byte same answer as the MMQ-off control apart from the GPU-budget line. |
| `mmq_sm121_gb10_safe.stdout` / `.stderr` | Final GB10-default candidate. | All 16 assignment lines correct; D2R active; 846.10 t/s prefill and 16.40 t/s generation. |
| `long_context_test.stdout` / `.stderr` | Built-in 30,474-token fact-recall regression. | Passed. |
| `vector_tests.stdout` / `.stderr` | Official logprob and local golden-vector regression suite. | Passed. |
