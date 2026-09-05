# Exact K partitions and guarded graph staging

Experiment revision: 0a293cf2. Production baseline: 88b00991 (same AMD
device implementation as bd09d89b). Overall target remains 1.5x in all
364 requested cases against c89459e3. Neither experiment is promoted.

## Split reduction

Partition K5120 as 4096+1024 and K6144 as 4096+2048. Both partitions
are exact powers of two, so no padded input/weight loads are required.
Compute both FP32 product reductions and add their results before the
existing FP16 store. Optional residual weights are still added in FP32.
This preserves real algebra and storage precision but changes rounding
order, so canonical correctness remains required (max absolute <=2e-3).

All seven shape groups, M1, four rates: 28/28 passed before and after
profiling. Down does not use this GEMV. Post-profile gate/up speedup
versus production was 1.145-1.167x, not better than the preceding full-K
experiment's 1.163-1.179x. Other shapes remain mixed/regressions.
No all-case 1.5x improvement is established. Focused tests after profiling:
298 passed, 50 warnings in 13.85s, including both K partitions, residual
on/off, block-N2/4/8 and output canaries. Synthetic tests do not establish
model quality. Ruff and whitespace checks passed.

Matched M1,K5120,N17408, block-N2, four waves:

| Metric | Previous full K | Split K |
| --- | ---: | ---: |
| Static instructions | 256 | 274 |
| Issued VALU | 6963200 | 6371328 |
| Issued SALU | 557056 | 905216 |
| Issued LDS | 69632 | 382976 |
| VGPR descriptor | 76 | 55 |
| Dynamic LDS bytes | 32 | 4096 |
| Scratch bytes | 0 | 0 |

The extra reduction/layout conversion offsets the removed padding:
VALU falls 8.5%, but issued LDS rises 5.5x. Do not substitute source-level
padding counts for emitted work. TTIR proves two unmasked loads over
4096/1024 extents; the generated ISA retains separate reduction handoffs.
For K6144 the split specialization instead uses 60 VGPR and 32 bytes LDS,
268 static instructions. No scratch/spills were emitted in either case.
Occupancy, scheduler stalls and bank conflicts were not measured.

The current matched capture also includes retained production kernels:
gate/up issued VALU/SALU/LDS = 5344256/487424/905216. Prior full-K
counts are from cdaab451's matched same-shape, same-device counter run
recorded in full_k_gemv_cdaab451.md; they are not inferred from static ISA.

JIT hashes:

- K5120 split: 178339739473009a0624818e0528d25d023b414bd13dbd849bd79a6d4bf71940
- K6144 split: 93d775ec815ef1c0c3c9ac6706073ca5584cdfc557247f28fe460942a6b294a0

## Graph staging (rejected)

Keep public layer guards and cache only a benchmark-local graph, its
weight references, and staging storage per shape/stream. Each invocation
copies the new activation into staging, replays the unchanged operator,
and clones the output so later calls cannot overwrite returned tensors.
Input staging's final reader precedes the next same-stream overwrite;
output staging's final reader is the clone before the next replay.
Outer graph capture and input/weight autograd take the original operator.
This single-threaded exploratory cache is not a production concurrency,
mutation, cache-budget or lifetime implementation.

The initial 48-case small-M sweep passed correctness but gate/up/down
usually slowed by about 20-25%. Full-KV was mixed. The extra copies
negate launch savings, so this is not a useful default. The post-profile
run additionally checks changed inputs, independence of previous outputs,
non-default streams and outer graph capture. Staging bytes are recorded
per row, excluding graph runtime/pool overhead and existing weight caches.
Post-profile result: 48/48 canonical passes, 48/48 fresh-input and prior-output
independence checks, 48/48 non-default-stream checks and 48/48 outer graph
checks. Compact accepted timing reports are stored alongside this note as
`split_k_0a293cf2_post.json` and `graph_staging_0a293cf2_post.json`.

Mapping trace: 508 retained GEMV launches, 615 copyBuffer launches,
256 hipGraphLaunch calls and 576 hipMemcpyAsync calls. This includes
setup, warmup and checks; do not call aggregate counts launches/forward.
The steady candidate dependency is input copy -> GEMV -> output clone,
versus the retained one-GEMV path. Mixed trace GEMV p50/mean/p95 was
28.240/29.838/33.200 us; copyBuffer was 3.200/6.373/5.560 us. These
instrumented, mixed-phase timings are attribution only, not performance
evidence. Graph staging generates no new GEMV arithmetic; no copy-kernel
instruction-count or scheduler claim is made from this trace.

## Reproduction and artifacts

MI355X gfx950, physical GPU0, BDF0000:83:00.0, unique ID
0x333ef6e01ec019b3, 256 CUs; Torch2.13.0+rocm10.0.0 and
Triton3.8.0+git4cff872c.rocm10.0.0. Accepted runs passed the strict
three-sample idle and pre-timing gates. Two immediately chained runs
were rejected while GPU residency/activity was still clearing after
the preceding process; they produced no accepted report. Retried only
after terminal process status and current idle state were verified.

Split timing command: `python scripts/benchmark_qvq_p32_amd_butterfly.py
--butterfly none --gemv-split-k --gemv-block-n 2 --baseline-amd-commit
88b00991 --full-sweep --m-values 1 --warmup 10 --iterations 30
--output /tmp/qvq-splitk-n2-post/report.json`.

Graph timing: replace split flags with `--graph-execute`, select shapes
full_kv/mlp_gate_up/mlp_down and M1/4/16/32. Same warmup and iterations.
Raw output: /tmp/qvq-graph-execute-post/report.json.

Split profiler: sudo rocprofv3 with process-local LD_LIBRARY_PATH
`/opt/rocm/core-10.0/lib:/opt/rocm/core-10.0/lib/rocprofiler-sdk`, counters
SQ_INSTS_VALU/SQ_INSTS_SALU/SQ_INSTS_LDS, regex
`folded_gemv_split_k_kernel|_qvq_p32_folded_gemv.*`, shapes
mlp_gate_up/full_kv/attn_out, M1, warmup1, iterations2, CSV output.

- Split counters: /tmp/qvq-splitk-profile/raw/ (counter_collection.csv)
- Split compiler artifacts: /tmp/qvq-splitk-profile-cache/C6BTS44UOMAJUBREQGHAKKGSLUBDWQKL2E633BE326NG2S7XDFAA/
- Graph mapping trace: /tmp/qvq-graph-profile/raw/ubuntu2404-mi350x/763051_kernel_trace.csv
- Graph HIP API trace: same directory, 763051_hip_api_trace.csv
- Graph trace command: rocprofv3 --kernel-trace --hip-trace --stats --output-format csv, graph benchmark M1 gate/up, warmup10/iterations50
- Test log: /tmp/qvq-splitk-tests.log

Backend lookup rechecked for the next direction: installed AITER is
7440ef72503e1c3fadc5be85a5c74eb7c9c34841; upstream main queried via gh
is 636098e5a462abfb2900efe623751e7a612b09e3. Installed FlyDSL and skinny
wrappers require same-type FP16/BF16 outputs; the prior lookup's FP32
claim was corrected, and the AMD skill validator passed. `wvSpltK`
supports activation M1..4; inspect native accumulation before testing.
