# H100 M960 W3 P32: single-stage TMA for row-five WGMMA

Measured 2026-09-23 UTC on NVIDIA H100, Llama 3.2 1B F6/P32/Rank-8
seed-7, GSM8K-Platinum B128, M960 prefill, FA2, GPU-local CPU affinity.
This is a compressed-weight kernel change: no dense weight materialization,
external workspace, environment opt-in, or quantized payload change.

## Mechanism

The W3 row-five CTA stages five M16 activation tiles plus compressed trellis,
bank IDs, and PGC levels. The two-stage pipeline consumed 111.36 KiB/CTA,
limiting residency to two CTAs/SM on H100. For this geometry only, use one
TMA stage. A full K256 stage is still decoded and consumed before the buffer
is reused; producer/consumer barriers retain the existing ordering. All
other rates, row geometries, and decode paths retain two stages.

| Gate M960/K2048/N8192 | Two stages | One stage |
| --- | ---: | ---: |
| Dynamic shared memory/CTA | 111.36 KiB | 64 KiB |
| Registers/thread | 107 | 90 |
| Active warps, % of peak | 15.00% | 22.07% |
| Eligible warps/scheduler-cycle | 0.76 | 0.94 |
| Executed warp instructions | 82.34M | 76.15M |

The isolated layer-0 raw-ABI oracle used the snapshot's actual P32 metadata,
50 paired GPU-event timings per projection, BM80/BN64, and FP32 output
comparison. Gate fell from 172.48 to 158.62 us (1.087x); down fell from
176.06 to 150.11 us (1.173x). Both outputs were bitwise equal. An earlier
four-to-eight-fragment WGMMA queue probe gained only about 0.4-0.5% while
raising registers from 107 to 121/thread, so it was not retained.

## Full-model gate

Each arm used its own matched runner and QVQ runfiles, a one-row warmup, then
the same 1,209-row continuous-batch request. Both control and candidate were
repeated on the same H100. Values below are ranges across two complete runs;
the numbers include all 16 layers and Rank-8 recovery.

| B128 GSM8K-Platinum | Merged control | Single-stage candidate |
| --- | ---: | ---: |
| Useful prefill tok/s | 36,137-36,173 | 37,268-37,271 |
| Padded prefill tok/s | 41,681-41,722 | 42,985-42,989 |
| Padded decode tok/s | 11,855-11,860 | 11,881-11,892 |
| Padded per-stream decode tok/s | 92.62-92.65 | 92.82-92.91 |
| Total wall time | 39.57-39.62 s | 38.72-38.74 s |
| Exact token streams | 1,209/1,209 | 1,209/1,209 |
| Correct / invalid | 542 / 0 | 542 / 0 |

This is approximately +3.08% useful/padded prefill throughput and -2.23%
wall time against the matched control; decode is not regressed. It is a
forward step, **not** completion of the 2x aggregate B128 prefill target.

The full evaluator artifacts remain outside Git at
`/root/qvq-profiler-artifacts/m960-one-stage-20260923/` (the four `*-full*.json`
records); large per-sample output JSON was intentionally not committed.

Build: `bazel build //gptqmodel_ext/qvq:libqvq_wgmma_raw` for the raw oracle;
ZML `llama_paged_token_runner` with CUDA `-c opt` and a local QVQ override for
the full gate. The final pinned ZML runner must be tested separately before
promoting the integration.
