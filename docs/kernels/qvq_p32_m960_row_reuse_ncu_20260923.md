# H100 P32 M960 row-reuse: matched NCU audit (2026-09-23 UTC)

Question: did PR #354 move the gate projection from compressed-weight decode
and scheduling overhead into a saturated WGMMA pipeline? **No.** Tensor-pipe
active cycles were 17.22% before and 17.15% after. The gain comes from
amortizing decode and barrier work across four M16 activation tiles, while the
new kernel remains short of enough eligible warps to fill issue slots.

The comparison used the same H100 (`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`),
real layer-0 W3 P32 gate metadata from the Llama 3.2 1B F6/P32/Rank-8 seed-7
snapshot, deterministic M960 FP16 activations, K2048, N8192, and exact FP32
output comparison. The old path is `p32_window_ampere_large_m2_kernel`; the
new path is raw ABI v3 algorithm 5, `launch_direct_rows<6, 960, 4>()`.

| Metric | Old large-M2 | New WGMMA M960 |
| --- | ---: | ---: |
| Warm local median | 286.34 us | 193.79 us (1.48x) |
| NCU launch duration | 287.55 us | 195.65 us |
| FP32 inner output | bitwise equal | bitwise equal |
| Executed instructions | 158.409M | 97.175M (-38.7%) |
| Tensor-pipe instructions | 7.864M | 3.932M (-50.0%) |
| Tensor-pipe active / elapsed cycles | 17.22% | 17.15% |
| Integer/logic ALU pipe utilization | 51.8% | 46.2% |
| Eligible warps / scheduler-cycle | 1.28 | 0.79 |
| Issue slots busy | 57.83% | 53.04% |
| Achieved occupancy | 31.30% | 15.02% |
| Registers / thread | 80 | 91 |
| Shared memory / CTA | 24.35 KiB static | 94.98 KiB dynamic |
| Spills | 0 | 0 |
| Shared bank-conflict counter | 11.007M | 8.748M |
| Short-scoreboard stalled warps / active cycle | 0.88 | 0.21 |
| Long-scoreboard stalled warps / active cycle | 0.10 | 0.52 |
| Barrier stalled warps / active cycle | 0.61 | 0.04 |
| Wait stalled warps / active cycle | 1.18 | 0.48 |
| DRAM throughput utilization | 2.40% | 3.62% |

The old M2 path and new WGMMA path execute different tensor instructions, so
their instruction counts are not FLOP counts. The counter used for tensor-pipe
activity is
`smsp__pipe_tensor_op_hmma_type_hmma_hgmma_qgmma_cycles_active.avg.pct_of_peak_sustained_elapsed`.
Its near-identical ~17% values are direct evidence against a WGMMA-throughput
bottleneck. The lower occupancy of the new path is a resource trade-off, not an
observed performance regression. NCU's `SM Busy` and pipe estimates alone do
not identify the critical dependency chain; the increased long-scoreboard
component and 44% no-eligible cycles make latency-hiding and TMA/shared
handoff worth inspecting next. No direct TMA-stall counter was collected; a
zero *multicast* TMA request counter would not establish zero ordinary TMA
traffic or stalls.

Reproduce the local timing with
`python -m scripts.benchmark_qvq_p32_m960_wgmma_h100` and the real snapshot,
GPU UUID, baseline `libzml_qvq.so`, and candidate `libzml_qvq_wgmma.so`.
For NCU, use `--profile-arm=baseline` or `--profile-arm=candidate` and
`--profiler-attached`; check that the GPU is idle **before** attaching the
profiler. Each arm warms 20 launches and then issues one traceable launch.
Filter `p32_window_ampere_large_m2_kernel` or
`qvq_p32_window_wgmma_m16_tma_kernel` with `--launch-skip 20 --launch-count 1`.
The two library SHA-256 values were
`d1f2c98043f9f1bc21a7088d5b26f592641cc04183c5fba02044db0316a2fdfc`
and `409fc307eaaa7a38a5551c403279cb35953e50d0c29731094962d0cfffb1ede3`.
QVQ source revision: `334a9ec80e31594896ab45982ce07f29f85021da`.

Raw NCU reports (not committed):
`/root/qvq-profiler-artifacts/m960-wgmma-20260923/old-m960-gate-ncu.ncu-rep`,
`new-m960-gate-ncu.ncu-rep`, `old-m960-gate-specific-ncu.ncu-rep`, and
`new-m960-gate-specific-ncu.ncu-rep`. The full B128 quality/performance gate
for the merged kernel remains 1,209/1,209 exact streams, 542 correct, about
36.9k useful prefill tokens/s. The model-wide 2x prefill target remains open.
