# H100 F6 P32 prefill: T6 shared-layout gate

This experiment targets the `p32_window_ampere_large_m2_kernel<6,0,0,3,6>`
specialization used by the Llama 3.2 1B F6/P32 B128, 960-token prefill path.
It does **not** change the WGMMA/TMA reuse-11 kernel, its CUTLASS operand
layout, the canonical P32 payload, or persistent model weights. Low HBM
throughput alone was not treated as an alignment fault: the measured T6 kernel
was limited by L1/TEX/shared decode and scheduler availability, with no spills.

## Mechanism

The six-row-group activation tile has 48 logical FP16 columns per row. Its
96-byte shared row stride advances the starting 32-bit bank index by 24 modulo
32, repeating after four rows. For this specialization only, the shared tile
uses 56 FP16 columns (112 bytes) per row, advancing by 28 modulo 32 and
repeating after eight rows. Logical input and matrix multiplication geometry
are unchanged. The 256 FP16 codebook values are also copied into a 512-byte
shared table once per CTA, replacing repeated irregular global gathers.

Padding alone reduced shared-read bank conflicts by about 95% but improved
isolated kernel latency by only 0.6%. Shared levels alone were 0.4% slower.
Together they reduced dependency stalls enough to produce a material gain.
This is a specialization-specific interaction, not a recommendation to add
generic shared padding or codebook replication to other P32 kernels.

## Matched kernel evidence

On the 132-SM H100 `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, the
coherent production-library A/B used M=960, K=2048, N=8192, F6, 20 warmups,
and 50 alternating CUDA-event samples. The control and candidate produced
bitwise-identical FP32 outputs.

| Metric | Control | Candidate |
|---|---:|---:|
| Median kernel latency | 328.672 us | 291.392 us |
| Shared-read bank conflicts, NCU replay | 16.319M | 10.051M |
| Scheduler cycles with no eligible warp | 45.13% | 39.92% |
| Static shared memory/CTA | 20.77 KiB | 24.35 KiB |
| Registers/thread | 80 | 80 |
| Spills | 0 | 0 |

The measured kernel latency fell 11.3% (1.128x throughput). The NCU replay
report is host-specific and is intentionally not committed:
`/tmp/qvq-t6-pad56-lut-ab-20260923.ncu-rep`.

## Full serving gate

The candidate QVQ source is compared against the exact control source
revision `39965f52b6297392c4c3d43f9f59a70d9a02976e`. Both used the same
ZML revision `f1ba31354cba155e6f6b9512f12be4cc0b67f728`, model snapshot,
GSM8K-Platinum 1,209-row reference/dataset hashes, release runner, B128,
prefill bucket 960, logical context 8192, FA2, rank-8 W8, 544 KV pages,
GPU-local CPU cores, and a 128-row warmup. Each complete candidate run had
1,209/1,209 exact token streams, 542 correct, and zero invalid outputs.

| Run | Useful prefill tok/s | Padded decode tok/s | Wall s |
|---|---:|---:|---:|
| Candidate A1 | 27,104.7 | 12,578.9 | 48.331 |
| Control B | 26,179.8 | 12,553.3 | 49.689 |
| Candidate A2 | 27,192.0 | 12,637.6 | 48.160 |

Candidate prefill throughput improved 3.5–3.9% against the matched control;
decode did not regress in this A/B/A gate. A third candidate run measured
27,165.6 useful prefill and 12,624.7 padded decode tok/s. The compact JSON
results remain in `/tmp/zml-qvq-t6-align-{w8-full,control-full,aba-full}-20260923.json`.

The isolated T6 kernel represented only part of prefill wall time, so its
12.8% local throughput gain cannot by itself deliver a 2x full-prefill target.
The next target should be chosen from a fresh end-to-end profile, not from HBM
GB/s alone. In particular, the WGMMA reuse-11 path already uses 221.952 KiB
shared memory per CTA and must not inherit this extra shared allocation.
