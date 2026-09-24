# W3 M960 transient-decode screen (2026-09-24)

This is an isolated H100 screen, **not** a production route or a qualified
GSM8K-Platinum result. The pre-BN128 full-suite B128 baseline was 60,362 useful
and 69,622 padded prefill tok/s. The goal remains 120,000 padded prefill tok/s.

These figures describe the pre-BN128 screen. The newer qualified pinned B128
reference is 62,845 useful / 72,486 padded prefill tok/s (1,209/1,209 exact
streams, 543 correct). Reaching 120,000 padded tok/s from that reference
requires reducing its 16.012 s cumulative prefill time to at most 9.672 s.

The probe reconstructs the canonical compressed P32 payload into a temporary
FP16 `[K,N]` matrix on every call, then performs FP16×FP16→FP32 GEMM. It does
not retain decoded weights between calls. Each projection needs 32 MiB of
temporary weight storage; a serving integration must measure graph-pool peak
and ensure the scratch is not accidentally promoted into a weight cache.

Both arms used the same real seed-7 snapshot layer metadata, same random M960
FP16 input, GPU-local CPU cores `0,1,3,4,12,13`, one H100, warm CUDA graphs,
and 100 alternating paired event rounds. The control was the production
compressed algorithm-5 geometry, BM160/BN64 for gate and BM80/BN64 for down.

| Layer / projection | Compressed WGMMA | Decode + GEMM | Local speedup | FP32 output |
| --- | ---: | ---: | ---: | --- |
| 0 gate K2048→N8192 | 140.864 µs | 86.368 µs | 1.63× | bitwise identical |
| 0 down K8192→N2048 | 150.352 µs | 79.424 µs | 1.89× | bitwise identical |
| 15 gate K2048→N8192 | 139.856 µs | 86.128 µs | 1.62× | bitwise identical |
| 15 down K8192→N2048 | 150.608 µs | 80.192 µs | 1.88× | bitwise identical |

The outputs were nonzero; for layer-0 gate all 7,864,320 output values were
nonzero and the maximum magnitude was 4.842. These results eliminate the
possibility that parity was merely two zero tensors. They do not establish
parity for all layers, inputs, graph replays, or token streams.

A second layer-0 gate/down pass raised the input scale from 0.02 to 1.0 and
replayed both captured graphs after changing the input tensor. Both direct
outputs and changed-input graph outputs remained FP32 bitwise identical. The
transient route measured 86.032 µs for gate and 80.128 µs for down in this
50-round stress pass; the respective compressed controls were 140.496 and
150.832 µs. This is still only two projections on one model snapshot.

A 16-row FP64 dot oracle on layer-0 metadata found identical error for both
routes, as expected from their FP32 bitwise parity. Gate maximum/mean absolute
errors were `1.330e-5`/`1.723e-6`; down errors were
`8.680e-5`/`1.384e-5`. This bounds the arithmetic difference for the sampled
rows, not the full model's post-quantization task score.

At 15 full-row gate and down calls per B128 prefill Step, simply replacing
the two kernel pairs could save roughly 1.9 ms of summed GPU kernel time.
That is a projection-only estimate, **not** an end-to-end prediction: overlap,
XLA scheduling, allocator behavior, and host overhead must be measured. The
most recent graph GPU span was about 10.54 ms and the full-suite prefill Step
averaged 13.79 ms, versus the 8.00 ms/Step required for 120k padded tok/s.

The follow-up branch adds a versioned W3-only framework-neutral decoder ABI
and one-kernel launch plan, with device-resident bank-alt selection. Its two
shape tests compare decode+FP32 GEMM bitwise against compressed WGMMA, verify
the launch plan, reject wrong transition bits, and replay a captured decoder
after changing the bank-alt tensor. Both tests passed. The local follow-up
branch wires the ABI into ZML; promotion still requires exact external admission and
launch proof, scratch lifetime checks, every relevant layer's FP32/FP64
oracle, and the full 1,209-row B128 quality and useful plus padded throughput
gate. No second model-weight copy may be cached.

An updated layer-0 probe against the merged compressed geometry (gate
BM160/BN128, down BM80/BN64) again passed bitwise FP32 output and changed-input
graph replay, with identical sampled FP64-oracle errors between arms:

| Projection | Current compressed WGMMA | Transient decode + GEMM | Local speedup |
| --- | ---: | ---: | ---: |
| Gate K2048→N8192 | 126.208 µs | 85.728 µs | 1.47× |
| Down K8192→N2048 | 151.680 µs | 79.840 µs | 1.90× |

At 15 prefill calls per projection, these local medians imply about 1.69 ms
of summed kernel opportunity per request, not a 120k end-to-end prediction.
The production path still needs exact XLA launch proof, measured scratch
lifetime, and full-suite validation before promotion.
