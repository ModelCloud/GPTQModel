# H100 M960 K2048 input Hadamard: one-CTA exact path (2026-09-24 UTC)

The framework-neutral raw ABI v2 now covers K2048 as well as K8192. For
K2048, one CTA owns each row and keeps all 11 FP16-rounded butterfly stages
in 4.125 KiB of padded shared memory. The K8192 two-stage route is unchanged.
The public ABI still requires caller-provided workspace for both widths;
K2048 does not read it. No dense weight cache or additional device allocation
is introduced. The normal ZML model path selects the raw K2048 transform only
for admitted M960 prefill shapes.

## Correctness

- The raw CUDA oracle passed K2048 and K8192 at M1/M16/M960, including
  changed-input CUDA graph replay: 11/11 tests passed.
- ZML's K2048 raw-versus-production-StableHLO test found zero mismatches over
  1,966,080 FP16 outputs. The complete ZML suite passed 223, with two skips.
- The matched one-stage and two-stage full GSM8K-Platinum arms produced the
  same 1,209/1,209 token streams, 543 correct answers, and zero invalid
  outputs. This establishes parity between the two K2048 CUDA algorithms on
  that workload, **not** parity against merged main: merged main scored
  544/1,209. The K2048 raw integration, whether one- or two-stage, retains
  a one-answer net difference that must be judged separately before default
  promotion.

## Warmed M960 GPU trace

Same H100, model, B128/M960, FA2, 544-page KV pool, 45% BFC pool, Rank-8
prefill disabled and decode enabled, auto terminal dead-row elimination, and
GPU-local CPU cores 0,1,3,4,12,13. Nsight Systems captured two prefill steps;
the table isolates the second, warmed step.

| Metric | K2048 two-stage | K2048 one-stage |
| --- | ---: | ---: |
| K2048 launches | 126 | 63 |
| K2048 GPU time | 1.102 ms | 0.884 ms |
| Whole prefill GPU span | 14.353 ms | 14.118 ms |
| W3 P32 GPU time | 4.532 ms | 4.548 ms |
| FA2 GPU time | 2.039 ms | 2.032 ms |

The launch and intermediate-traffic reduction saves about 0.22 ms in K2048
and 0.24 ms in the complete warmed GPU span. The W3 compressed P32 decoder
remains the largest single family, so this isolated change cannot close the
remaining gap to the 2x aggregate prefill target.

Local profiler reports (not committed):
`/var/tmp/b128-had2048-clean-native-20260924.nsys-rep` and
`/var/tmp/b128-had2048-single-native-20260924.nsys-rep`.

## Full continuous B128 suite

Each arm evaluated the same 1,209 ordered GSM8K-Platinum requests, the same
model and scoring reference, an optimized runner, and the same pinned H100
CPU cores. Both arms were repeated in a single server process; the candidate
was also rerun after the control. Useful tokens exclude inactive padding;
padded tokens include all issued B128 slots.

| Arm | Useful prefill tok/s | Padded prefill tok/s | Wall | Correct |
| --- | ---: | ---: | ---: | ---: |
| One-stage first | 45,542 | 52,529 | 35.885 s | 543/1,209 |
| One-stage warm | 46,133 | 53,210 | 35.524 s | 543/1,209 |
| Two-stage first | 45,015 | 51,920 | 36.107 s | 543/1,209 |
| Two-stage warm | 45,404 | 52,369 | 35.831 s | 543/1,209 |
| One-stage after control | 45,774 | 52,797 | 35.712 s | 543/1,209 |

Warm candidate-versus-control useful prefill is +1.61%; the candidate after
control is +0.82%. The one-stage result is a modest E2E gain, not the 2x
goal. The current best measured 46,133 useful tok/s is 1.71x the original
27,023 tok/s baseline; 2x requires at least 54,046 useful tok/s.

Raw full-suite results (not committed):
`/var/tmp/b128-had2048-single-full-repeat-20260924.json`,
`/var/tmp/b128-had2048-twostage-matched-control-repeat-20260924.json`, and
`/var/tmp/b128-had2048-single-full-after-control-20260924.json`.

## W3 gate BM160/R10 follow-up

The W3 K2048→N8192 gate projection can use ten M16 row tiles per CTA
(BM160), reducing six CTA rows over M960 versus twelve for BM80/R5. This
variant retains the compressed P32 weights and the exact FP32 output
contract. It is admitted only for the gate shape; the K8192→N2048 down
projection remains BM80/R5 because an earlier R6 down probe regressed.
The raw ABI and ZML dispatch select BM160 automatically by shape.

The source XLA composite rewriter must explicitly admit BM160 as well as the
native raw ABI. Before that admission, ZML silently compiled the gate into a
dense TF32 GEMM even though the BM160 CUDA kernel itself passed the oracle.
Nsight Systems showed 15 `sm90_xmma_gemm_f32f32` calls instead of 15 native
R10 gate calls. This is why an ABI-level speedup alone is insufficient as an
end-to-end gate. After the XLA fix, the warm two-step trace contains 15
native R10 gate and 15 native R5 down calls per step, with no dense fallback.
The warm GPU span fell from 14.118 to 13.824 ms; W3 P32 time fell from
4.548 to 4.321 ms.

The matched full-suite A/B/A below used the same rebuilt XLA, QVQ library,
model, pinned CPU cores, B128/M960 and 544-page KV pool. The only change
between arms was gate BM160/R10 versus BM80/R5; every one of 1,209 output
token streams was bitwise identical across arms, with 543 correct and zero
invalid. This verifies R10 does not add quality drift beyond the separate
K2048 integration described above.

| Arm | Useful prefill tok/s | Padded prefill tok/s | Correct |
| --- | ---: | ---: | ---: |
| R10 first | 46,324 | 53,431 | 543/1,209 |
| R5 matched control | 45,764 | 52,785 | 543/1,209 |
| R10 after control | 46,205 | 53,294 | 543/1,209 |

R10 gives +0.96–1.22% versus the matched control. The best observed useful
prefill is now 46,324 tok/s, or 1.71x the original 27,023 baseline; the
54,046 tok/s 2x goal remains open. Enabling the XLA prefill command buffer
with the default LHS scheduler and the same R10 binary instead measured
45,551 useful / 52,539 padded tok/s, so LHS prefill was not promoted.
The separately validated concurrent scheduler is documented in ZML-Ultra.

Local raw results (not committed):
`/var/tmp/b128-had2048-r10-native-fixed-full-20260924.json`,
`/var/tmp/b128-had2048-r5-native-fixed-control-20260924.json`,
`/var/tmp/b128-had2048-r10-native-fixed-full-repeat-20260924.json`, and
`/var/tmp/b128-had2048-r10-graph-prefill-full-20260924.json`.

## Two-consumer BN128 screen

After the concurrent prefill gain, a 50-round same-library raw-ABI screen
tested the existing BM64/BN128 two-consumer geometry against the selected
one-consumer W3 shapes on real layer-0 snapshot metadata. All FP32 outputs
were bitwise equal, but both BN128 candidates regressed in isolated latency:

| Projection | Selected geometry | Selected median | BN128 candidate median | Candidate speedup |
| --- | --- | ---: | ---: | ---: |
| Gate | BM160/BN64 | 144.064 µs | 181.216 µs | 0.795x |
| Down | BM80/BN64 | 150.400 µs | 174.528 µs | 0.862x |

No production dispatch was changed. The next prefill work should not assume
that adding a second N64 consumer improves either of these W3 shapes.

## W3 R10 decode depth and build isolation

Nsight Compute found the W3 BM160/R10 gate at 128 registers/thread and
104.96 KiB shared memory/CTA, with 0.52 eligible warps/scheduler and 63.49%
of scheduler cycles having no eligible warp. DRAM throughput was only 5.10%,
so waiting on HBM was not the principal bottleneck. The W3 BM80/R5 down
projection had 0.94 eligible warps/scheduler, 64 KiB shared memory/CTA, and
6.12% DRAM throughput. These are single-kernel observations, not additive
end-to-end times.

The R10 W3 path now keeps six decoded WGMMA source fragments in flight
instead of four. Other geometries retain their existing decode depths. A
50-round, real layer-0, raw-ABI test found gate 142.464 → 137.792 µs
(+3.39%) with bitwise FP32 output parity. Down was effectively unchanged at
150.080 → 150.016 µs. The full B128/M960 1,209-request control and two
candidate runs used the same runner, checkpoint, FA2 backend, 544-page KV
pool, pinned CPU cores, and scoring reference. All 1,209 generated streams
matched between control and candidate, with 543 correct and zero invalid:

| Arm | Useful prefill tok/s | Padded prefill tok/s | Wall |
| --- | ---: | ---: | ---: |
| Depth 4 control | 47,986 | 55,347 | 33.096 s |
| Depth 6 first | 48,280 | 55,686 | 32.914 s |
| Depth 6 repeat | 48,087 | 55,464 | 33.131 s |

This is a bounded, small prefill gain, not evidence of reaching the 2x goal:
54,046 useful tok/s remains the target. The direct native B128 prefill trace
still puts compressed P32 WGMMA at 7.295 ms summed node time, FA2 at 2.070
ms, and P32 Hadamard epilogue at 1.996 ms. Node times can overlap and must
not be added to estimate GPU wall time; the warmed graph span was 12.993 ms
under node tracing.

The first concurrent shard rebuild failed linking because the W4 object
contained W6 symbols. NVCC was compiling rate-family objects concurrently
with temporary intermediates in a shared directory. Each shard now passes
`--objdir-as-tempdir`, which places intermediates beside its distinct object
file. The rebuilt W4/W5/W6/W7 objects each export only their corresponding
rate-family entry point, and the shared library links successfully. This is
a build-integrity fix independent of the depth-six performance result.

Profiler and full-suite JSON binaries remain local, not in the repository:
`/var/tmp/qvq-m960-r10-gate-20260924.ncu-rep`,
`/var/tmp/qvq-m960-r5-down-20260924.ncu-rep`, and
`/var/tmp/b128-had2048-r10-depth{4-matched-control,6-full,6-full-repeat}-20260924.json`.
