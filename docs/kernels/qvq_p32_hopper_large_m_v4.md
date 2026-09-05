# Hopper large-M P32 v4

This phase starts from merged PR #114 (`9dcaf07e`) and targets another
1.25x full-MLP speedup on the physical 132-SM NVIDIA H100. All timings use
warmed CUDA Graph replay measured by CUDA events; CPU and container scheduling
are outside the timed region.

## Reuse-11 gate/up kernel

The merged gate/up kernel decodes one P32 weight tile and reuses it for eight
M16 activation tiles, or 128 rows. At logical M512 that produces four row CTA
waves. The H100 shared-memory limit cannot hold twelve double-buffered input
tiles, but it can hold eleven:

```text
11 * M16 * K256 * sizeof(fp16) * 2 stages = 180,224 bytes
P32 trellis, selectors, levels, and barriers  = ~40 KiB
total dynamic shared memory                  = ~221 KiB
```

The runtime therefore pads only the grouped gate/up inner operand from 512 to
528 rows and launches three M176 CTA waves:

```text
merged:    512 / 128 = 4 row waves
reuse-11:  528 / 176 = 3 row waves
```

The extra sixteen zero rows add 3.125% tensor-core work, while the P32 decode
and address stream is executed 25% fewer times. The first 512 outputs retain
the same K traversal, WGMMA instruction sequence, FP32 accumulation order, and
FP16 recovery boundary. Padded rows are discarded before recovery.

The specialization is deliberately narrow: physical NVIDIA H100, grouped
Llama 2048-to-8192 gate/up, unsplit children, and logical M512. Other devices,
shapes, rates, and row counts retain the merged production path.

## Rejected decode lookup

An exact 1 MiB table mapping `(bank, 16-bit state)` directly to two packed
FP16 levels removed affine PGC arithmetic but changed regular shared-memory
loads into data-dependent cache lookups. W3/M512 full MLP regressed from
337.982 to 665.511 microseconds. The experiment was removed without a commit.

## Rejected one-stage reuse-16

A single-stage M256 kernel fit in shared memory and was exact and graph-safe,
but exposed TMA latency. W3/M512 regressed from 337.982 to 389.771 microseconds.
It was removed without a production commit.

## M512 result

| Weight | Gate/up M×K×N | Down M×K×N | Merged main | Reuse-11 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| W2 | 512×2048×8192 | 512×8192×2048 | 331.784 µs | 319.615 µs | 1.038× | 0.515× | 0.370× | Yes |
| W2.5 | 512×2048×8192 | 512×8192×2048 | 340.222 µs | 335.828 µs | 1.013× | 0.490× | 0.352× | Yes |
| W3 | 512×2048×8192 | 512×8192×2048 | 337.982 µs | 325.754 µs | 1.038× | 0.505× | 0.363× | Yes |
| W3.5 | 512×2048×8192 | 512×8192×2048 | 335.194 µs | 328.096 µs | 1.022× | 0.501× | 0.361× | Yes |

Geometric speedup over merged main is **1.028x**. Maximum absolute error over
the four complete-MLP cases is `1.0133e-6`, below the locked `2e-3` limit.
The low-level and runtime tests require bit-exact output and CUDA Graph replay.

Artifact:
`artifacts/qvq_hopper_large_m/v4_reuse11_m512_candidate.json`.

This is the first v4 win, not the complete 1.25x target. Nsight Compute and
source-correlated SASS analysis are recorded after committing the exact source.
