# QVQ A41/R0 Phase 5: H100 benchmark

Phase 5 reduces the complete Llama 3.2 1B QVQ MLP latency by **1.297x**
geometric mean, or **22.9%**, relative to the unfused production QVQ path.
Its paired-recovery stage reduces the gate/up projection site by **1.391x**, or
**28.1%**, relative to the committed Phase-4 benchmark.  The second stage,
which combines the rounded gate/up product with the down input transform, adds
another **1.019x**, or **1.9%**, over paired recovery alone.

All outputs in the benchmark are bit-exact to the ordinary QVQ MLP.  This is an
execution optimization; it does not change quantized weights or substitute an
activation approximation.

## Complete Llama 3.2 1B MLP

Each row measures this entire sequence:

```text
gate: M x 2048 x 8192
up:   M x 2048 x 8192
SiLU and FP16 gate*up
down: M x 8192 x 2048
```

“Versus” is comparator latency divided by Phase-5 latency.  A value above 1
means QVQ is faster; a value below 1 means the W4 comparator is faster.
“Better” compares the complete Phase-5 path with the preceding paired-recovery
stage in the same run.  All 20 rows improved.

| Rate | M | Phase 5 us | vs plain QVQ | Marlin W4 us | vs Marlin | Machete W4 us | vs Machete | Better than previous stage | QVQ effective TFLOP/s | Marlin effective TFLOP/s | Machete effective TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| W2 | 1 | 148.429 | 1.341x | 29.786 | 0.201x | 52.111 | 0.351x | Yes | 0.678 | 3.380 | 1.932 |
| W2 | 2 | 148.694 | 1.305x | 32.138 | 0.216x | 51.540 | 0.347x | Yes | 1.354 | 6.265 | 3.906 |
| W2 | 4 | 149.245 | 1.299x | 32.382 | 0.217x | 51.850 | 0.347x | Yes | 2.698 | 12.435 | 7.766 |
| W2 | 8 | 150.091 | 1.294x | 29.678 | 0.198x | 51.230 | 0.341x | Yes | 5.365 | 27.134 | 15.720 |
| W2 | 16 | 144.734 | 1.291x | 32.973 | 0.228x | 51.583 | 0.356x | Yes | 11.128 | 48.847 | 31.224 |
| W2.5 | 1 | 149.598 | 1.311x | 29.786 | 0.199x | 52.111 | 0.348x | Yes | 0.673 | 3.380 | 1.932 |
| W2.5 | 2 | 150.232 | 1.301x | 32.138 | 0.214x | 51.540 | 0.343x | Yes | 1.340 | 6.265 | 3.906 |
| W2.5 | 4 | 150.756 | 1.292x | 32.382 | 0.215x | 51.850 | 0.344x | Yes | 2.671 | 12.435 | 7.766 |
| W2.5 | 8 | 151.495 | 1.291x | 29.678 | 0.196x | 51.230 | 0.338x | Yes | 5.316 | 27.134 | 15.720 |
| W2.5 | 16 | 146.130 | 1.285x | 32.973 | 0.226x | 51.583 | 0.353x | Yes | 11.022 | 48.847 | 31.224 |
| W3 | 1 | 150.087 | 1.300x | 29.786 | 0.198x | 52.111 | 0.347x | Yes | 0.671 | 3.380 | 1.932 |
| W3 | 2 | 150.201 | 1.295x | 32.138 | 0.214x | 51.540 | 0.343x | Yes | 1.340 | 6.265 | 3.906 |
| W3 | 4 | 150.878 | 1.293x | 32.382 | 0.215x | 51.850 | 0.344x | Yes | 2.669 | 12.435 | 7.766 |
| W3 | 8 | 151.619 | 1.291x | 29.678 | 0.196x | 51.230 | 0.338x | Yes | 5.311 | 27.134 | 15.720 |
| W3 | 16 | 146.103 | 1.288x | 32.973 | 0.226x | 51.583 | 0.353x | Yes | 11.024 | 48.847 | 31.224 |
| W3.5 | 1 | 150.026 | 1.307x | 29.786 | 0.199x | 52.111 | 0.347x | Yes | 0.671 | 3.380 | 1.932 |
| W3.5 | 2 | 150.278 | 1.293x | 32.138 | 0.214x | 51.540 | 0.343x | Yes | 1.340 | 6.265 | 3.906 |
| W3.5 | 4 | 151.193 | 1.291x | 32.382 | 0.214x | 51.850 | 0.343x | Yes | 2.663 | 12.435 | 7.766 |
| W3.5 | 8 | 151.677 | 1.292x | 29.678 | 0.196x | 51.230 | 0.338x | Yes | 5.309 | 27.134 | 15.720 |
| W3.5 | 16 | 146.426 | 1.287x | 32.973 | 0.225x | 51.583 | 0.352x | Yes | 10.999 | 48.847 | 31.224 |

Effective throughput uses the same dense-equivalent logical work for every
implementation:

\[
\operatorname{FLOPs}=2M(2048\cdot8192+2048\cdot8192+8192\cdot2048).
\]

This is a useful execution-efficiency normalization, not a claim that the
kernels execute identical instructions or move identical bytes.  W2.5 does
the same logical matrix multiplication as W4, but its P32 decoder performs
additional integer state reconstruction while moving fewer packed weight
bits.  Marlin and Machete execute W4 dequantization, so their effective
TFLOP/s values are figurative W4 baselines rather than accuracy-equivalent
comparisons.

## Projection-site comparison with Phase 4

The projection-only driver measures grouped QKV and grouped gate/up through
their complete input/output transforms, but excludes activation and down.  The
QKV source is unchanged and its geometric latency changes by less than 0.05%.
Nine QKV rows are numerically faster and eleven are slower, consistent with
run-to-run noise.  Gate/up improves in all 20 rows.

| Site | Phase 5 vs plain QVQ | Phase 5 vs Phase 4 | vs Marlin W4 | vs Machete W4 | Better than Phase 4 |
|---|---:|---:|---:|---:|---:|
| QKV | 2.211x | 1.000x | 1.249x | 0.805x | 9/20 |
| gate/up | 1.775x | 1.391x | 0.270x | 0.549x | 20/20 |
| QKV + gate/up model sites | - | 1.215x | - | - | 20/20 |

Summing the two projection sites per M/rate row lowers latency by **17.7%**
geometric mean versus Phase 4.  The complete MLP result is the more realistic
number because it includes the activation, product, and down projection.

## Profiler validation

The matched W3/M1 Nsight Systems trace shows the Phase-4 gate/up epilogue as
two serial FP32 Hadamard recoveries with a median of approximately 19.25 us
each.  Phase 5 replaces them with one paired launch whose trace median is
approximately 18.88 us.  The fused product/down-input precondition appears as
one approximately 16.91 us launch in the traced mixed workload.

Focused Nsight Compute replays report:

| W3/M1 kernel | Grid | Registers/thread | Dynamic shared memory/block | Executed instructions | Achieved occupancy | Local/shared spills |
|---|---:|---:|---:|---:|---:|---:|
| paired gate/up recovery | 2 blocks | 20 | 33.79 KiB | 204,352 | 49.46% | 0 / 0 |
| gate/up product + down precondition | 1 block | 22 | 16.90 KiB | 95,264 | 49.87% | 0 / 0 |

At M1 these kernels cannot fill 132 H100 streaming multiprocessors: there are
only two recovery rows and one precondition row.  Their low whole-device
throughput is therefore launch geometry, not evidence of an HBM conflict.  The
optimization wins by running the two recovery rows in one grid and deleting
intermediate allocations and launches.  Native small-M multi-row persistence
or fusion across layer boundaries would be needed to expose more parallel work.

## Method and controls

- Device: physical H100
  `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, PCI
  `00000000:44:00.0`, SM 9.0, 132 streaming multiprocessors.  The H200 was
  hidden with `CUDA_VISIBLE_DEVICES=1`.
- Before formal timing, the harness samples the H100 at least three times and
  requires 0% utilization and 0 MiB allocated memory.  Nsight itself reserves
  4 MiB, so profiler-only runs allow exactly that reservation after verifying
  0% utilization.
- M: 1, 2, 4, 8, and 16.  QVQ uses its production zero-padded M16 Hopper path;
  the W4 controls use their native M.
- Timing: warmed CUDA Graph replay bracketed by CUDA events, 20 replays per
  sample, 50 samples, and 20 warmups.  Host launch gaps, CPU starvation, and
  container scheduling are outside the timed interval.
- Marlin and Machete use symmetric GPTQ W4, group size 128, FP16 activations,
  and the same Llama projection geometry.
- The raw artifacts record the git base and a source fingerprint that is
  checked again after every matrix.

CUDA Compute Sanitizer is not installed on this host.  The implementation is
still covered by non-default-stream, overflow, repeatability, CUDA Graph, and
contract-guard tests on the H100, but sanitizer results are not claimed.

## Reproduction

- Full-MLP driver: `scripts/benchmark_qvq_a41_phase5_mlp.py`
- Full-MLP result: `artifacts/a41_phase5_h100/fused_mlp_vs_baselines.json`
- Projection result:
  `artifacts/a41_phase5_h100/production_grouped_vs_baselines.json`
- Kernel design: `qvq_a41_r0_phase5_fused_mlp.md`
