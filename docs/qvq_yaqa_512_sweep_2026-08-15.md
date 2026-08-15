# Llama 3.2 1B QVQ YAQA 512-row sweep (2026-08-15)

This log records the four-layer Q/K/V/O comparison of canonical V2 Block-LDLQ, canonical V2 with YAQA, and
V2B2-P32 with YAQA family reselection.

## Configuration

| Item | Value |
|---|---|
| Model | `unsloth/Llama-3.2-1B-Instruct`, snapshot `5a8abab4a5d6f164389b1079fb721cfab8d7126c` |
| Quantized scope | Real decoder layers 0--3, all 16 Q/K/V/O projections |
| Calibration | `neuralmagic/calibration` `llm.parquet`, rows 0--511, 188,256 non-padding tokens |
| Evaluation | Rows 512--1023, 512 disjoint full-length rows, 172,367 scored tokens |
| YAQA Sketch-B | Rows 1024--1535, 512 further-disjoint full-length rows, 163,324 valid tokens |
| YAQA collection batch | 8, 64 batches, padding excluded from factor accumulation |
| Evaluation batching | Batch 1, no row concatenation, no sequence-length truncation |
| Metrics | Exact token-weighted means; percentile fields in JSON are row-weighted auxiliary quantiles |
| Software | GPT-QModel Ultra `bf4a710f`, Python 3.14 free-threaded (`PYTHON_GIL=0`), Torch 2.13.0+cu130 |
| CUDA | CUDA 13.0, native QVQ kernels |

The Sketch-B factors were collected once and loaded from the validated shared cache for every arm. YAQA used the
shared stabilized block-LDL factorization introduced in `cc37ccc7`; no Cholesky failure occurred across the 64
YAQA module quantizations represented below.

## Completed results

`Layer KL` is the mean forward KL across the outputs of decoder layers 0--3. `Final KL` and Top-N are measured at
the final vocabulary logits after the four-layer truncated model.

| Rate | Arm | BPW | Weight rel-L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1.5 | V2 | 1.50000 | 0.462584 | 0.006305 | 0.017484 | 0.079240 | 0.270476 | 51.52% | 56.17% | 56.73% | 512.46 s |
| W1.5 | V2+YAQA | 1.50000 | 0.625811 | 0.091006 | 0.098095 | 0.039655 | 0.150286 | 62.03% | 66.14% | 67.13% | 672.29 s |
| W2 | V2 | 2.00000 | 0.332410 | 0.002197 | 0.006116 | 0.036377 | 0.131938 | 62.39% | 67.24% | 68.29% | 523.77 s |
| W2 | V2+YAQA | 2.00000 | 0.477162 | 0.018192 | 0.020307 | 0.014350 | 0.061103 | 73.59% | 76.97% | 77.84% | 688.51 s |
| W2.5 | V2 | 2.50000 | 0.237991 | 0.000945 | 0.003080 | 0.015032 | 0.060171 | 73.41% | 76.83% | 77.87% | 509.46 s |
| W2.5 | V2+YAQA | 2.50000 | 0.357409 | 0.006466 | 0.007221 | 0.005933 | 0.027478 | 81.60% | 83.75% | 84.59% | 666.13 s |
| W3 | V2 | 3.00000 | 0.170854 | 0.000470 | 0.001348 | 0.006415 | 0.029238 | 80.62% | 83.11% | 83.94% | 566.15 s |
| W3 | V2+YAQA | 3.00000 | 0.262786 | 0.003194 | 0.003539 | 0.002793 | 0.013363 | 86.68% | 88.24% | 88.94% | 742.48 s |

## V2-baseline deltas

| Rate | YAQA Layer-KL delta | YAQA Final-KL delta | Top-1 delta |
|---|---:|---:|---:|
| W1.5 | -50.0% | -44.4% | +10.51 pp |
| W2 | -60.6% | -53.7% | +11.20 pp |
| W2.5 | -60.5% | -54.3% | +8.19 pp |
| W3 | -56.5% | -54.3% | +6.06 pp |

YAQA increases raw weight relative L2 and the local QKVO KL at every rate while sharply reducing layer and final
logit KL. This is expected evidence that the two-sided objective is selecting a better propagated error direction,
not merely a lower-MSE weight reconstruction.

## V2B2-P32+YAQA family-reselection results

These arms evaluated all three alternative B2 families for every module under the YAQA objective, then retained
the best complete module result. They ran on dedicated physical GPUs 4--6 with verified process masks.

| Rate | Arm | BPW | Weight rel-L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1.5 | V2B2-P32+YAQA | 1.53125 | 0.614660 | 0.074820 | 0.080649 | 0.035609 | 0.138015 | 63.15% | 67.24% | 68.25% | 1012.80 s |
| W2 | V2B2-P32+YAQA | 2.03125 | 0.467218 | 0.016472 | 0.018225 | 0.013133 | 0.056970 | 74.55% | 77.59% | 78.66% | 980.17 s |
| W2.5 | V2B2-P32+YAQA | 2.53125 | 0.348671 | 0.006070 | 0.006788 | 0.005581 | 0.025907 | 81.84% | 84.28% | 85.02% | 967.36 s |

| Rate | Layer-KL delta vs V2+YAQA | Final-KL delta vs V2+YAQA | Top-1 delta vs V2+YAQA |
|---|---:|---:|---:|
| W1.5 | -10.20% | -8.16% | +1.12 pp |
| W2 | -8.48% | -6.76% | +0.96 pp |
| W2.5 | -5.93% | -5.72% | +0.24 pp |

The banked advantage remains positive after YAQA at every supported rate. Its magnitude decreases as rate rises,
from an 8.16% Final-KL reduction at W1.5 to 5.72% at W2.5.

### Selector and family telemetry

| Rate | Alternative occupancy | Entropy | Selector histogram | Module family histogram 0/1/2/3 |
|---|---:|---:|---|---|
| W1.5 | 50.08% | 0.999998 bits | 654345 / 656375 / 0 / 0 | 0 / 7 / 4 / 5 |
| W2 | 49.93% | 0.999999 bits | 656283 / 654437 / 0 / 0 | 0 / 5 / 5 / 6 |
| W2.5 | 50.01% | 1.000000 bits | 655243 / 655477 / 0 / 0 | 0 / 6 / 8 / 2 |

All three alternative families win complete modules at every rate. Near-maximal selector entropy indicates that
the gain is not produced by a sparse rescue path: YAQA actively uses both canonical and alternative segment
manifolds across approximately half of the P32 segments.

## W3.5 completion

W3.5 does not support V2B2-P32, so it compares canonical V2 with V2+YAQA. Both arms ran on dedicated physical GPU
7 with a verified process mask.

| Rate | Arm | BPW | Weight rel-L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W3.5 | V2 | 3.50000 | 0.123013 | 0.000241 | 0.000633 | 0.003223 | 0.015064 | 85.85% | 87.49% | 88.12% | 374.04 s |
| W3.5 | V2+YAQA | 3.50000 | 0.190856 | 0.001605 | 0.001774 | 0.001390 | 0.006748 | 90.27% | 91.27% | 91.83% | 430.78 s |

At W3.5, YAQA reduces Layer KL by 56.86% and Final KL by 55.20%, while improving Top-1 agreement by 4.42
percentage points.

## RAM-fix validation

The original evaluation implementation retained every full-vocabulary row and grew each worker to 335--364 GiB
RSS. Commit `bf4a710f` changed evaluation to compare the immutable dense replay model and quantized model one row
at a time, immediately reducing each row into exact token-weighted accumulators. Under the complete 512-row run,
worker RSS remained between 3.4 and 6.3 GiB and host available RAM remained approximately 1.8 TiB.

## Timing caveat and follow-up

An orchestration error scoped `CUDA_VISIBLE_DEVICES` to a shell builtin instead of the Python process, so the
W1.5--W3 V2 and V2+YAQA workers shared physical GPU 0 rather than the intended GPUs 4--7. Their output metrics
remain valid because each worker had an independent CUDA context and deterministic inputs, but those wall times
are contention measurements and must not be used as single-GPU performance benchmarks. The V2B2-P32+YAQA and
W3.5 runs used verified per-process masks on dedicated physical GPUs 4--7, so their wall times are valid dedicated-
GPU measurements.
