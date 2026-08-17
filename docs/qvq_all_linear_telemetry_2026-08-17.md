# QVQ all-linear telemetry sweep — 2026-08-17

## Workload

- Model: Llama 3.2 1B Instruct, all 16 decoder layers.
- Modules: Q/K/V/O plus gate/up/down (112 linear modules); embeddings and LM head remain dense.
- Calibration: 512 full, independent rows at offsets 0–511; batch 1; no concatenation or length limit.
- Evaluation: 512 disjoint rows at offsets 512–1023.
- YAQA Sketch-B: 512 further-disjoint rows at offsets 1024–1535; batch 8.
- Hardware: NVIDIA PG506-230/232, compute capability 8.0, 124 SMs, CUDA 13.0, Torch 2.13.0+cu130.

## Completed V2 phase attribution

`baseline_encode` contains `block_ldl_viterbi` and must not be added to it. The MLP gate and final evaluation are
wall-clock stages, while Viterbi and evaluation subphases are CUDA-event work sums.

| Rate | Total (s) | Encode wall (s) | MLP gate (s) | Final eval (s) | Viterbi GPU (s) | Local replay GPU (s) | Dense forward GPU (s) | Quant forward GPU (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | 692.6 | 114.1 | 308.2 | 59.2 | 112.8 | 20.3 | 11.7 | 10.1 |
| W1.5 | 710.3 | 126.2 | 297.7 | 62.5 | 124.9 | 23.2 | 11.4 | 10.2 |
| W2 | 647.0 | 107.2 | 287.4 | 57.8 | 106.3 | 20.3 | 11.2 | 9.8 |
| W2.5 | 663.8 | 112.6 | 296.4 | 59.1 | 111.6 | 20.2 | 11.6 | 10.3 |
| W3 | 684.7 | 129.3 | 297.6 | 58.9 | 128.2 | 20.3 | 11.6 | 10.2 |
| W3.5 | 700.9 | 112.7 | 319.8 | 59.5 | 111.8 | 20.3 | 11.9 | 10.3 |

The top quantization stages are MLP acceptance and native Viterbi. The top final-evaluation GPU work is grouped
local replay, followed by dense model forward; quantized forward is a close third.

## MLP acceptance exact-KL A/B

The gate previously invoked the general CPU diagnostic metric suite for every near-threshold candidate, although
acceptance consumes only final-logit KL and Top-1/5/10. The optimized path recomputes only CPU FP32 KL and retains
the existing deterministic CUDA Top-N values.

| Arm | Model | Layers | Rows | Proposals | Total (s) | Metrics (s) | Candidate forward (s) | Max KL delta | Top-N delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline full CPU diagnostics | Llama 3.2 1B | 16 | 8 | 112 + baseline | 91.775 | 88.991 | 7.407 | — | — |
| Lean exact CPU KL | Llama 3.2 1B | 16 | 8 | 112 + baseline | 45.736 | 43.319 | 6.908 | 9.83e-7 | 0 |

The end-to-end gate speedup is **2.01x**, the metric phase speedup is **2.05x**, all Top-1/5/10 fingerprints are
identical, and worst KL drift remains below the declared `1e-6` quantization-analysis tolerance.

## Rejected Viterbi experiments

Nsight Compute on the W2 FP16-codebook recurrence reported 72.23% SM throughput, 56.27% L2 throughput, 64
registers/thread, 40.96 KiB dynamic plus 8.19 KiB static shared memory per CTA, and 48.75% achieved occupancy. The
dominant sampled stalls were math-pipe throttle, dispatch, long scoreboard, and wait.

- Replacing the final serial 1,024-entry reduction with a warp/block reduction was not consistently faster across
  W1–W3.5 and was rejected.
- A 512-thread CTA plus maximum shared-memory carveout preserved exact states but regressed a matched raw two-pass
  496-sequence benchmark (W3: 19.991 ms to 22.546 ms; W3.5: 18.187 ms to 19.454 ms) and was rejected.

Profile artifact: `/private/monster/data/model/qvq_all_linear_telemetry_20260817/reports/viterbi-w2-sm80.ncu-rep`.

## Final-evaluation local replay A/B

The local reconstruction diagnostic previously disabled TF32 for 112 large FP32 projection replays per row. TF32
is now scoped only to those diagnostic GEMMs; dense and quantized model forwards retain their original numerical
policy. A matched 512-row, all-linear, 16-layer real-model replay produced:

| Metric | FP32 baseline | Scoped TF32 | Speedup / absolute drift |
|---|---:|---:|---:|
| Evaluation wall time | 58.256 s | 42.104 s | 1.38x |
| Local replay GPU work | 20.273 s | 4.255 s | 4.77x |
| Local relative L2 | 2.091496e-4 | 2.091536e-4 | 3.99e-9 |
| Local KL | 5.122090e-6 | 5.121487e-6 | 6.03e-10 |
| Final-logit KL | 0 | 0 | 0 |
| Top-1/5/10 | identical | identical | 0 |

A separate 16-layer, 320-token raw-output stress microbenchmark measured 4.83x replay speedup, 2.78e-4 maximum
absolute output drift, and 7.69e-6 relative L2, all inside the `2e-3` inference-analysis tolerance.

## YAQA feedback A/B

Telemetry from the full YAQA sweep exposed a lifecycle gap: canonical V2+YAQA still used the shrinking-suffix
feedback path, while the exact incremental CUDA recurrence was enabled only by the B2 wrapper. The production
dispatcher now enables the incremental path for every CUDA YAQA format within its measured 2,048-dimension
envelope, including the independent bank-zero oracle.

| Shape | Hessian | Path | Wall (s) | Feedback GPU (ms) | Viterbi GPU (ms) | Weight/state parity |
|---|---|---|---:|---:|---:|---|
| 2048 x 2048 | identity | suffix | 3.678 | 2468.7 | 990.4 | reference |
| 2048 x 2048 | identity | incremental | 1.460 | 2.9 | 932.2 | bit-exact |
| 1024 x 1024 | nontrivial dense SPD | suffix | 1.084 | 389.0 | 524.9 | reference |
| 1024 x 1024 | nontrivial dense SPD | incremental | 0.519 | 1.6 | 449.1 | bit-exact |

This is a **2.52x** module speedup at 2,048 square and **2.09x** with a nontrivial 1,024 square Hessian. Exact
CUDA parity now covers canonical V2, B2-P32, and B4-P64 at every half-step W1--W3.5; the production-dispatch test
also verifies that canonical V2+YAQA selects the optimized recurrence.

## Rejected final-forward experiments

The dense and quantized model forwards remain the two largest individual final-evaluation phases. Two matched
512-row experiments were rejected:

| Experiment | Baseline wall (s) | Candidate wall (s) | Result |
|---|---:|---:|---|
| Concurrent dense/quantized CUDA streams | 42.104 | 43.086 | 2.3% regression; metrics bit-identical |
| Explicit FlashAttention 2 | 42.104 | 47.604 | 11.6% regression; floating-point ordering changed |

The forward pair already saturates the same SM resources under batch-1 full-row evaluation, so stream overlap
causes contention. Transformers' default SDPA is faster than FA2 for this exact variable-length workload.
