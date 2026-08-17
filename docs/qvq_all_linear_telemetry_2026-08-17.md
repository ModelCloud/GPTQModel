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

The incremental update now uses one in-place `addmm(beta=1, alpha=-1)` for its second projection and subtraction,
removing one full-matrix workspace and one launch per anti-diagonal. At 1,024 square, update GPU work fell from
39.27 ms to 34.32 ms (1.14x); at 2,048 square, combined commit/update work fell from 469.96 ms to 465.91 ms
(1.009x). End-to-end module wall time changed by less than 0.2%, so the retained benefit is primarily lower
workspace and clearer `yaqa_feedback_update` telemetry. Weight/state outputs remained bit-exact in the W1--W3.5
gate.

## Rejected final-forward experiments

The dense and quantized model forwards remain the two largest individual final-evaluation phases. Two matched
512-row experiments were rejected:

| Experiment | Baseline wall (s) | Candidate wall (s) | Result |
|---|---:|---:|---|
| Concurrent dense/quantized CUDA streams | 42.104 | 43.086 | 2.3% regression; metrics bit-identical |
| Explicit FlashAttention 2 | 42.104 | 47.604 | 11.6% regression; floating-point ordering changed |

The forward pair already saturates the same SM resources under batch-1 full-row evaluation, so stream overlap
causes contention. Transformers' default SDPA is faster than FA2 for this exact variable-length workload.

## Sketch-B accumulator residency A/B

Sketch-B previously copied every quadratic input/output Gram update to CPU and accumulated it there after every
batch. For wide gate/up projections this repeatedly moved the complete factor footprint over PCIe and introduced
host synchronization inside every module hook. The collector now retains accumulators on CUDA when an
allocator-aware check can preserve at least 25% of device memory (and at least 16 GiB), then performs one final
host transfer. CPU remains the exact low-memory fallback.

Matched real Llama 3.2 1B A/B: first two layers, all four gate/up modules, 32 independent full rows, Sketch batch
8, FP32 factors, checkpointed full-model backward.

| Accumulator | Wall (s) | CUDA work (s) | Peak VRAM (GiB) | Factor footprint (GiB) | Max factor drift |
|---|---:|---:|---:|---:|---:|
| CPU per-batch | 7.604 | 6.976 | 7.79 | 1.06 | reference |
| CUDA then one host transfer | 4.127 | 2.647 | 8.86 | 1.06 | 4.06e-7 |

The end-to-end collection speedup is **1.84x** with 1.07 GiB extra peak VRAM. The measured drift is below the
declared `1e-6` quantization tolerance. Tiny-model CUDA/CPU accumulation, activation-checkpointing, repeated-seed,
and factor-output tests remain bit-exact; telemetry now records residency, allocated factor bytes, capture wall/CUDA
time, and final transfer time.

## Near-threshold acceptance CUDA KL recheck

The full W2.5 V2+YAQA run exposed the next evaluation bottleneck after the lean metric change: 61 near-threshold
proposals caused 496 historical CPU KL rechecks. Metrics consumed 215.466 seconds while candidate forwards used
only 9.843 seconds. The exact recheck now uses ordinary CUDA FP32 log-softmax and reduction; deterministic
Top-1/5/10 continues to use the native tie-stable evaluator.

| Case | Device / dtype | CPU reference | CUDA recheck | Absolute KL delta | Contract |
|---|---|---:|---:|---:|---:|
| Representative 384 x 128256 logits | A100 / FP16 | 0.711 s | 0.063 s cold | 1.49e-7 | <=1e-6 |
| Ten scale/noise cases | A100 / FP16+BF16 | — | — | 3.47e-7 max | <=1e-6 |

The warm CUDA reduction is millisecond-scale, so the expected improvement for the 215-second W2.5 metric phase is
substantially larger than 10x without changing an acceptance decision within the required numerical contract.

## Completed full-model YAQA sweep

This matched sweep quantized all 112 Q/K/V/O/gate/up/down projections in all 16 Llama 3.2 1B layers. V2 and
B2-P32 used the same 512 calibration rows, 512 disjoint evaluation rows, and 512 further-disjoint Sketch-B rows.
Rows were processed at batch 1 without concatenation or truncation; Sketch-B used batch 8.

| Rate | Arm | Rel L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | V2 + YAQA | 0.546288 | 0.694820 | 1.059582 | 1.615112 | 0.942185 | 64.97% | 58.97% | 58.11% |
| W1 | B2-P32 + YAQA | 0.529927 | 0.644295 | 0.943450 | 1.391596 | 0.800008 | 67.64% | 61.10% | 60.14% |
| W1.5 | V2 + YAQA | 0.406340 | 0.155091 | 0.319266 | 0.407024 | 0.248925 | 82.65% | 74.27% | 73.64% |
| W1.5 | B2-P32 + YAQA | 0.389116 | 0.134656 | 0.270713 | 0.370088 | 0.214388 | 83.33% | 75.69% | 75.15% |
| W2 | V2 + YAQA | 0.293481 | 0.048756 | 0.118903 | 0.127933 | 0.091508 | 88.80% | 82.59% | 82.21% |
| W2 | B2-P32 + YAQA | 0.286913 | 0.044344 | 0.111459 | 0.126684 | 0.088402 | 89.66% | 82.90% | 82.53% |
| W2.5 | V2 + YAQA | 0.201517 | 0.019941 | 0.047538 | 0.047341 | 0.033498 | 92.85% | 88.35% | 88.19% |
| W2.5 | B2-P32 + YAQA | 0.196658 | 0.018809 | 0.045044 | 0.041130 | 0.031463 | 93.20% | 88.83% | 88.55% |
| W3 | V2 + YAQA | 0.142375 | 0.009995 | 0.022475 | 0.019315 | 0.014759 | 95.08% | 91.92% | 91.78% |
| W3 | B2-P32 + YAQA | 0.141947 | 0.009436 | 0.022751 | 0.020499 | 0.015194 | 95.35% | 91.77% | 91.58% |
| W3.5 | V2 + YAQA | 0.107992 | 0.004975 | 0.012768 | 0.012918 | 0.008952 | 96.32% | 93.60% | 93.42% |
| W3.5 | B2-P32 + YAQA | 0.099910 | 0.004619 | 0.010688 | 0.009418 | 0.006952 | 97.01% | 94.26% | 94.07% |

### B2-P32 delta from V2

KL/Rel-L2 columns report relative reduction (positive is better); Top-N columns report percentage-point change.

| Rate | Rel L2 | Local KL | Live KL | Layer KL | Final KL | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | +2.99% | +7.27% | +10.96% | +13.84% | +15.09% | +2.67 | +2.13 | +2.03 |
| W1.5 | +4.24% | +13.18% | +15.21% | +9.07% | +13.87% | +0.68 | +1.42 | +1.51 |
| W2 | +2.24% | +9.05% | +6.26% | +0.98% | +3.39% | +0.86 | +0.31 | +0.33 |
| W2.5 | +2.41% | +5.68% | +5.25% | +13.12% | +6.07% | +0.36 | +0.48 | +0.35 |
| W3 | +0.30% | +5.60% | -1.23% | -6.13% | -2.95% | +0.27 | -0.16 | -0.21 |
| W3.5 | +7.48% | +7.15% | +16.30% | +27.10% | +22.34% | +0.70 | +0.66 | +0.65 |

B2-P32+YAQA wins the final KL at five of six rates. W3 is the exception: local KL improves 5.60%, but live,
layer, and final KL regress, demonstrating again that the local objective is not a sufficient promotion gate.

## Full-run hotspot closure

W2.5 is representative of the common-rate timing profile.

| Area | Before | After | Speedup / finding | Accuracy |
|---|---:|---:|---:|---|
| MLP acceptance metric phase | 215.466 s | 33.092 s | 6.51x | CUDA KL within 3.47e-7 of CPU; deterministic Top-N |
| Complete MLP acceptance | 222.159 s | 40.800 s | 5.45x | same acceptance policy |
| Final local replay | 20.273 s | 4.215 s | 4.81x | local Rel-L2 drift 3.99e-9; final metrics exact |
| Sketch-B, 2-layer gate/up micro A/B | 7.604 s | 4.127 s | 1.84x | factor drift 4.06e-7 |
| Canonical square YAQA feedback, 2048 | 3.678 s | 1.460 s | 2.52x | weights/states bit-exact |

The remaining W2.5 B2 quantization work is dominated by Sketch-B capture (about 521 seconds across O, gate/up,
and down) and exact YAQA encoding. Within the selected encodes, feedback plus incremental update consumes about
170 GPU-seconds and canonical plus segmented Viterbi consumes about 165 GPU-seconds. B2 intentionally performs
an independent canonical encode and a segmented-family encode so rejection can restore exact V2 bytes.

The final 512-row evaluator now takes 41.47 seconds. Its largest GPU phases are dense forward (10.83 seconds) and
quantized forward (9.73 seconds), followed by layer metrics (6.28 seconds). Concurrent model streams and explicit
FA2 were both measured regressions and remain disabled.

A wide YAQA TF32 feedback experiment was also rejected. It preserved states/weights in the tested 1024 x 4096
identity and SPD probes but improved wall time by only 1.07--1.11x; process-global TF32 state and unproven near-tie
behavior do not justify weakening the exact quantization contract.

## Batched acceptance reduction and bounded exact rechecks

The remaining 33.09-second acceptance metric phase was not limited by the native fused reducer. On the exact
2,702-token by 128,256-vocabulary geometry, one fused KL/Top-10 pass takes 34.8 ms on the local `sm_80` A100-class
GPU. The cost came from invoking the reducer once per source row and from 528 broad PyTorch log-softmax rechecks.

The evaluator now caches the immutable teacher logits once in allocator-bounded FP32 CUDA storage, streams each
proposal into one flat FP32 buffer, and invokes the fused reducer once per proposal. Baseline and genuinely
ambiguous proposals reuse that same buffer for the exact KL check, avoiding a second suffix forward. Exact
rechecks are limited to the measured `1e-6` fused-KL uncertainty interval around the acceptance boundary. Top-N
uses deterministic value-descending/index-ascending indices and therefore needs no floating-point margin.

Matched 113-proposal scheduling microbenchmark, including the historical 66 exact rechecks:

| Path | Device | Shape | Wall (s) | Speedup | KL delta | Top-1 delta | Top-5 delta | Top-10 delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Rowwise + 66 broad rechecks | PG506-230 `sm_80` | 2702 x 128256 | 24.437 | reference | reference | reference | reference | reference |
| Batched + bounded recheck | PG506-230 `sm_80` | 2702 x 128256 | 3.964 | **6.16x** | 2.17e-19 | 6.38e-8 | 1.11e-16 | 0 |

The tiny Top-1 fraction delta is only the order used to average identical integer token decisions. It is far below
the `2e-3` inference tolerance; KL remains far below the `1e-6` quantization gate. Telemetry records batched calls,
allocator fallbacks, persistent teacher-cache bytes, and transient candidate-buffer bytes.

## Factored anti-diagonal YAQA feedback update

The incremental CUDA recurrence previously rebuilt each anti-diagonal update with two full-matrix GEMMs even
though the committed matrix is zero outside a sparse set of 16 x 16 tiles. The optimized path batches each tile's
right projection and applies their concatenated input factors as one low-rank update. It removes the mostly-zero
dense intermediate and caches every row-index tensor with the immutable anti-diagonal schedule.

Matched Llama 3.2 1B layer-0 q_proj, 2048 x 2048, W2.5 B2-P32+YAQA, sampled-96 family selection, three measured
runs after warm-up:

| Path | Feedback-update GPU | Module median | Phase speedup | Module speedup | Accuracy |
|---|---:|---:|---:|---:|---|
| Full-matrix anti-diagonal GEMMs | 913.61 ms | 3.295 s | reference | reference | reference artifact |
| Factored low-rank update | 275.09 ms | 2.635 s | **3.32x** | **1.25x** | weights/states/selectors/family bit-exact |

The CUDA reference gate also compared the incremental result against the direct YAQA recurrence across W1--W3.5,
canonical V2, B2-P32, B4-P64, and both 32 x 48 and 48 x 32 rectangular B2 geometries: 12/12 focused cases passed.

An implicit-PGC16 emission experiment was rejected. It was bit-exact across 72 constrained, unconstrained,
weighted, and unweighted rate/bank cases, but integer state mixing increased the real segmented phase from about
1.09 seconds to 1.94 seconds and module time to 4.14 seconds. The measured 99.97% codebook L2 hit rate makes the
cached table loads cheaper than reconstructing every state in the recurrence.

With dense feedback removed, overlapping the independent canonical and selected B2-family artifacts becomes a
small repeatable win even when sampled selection leaves only one complete alternative. Matched medians on two
PG506-230 `sm_80` GPUs were 2.499 s and 2.521 s versus the 2.635 s serialized median (1.05x). A focused CUDA test
confirms the fixed-family parallel and serialized artifacts are bit-exact. Full three-family reselection retains
its existing concurrent-stream path.

## Midpoint-only segmented provisional traceback

Tail-biting consumes only the provisional state overlap at transition 63. The SM80 midpoint specialization stores
backpointers for transitions 63--126 and traces only to that overlap, instead of materializing the complete
provisional states, selectors, and loss. A complete W1--W3.5 B2/B4 sweep found this profitable only at W1 and W3.5,
so production dispatch is rate-gated; W1.5--W3 retain the faster complete provisional implementation.

Matched Llama 3.2 1B layer-0 q_proj, sampled-96 B2-P32+YAQA family selection, three measured runs after warm-up:

| Rate | Complete provisional | Midpoint provisional | Module speedup | Segmented-Viterbi GPU | Accuracy |
|---|---:|---:|---:|---:|---|
| W1 | 2.694 s | 2.643 s | 1.02x | 1325.95 -> 1278.72 ms | bit-exact artifact |
| W3.5 | 3.120 s | 2.971 s | 1.05x | 1579.04 -> 1455.68 ms | bit-exact artifact |

CUDA 13.0 (`nvcc 13.0.88`) tests cover all six rates, B2-P32 and B4-P64, weighted and unweighted objectives,
batches 1/3/17, and three deterministic repeats: 72/72 passed with exact overlap equality. The middle-rate
regressions measured approximately 1--5%, which is why they are not routed to this specialization.
