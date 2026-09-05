# Twenty P32 experiments

Status: preparation only; **0/20 experiments completed**. No candidate is promoted.

Baseline: origin/main `a75e6f732d49041535ff0b0890d4002b3fc0d7ae`.
All experiment commits and results belong to the single branch `experiments/p32-twenty` and its PR.
Commit and push after each completed experiment, including rejected results and profiler evidence.

## Inputs still to lock

The user must identify the accepted P32 teacher by checkpoint path/model revision.
The best checkpoint documented in `docs/qvq_vaqa_best_log.md` is absent at its recorded
`/root/qvq-results/calibration-fisher-scaling-v2/llama32-1b-f6_yaqa182_nm10000-anchor-up4-l6-l8` path.
Other local QVQ checkpoints cannot establish retention of that checkpoint's advantage.
Record hashes of all teacher tensors, metadata, tokenizer and original BF16 checkpoint before running.
The original `/monster/data/model/Llama-3.2-1B-Instruct` directory is available.

Calibration must use ordinary C4 or FineWeb text, with pinned dataset revision and document IDs.
Keep calibration, tuning, held-out perplexity and downstream evaluation disjoint. Record sample counts,
sequence lengths, seeds, prompt/template configuration and evaluation versions before candidate selection.
Never calibrate on evaluation benchmarks. Synthetic tests establish kernel correctness only.

## Common scorecard

For every candidate, preserve raw per-case results and record:

- Canonical FP32 P32 reconstruction and forward teacher; original BF16 secondary reference.
  Disable TF32/reduced precision modes for the canonical FP32 reference. Record all dtype boundaries.
- Layer mean/max absolute error, relative L2 and cosine on identical inputs; propagated logits KL
  (teacher to candidate), top-1/5/10 agreement, perplexity, and downstream task scores.
  Declare top-k agreement semantics and aggregation before running. Keep kernel and model metrics separate.
- M = 1, 2, 4, 8, 16, 32, 128, 512, 2048, with actual N/K and projection roles;
  actual Llama prefill and decode, including the full transforms, decoder, correction and epilogue.
- Warmed latency distributions and baseline/candidate speedups. Target >=2x end-to-end linear latency
  improvement, stretch >=4x. Model inference and held-out quality must support advancement.
- Tensor Core utilization, executed integer/FP32 instructions, registers, spills, occupancy, shared memory,
  L2 traffic and decoder/GEMM overlap. Bind source-correlated SASS and profiler reports to revisions/builds.
- Effective BPW = 8 * total representation bytes / represented logical weights. Include packed data,
  checkpoints, padding, LUTs, banks, scales, exception indices/values and correction factors. Report resident
  decoded caches separately and count shared tables once with an explicit amortization denominator.
- Numerical failures, unsupported cases and unavailable metrics explicitly; never substitute zero.

Localized inference gates: finite values, mean absolute drift <=0.002 AND maximum absolute drift <=0.046875
per case against the same-input canonical operator. Exactness claims additionally require exact reconstructed
values and preserved checkpoint bits where applicable. Passing these tolerances does not prove bitwise math.
Model-quality gates and uncertainty policy must be locked with the teacher before selection.

## Ordered ledger

All entries are pending teacher quantization, baseline evaluation, and execution.

| Wave | ID | Experiment | Required sweep |
|---|---:|---|---|
| 1 | 1 | Decoder cost decomposition | traversal, bank, lookup, SU/SV, Hadamards, accumulation; isolated/fused |
| 1 | 2 | Decode once across rows | reuse 2/4/8/16 |
| 1 | 5 | Transition LUT | state, bank XOR, PGC and code indices; LUT sizes |
| 1 | 7 | Short low-precision partials | FP16/BF16; promotion K=16/32/64/128/256 |
| 1 | 10 | Lossless repack | warp order, boundaries, padding; inverse round trip |
| 2 | 3 | Persistent decoded tiles | tile shape, reuse, registers/shared storage |
| 2 | 4 | Warp-specialized pipeline | producer/consumer warps, double buffers |
| 2 | 6 | Vector codebook outputs | half2/BF16 pairs/fragment layout |
| 2 | 8 | Blockwise promotion | local/periodic/final FP32 reductions; deterministic orders |
| 2 | 9 | Output supertiles | neighboring channels, QKV, gate/up |
| 3 | 11 | Independent trellis tiles | 64/128/256 weights |
| 3 | 12 | Checkpointed states | every 8/16/32 transitions |
| 3 | 13 | Multi-symbol tables | 2/4/8 steps |
| 3 | 14 | GPU-aligned banks | permutations, signs, additive bases |
| 3 | 15 | Additive codebooks | two-codebook sizes and residual |
| 3 | 16 | Signed basis | basis dimension and residual |
| 4 | 17 | INT4 plus exceptions | exception density and full BPW |
| 4 | 18 | Hybrid native/trellis | calibration-weighted tile sensitivity and allocation |
| 4 | 19 | Native plus output-fitted recovery | rank 16/32/64/128 |
| 4 | 20 | Joint native/recovery optimization | iterations and per-layer rank allocation |

Experiments 7 and 8 preserve checkpoint bits but can change arithmetic. Experiments 11 and 14–18 may change
reconstructed weights; classify those as approximation unless exact value equivalence is demonstrated.
A representation change alone does not guarantee exact reconstruction.

For 19, fit rank-constrained D to Z = Y_P32(X) - Y_native(X; Q, scales, activation quantization), minimizing
||Z-XD||_F^2. Obtain Z from the actual deployed native function; a weight-only residual is insufficient.
For 20, alternate native quantization of W_T-AB and fitting real output residuals; account for all rank storage.

## Host preflight (2026-09-05)

- Physical GPU 0, PCI 00000000:25:00.0, UUID GPU-cb9e7784-cf50-203d-4f0d-5c622a89b1f2.
- NVIDIA PG506-230, compute capability 8.0, 124 SMs, Torch reports 97457 MiB memory.
- Driver 610.43.02; `/root/work/venv/bin/python`: Torch 2.13.0+cu130, CUDA 13.0.
- Nsight Compute executable exists at `/usr/local/bin/ncu`; counter permission not yet tested.
- Initial inventory showed no running GPU processes. Formal three-sample and pre-timing idle gates pending.
- This host supports Ampere experiments. H100/H200 FP8 and Blackwell NVFP4 performance require those devices.

Validation so far: repository checkout and runtime property query only; no kernel benchmarks or model evaluation.

### Follow-up preflight

Nsight Compute 2026.2.1 successfully collected a `SpeedOfLight` section from a Torch elementwise addition.
This is a profiler access probe, not a P32 experiment or benchmark. Three idle samples each showed 0 MiB,
0% utilization and no compute process on the selected UUID. The probe's exact script, log and raw CSV are
in `artifacts/p32_twenty/preflight/`. Reproduction from repository root:

```bash
ncu --profile-from-start off --section SpeedOfLight --launch-count 1 --csv \
  --log-file /tmp/p32-counter-probe.csv \
  /root/work/venv/bin/python artifacts/p32_twenty/preflight/counter_probe.py
```

The script intentionally identifies this host's physical GPU by UUID; substitute a verified target UUID
when reproducing on another host. No P32 bottleneck conclusion is drawn from this probe.

A config inventory of `/monster/data/model/Llama-3.2-1B*QVQ*/quantize_config.json` found 49 configs,
all with top-level format `qvq`; a text search found no `p32` or `v2b2` marker in those configs.
The documented best checkpoint was not found by its name under `/monster`, `/tmp`, `/root/work`, or `/qvq`.
Teacher identification/access remains unresolved.

Existing integration points inspected:

- `qvq_dense_oracle_forward` in `gptqmodel/nn_modules/qlinear/qvq.py` reconstructs full FP32 operators,
  including the layer metadata. Audit activation settings and math modes before fixing it as teacher.
- `scripts/benchmark_qvq_p32_ampere.py` provides window/planar timing and runtime device checks, but its
  generated fixture weights are not accepted checkpoint weights or model-quality evidence.
- `scripts/benchmark_qvq_rotation_full_model_cuda.py` re-quantizes and selects WikiText validation streams;
  it cannot be used unchanged for this fixed-checkpoint, ordinary-text-calibration study.

Experiment completion remains 0/20. No kernel source or production dispatch has changed.


### Authorized teacher: F6 seed 7

The user selected F6 seed 7 on 2026-09-05 and authorized fresh Llama 3.2 1B quantization.
This resolves checkpoint identity: use the resulting immutable checkpoint for all twenty experiments.
`scripts/p32_twenty/f6_seed7.json` copies the documented F6 config, changing only YAQA seed 0 to 7.
The study retains latest-main quantizer code, rather than reproducing the historical quantizer revision.

To retain the original ordinary-text calibration requirement, `prepare_c4.py` uses the local C4 training shard,
with URL and normalized whole-document deduplication across all streams and up to 512 tokens per document.
This is an F6-settings C4 variant, not a historical F6 quality reproduction. No historical score is inherited.
The checkpoint must establish its own BF16 comparison before any post-quant advantage is claimed.

Prepared streams: lifecycle 128 documents, teacher calibration 10178, candidate calibration 512,
tuning 128, heldout 256. Corpus bindings and ordered document hashes are stored in
`/root/work/p32-twenty-data/manifest.json`. No benchmark is used for calibration.
Exact duplicate/URL separation is not a claim of semantic near-duplicate decontamination.

Run `bash scripts/p32_twenty/quantize_teacher.sh` after `prepare_c4.py`.
Output: `/root/work/p32-twenty-data/f6-seed7-c4-checkpoint`.
The launcher pins the physical UUID, passes a strict idle gate and caps build/CPU parallelism.
Environment: `/root/venv-py3.14t`, Python 3.14, Torch 2.15.0.dev20260817+cu130, Transformers 5.15.0.
The earlier Python 3.13 probe environment lacks model dependencies and is not used for quantization.
