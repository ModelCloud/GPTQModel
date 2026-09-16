# Corrected QVQ-GSQ W3/P32 validation

## Outcome

The corrected GSQ integration is operational and compatible with the QVQ
`qvq_v2b2_p32` contiguous-window payload on Llama 3.2 1B. It is not a no-op:
87 of 112 projections accepted 265 legal tile changes. All 112 packed
round-trip checks passed and no module regressed on the guarded
`normalized_prepared_yaqa_fisher` objective.

The full disjoint GSM8K-Platinum result does **not** show post-quant quality
improvement. The corrected checkpoint scored 516/1,209 (42.6799%), versus
537/1,209 (44.4169%) for the incoming YAQA checkpoint represented by the
original zero-change GSQ run. The observed delta is -21 answers, or -1.7370
percentage points.

## Quantization evidence

- Format: uniform W3, QVQ V2B2 P32, vector size 2, 16x16 tiles, contiguous
  trellis window 16, two banks.
- Fisher source: 10,178 independent YAQA sequences and 3,961,260 valid tokens.
- Fisher storage: exact dense float32 factors, 15,334,375,424 bytes.
- Projections: 112.
- GSQ changes: 87 improved modules, 25 retained baseline modules, 265 changed
  tiles.
- Aggregate guarded objective: 0.1488790203584358 to
  0.14886338604264893, a 0.0105014% relative decrease.
- Strict module improvements: 87; regressions: 0.
- Packed round-trip verifications: 112/112.
- Quantization wall time: 6,630.350 seconds; save time: 1.263 seconds.

The previous implementation reported 0 changed modules and 0 changed tiles;
all 112 before/after objectives were bit-identical. The corrected model shard
hashes differ from that no-op checkpoint.

## Performance evidence

The original GSQ coordinate initializer was launch- and synchronization-bound:
14.868 seconds, 830,475 CUDA launches, and 135,064 stream synchronizations for
the profiled real `8192 x 2048` projection. Batched coordinate scoring, direct
P32 edge edits, quartet decode reuse, and early stopping reduce the production
path to 1.047 seconds, a 14.2x speedup. The optimized profile has 12,023 CUDA
launches and 1,046 stream synchronizations.

Across the full model, GSQ refinement used 49.635 aggregate GPU-seconds. The
incoming YAQA baseline encoder used 1,128.533 aggregate GPU-seconds and remains
the dominant post-capture cost. Nsight Compute measured its dominant guarded
FP32 GEMM at 86.17% SM throughput; replacing it with a custom kernel is not
justified without changing the exact FP32 acceptance contract.

## SM90 and inference evidence

The P32 CUDA device bodies, ABI, build targets, and runtime smoke test now admit
SM90 while retaining the SM100+ guard. The corrected checkpoint loaded and ran
with the ZML native paged runner on an NVIDIA H100 using FA2 attention,
batch size 8, context 8,192, and decode graph mode.

- Canary: 8/8 complete, 0 invalid, 0 incomplete.
- Full run: 1,209/1,209 complete, 0 invalid, 0 incomplete.
- Full inference wall time: 811.586 seconds.
- Full score: 516/1,209 (42.6799%).

## Paired post-quant comparison

The corrected and prior runs used identical saved prompts and targets for all
1,209 examples.

- Correct under both: 394.
- Wrong under both: 550.
- Prior correct, corrected wrong: 143.
- Prior wrong, corrected correct: 122.
- Net: -21 correct answers.
- Predictions changed: 1,180; extracted numeric answers changed: 693.
- Two-sided exact paired sign/McNemar p-value: 0.21915.

Thus the observed regression is not significant at the 5% level, but there is
also no evidence that enabling GSQ improves GSM8K. The Fisher proxy improvement
is real and packing-safe; it does not transfer positively to this downstream
metric under the tested configuration.

## Data isolation

Strict disjointness validation passed. Quantization used the frozen
`nm-calibration/llm.parquet` and `yaqa182-nm10000.parquet` bindings. Evaluation
used only `madrylab/gsm8k-platinum`, config `main`, split `test`. GSM8K was not
used for quantization, GSQ candidate selection, or parameter tuning.

## Artifacts

- Quantization report:
  `/root/qvq-results/w3-p32-gsq-corrected-20260916/quantize_run.json`
- Corrected checkpoint:
  `/root/qvq-results/w3-p32-gsq-corrected-20260916/model`
- Full evaluation:
  `/root/qvq-results/w3-p32-gsq-corrected-20260916/evaluations/zml-sm90-full1209/evaluation.json`
- Raw evaluation samples:
  `/root/qvq-results/w3-p32-gsq-corrected-20260916/evaluations/zml-sm90-full1209/evaluation.samples.jsonl`
- Nsight/performance report: `performance.md` and
  `/root/qvq-results/gsq-performance-20260916/`.
