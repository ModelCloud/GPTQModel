# Experiment coverage ledger

The historical F6 seed-7 checkpoint is the fixed read-only teacher. No numbered
experiment yet satisfies the entire requested scorecard. A passing subtest or a
completed worker does not mean the experiment is complete. The 3e-3 local inference
MAE gate is approved; max error remains 0.046875. Human exception review follows
AGENTS.md. All changes and evidence share PR #137.

| # | Experiment | Current evidence / outstanding implementation |
|---:|---|---|
|1|Decoder decomposition|Real-layer transform/decoder timing and four scoped Nsight captures; instruction-class breakdown and overlap remain.|
|2|Decode reuse across rows|Five row-group limits, 45 passing cases on one projection; launch/concat confounding remains, controlled fused sweep needed.|
|3|Persistent decoded tiles|New persistent scheduling/register-pressure sweep remains.|
|4|Warp-specialized pipeline|Producer/consumer decode-MMA implementation and overlap measurement remain.|
|5|Transition LUTs|Compact transition analysis available; actual GPU LUT implementation/sweep remains.|
|6|Vectorized codebook output|Existing window kernel is comparator; dedicated packed-output ablation remains.|
|7|Short reduced-precision accumulation|Promotion intervals 16–256 and real-model validation remain.|
|8|Blockwise FP32 promotion|Reduction-order/promotion variants remain.|
|9|Output supertiles|Neighboring projections/channels implementation and matched evaluation remain.|
|10|Lossless repack|All 94 projections bit/value exact; 108 layers, bounded model PPL/logits/ARC and large prefill speedups measured; full quality/profiling coverage remains.|
|11|Independent trellis tiles|Requantized 64/128/256-tile exports and total BPW sweep remain.|
|12|Checkpointed states|Stored 8/16/32/64-step states and independent GPU blocks tested across 94 projections; slower than direct windows in initial state-only runs, full operator remains.|
|13|Multi-symbol LUT|Factored affine alternative measured under #24; explicit LUT storage/traffic study remains.|
|14|GPU-aligned banks|Constrained learned-bank implementation/calibration remains.|
|15|Additive codebooks|CPU fitting reference/tests; calibrated GPU/model integration remains.|
|16|Signed-basis P32|CPU fitting reference/tests; calibrated GPU/model integration remains.|
|17|INT4 + exceptions|Sparse residual reference; deployed base/exception export and calibration remain.|
|18|Hybrid native/trellis|Sensitivity selection, tile dispatch, and full BPW/latency tradeoff remain.|
|19|Native base + low rank|Four W4A16 projections refit at FP16 boundary on 8192 tokens; down rank16 passes 9/9 and bounded model run has mixed PPL/ARC effects. Full coverage/export remains.|
|20|Joint optimization|CPU alternating-callback reference; deployed alternating optimization remains.|
|21|Associative scan|Exact GPU states across all 94 projections; unfused state-only scan slower overall than direct extraction. Fused scorecard remains.|
|22|All-start decoding|CPU symbolic all-start reference; redundant GPU all-start or independently justified symbolic GPU variant remains.|
|23|Sparse checkpoints|8/16/32/64-step metadata and GPU decode measured exactly across 94 projections with full sidecar BPW; fused/model study remains.|
|24|Super-symbol automaton|2/4/8-step compact affine GPU scans exact across 94 projections; no overall state-only speed win. Fused lookup/MMA and LUT alternatives remain.|
|25|Bit-sliced decoder|CPU reference exact on sampled real tiles; actual Boolean GPU kernel remains.|
|26|GF(2) jump-ahead|CPU matrix reference exact; GPU compact affine scan #21 is related evidence, matrix-form implementation remains.|
|27|Tensor-product codebook|CPU reference/tests; calibrated learned representation, GPU execution and quality/BPW sweep remain.|
|28|Signed/ternary basis|CPU reference/tests; trained GPU representation and model study remain.|
|29|Native + low-rank + sparse/P32|CPU residual components; deployed joint budget optimization remains.|
|30|Sparse Walsh spectrum|CPU local-Walsh reference/tests; calibrated GPU/model representation remains.|

Authoritative evidence links: [window model](WINDOW_MODEL.md), [row reuse](ROW_REUSE.md),
[GPU scan](GPU_SCAN.md), [super-symbols](GPU_SUPERSYMBOL.md),
[CPU references](REFERENCE_IMPLEMENTATIONS.md), [teacher snapshot](F6_SEED7_SNAPSHOT.md).
Runtime progress logs live outside Git; record completed measurements before
claiming a row has advanced. Small C4/ARC slices currently do not establish P32's
post-quant advantage over original BF16.
