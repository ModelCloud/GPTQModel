# Experiment coverage ledger

The historical F6 seed-7 checkpoint is the fixed read-only teacher. No numbered
experiment yet satisfies the entire requested scorecard. A passing subtest or a
completed worker does not mean the experiment is complete. The 3e-3 local inference
MAE gate is approved; max error remains 0.046875. Human exception review follows
AGENTS.md. All changes and evidence share PR #137.

| # | Experiment | Current evidence / outstanding implementation |
|---:|---|---|
|1|Decoder decomposition|Real-layer transform/decoder timing and four scoped Nsight captures; instruction-class breakdown and overlap remain.|
|2|Decode reuse across rows|Fused BM16/32/64 split16 q sweep and layer1-down BM32 pass 36/36; modest/no gains against existing window. More configurations/model/profile remain.|
|3|Persistent decoded tiles|New persistent scheduling/register-pressure sweep remains.|
|4|Warp-specialized pipeline|Producer/consumer decode-MMA implementation and overlap measurement remain.|
|5|Transition LUTs|GPU index/pair LUT decoders exact across all 94 projections; both slower overall in materialization, fused LUT study remains.|
|6|Vectorized codebook output|Packed half2 output exact across all 94 projections at 1/4 warps; fused MMA-fragment output and model validation remain.|
|7|Short reduced-precision accumulation|Real five-projection arithmetic sweep recorded; layer1-down FP16/BF16 partials fail some/all cases. Integrated decoder/model and executed profiling remain.|
|8|Blockwise FP32 promotion|Blockwise FP32 promotion16–256 passes all 225 arithmetic-isolation cases across five projections; integrated kernel/model/profile remain.|
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
|20|Joint optimization|Two deployed rank16 rounds on four projections measured; calibration selects step0 for q/k, step2 for gate/down. Full rank allocation/model study remains.|
|21|Associative scan|Exact GPU states across all 94 projections; unfused state-only scan slower overall than direct extraction. Fused scorecard remains.|
|22|All-start decoding|CPU symbolic all-start reference; redundant GPU all-start or independently justified symbolic GPU variant remains.|
|23|Sparse checkpoints|8/16/32/64-step metadata and GPU decode measured exactly across 94 projections with full sidecar BPW; fused/model study remains.|
|24|Super-symbol automaton|2/4/8-step compact affine GPU scans exact across 94 projections; no overall state-only speed win. Fused lookup/MMA and LUT alternatives remain.|
|25|Bit-sliced decoder|GPU bit-plane decoder exact across all 94 projections at 1/4 warps; state-only pipeline slower than direct windows. Fused/model scorecard remains.|
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

Additional partial evidence: [accumulation isolation](ACCUMULATION.md) records
432 passing real layer-0 cases for experiments 7/8; execution profiling and
integrated model results remain. [Bounded GSM8K](GSM8K_PARTIAL.md) records the
completed BF16/window/joint-recovery arms; FP32 teacher evaluation is running.

Experiments **31–40** are authorized in [the small-rank queue](LOW_RANK_QUEUE.md).
The focused 31/32/33/35 layer-0 down sweep is executed; model jobs are automatically
dispatched when their exports are ready. Full completion is still unproven.

[Small-rank model results](LOW_RANK_MODEL.md): all 28 focused model runs complete;
full ARC and broader activation replays dispatched. All-16-down calibration and
held-out captures are complete; per-layer native fits/selection remain.
