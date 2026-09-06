# Experiment coverage ledger

The historical F6 seed-7 checkpoint is the fixed read-only teacher. No numbered
experiment yet satisfies the entire requested scorecard. A passing subtest or a
completed worker does not mean the experiment is complete. The 3e-3 local inference
MAE gate is approved; max error remains 0.046875. Human exception review follows
AGENTS.md. All changes and evidence share PR #137.

| # | Experiment | Current evidence / outstanding implementation |
|---:|---|---|
|1|Decoder decomposition|Real-layer timing covers 12 projections and all nine row counts; expanded fused-window Nsight wave22 adds eight projections at M=1/16/2048 with registers, occupancy, shared memory, instructions, scheduler, warp-state, DRAM/L2, and source-counter CSVs. Tensor-Core roofline has no eligible metric for this Triton `_gemm`, and decoder/GEMM overlap remains.|
|2|Decode reuse across rows|Existing row-group reuse limits 2, 4, 8, and 16 now cover all 576 cases, with matched baselines for all 12 M values; extra launches/concatenation make all limits slower than no-reuse window as M grows. Direct fragment reuse remains.|
|3|Persistent decoded tiles|New persistent scheduling/register-pressure sweep remains.|
|4|Warp-specialized pipeline|Split-1, split-2, and split-4 direct decode/MMA tile baselines now cover 12 projections and 12 row counts with 1,872/1,872 local passes; split-K variants remain below split-1 performance and below the 2x target. Producer/consumer overlap implementation remains.|
|5|Transition LUTs|GPU index/pair LUT decoders exact across all 94 projections; both slower overall in materialization, fused LUT study remains.|
|6|Vectorized codebook output|Packed half2 output exact across all 94 projections at 1/4 warps; fused MMA-fragment output and model validation remain.|
|7|Short reduced-precision accumulation|Real five-projection arithmetic sweep recorded; layer1-down FP16/BF16 partials fail some/all cases. Integrated decoder/model and executed profiling remain.|
|8|Blockwise FP32 promotion|Integrated BM64/BN64 fused-window K16/K32/K64/K128/K256 arms pass 143/144, 143/144, 142/144, 142/144, and 132/144. The per-module K256 policy with an exact layer-1 down fallback passes 144/144 and reaches 1.20x at M=2048 in the local median. Wave20 model runs are complete, but matched timing repeats and downstream quality remain.|
|9|Output supertiles|Neighboring projections/channels implementation and matched evaluation remain.|
|10|Lossless repack|All 94 projections bit/value exact; 108 layers, bounded model PPL/logits/ARC and large prefill speedups measured; full quality/profiling coverage remains.|
|11|Independent trellis tiles|Requantized 64/128/256-tile exports and total BPW sweep remain.|
|12|Checkpointed states|Stored 8/16/32/64-step states and independent GPU blocks tested across 94 projections; slower than direct windows in initial state-only runs, full operator remains.|
|13|Multi-symbol LUT|Factored affine alternative measured under #24; explicit LUT storage/traffic study remains.|
|14|GPU-aligned banks|Constrained learned-bank implementation/calibration remains.|
|15|Additive codebooks|CPU fitting reference/tests; calibrated GPU/model integration remains.|
|16|Signed-basis P32|CPU fitting reference/tests; calibrated GPU/model integration remains.|
|17|INT4 + exceptions|Deployed fixed-base sparse-only sweeps: 1/2/4/8 target channels, 0–512 exceptions, all tested local cases fail. Broader budget/channel selection, model and profile coverage remain.|
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
|29|Native + low-rank + sparse/P32|Deployed rank6/8 plus sparse exports and bounded/full-ARC evidence recorded; joint budget optimization, retained P32 blocks and fused sparse execution remain.|
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

[Targeted rank8 results](RANK8_TARGETED.md) add a passing alpha1 rank8 fit,
calibration-fitted sparse exceptions, fused expansion/add with Graph timings and
executed Nsight/SASS evidence. [Full ARC](FULL_ARC.md) and canonical GSM8K-128
are complete. Native-GEMM fusion and fused-model validation remain outstanding.

[All-down rank screening](ALL_DOWN_RANKS.md): layers1–15 fitted ranks0–16;
none passes every canonical case, so all retain window. 105 exports and 945
reload metric comparisons are recorded. This does not complete broader/model
validation for possible future alternatives.

## Updated low-rank coverage

- 31/32/33/35: fixed-base rank/fit/FP32-FP16 studies executed on layer0; broader
  factor-precision options, maximum-aware objectives and all-layer coverage remain.
- 34: rank-plus-sparse sweeps executed; targeted channel selection uses development
  errors, so final untouched confirmation remains necessary.
- 36: all 15 calibration fits and 15 broader replays completed (360 exports,
  3,240 local reload cases, 9,000 broader cases). Cross-subset model checks are
  running; full stability/sensitivity interpretation remains.
- 37: layers1–15 initial ranks0–16 fail strict local coverage; remain window.
- 38: layer0-only model replacement evaluated; successful deeper replacements
  and nontrivial progressive replacement remain.
- 39: opt-in fused expansion/add integrated, 9/9 local and Graph checks, full ARC,
  C4/logits, fresh profiles and post-profile checks recorded. This is not fusion
  of the native base with correction, nor proven 2x full-model speedup.
- 40: bounded joint rank8/16 fitting executed; joint per-layer budget/rank selection
  and broader optimization remain.

See [calibration stability](CALIBRATION_STABILITY.md) and
[fused model evidence](FUSED_RECOVERY_MODEL.md). These updates do not close any
experiment's missing full scorecard items or replace experiments 1–30.

[Window model wave20](WINDOW_MODEL_WAVE20.md) completed eight full-model arms
(production control, K0/K16/K32/K64/K128/K256, and the per-module policy) on
the exact seed-7 snapshot. The wave records bounded C4-style PPL and full
prefill/decode timing. The production control has an anomalous M=1 median, so
model speed conclusions await matched repeats; ARC/GSM8K and full profiling
remain open.

[Window model repeat wave21](WINDOW_MODEL_REPEAT_WAVE21.md) reran the production
window and per-module policy on four GPUs each. It resolves the wave20 M=1
control anomaly and measures a pooled policy speedup of 1.294x at M=512 and
1.366x at M=2048, while policy remains slower at smaller M and in decode.
Downstream quality and complete profiling remain open.

[Window quality wave23](WINDOW_QUALITY_WAVE23.md) is complete: two production
window repeats score 390/1172 raw ARC, 428/1172 normalized ARC, and 59/128
GSM8K; two policy repeats score 390/1172 raw ARC, 429/1172 normalized ARC, and
58/128 GSM8K. The differences do not establish a quality advantage.
[The incremental record](WINDOW_QUALITY_WAVE23_PARTIAL.md) has two
identical production-window ARC repeats at 390/1172 raw and 428/1172
length-normalized, plus two identical production-window GSM8K repeats at
59/128. Two policy ARC repeats now agree at 390/1172 raw and 429/1172
length-normalized; policy GSM8K completed at 58/128 on both repeats.

[Window model tile wave24](WINDOW_MODEL_WAVE24.md) is complete across control
plus seven exact-K0 BM/BN/split arms. BM64/BN32 split1 is the fastest single
run at M=2048 (1.425x), BM64/BN64 split1 reaches 1.401x, and every fused arm
is slower at M=1 and decode. These model timing gains still need matched
repeats before promotion.

[Window model repeat wave25](WINDOW_MODEL_REPEAT_WAVE25.md) repeats the two
leading geometries across four GPUs each. BM64/BN32 reaches a pooled 1.423x at
M=2048 and BM64/BN64 reaches 1.402x; both remain slower at low M and decode.
The large-prefill gain reproduces, but no full-model 2x result exists.

[Window model wave26](WINDOW_MODEL_WAVE26.md) is complete. Four matched
BM64/BN32 per-module-policy runs reach 1.354x at M=512 and 1.378x at M=2048
against four same-wave controls, while remaining slower at low M and decode.

[BM64/BN32 SASS and algebraic waves 27–32](BM64BN32_SASS_WAVE27_32.md) are
complete on all eight mapped GPUs. Exact address/bank rewrites preserve gates
but do not improve the kernel. Scalar BM64/BN32 with four warps and three
stages is the best launch arm at about 1.29x on the layer-0 down projection.
The exact pair-LUT fold reduces integer SASS instructions by about 32% but
raises long-scoreboard stalls to about 50% and is slower. Cache-qualified LUT
loads also fail to advance. The next 3x work must remove tile staging and
transform boundaries while preserving the M>=512 dispatcher.

The same record now includes resident-word waves 33–34. Resident staging
reduces integer SASS by about 14% and registers from 64 to 56, but raises
barrier/MIO pressure and reaches only about 1.25x pooled M=2048 layer speedup.
Resident launch/stage wave35 and Nsight wave36 extend this result: all 72
frontier cases pass, while the best w4/s2 arm reaches only about 1.27x on the
strongest representative M=2048 projection. Wave36 reports 56 registers,
51.6% achieved occupancy, 12.19B integer instructions, 19.5% barrier stalls,
and 14.2% MIO stalls across eight profiled projections. Resident staging is
therefore rejected as the 3x path; transform fusion and cross-row reuse are
still required. BM128 cross-row reuse wave37 then passed 72/72 local cases but
reached only 1.398x pooled M=2048 layer speedup and was slower below M=512.
The current exact large-prefill candidate remains BM64/BN32; BM128 is retained
as a measured reuse ceiling, not a dispatcher replacement. Nsight wave38
measured 87 registers/thread, 16 KiB shared memory, 30.3% achieved occupancy,
and 23.2% long-scoreboard stalls; it cut integer SASS by about 50% but could
not convert that reduction into a 3x runtime result.
Transform wave39 added the existing CUDA Hadamard primitive with SU/SV fused
into the scorecard boundaries. Standalone input/output checks were bitwise
equal; all 72 BM64/BN32 cases passed, with pooled speedup 1.539x at M=512 and
1.787x at M=2048, while low-M direct decode remains slower. This is a fairer
layer scorecard but still does not establish a 3x result. Nsight wave40
bracketed the same BM64 decode/MMA call on all eight GPUs and reproduced 64
registers, 45.6% achieved occupancy, 14.72B integer SASS, and 13.8%
long-scoreboard stalls. The transform fold changes boundary cost only; it does
not yet overlap transform work with MMA.
