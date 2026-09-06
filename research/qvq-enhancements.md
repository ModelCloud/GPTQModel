# QVQ-specific quantization and recovery enhancements

These are implementation findings at the
[audited revision](https://github.com/ModelCloud/QvQ/tree/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac),
not separate publications or evidence that all combinations are production defaults.
External foundations are [QTIP](qtip.md), [YAQA](yaqa.md),
[EoRA](eora.md) and [rotation/scaling](rotation-and-smoothing.md).

## Implementation map

| Enhancement | Evidence | What it changes / boundary |
|---|---|---|
| Codebooks, bank candidates and trellis rounding | [QVQ math](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq.py); [V4 candidates](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_v4_candidates.py) | Representable weight candidates and selection; do not equate all versions |
| YAQA real-Fisher weighting | [collector](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_yaqa.py) | Rounding objective; A8 collection is not A4 proof |
| Fixed-trellis output alignment | [alignment](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/qvq_output_alignment.py) | Trainable SU/SV around fixed inner weights; transactional replay/acceptance |
| Propagation-shaped spectral correction | [spectral helpers](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_spectral.py) | Candidate mode selection using realized deltas and cross-fit machinery |
| SwiGLU-aware scaling and candidate selection | [SwiGLU helpers](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/swiglu.py) | Joint gated-product sensitivity rather than isolated GEMM error |
| Hessian-weighted adjacent rounding | [QUBO/Ising helpers](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/adjacent.py); [model integration](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/adjacent_model.py) | Fixed-codebook neighboring-code choices; exact and heuristic solver status differ |
| Viterbi survivor pruning | [pruning policy](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_pruning.py) | Exact search acceleration policy, not weight sparsification |
| Transform placement | [planner](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_transform_planner.py); [runtime](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/qvq_transform_runtime.py) | Placement of equivalent transforms; validate whole graph |
| Rate/sensitivity analysis | [profiler](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/sensitivity/profiler.py); [SLQ utilities](slq.md) | Bit allocation / diagnosis; no automatic quality guarantee |
| Lossless windows and runtime output-residual fitting | [P32](p32.md); [EoRA and native recovery](eora.md) | Storage layout versus additive recovery are distinct |

## Scientific distinctions that affect implementation

For the dense SwiGLU product, the helper scales the up-projection's output
channels and inversely scales the down-projection's input columns. It leaves the
gate unchanged because SiLU is not homogeneous. Arbitrarily scaling the gate and
dividing down is not an equivalent transformation. Candidate evaluation should
observe the complete gated product and downstream projection.

Output alignment exposes SU/SV parameters around a fixed decoded inner weight.
These fitted transform parameters differ from the [NVFP4 paper's calibrated
encoding scales](nvfp4-hybrid-ptq.md). A differentiable surrogate needs validation
against the real installed decoder, including dtype boundaries.

A favorable first-order spectral score is a candidate-screening signal, not
proof of an improvement after nonlinear model propagation. Use the realized
serialized delta and disjoint replay. Similarly, an exact adjacent-rounding
solution is exact only for its specified fixed-codebook objective and solver
completion state, not for end-to-end model quality.

Exact survivor pruning aims to retain the baseline recurrence's answer. It
must not be described as approximate weight pruning. Preserve strict/fallback
policy behavior and test the selected native implementation.

## Proposed recovery workflow

Record which of these stages ran and in what order; do not infer a recipe from
the checkpoint's W label. Separate changes in representation, objective,
transform parameters, native precision and additive factors.

Preserve the installed baseline and evaluate candidates after serialization.
Use real calibration/evaluation inputs, state the teacher target, and measure
local error, propagated agreement and tasks separately. Do not duplicate an
existing recovery fitter solely because a new activation format is proposed.
