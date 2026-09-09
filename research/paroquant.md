# ParoQuant: learned pairwise rotations

## Sources and finding

[ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference,
v2](https://arxiv.org/html/2511.10645v2);
[authors' implementation](https://github.com/z-lab/paroquant).

ParoQuant uses scaled pairwise rotations to improve quantization geometry, with
runtime kernels designed around those transforms. This is calibration-time
optimization of a transformed representation, unlike simply estimating an absmax
scale or adding a post-quantization residual branch.

## Repository evidence

[gptqmodel/quantization/paroquant/optimization.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/quantization/paroquant/optimization.py) explicitly describes
learning channel scales/Givens angles, optimizing transformed-domain quantization,
and exporting packed tensors matching the pseudo-quantized layer.
[gptqmodel/looper/paroquant_processor.py](https://github.com/ModelCloud/QvQ/blob/4cbd1bc564c5faff7ce9fd0911b16a42ffefbeac/gptqmodel/looper/paroquant_processor.py) provides the lifecycle.
The method enum is METHOD.PARO with serialized value `paroquant`.

## Recovery implications

Track learned rotations, channel scales, quantizer parameters and runtime
precision independently. A pseudo-quantized calibration win must survive
packing and the actual transform kernel.

[EoRA](eora.md) factors fitted in one coordinate system cannot be applied in
another without a consistent basis change. Compare additional recovery only
after the deployed ParoQuant operator is validated. Do not attribute its learned
scale/rotation procedure to the calibration-only [NVFP4 paper](nvfp4-hybrid-ptq.md).

## GSQ adapter groundwork (2026-09-09)

The repository's ParoLinear inherits AWQ packed storage through AwqKomodoLinear,
but runtime first applies exported reciprocal channel scales and ordered pairwise
rotations. With row-vector convention, write this transform as X S R. A frozen
rotation GSQ adapter therefore needs teacher W S^-1 R and calibration X S R.
For transformed candidate error E, the original-domain error is E R^T S, and
the reconstruction quadratics agree. Using the optimizer's unexported channel
parameters in place of the stored reciprocal scales would fit the wrong problem.

`gptqmodel/quantization/gsq_paro.py::paro_gsq_basis` prepares that coordinate
system using export-rounded angles/scales and at least FP32 accumulation.
Independent dense rotation matrices check orientations, cancellation and
reconstruction-error equivalence across identity/three-stage rotations and
group sizes 16, 32 and channelwise. FP16/BF16/FP32 input checks cover promotion
and input nonmutation; invalid/nonrepresentable scales and overlapping pairs
are rejected. These are algebra/corner-case checks, not model-quality evidence
or native transform parity.

`refine_paro_export` now fits the frozen transformed-domain affine grid with
the AWQ export-aware scalar fitter. It supports fixed or learned group scales,
keeps rotation metadata untouched, and reconstructs replay weights from the
actual packed grid. Controlled improving-candidate fixtures pass through the
real AWQ CPU packer and independently reproduce the original-domain objective.
Disabled and exact-baseline cases preserve the original export tensors. The
low-level result deliberately leaves initializer train/validation diagnostics
separate from the GSQ before/after objective. No real-model gain or native
ParoQuant lifecycle support is established by these fixtures.

ParoConfig now accepts `gsq=None` (default) or a GSQConfig/dictionary. The
module-scope processor applies the fitter after rotation optimization and before
result export/replay, carries separate GSQ diagnostics, and recomputes reported
Smooth-L1 replay losses after an improvement. Initializer losses remain in the
diagnostics. The fitter never uses validation rows to select a checkpoint;
implicit validation tails are excluded even when the initializer's short-input
split overlaps. No-calibration fallback, disabled and unmatched modules retain
the initializer result.

The focused CPU suite passes 28 tests, including config round-trip, actual AWQ
packing, row-selection isolation and processor state application. The broader
ParoQuant config/processor selection has 81 passes, 3 skips and 3 failures; all
three failures also reproduce at the preceding unchanged commit and involve
grouped-processor fixtures. This is not a green whole-ParoQuant suite.

Those fixtures were subsequently updated to supply the module-tree role flags
used by the current processor: Q/K/V roles drive compute-block grouping and
the routed-expert role selects clone fallback. The combined focused CPU run
then passed 112 tests with 7 skips; the final expanded test matrix passes 112
with 11 skips (including the eight separately executed GPU cases). The earlier
failure logs remain archived.

Native SM80 checks now pass eight forced-improvement export/reload cases:
fixed/learned GSQ scales, M=1/17, K=N=128, W4/group128, and krot=1/8.
All preserve packed state under strict reload into ParoLinear, pass eager
output gates against the reconstructed original-domain reference, and pass
three changed-input CUDA Graph replays per case. Captured and eager outputs
match exactly. These controlled tensors establish runtime correctness, not
real-model recovery. The [native log and checks](../artifacts/gsq-paro/native-lifecycle/)
bind the executed tests and measured drift. This is one SM80 device and these
shapes only; no other architecture or broad performance claim follows.

Remaining work: grouped-scope calibration/result binding (enabled GSQ currently
rejects `layer` and `compute_block` scopes explicitly), broader native ParoQuant
packing/reload/rotation validation, and real calibrated layers. The new public
control is experimental and disabled by default. These pending scopes remain
part of the broader compatibility goal; they are not deemed mathematically
incompatible.
