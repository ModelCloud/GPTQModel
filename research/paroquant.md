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

Real Llama 3.2 1B block-0 Q/K/V validation subsequently completed: three arms
per projection (baseline, fixed GSQ, learned-scale GSQ), W4/group128/krot8,
base seed7, actual module optimizer and native ParoLinear pack/save/reload.
The saved 16 calibration documents were split 12/4 for training/initializer
validation; all 32 held-out documents remained outside optimization. All 288
native checks pass (worst mean 0.0003560413, max 0.008094788). Both GSQ arms
retain baseline payloads exactly for all projections; no recovery is claimed.
See the [real-layer report](../docs/experiments/gsq-paro-real-layers.md) and its
manifest-bound artifacts. Grouped binding and final-model propagation remain
pending; this result is not a complete native model export.

Implicit-calibration follow-up: matching GSQ modules now reserve disjoint
whole-sequence prefix/suffix streams before activation concatenation. Short
calibration sets previously allowed the processor to pass overlapping streams
as explicit fitter inputs. One sequence leaves validation empty so the module
path uses its existing internal row split; disabled/unmatched controls preserve
the previous selection. Seven regression cases include the layer-capture filter
route. The focused processor/config/GSQ suite passes 119 tests with 11 skips
(CPU only); this does not change the explicit 12/4-document real experiment.

Grouped GSQ binding is now implemented locally for clean-input `layer` and
`compute_block` initializers: each exported module receives the same optional
format-aware local fitter using captured training/validation streams. Metadata
identifies the initializer scope and group validation loss separately from
module refinement; group loss is explicitly not recomputed, and the process log
labels it `group_initializer_before_gsq`. Fitting time is included in module
duration. No grouped objective recovery follows from local improvement.
The clean-target/noisy-input mode still rejects enabled grouped GSQ pending
paired module activation binding. Grouped native and real-model checks remain
pending; CPU config round-trips and stream/diagnostic binding pass.

Controlled grouped lifecycle checks now route the actual GSQ fitter through
`_quantize_layer` for both scopes, force an improving export, and verify packed
state application, separated calibration counts, initializer loss preservation
and cleanup. Only the group initializer is replaced with a deterministic test
fixture. These tests establish lifecycle behavior, not real-model quality.

Grouped explicit calibration now fails if either requested stream is missing,
rather than substituting an implicit split. Final focused CPU validation: 126
passed, 11 skipped, 47 deselected; log archived in
`artifacts/gsq-paro/processor-cpu/grouped-gsq.log.gz`. No GPU execution is claimed
for the newly bound grouped path.

The low-level export fitter now accepts optional row-aligned `teacher_inputs`
for clean targets and `inputs` for noisy runtime activations. In frozen export
coordinates it adds D=(Xclean-Xnoisy)^T Xnoisy to the scalar asymmetric
quadratic. The omitted constant is ||(Xclean-Xnoisy) Wteacher^T||^2; normalization
remains the noisy-input teacher energy. Its reported objective can therefore
be negative and must not be labeled normalized clean-target MSE.
Independent direct-output algebra checks and actual fixed/learned-scale fitting
through AWQ packing pass on controlled fixtures. Identical clean/noisy inputs
retain exactly the unpaired tensor outputs and history. These checks are
correctness evidence only. Processor collection of aligned clean/noisy module
activations is not yet bound, so the public noisy-input guard remains active.

Paired processor infrastructure now captures clean activations through the
shared pristine replay context and noisy activations through the normal module
hook. Records carry batch/invocation IDs; duplicates, missing calls and shape
mismatches fail alignment. Capture is transactional, copies input storage,
preserves existing hooks, and protects its bookkeeping during parallel replay.
Paired streams use matching sequence and row selections before the asymmetric
fitter; improved replay losses compare noisy outputs against clean targets.
Temporary clean/noisy buffers are cleared after layer quantization.

Controlled integration tests run pristine capture through noisy hook capture,
alignment, actual GSQ fitting and export application for both grouped scopes.
They replace the group initializer with a fixture, so they do not establish
real grouped optimization or model-quality recovery. Focused CPU validation:
142 passed, 11 skipped, 44 deselected. The configuration guard remains active
pending actual grouped replay/native/model verification.
