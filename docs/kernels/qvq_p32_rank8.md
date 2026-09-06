# P32 window rank-8 reference integration

Baseline: `0cbb57c7` (main including PR #137). The window representation and
all CUDA decoders remain unchanged. This change adds rank-8 state to
`QVQLinear`, its quantization result, and the grouped Hopper consumer.
Existing output "recovery" still means Hadamard/SV/bias; new tensor names are
`rank8_A`, `rank8_B`, and `rank8_metadata`.

For row-major activations the contract is:

```
x_prime = existing_input_transform(x)
inner = existing_window_dispatch(x_prime)          # existing FP32 result
hidden = fp16(fp32(x_prime) @ fp32(rank8_A))
corrected = fp32(inner) + fp32(hidden) @ fp32(rank8_B)
y = existing_output_hadamard_sv_bias(corrected)
```

The separate-reference correction requires TF32 disabled. It has explicit
FP16 hidden rounding and FP32 expansion/addition; it does not fold A/B into
weights. Recovery-off does not inspect correction tensors in forward and
retains existing dispatch and arithmetic. No A100 M512 crossover is installed.

## Quantization and artifact ownership

`quantize_qvq_linear(..., rank8_calibration=Rank8Calibration(...))` retains the
original FP weight, completes normal P32 quantization, fits the completed
runtime module, and returns optional rank-8 buffers in `serialized_tensors()`.
The calibration processor also accepts explicit per-module calibration through
`set_rank8_calibration(name, calibration)`. Collection of original activations
and document provenance remains the caller's responsibility; there is no
implicit reuse of propagation gates or evaluation benchmarks. Automatic
whole-model document capture is not implemented. Atomic SwiGLU selection and
output alignment must finish before fitting and are currently rejected by the
processor attachment.

The initial fitter is bounded-calibration CPU FP64 reduced-rank regression
with `gelsd`, rcond 1e-5. It fits the output predicted by least squares, not a
weight-space SVD. The two objectives are ordinary output L2 and output L2
weighted by calibration residual row energy. B is fit in final-output
coordinates then transformed by inverse SV/Hadamard into inner coordinates.
This matters for nonuniform SV and nonsymmetric composite Hadamards.

Both candidates are scored with FP16 serialized factors, the deployed inner
operator and its output transform on training and disjoint held-out documents.
Validation requires MSE improvement and no p99 absolute-error regression on
both splits. Balanced selection additionally requires the configured minimum
relative MSE improvement (default 1%). Quality accepts every validated module.
No synthetic test establishes real-model quality or promotion eligibility.

The ordinary checkpoint remains canonical planar P32 with optional rank-8
buffers; old checkpoints require no additional storage. The explicit unified
window exporter stores window words instead of planar words (never both),
actual codebook levels, selectors, transforms, bias, factors and fit metadata.
Loading reconstructs canonical planar ownership for existing QVQLinear and
uses its existing transient window cache. This is the same reversible layout,
not another rank-8 weight format. The package and standard state_dict both
reload with correction off until explicitly prepared.

SHA256 binds factors to exact window bytes, levels, bank state, dimensions,
rate, codebook version, transform flags, SU/SV and bias. Separate hashes bind
factors and original teacher parameters. Hashing happens before capture;
identity/version guards reject state mutation during eager execution. Do not
mutate captured graph buffers or tensor `.data` behind the ownership boundary.

`save_window_package` reports actual file bytes including the Torch container.
`window_package_storage` separately reports tensor bytes, per-module window
and recovered BPW, weight-count-weighted average BPW and the analytic FP16
rank-8 increment `16*8*(K+N)/(K*N)`. These are supplied-module totals, not a
whole-model inventory unless every model tensor has been accounted for.

## Runtime and external policy

```python
from gptqmodel.quantization.qvq_rank8 import P32WindowConfig, prepare_rank8

prepare_rank8(layer, P32WindowConfig(recovery_mode="auto", quality_mode="balanced"))
y = layer(x)  # same module, same input/output transform boundaries
```

`qvq_p32_window_linear(layer, x, config)` is the Python entry point over the
module-owned operands. Configure outside graph capture. Policy is immutable
per capture; separate fast/balanced/quality captures must be owned by the
execution graph caller. Tests retain and replay off/on graphs after changing
the eager policy. There is no graph manager that silently changes modes.

`auto`/`production_window` preserve existing dispatch. Explicit `hopper_m16`
and `hopper_direct_decode_mma` expose existing M16 and row-reuse WGMMA
consumers, split count and M range for external correctness/timing comparisons.
Unsupported devices, activation contracts and shapes fail explicitly.
`window_tuning_key` includes physical product name/UUID/memory/SM count,
shape, rate, M, TP world/rank, transforms, quality/recovery state and caller
build identity. It is a key builder, not a ZML autotuner or StableHLO lowering.
Grouped dispatch retains its existing geometry policy.

Grouped QKV and gate/up apply child corrections to completed FP32 outputs using
exactly the shared padded activation that fed WGMMA, then run existing child
or paired output transforms. Each child may independently enable correction.
Folded alternatives without that activation boundary are bypassed. Whole-MLP
fusion falls back to grouped gate/up plus ordinary down forward while rank-8
is active; raw split partials cannot accept correction before reduction.

## Remaining work and promotion boundary

The reference and fused output epilogue are implemented. Fully fused input projection,
complete native pipeline fusion, FP8 factors, rank-16 Tensor Core sweeps,
externally selectable BM/BN/stages, native ABI v3/StableHLO lowering, ZML
latency tuning, automatic graph construction, real-model quality evaluation,
TP1/2/4/8 and H100/H200 performance promotion remain unimplemented/unvalidated.
`fused_epilogue` runs expansion/add/Hadamard/SV/bias in one kernel.
`fully_fused` remains unsupported; it does not alias the partial fusion.
No <=3–5% overhead or model speed/quality claim follows from these tests.

## Validation of this WIP

- CPU regression: `CUDA_VISIBLE_DEVICES='' python -m pytest -q
  tests/test_qvq_window_recovery.py tests/test_qvq_v2b2_p32_window.py
  tests/test_qvq_grouped_runtime.py --disable-warnings --maxfail=1`:
  79 passed, 179 skipped (before the final quantization smoke was added).
- Final H200 run: `CUDA_DEVICE_ORDER=PCI_BUS_ID
  CUDA_VISIBLE_DEVICES=GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea
  python -m pytest -q tests/test_qvq_window_recovery.py
  tests/test_qvq_grouped_runtime.py --disable-warnings --maxfail=1`:
  **50 passed, 86 skipped**. The new rank-8 file itself passes all 26 tests.
  Skips include H100-specific and opt-in real-model tests.
- H200 single-operator matrix: M1/16/64/512, K256/N256, FP16 activations,
  W3, auto/M16/large-M consumers, correction off/on, ten graph replays.
  The independent FP32-inner correction composition agrees exactly at the
  returned FP16 boundary. Retained off/on graphs replay correctly after eager
  mode changes. Grouped QKV and gate/up at M1/64 pass both local absolute-error
  limits and exact eager/graph replay with independent child flags.
- CPU smoke runs actual P32 quantization, rank-8 fitting, serialization and
  reload. Synthetic low-rank-teacher fixtures validate both Hadamard-axis
  settings and nonuniform SV; they are algebra tests, not model-quality data.
- New Python files pass Ruff. Comparing Ruff results for changed existing
  files with `HEAD` found no additional findings; their pre-existing lint
  findings remain. `git diff --check` passes.

Hardware: physical GPU0, PCI `00000000:1C:00.0`, NVIDIA H200,
UUID `GPU-0c667065-5c47-38ce-0b0a-d211392ce9ea`, driver 610.57.04,
143771 MiB; Torch `2.15.0.dev20260901+cu130`, Triton `3.8.0+gitc01b6774`.
This was correctness-only, without performance-isolation or Nsight promotion
claims. No CUDA/C++/Triton kernel source or compiler flags were changed.
Logs: `/tmp/qvq-rank8-final.log`, `/tmp/qvq-rank8-regression.log`,
`/tmp/qvq-rank8-quantize.log`. The initial graph-policy rejection is retained
in `/tmp/qvq-rank8-h200.log`; explicit FP16 dispatch now preserves the existing
BF16 overflow-rescue consumer during capture.


## Fused output epilogue (00a71f4b)

`recovery_kernel="fused_epilogue"` retains the reference FP32 projection/FP16
hidden boundary and fuses expansion, FP32 base addition, existing rounded
output Hadamard, SV and bias into one Triton kernel. QVQLinear and grouped
QKV/gate-up consume it through the same operator. One CTA owns an output row;
no persistent scratch or grid barrier is introduced. Supported output widths
are powers of two from16 through16384 on SM90, with optional Hadamard/bias and
strided grouped row storage. Unsupported widths fail explicitly in this
opt-in implementation; automatic dispatch remains unchanged.

The butterfly keeps each historical FP16 rounding point and rescues an
individual overflowing conversion in FP32, matching modes3/4 in
`qvq_hadamard_cuda.cu`. FP32 addition follows the completed eight-product rank
reduction. `enable_fp_fusion=False` prevents contracting that addition into
rank products. Source-correlated SASS confirms separate FMUL/FADD rank
instructions; later FFMA instructions implement correctly rounded division.
Triton's half2 folding preserves the tested half rounding boundaries.

At matched M16/K2048/N2048/W3, Nsight counts13 baseline kernels versus9 fused,
and 2,977,948 versus2,486,932 executed warp instructions (16.5% fewer).
The fused output kernel executes135,616 instructions, uses47 registers/thread,
8KiB shared memory,128 threads/CTA, and no spills. The old Hadamard alone
uses27 registers/thread,8.25KiB shared memory and1024 threads/CTA.
The fused kernel has substantial shared exchange work: the NCU shared-conflict
metric is1568 versus6 for the padded old Hadamard, and measured active-warp
occupancy is6.24% versus49.59% on this small M16 grid. These are optimization
opportunities, not evidence that fewer instructions guarantees lower latency.
The expanded correction and its casts/add/global-memory boundary disappear;
the input projection and its casts/reduction remain separate.

Post-profile tests: **87 passed,86 skipped**, including output widths through
16384, per-operation overflow rescue, no-bias/strided outputs, independent
single/grouped composition, and ten graph replays per case. Skips are mostly
H100 and opt-in real-model tests. Performance remains synthetic operator
scope; no real-model or H100 promotion is implied.

Artifacts: `/tmp/p32-rank8-{reference,fused}-nsys.nsys-rep`,
`/tmp/p32-rank8-{reference,fused}-ncu.ncu-rep`, corresponding `-sass.csv`,
`-metrics.csv`, and `-opcode-summary.json`; correctness log
`/tmp/p32-rank8-fused-postprofile-tests.log`. The full remaining requirements
are tracked in [the phase ledger](qvq_p32_rank8_plan.md).

Post-profile graph medians, matched K=N2048/W3/FP16 on the H200:

| M | Window off us | Separate rank8 us | Fused output rank8 us | Rank8 implementation speedup | Remaining recovery overhead |
|---:|---:|---:|---:|---:|---:|
|1|33.115|47.560|41.808|1.138x|26.25%|
|16|31.586|50.392|43.881|1.148x|38.93%|
|128|35.019|56.571|48.428|1.168x|38.29%|
|512|65.843|99.847|87.404|1.142x|32.75%|
|2048|238.922|296.029|265.462|1.115x|11.11%|

Mean and p95 improve as well; the machine-readable
[summary](results/p32_rank8_h200_epilogue.json) retains those values. All
preflight and pre-timing exclusivity checks passed, with no foreign compute
processes. This meets the local opt-in implementation gates at the measured
scope but does **not** meet the final <=3–5% recovery overhead target.
