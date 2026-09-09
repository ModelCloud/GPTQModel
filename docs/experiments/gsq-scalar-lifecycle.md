# Scalar GSQ integration work

This extends the optional control beyond the verified [QVQ tile adapters](gsq-qvq-lifecycle.md).
The scalar paths remain experimental. Real GPTQ and RTN W4 projection results
below do not qualify complete-model exports or the untested backends/methods.

## Current implementation

`GPTQConfig`, `AWQConfig` and `RTNConfig` accept a nested `GSQConfig`, defaulting
to `None`. `GSQConfig(enabled=True, learn_scales=True)` additionally trains
positive group scales; `learn_scales=False` keeps scales fixed. QVQ currently
rejects scale learning because its tile adapter has a different scale contract.
The shared control describes the search; each quantization method owns its
candidate adapter and calibration objective.

- GPTQ retains original-column weights and calibration Hessian before its
  sequential quantizer consumes them. GSQ follows GPTQ's final inverse ordering,
  using the returned `g_idx`, scales and zeros. Embedding metrics stay diagonal.
  GPTAQ/FOEM and mock quantization currently reject enabled GSQ pending their
  own objective adapters.
- AWQ GEMM/GEMV/GEMV_FAST/LLM-AWQ snapshot scaled dense weights before clipping. GSQ runs after AWQ
  rounding and optional adjacent refinement, using the corresponding scaled
  activations. It runs before EoRA residual capture and packing. A fallback
  without activations explicitly uses weight MSE.
- RTN uses weight reconstruction without manufacturing calibration data. It
  supports GPTQ exports and the same four AWQ storage adapters.

Scalar candidates are legal affine integer codes, with fixed zero-points and
group ownership. The grid uses all codes when the candidate budget allows it,
otherwise a bounded neighborhood of the baseline assignment. Initial logits
use `-0.5 * (candidate_code - baseline_code)^2`, following the distance prior in
the [pinned author implementation](https://github.com/IST-DASLab/GSQ/blob/03fc16484c369e3127225615d5e03e8d3a6043e3/src/quantization/gumbel_quantizer_2bit.py).
An equal prior for distant alternatives failed even elementary W4/W8 fitting
fixtures; the distance prior passes them. This implementation uses Adam, a
private seeded Gumbel stream and geometric temperature decay, rather than
claiming reproduction of the author's full optimizer and schedules.

Hard checkpoints are scored from reconstructed packed codes and the actual
stored scale values in FP32 canonical arithmetic. This is separate from FP16
or BF16 runtime multiplication/accumulation error. GPTQ and AWQ GEMM have
different inverse-arithmetic/casting conventions; the AWQ adapter casts its
scale table before code reconstruction. The unchanged baseline is retained
unless a hard checkpoint improves the calibration objective. Held-out data
must never select the checkpoint.

## Remaining compatibility and verification work

CPU checks on 2026-09-09: 36 scalar helper/lifecycle fixtures cover W2/W3/W4/W8,
activation ordering, diagonal/rank-deficient metrics, default-off tensor/RNG
parity, scale learning, and actual AWQ GEMM packing across FP16/BF16/FP32. The
combined scalar/GPTQ/AWQ/RTN/QVQ regression command passed 10,320 cases with 1,079
skips in 211.68 seconds; most cases are existing randomized GPTQ Hessian tests.
A separate QVQ config/processor and weight-only/AWQ processor run passed 48
cases with one accelerator skip. All runs hid CUDA; skips do not establish
accelerator support. Those historical Ruff and `git diff --check` results covered the then-current
working changes only, not every committed file in the PR. A later broad audit
found 94 Ruff findings (including historical artifact scripts and package
re-exports) and trailing whitespace in committed raw logs; the branch-wide
checks were not clean at that audit. Subsequent cleanup makes Ruff over all
changed Python paths and `git diff --check origin/main` pass. Original artifact
bytes are preserved under `artifacts/gsq-review/raw-archive` with SHA256 bindings.
An expanded 56-case scalar suite adds malformed metadata, mixed-device input,
storage underflow and finite-objective/gradient/checkpoint failure cases. It
achieves 100% CPU line/branch coverage of `gsq_scalar.py` (138 statements, 60
branches); this is not whole-package or GPU branch coverage. No scalar quality
gain is claimed from fixtures. See the [coverage data](../../artifacts/gsq-scalar/validation/scalar-coverage.json).

The user-requested goal includes every compatible method. This inventory keeps
that scope open; a missing adapter does not establish mathematical incompatibility.

| Method/path | Current evidence or remaining work |
|---|---|
| QVQ P32 and ordinary W4–W8 | Published draft PR has all-rate GPU checks and real F6/seed7 W2.5/W4/W8 measurements; no gain in those lifecycle runs |
| GPTQ | Real F6/seed7 W4 QKV, config/packed reload and Torch GPU layer checks pass; other rates/backends and complete-model exports pending |
| AWQ GEMM | Real F6/seed7 QKV scale search, clipping, GSQ, packing/reload and Torch GPU checks pass; fixed GSQ has mixed small propagated effects, scale learning retains baseline; other backends and complete-model exports pending |
| AWQ GEMV/GEMV_FAST/LLM-AWQ | Format-specific scalar adapters, CPU packed-objective and native GPU fixture checks pass; real-model scale-search and propagation validation pending |
| AWQ Marlin/BitBLAS | Real GEMM-to-Marlin reload/repack/forward passes 18 cases; BitBLAS unavailable in the current environment; direct-format GSQ requests remain rejected pending packer/lifecycle verification |
| RTN | Real F6/seed7 W4 QKV and Torch GPU reload checks pass with mixed quality effects; remaining formats/backends and complete-model exports pending |
| GPTAQ | Real F6/seed7 block-1 QKV paired inputs, config/packed reload, Torch GPU and F6 propagation pass; both GSQ arms retain baseline; complete-model exports and other rates/backends pending |
| FOEM | Original beta-based initializer followed by optional reconstruction fitting; real block-1 alpha0/beta0.2 QKV, reload/Torch GPU and F6 propagation pass; other coefficients/backends and complete-model exports pending |
| QQQ | Fixed-scale GSQ bound to quantizer, paired-moment collector and verified producer packing; 54 CPU contract checks and four native GSQ reload/graph cases pass; native saturation fixed and profiled on SM80 ([audit](../../artifacts/gsq-qqq/saturation-native/README.md)); real QKV W4A8 grouped/channelwise reconstruction and grouped full-model propagation pass with baseline retained ([report](gsq-qqq-real-layers.md)); complete native exports and broader lifecycle validation pending ([audit](../../research/qqq.md#gsq-adapter-audit-2026-09-09)) |
| ParoQuant | Optional module-scope GSQ config and processor binding; frozen exported rotations, AWQ packing, CPU checks and eight SM80 native reload/graph cases; real QKV baseline/fixed/learned arms pass 288 native checks with identical payloads ([report](gsq-paro-real-layers.md)); clean-input grouped post-fit binding passes CPU lifecycle checks; real layer-0 grouped pilots pass 576 native checks with fixed-GSQ decoder-MSE regressions ([report](gsq-paro-real-group.md)); real layer-1 paired pilots pass 576 native checks with fixed-GSQ MSE regressions ([report](gsq-paro-real-paired.md)); full-model propagation and broader orchestration validation pending ([notes](../../research/paroquant.md)) |
| EXL3 | Backend-owned trellis payloads require their own candidate/decoder and lifecycle binding |
| FP8 | Nonuniform floating code grid and overflow/scale semantics require a distinct adapter |
| bitsandbytes FP4/NF4 | Codebook and nested scale metadata require a distinct adapter |
| GGUF | Audit each supported tensor type; the container is not one quantization grid |
| MXFP4 | E2M1 assignments and microscale representation require their own export-aware adapter |

The implementation is unfinished until those boundaries are resolved and the
compatible paths are verified. The previous QVQ results must not be cited as
validation of these scalar paths.

## Real F6/seed7 scalar experiment

`scripts/validate_gsq_scalar_layers.py` reuses the historical F6 snapshot/data
hash audit and seed-7 token selection: 16 stratified YAQA/NM calibration
documents (3,767 tokens) and 32 locked, disjoint held-out documents (6,367
tokens), capped at 256 tokens per document. It verifies and archives the
executed sources and retains the exact inputs, dense teacher logits, real
projection weights/activations, per-arm configs and packed exports locally.

The full block-0 Q/K/V matrices are 2048x2048, 512x2048 and 512x2048. Both methods
use W4, group size 128, symmetric GPTQ v2 storage, and FP16 fitting weights.
GPTQ uses activation ordering with original-column group ownership and real
dense-model activations, weighted by the F6 source weights (YAQA 1.25, NM 1).
RTN has no calibration-dependent fitting objective. Its recorded activations
are used for evaluation only. All comparisons use 100 GSQ steps, seed 7,
candidate budget 33 (the full 16-code W4 scalar grid), and learning rate 0.1.

Only these three projections are replaced in the full F6 model. The remaining
snapshot projections and endpoint/norm tensors are preserved. Full-model
metrics use FP32 canonical reconstruction against the dense FP32 teacher;
these are not uniform whole-model W4 quantizations or full native-model runs.
Both configs and packed tensors are saved/reloaded. Each arm additionally runs
the public Torch GPU backend on 16 real held-out input rows per full projection,
against its FP32 canonical reconstruction on identical FP16 inputs.

| Method | Arm | KLD | Logit MSE | Top-1 agreement |
|---|---|---:|---:|---:|
| GPTQ | Baseline | 0.110326442 | 0.444875231 | 85.4406% |
| GPTQ | GSQ fixed scales | 0.110326442 | 0.444875231 | 85.4406% |
| GPTQ | GSQ learned scales | 0.110326442 | 0.444875231 | 85.4406% |
| RTN | Baseline | 0.121806210 | 0.458945587 | 85.2050% |
| RTN | GSQ fixed scales | 0.121806210 | 0.458945587 | 85.2050% |
| RTN | GSQ learned scales | 0.117856072 | 0.505603363 | 84.6081% |

GPTQ and fixed-scale RTN retained byte-identical packed tensors. Learned-scale
RTN changed 694,563 Q codes, 194,959 K codes and 194,175 V codes, plus all 49,152
group scales. Zero-points and group indices stayed exact. Its fitting weight
NMSE fell 5.069%, 8.536% and 7.983%, respectively. All arms have exactly
3,293,184 packed tensor bytes across QKV; `.pt` zip file sizes vary slightly
with filename length and are not used to infer tensor storage changes.

Learned-scale RTN's measured full-model KLD is 3.243% lower, but logit MSE is
10.166% higher and top-1 agreement falls 0.5968 percentage points. These are
mixed effects, not an overall quality improvement. Paired document bootstrap
95% intervals also separate these directions: KLD delta [-0.008754, -0.003196],
MSE delta [0.044293, 0.066198], top-1 fraction delta [-0.009592, -0.002631]. Those
intervals average documents; the table above weights tokens. Top-5/top-10
agreement also declines. KLD measures distributions, logit MSE measures raw
logits, and top-N agreement measures ranked-token overlap with the dense
teacher; none is task accuracy. A lower fitting loss does not guarantee
improvement in each of these different downstream measurements.

All 18 localized Torch GPU cases pass both existing gates independently
(mean absolute error <= 0.002 and maximum <= 0.046875). The worst mean is
0.0002450 and worst maximum is 0.0087395. Runs used physical GPU 0,
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, PCI DE:00.0, PG506-230/SM80 with
124 SMs and 96 GiB; three idle preflights passed per run and leases were
released. This is correctness evidence for Torch GPU execution, not a speed
claim or qualification of ExLlama/Marlin/BitBLAS or other native backends.

Raw evidence: [GPTQ report](../../artifacts/gsq-scalar/gptq-w4-seed7-v2/report.json),
[RTN report](../../artifacts/gsq-scalar/rtn-w4-seed7-v2/report.json), and
[independent packed-tensor audit](../../artifacts/gsq-scalar/validation/report.json).
The initial GPTQ runner attempt stopped before baseline quantization because
`length_aware` was passed at the wrong config level; its failed report and
capture artifacts remain local. The successful runs use `hessian.length_aware`.
No failed run is counted as a quality result, and these selected-module
artifacts are not published under the complete-model snapshot directory.

## Additional AWQ format audit

The GEMM adapter cannot be selected solely from `METHOD.AWQ`. Source inspection
at `dfdd1564f` identifies these distinct contracts:

- `gemv_awq.py::pack` computes `zero * source_scale` before casting the scale
  table to FP16, then reconstructs codes with that source offset and the stored
  FP16 denominator. GEMM instead computes its offset from the already-stored
  scale table. GEMV stores integer zeros and packs codes along input columns.
- `gemv_fast_awq.py::pack` uses the GEMV inverse arithmetic, a separate four-row
  interleaved code layout, and a stored FP16 additive offset
  `-round_fp16(stored_scale * zero)`. Its canonical decoded operator is
  `stored_scale * code + stored_offset`; for asymmetric zeros it can differ
  from `stored_scale * (code - zero)`. The LLM-AWQ subclass shares this packer.
- At that revision, both GEMV packers converted rounded weights to integers
  without saturation. The current integration shares their source-offset /
  stored-FP16-scale inverse, clamps codes before integer conversion and rejects
  invalid metadata, including FP16 scale underflow. Normal in-range codes remain
  exact against the original per-column reference. Out-of-range codes now
  saturate instead of corrupting neighboring packed bits.

These are identified implementation requirements, not a claim that GEMV or
GEMV_FAST is mathematically incompatible with GSQ. In particular, learned
scales must be scored after both the returned source dtype and the packer's
storage dtype conversions, including the separately stored fast-path offset.

The new adapters implement those requirements. `AWQConfig.gsq` and
`RTNConfig.gsq` accept GEMM, GEMV, GEMV_FAST and LLM-AWQ exports at W4. GEMV-family
group sizes are -1, 32, 64 and 128. Group size 16 remains rejected for enabled
GSQ because the existing GEMV storage-width helper does not implement it.
Marlin and BitBLAS AWQ still require their own adapters. These exclusions are
open implementation work, not blanket mathematical incompatibility claims.

GEMV and GEMV_FAST also now override the inherited GEMM-layout dequantizer.
Their reference decoders return [in,out] weights with integer zeros for GEMV
and stored additive offsets for GEMV_FAST/LLM-AWQ. Native FP16 reconstruction
and canonical FP32 reconstruction are checked separately. This makes
post-pack error measurement use the actual saved layout.

The native fixture uses a 128x256 projection, group size 128, asymmetric W4,
GSQ scale learning, and exact packed save/reload. All six native cases pass
the unchanged mean <= 0.002 and max <= 0.046875 gates:

| Backend | Input tokens | Mean absolute error | Maximum absolute error |
|---|---:|---:|---:|
| GEMV | 1 | 0.000083676 | 0.000706434 |
| GEMV_FAST | 1 | 0.000187241 | 0.000849247 |
| LLM-AWQ | 1 | 0.000187241 | 0.000849247 |
| GEMV | 16 | 0.000212935 | 0.001420975 |
| GEMV_FAST | 16 | 0.000220199 | 0.001721859 |
| LLM-AWQ | 16 | 0.000220199 | 0.001721859 |

These are synthetic native-kernel correctness fixtures, not real-model quality
measurements. They ran on the same exclusively leased physical GPU 0 / SM80
device described above. There is no new kernel or inference-format payload.
The CPU suite additionally checks all three source dtypes, all four supported
group sizes, source/storage casting of improving hard checkpoints, both public
AWQ and RTN hooks, and malformed/tiny-scale/endpoint cases.

Final CPU checks pass 141 cases with six CUDA-only skips. The six native cases
then pass separately after pack/save/reload. Coverage is 100% of the two measured
helpers (205 statements and 92 branches), not whole-lifecycle coverage. Raw
[coverage](../../artifacts/gsq-scalar/gemv-validation/coverage.json) and
[native log](../../artifacts/gsq-scalar/gemv-validation/native.log) preserve this scope.

The real-input regression audit also exposed a pre-existing local-grid optimizer
issue: FP16 AWQ codes left logits and Adam state in FP16, where the optimizer
epsilon underflows. Packing retains its original arithmetic; candidate codes
are now converted to FP32 before optimization. The 16-case audit uses an 8x64
real block-0 K slice and 128 actual calibration rows. Twelve GPTQ/full-grid AWQ
histories and output tensor sets exactly match commit `22d98963a`; the four
local-grid AWQ cases intentionally change optimizer precision. Both old FP16
cases failed with a non-finite objective; all four corrected cases finish with
finite histories and retain the best hard checkpoint including baseline.
BF16 local-grid trajectories also change. This is correctness evidence, not a
new model-quality claim. See the [audit report](../../artifacts/gsq-scalar/gemv-validation/prior-path-regression-v3.json)
and `scripts/validate_gsq_scalar_compatibility.py` for reproduction.

Real AWQ scale-search, scaled-normalization propagation and complete-model
export validation remain open, along with the remaining adapters in the inventory.

### Real AWQ calibration preparation

`scripts/prepare_gsq_awq_calibration.py` now runs the actual AWQ QKV group scale
search on Llama 3.2 1B block 0, using the audited F6/seed7 selection: 16 training
documents and 3,767 valid tokens. It uses W4 asymmetric groups of 128, FP16
weights/activations, the default 20-ratio search, and uniform token weighting.
The YAQA source weights are not applied to this AWQ calibration. Held-out
documents are captured separately and never enter scale selection.

Documents use separate causal blocks and reset rotary positions. Joined versus
separate-document attention has mean absolute drift 0.000008168 and maximum
0.000366211. After folding the selected scales into RMSNorm and QKV, a fresh
embedding/RMSNorm/attention forward has mean drift 0.000010817 and maximum
0.000488281 versus the original attention. Both checks pass the existing
0.002 / 0.046875 gates independently on physical GPU 0 (the leased SM80 above).
Cached-feature division and fresh scaled RMSNorm differ slightly; both are
preserved so subsequent fitting and deployment checks use their correct domains.

Scale search restores every original block tensor before applying its winner.
Selected channel scales range from 0.167236 to 5.980469; the AWQ attention search
loss is 0.000001504815. This is calibration preparation, not a GSQ improvement or
a full AWQ model validation. The full original/scaled block, calibration and
held-out features, exact tokens, dense hashes and executed source are preserved
locally under `artifacts/gsq-scalar/awq-calibration-seed7-v2/`; its compact
[report](../../artifacts/gsq-scalar/awq-calibration-seed7-v2/report.json) is tracked.
The next step is matched baseline/fixed-scale/learned-scale GSQ with actual
clipping and packing, followed by F6 full-model propagation with the scaled
RMSNorm retained. Complete-model export and remaining method adapters stay open.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4 MAX_JOBS=4 \
python -m gpu_allocator.cli run -n 1 --style uuid -- \
python -m scripts.prepare_gsq_awq_calibration \
  --source artifacts/gsq-scalar/gptq-w4-seed7-v2 \
  --output artifacts/gsq-scalar/awq-calibration-seed7-v2
```

### Real AWQ GEMM GSQ comparison

The subsequent `awq-w4-seed7-v2` run completes all three full block-0 QKV arms
using that scale-search fixture. Actual AWQ clipping is applied to V; Q/K retain
AWQ's clipping exclusions. GSQ receives the scaled preclip FP16 teacher and
uniform scaled calibration activations, with seed 7, 100 steps and candidate
budget 33 (all 16 W4 codes). Hard checkpoints are scored after actual AWQ inverse
arithmetic and stored scales. Each arm saves/reloads its config and GEMM payload.
The original F6's other projections remain unchanged; its FP32 canonical model
retains the AWQ-scaled block-0 RMSNorm for 32 locked documents / 6,367 tokens.
This is not a full AWQ model export or full native-kernel model run.

| Arm | Final KLD | Logit MSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.117415165 | 0.452552406 | 85.738967% | 82.425004% | 82.479975% |
| Fixed-scale GSQ | 0.117606072 | 0.453133418 | 85.738967% | 82.484687% | 82.384168% |
| Learned-scale GSQ | 0.117415165 | 0.452552406 | 85.738967% | 82.425004% | 82.479975% |

Fixed-scale GSQ changes 326 V codes and no Q/K codes. V calibration NMSE falls
from 0.007332732 to 0.007268057, but final KLD rises 0.162592% and MSE rises
0.128386%. Paired document-bootstrap 95% intervals for their mean deltas are
[0.000147024, 0.000261665] and [0.000405335, 0.000863578]: small clear negatives
on this locked sample. Top-1 is noise-consistent (interval includes zero), Top-5
is a small clear positive, and Top-10 a small clear negative. These are mixed
effects, not an overall recovery claim. Learned-scale GSQ retains every baseline
tensor and every per-document metric exactly. Defaults remain disabled.

All nine reloaded AWQ Torch GPU cases pass on 16 real held-out activation rows
per projection. Worst mean/max drift against the matching FP32 operator is
0.000242425 / 0.006502151, within 0.002 / 0.046875 independently. All arms have
3,268,608 QKV tensor bytes and identical zero points; fixed GSQ also preserves
all scales. Independent nibble/order decoding matches the real Q baseline.
The first attempt stopped before GSQ at strict reload because the experiment
omitted AWQ buffer registration; its log is preserved locally. The corrected
run uses `register_buffers=True` and completed under a new exclusive lease.

Evidence: [raw report](../../artifacts/gsq-scalar/awq-w4-seed7-v2/report.json),
[payload audit](../../artifacts/gsq-scalar/awq-w4-seed7-v2/payload-audit.json).
Exact tokens, teacher logits, payloads, calibration fixture and executed source
remain in the local experiment directories. Reproduce with
`scripts.validate_gsq_scalar_layers --method awq --awq-calibration
artifacts/gsq-scalar/awq-calibration-seed7-v2`, first with `--prepare` and a fresh
`--output`, then under the GPU allocator. Audit with
`scripts.analyze_gsq_awq_results RUN --output NEW_JSON`.

### GPTAQ asymmetric lifecycle hook

`GPTQConfig(gptaq={"alpha": 0.5}, gsq=GSQConfig(enabled=True))` now routes
GPTAQ through the optional post-quantizer GSQ wrapper. It clones original-column
H and native-minus-current cross moments before ordering, damping or release.
The original GPTAQ quantizer remains the initializer; GSQ uses its final group
indices and the asymmetric quadratic described in `research/gsq-rco.md`.
Its diagnostic name is `asymmetric_quadratic_without_constant`: negative scores
are possible because the candidate-independent native residual is omitted.
Missing or consumed paired statistics reject enabled refinement. FOEM remains
unsupported until its separate first-order contract is implemented.

Seventeen focused asymmetric tests pass, including activation order on/off,
config round-trip, explicit native-output reconstruction, and byte-identical
disabled/unmatched-module bypass against the original quantizer implementation.
Config checks pass 145 cases with 32 skips. The broader CPU regression command
(`tests/test_gsq_asymmetric.py tests/test_gsq_scalar.py tests/test_gptaq.py -k
'not TestQwen'`) records 10,066 passed, 1,002 skipped, one deselected and **11
failed** Hessian-reference cases. All 11 seeds fail identically with the pinned
pre-change GPTAQ class at `11993f31d`; their H and cross moments match exactly
between versions. The audit uses the unchanged shared accumulation base and
does not invoke quantization. No tolerance was relaxed. See
[the reproduction audit](../../artifacts/gsq-scalar/gptaq-hook/regression-audit.json)
and `scripts/audit_gptaq_gsq_regressions.py`. The three additional bypass cases
ran in the separate 17-case focused check after that broader run started.
These checks do not establish real-model export, GPU parity or propagated
quality for GPTAQ GSQ; those remain open.

### Real GPTAQ block-1 verification

The subsequent `gptaq-block1-w4-seed7` run uses full block-1 Q/K/V, where the
preceding F6 quantized layer produces actual upstream input error. Dense-native
and canonical-F6 inputs are captured for identical documents and token positions.
Their unweighted relative squared difference is 0.008006822 on the 3,767 training
tokens and 0.009709677 on the 6,367 locked held-out tokens. The latter captures
are used only for validation. Original seed-7 source weighting applies equally
to both sides of the training pair.

The experiment uses GPTAQ alpha 0.5, W4/group128/symmetric GPTQ v2 storage,
activation ordering on, and GSQ seed 7 / 100 steps / candidate budget 33. Every
packed objective is checked against the explicit activation-pair quadratic,
including its omitted constant. Config and packed state round-trip exactly.
Both fixed-scale and learned-scale GSQ retain every baseline tensor. All three
arms have 3,293,184 QKV tensor bytes, identical per-document metrics, final KLD
0.107260829, logit MSE 0.445125877, Top-1 85.597613%, Top-5 83.072090%, and Top-10
82.622899%. Paired delta intervals are zero because the outputs are identical.
There is no GSQ quality improvement in this run.

Nine actual reloaded Torch GPU projection checks on real held-out inputs pass
independently; worst mean/max absolute drift is 0.000294336 / 0.004081726 against
the FP32 canonical operator, below 0.002 / 0.046875. The leased physical GPU 0 is
the same SM80 device documented above. Final-model evaluation uses canonical
FP32 F6 operators with only block-1 QKV replaced; this is not a full GPTAQ model
export or a full native-kernel model run. Other rates/backends and complete-model
exports remain pending.

Evidence: [run report](../../artifacts/gsq-scalar/gptaq-block1-w4-seed7/report.json)
and [payload/input-pair audit](../../artifacts/gsq-scalar/gptaq-block1-w4-seed7/payload-audit.json).
Exact paired activations, native teacher logits, weights, configs and executed
source remain locally under the run directory. The shared
`scripts/gsq_f6_reference.py` extracts the prior runner's canonical F6 installation
unchanged so capture and evaluation use the same operator definitions.

```bash
python -m scripts.validate_gsq_scalar_layers --prepare --method gptaq --layer 1 \
  --output artifacts/gsq-scalar/gptaq-block1-w4-seed7
python -m gpu_allocator.cli run -n 1 --style uuid -- \
  python -m scripts.validate_gsq_scalar_layers --method gptaq --layer 1 \
  --output artifacts/gsq-scalar/gptaq-block1-w4-seed7
python -m scripts.analyze_gsq_scalar_results artifacts/gsq-scalar/gptaq-block1-w4-seed7 \
  --output artifacts/gsq-scalar/gptaq-block1-w4-seed7/payload-audit.json
```

### FOEM initialization and optional GSQ

FOEM now retains its existing beta-dependent quantizer updates as the initializer,
then uses the scalar GSQ wrapper. Beta is not a trainable quantization scale or
a second final reconstruction target. At alpha=0, GSQ fits the original teacher
against propagated calibration inputs using FOEM's directly accumulated H.
At nonzero alpha it also uses the original-column native/current cross moment.
Both are copied before the initializer consumes them. Diagnostics retain the
initializer name and beta. The disabled path matches the original FOEM quantizer.

Integration exposed and fixed a pre-existing config issue: `_update_meta_payload`
omitted FOEM coefficients whenever `gptaq` was absent. FOEM-only configs now save
and restore alpha, beta and device independently of GPTAQ. Nine dedicated JSON
round-trip cases cover GSQ absent, disabled and enabled. Another 231 focused
scalar/asymmetric/config cases pass with 32 skips, including alpha 0/.5, beta
0/.2 and activation ordering on/off. These are correctness tests, not quality
evidence for every coefficient choice.

The real `foem-block1-w4-seed7` run uses the same full block-1 QKV, source audit,
16 training / 32 locked documents, F6 propagated inputs and protocol as GPTAQ
above, with default FOEM beta 0.2 and alpha 0. W4/group128/symmetric GPTQ v2,
activation ordering on, GSQ seed7/100 steps/33-candidate budget. Both GSQ arms
retain all baseline tensors and per-document metrics exactly: KLD 0.109141164,
logit MSE 0.440341529, Top-1 85.220669%, Top-5 83.059526%, Top-10 82.640176%.
All arms use 3,293,184 QKV tensor bytes. There is no GSQ gain in this run.

All nine packed/reloaded Torch GPU projection checks pass on the same leased
physical GPU 0 / SM80: worst mean/max drift is 0.000294089 / 0.004364967 versus
the matching FP32 operator, within the independent 0.002 / 0.046875 limits.
The explicit activation objective matches the exported payload. Final-model
propagation uses canonical F6 FP32 operators with only block-1 QKV replaced;
complete FOEM-model export and full native-model execution remain unverified.
Real nonzero-alpha FOEM and other coefficients/backends remain outside this run.

Evidence: [report](../../artifacts/gsq-scalar/foem-block1-w4-seed7/report.json),
[payload audit](../../artifacts/gsq-scalar/foem-block1-w4-seed7/payload-audit.json).
Use the preceding preparation/run/audit commands with `--method foem --layer 1
--asymmetric-alpha 0 --foem-beta 0.2` and a fresh output directory.

### AWQ runtime/export distinction

The AWQ processor's default format selector explicitly rejects MARLIN and
BITBLAS; a model-supplied `qlinear_kernel` can bypass that selector, so the enum
table alone does not prove or disprove a complete direct-export path. Keep direct
export qualification separate from loading a GEMM checkpoint into another runtime.
Marlin's `post_init` repacks GEMM words and permutes scales/zero points. BitBLAS
has both a GEMM-repack path and a direct scalar packer; the latter reconstructs
codes in source scale precision before storing scales in its runtime dtype.
That distinction needs its own cast-boundary checks and must not be assumed
identical to the GEMM packer for every source/runtime dtype pair.

`scripts/validate_gsq_awq_marlin.py` checks the already-saved real AWQ GEMM
baseline and GSQ payloads through actual Marlin reload/repacking and forward
against their canonical FP32 operators on real held-out activations. BitBLAS
is not installed in `/root/venv-py3.14t`; no BitBLAS execution result is claimed.

The real Marlin run completes 18 cases: baseline/fixed-scale/learned-scale arms
for full Q/K/V, with one and 16 real held-out activation rows. Every saved GEMM
payload reloads exactly before actual Marlin repacking and native forward.
Worst mean/max drift against its FP32 canonical operator is 0.000606514 /
0.014185905, passing the existing 0.002 / 0.046875 gates independently. This
includes the fixed-scale V payload's 326 changed codes. The leased physical GPU
0 / SM80 and exact Python/CUDA source hashes are recorded in the
[runtime report](../../artifacts/gsq-scalar/awq-w4-seed7-v2/marlin-runtime-v2.json).
This is runtime-conversion correctness, not direct Marlin export, a full-model
load/generation check, a speed claim, or additional held-out quality evidence.

The first attempt completed JIT compilation but stopped in `marlin_padded_nk`
before native forward: `math.lcm` lacked its module import. Adding `import math`
fixes the existing runtime failure; no kernel source or numerical operation was
changed. The failed attempt/log remain local. The corrected run used a fresh
report and lease. Reproduce with `scripts.validate_gsq_awq_marlin --run
artifacts/gsq-scalar/awq-w4-seed7-v2 --output NEW_JSON` under the GPU allocator.

The initial CPU command `pytest tests/test_marlin_jit.py -k 'pad or align'`
recorded 10 passed, three failed and 44 deselected; the
[original failure output](../../artifacts/gsq-scalar/awq-w4-seed7-v2/marlin-padding-cpu.txt)
is preserved. The follow-up fixes the missing `BACKEND` test import and corrects
GPTQ fixtures to respect the existing 32-value checkpoint alignment. A valid
224-column fixture exercises padding to 256; a 200-column fixture explicitly
checks rejection. Group size 128 with 288 input columns exercises the existing
activation-order restriction on K padding. AWQ's explicit-backend padding rule
remains covered separately. Production selectors and error gates are unchanged.

The full CPU command `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4
/root/venv-py3.14t/bin/python -m pytest -q tests/test_marlin_jit.py` now reports
**42 passed, 15 skipped** in 12.39 seconds. The skipped cases require CUDA; this
CPU result does not replace the 18 real GPU runtime cases above. See the
[full corrected CPU output](../../artifacts/gsq-scalar/awq-w4-seed7-v2/marlin-cpu-corrected.txt).
