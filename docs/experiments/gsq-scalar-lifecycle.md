# Scalar GSQ integration work

This extends the optional control beyond the verified [QVQ tile adapters](gsq-qvq-lifecycle.md).
The scalar paths remain experimental: CPU fitting and packing checks are not
evidence of a real-model quality improvement or complete backend support.

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
- AWQ GEMM snapshots scaled dense weights before clipping. GSQ runs after AWQ
  rounding and optional adjacent refinement, using the corresponding scaled
  activations. It runs before EoRA residual capture and packing. A fallback
  without activations explicitly uses weight MSE.
- RTN uses weight reconstruction without manufacturing calibration data. Its
  current adapter is limited to GPTQ exports.

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
accelerator support. Ruff and `git diff --check` passed for the changed files.
Scalar line/branch coverage, real-model quality and native runtime gates remain
pending. No scalar quality gain is claimed from these fixtures.

The user-requested goal includes every compatible method. This inventory keeps
that scope open; a missing adapter does not establish mathematical incompatibility.

| Method/path | Current evidence or remaining work |
|---|---|
| QVQ P32 and ordinary W4–W8 | Published draft PR has all-rate GPU checks and real F6/seed7 W2.5/W4/W8 measurements; no gain in those lifecycle runs |
| GPTQ | Scalar lifecycle and original-column objective fixtures implemented; real calibration/model, save/reload/native backend matrix pending |
| AWQ GEMM | Preclip teacher hook and actual CPU packer checks implemented; full scale-search lifecycle, real-model and native backend validation pending |
| AWQ GEMV/GEMV_FAST/Marlin/BitBLAS | Audit separate packers, stored scales and code conventions; currently rejected by enabled AWQ GSQ config |
| RTN | Scalar GPTQ export hook implemented; remaining export adapters and real-model/native checks pending |
| GPTAQ/FOEM | Preserve asymmetric/first-order targets rather than substitute ordinary GPTQ reconstruction; currently rejected |
| QQQ | Audit W4A8 deployed activation and multi-scale contract before reusing scalar assignments |
| ParoQuant | Fit in the learned rotation basis, preserve exported transforms and quantizer metadata |
| EXL3 | Backend-owned trellis payloads require their own candidate/decoder and lifecycle binding |
| FP8 | Nonuniform floating code grid and overflow/scale semantics require a distinct adapter |
| bitsandbytes FP4/NF4 | Codebook and nested scale metadata require a distinct adapter |
| GGUF | Audit each supported tensor type; the container is not one quantization grid |
| MXFP4 | E2M1 assignments and microscale representation require their own export-aware adapter |

The implementation is unfinished until those boundaries are resolved and the
compatible paths are verified. The previous QVQ results must not be cited as
validation of these scalar paths.
