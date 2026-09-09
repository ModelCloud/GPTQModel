# Optional GSQ in the QVQ quantization lifecycle

`QVQConfig.gsq` is a separate optional stage alongside `yaqa`. Both `gsq=None`
and `GSQConfig(enabled=False)` disable refinement. Older configs load disabled.
The control is serialized in quantize_config.json, not in inference tensors.

```python
from gptqmodel.quantization import GSQConfig, QVQConfig

config = QVQConfig(
    bits=2.5,
    format="qvq_v2b2_p32",
    rounding="yaqa",
    gsq=GSQConfig(
        enabled=True,
        seed=7,
        steps=100,
        candidates=33,
        modules=(r"self_attn\.[qkv]_proj$",),
    ),
)
```

This is GSQ-inspired fixed-scale, whole-tile candidate optimization, not a
reproduction of scalar GSQ. Initialization, scales and banks remain YAQA-owned.
The optional solve runs before packing; any rank-8 correction is fitted afterward.
It updates states, decoded inner weight, reconstructed weight and diagnostics
together. Normal packing and the existing exact P32 decode check remain authoritative.
Unsupported combinations (spectral refinements, replay, alignment, activation
quantization, scale search, folded/deployed-target transforms) reject explicitly.
The candidate-bank limit is 1 GiB of decoded FP32 weights by default; peak memory
also includes candidate copies, gradients, metrics and the surrounding quantizer.

## Objective and scope

Let `E = W_candidate - W_teacher` in the normalized inner basis. With prepared
YAQA matrices `H=L_H L_H^T` and `G=L_G L_G^T`, the fitter minimizes
`||L_H^T E L_G||_F^2 / ||L_H^T W_teacher L_G||_F^2`. This equals normalized
`tr(G E^T H E)`. Both matrices include YAQA's existing damping/stabilization;
the fitter adds none. This differs from the earlier raw-activation MSE screen.
Candidate zero is always eligible. The best hard calibration checkpoint is
selected without looking at held-out data. This local guard does not certify
improved final-model quality.

## Real W2.5 P32 lifecycle run

The original F6 seed-7 Llama 3.2 1B snapshot, dense source, original YAQA/NM
calibration corpus, locked evaluation data and protocol are the same as the
[prior W2.5 experiment](gsq-p32-f6-seed7-w25-qkv.md). Full block-0 Q/K/V are
freshly quantized, with and without the lifecycle GSQ stage. Other F6 modules
remain unchanged. The comparison uses 16 stratified calibration documents
(3767 tokens) and 32 locked held-out documents (6367 tokens), not all 10178
historical YAQA calibration rows. All three full projections use W2.5, 33
candidates, 100 steps and seed 7. This is not a uniform-W2.5 full-model quantization.

| Arm | Final KLD | Logit MSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| YAQA | 0.110186534 | 0.446268583 | 86.0374% | 83.0407% | 82.7815% |
| YAQA + lifecycle GSQ | 0.110186534 | 0.446268583 | 86.0374% | 83.0407% | 82.7815% |

The hard guard retained every baseline tile. All per-document metrics and paired
deltas are identical, so this establishes no quality improvement. It does not
negate the earlier activation-MSE experiment's small propagated regression:
these are different fitting objectives. Keep GSQ disabled by default.

[Raw report](../../artifacts/gsq-p32/lifecycle-w25-seed7/report.json) records
full metrics, source hashes, factors, configuration and GPU identity. The local
experiment directory retains exact token selections, Fisher matrices, complete
projection exports, dense teacher logits and executed source copies. These are
partial-model artifacts, not a newly published model snapshot.

Execution: physical GPU 0, UUID `GPU-737e2423-874a-23a4-1126-dfbe3e77c294`,
NVIDIA PG506-230 (SM80), under an exclusive allocator lease and three idle
samples. Full-model evaluation uses the FP32 canonical reference. Separate
GPU tests exercise the public CUDA backend across W1–W3.5 and exact serialized
reload, using the existing mean <= 2e-3 / max <= 0.046875 localized FP16 gates.
The first test assumed FP32 CUDA compute; it was corrected to the backend's
actual FP16 contract, not used to widen those existing gates.

Validation at this stage: 36 config/math tests, nine GPU lifecycle tests and
one CPU processor test pass. The processor test collects real backward Fisher
factors in a small toy causal model and exercises fitting, installation and
save/reload; it is a correctness fixture, not model-quality evidence.

```bash
PYTHONPATH=. /root/venv-py3.14t/bin/python -m scripts.validate_qvq_gsq_layers \
  --prepare --gsq-lifecycle --target-bits 2.5 --output artifacts/gsq-p32/lifecycle-w25-seed7
PYTHONPATH=. MAX_JOBS=4 /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- \
  /root/venv-py3.14t/bin/python -u -m scripts.validate_qvq_gsq_layers \
  --gsq-lifecycle --target-bits 2.5 --output artifacts/gsq-p32/lifecycle-w25-seed7
```

Use a new output directory for each run. `--gsq` remains the older activation-MSE
experiment; `--gsq-lifecycle` tests the public quantizer's Fisher objective.

## GPTQ, AWQ and weight-only quantization

The [author implementation](https://github.com/IST-DASLab/GSQ/tree/03fc16484c369e3127225615d5e03e8d3a6043e3)
supports GPTQ (default) or RTN initialization, followed by scalar GSQ. It does
not list an AWQ initializer. A scalar fitter could follow AWQ rescaling, but
that is a proposed combination: preserve its compensating transformations,
group scales, zero-point convention and export layout, then measure separately.
The QVQ tile adapter does not support GPTQ/AWQ scalar codes.

GSQ is weight-only: calibration activations inform optimization, but GSQ itself
does not quantize inference activations. This terminology does not mean
calibration-free. It does not imply support in this repository's separately
named `WeightOnlyConfig` dispatcher. No GPTQ/AWQ GSQ config option is added here.
