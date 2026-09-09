# Optional FP8 GSQ: implementation and validation

The local implementation adds `FP8Config.gsq`, defaulting to `None`, with
module-regex selection and per-module dynamic overrides. The weight-only
processor runs fitting after ordinary FP8 packing and before releasing the
original dense teacher. It writes the selected FP8 bytes directly, retaining
the stored inverse scales, and saves diagnostics in module state and the
processor log. This lifecycle uses weight reconstruction, not calibration
activations; the low-level fitter separately accepts input activations.

The adapter enumerates finite storage encodings for E4M3/E5M2 FN/FNUZ grids.
Candidate zero retains the exact baseline bytes, including signed zero.
Other candidates alternate adjacent finite grid values. Tensor, row and block
inverse scales decode those candidates into the reconstruction objective.
Adam fits Gumbel-Softmax logits; only a strictly improving hard checkpoint can
replace the baseline. This remains GSQ-inspired fitting, not a reproduction
of the paper's complete staged optimization. Scale learning is explicitly
rejected; E8M0 is not an enabled GSQ weight format.

The CPU runtime previously interpreted an inverse-scale table entirely below
or equal to one as multiplicative scales through the generic decoder's legacy
heuristic. The explicit inverse-scale FP8 module now bypasses that heuristic
for this range. Generic checkpoint decoding retains its existing behavior.

## CPU evidence

On 2026-09-09, with CUDA hidden, the command below passed **83 tests** in
3.96 seconds using `/root/venv-py3.14t/bin/python`:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest -q \
  tests/test_gsq_fp8.py tests/test_fp8.py tests/test_weight_only_config.py
```

The checks cover finite payload enumeration, scale geometry, malformed and
zero-energy inputs, normalization overflow, private RNG, config serialization,
dynamic overrides and processor finalization. A controlled non-optimal packed
initializer forces the real fitter to select changed bytes; those bytes survive
on-disk state-dict save, strict reload and CPU forward. Disabled and unmatched
module selections retain the same controlled baseline. Module creation and the
outer pack dispatcher are stubbed in this fixture; the actual FP8 packer,
fitter, finalizer, state serialization, decoder and forward run.

These synthetic fixtures establish implementation correctness only. They are
not real-model quality evidence or complete-model export validation. GPU eager
and graph execution, real Llama weights with disjoint calibration/held-out data,
activation-aware lifecycle integration, scale learning, and full-model native
exports remain pending. No accuracy recovery or default promotion is claimed.

## Real Llama QKV, seed 7

The subsequent `real-qkv-seed7-v2` run completed on physical GPU 0,
PG506-230 SM80 (`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`). It used real
Llama 3.2 1B block-0 Q/K/V weights and the verified F6 campaign selection:
16 calibration documents (3767 tokens), 32 disjoint held-out documents
(6367 tokens), YAQA calibration weight 1.25 and NM weight 1.0.

| Projection | Baseline held-out MSE | Calibrated GSQ MSE | Relative change | Changed codes |
|---|---:|---:|---:|---:|
| Q | 0.000169367950 | 0.000151717894 | -10.4211% | 1882 |
| K | 0.000270709680 | 0.000247939053 | -8.4115% | 2192 |
| V | 0.000003895171 | 0.000003844744 | -1.2946% | 112 |

Weight-only GSQ retained the baseline in all three projections. The calibrated
low-level fitter used 100 steps, three adjacent candidates, seed 7 and frozen
row inverse scales. These promising local results motivate activation-aware
lifecycle integration and propagated evaluation; they do not establish final
model recovery. No KL, Top-K, task accuracy or paired uncertainty was measured.

All 288 FP32 runtime checks after save/reload passed with zero drift against
FP32 decoded-weight matmul. SM80 uses the CPU FP8 decoder plus CUDA matmul;
this is not native FP8 tensor-core or CUDA-graph validation. The original runner
retained per-document runtime checks but only aggregate quality MSE, so a future
run must preserve per-document quality errors for uncertainty analysis.

See the [run record](../../artifacts/gsq-fp8/real-qkv-seed7-v2/model_run.md),
[raw report](../../artifacts/gsq-fp8/real-qkv-seed7-v2/report.json), and
[manifest](../../artifacts/gsq-fp8/real-qkv-seed7-v2/manifest.json).

## F6 propagation diagnostic

A subsequent eager FP32 run substituted each saved FP8 QKV arm into canonical
F6 and compared full-model logits against the same dense teacher on all 32
held-out documents. Other operators retained canonical F6 decoding.

| Arm | KL | Logit MSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| F6 control | 0.1081267801 | 0.4404727656 | 85.48767% | 83.03125% | 82.92602% |
| Ordinary FP8 QKV | 0.1089368030 | 0.4413371391 | 85.40914% | 83.03440% | 82.89461% |
| Weight-only GSQ | 0.1089368030 | 0.4413371391 | 85.40914% | 83.03440% | 82.89461% |
| Calibrated GSQ | 0.1081454614 | 0.4410261719 | 85.42485% | 83.05638% | 82.83336% |

Versus ordinary FP8, calibrated GSQ's KL delta is -0.0007913416, with paired
95% document-bootstrap CI [-0.0009525260, -0.0006702904]: a clear positive
within this sample. Top-10 falls 0.0612533 percentage points, CI
[-0.0904624, -0.0286192] pp: a clear negative. MSE, Top-1 and Top-5 changes
are noise-consistent. Bootstrap uses 10000 resamples, seed 7, token-weighted
means. This mixed result merits expanded disjoint validation; it does not
justify default promotion. Weight-only GSQ preserves every baseline logit hash.

All per-document logits, metrics and execution logs are retained in
[the propagation record](../../artifacts/gsq-fp8/propagation-seed7-v1/evaluation.md).
This diagnostic does not establish native FP8 hardware, graph replay, paged
production inference, full-model export or task accuracy.
