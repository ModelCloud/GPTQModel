# Real QQQ W4A8 / GSQ projection validation

The complete block-0 Q/K/V projections from Llama 3.2 1B Instruct were
quantized through QQQ W4/group128 with GSQ disabled and fixed-scale GSQ
enabled (100 steps, seed 7). GSQ retained the baseline in every projection.
All packed tensors, not just nibble assignments, are exactly equal between arms.
No recovery improvement is established.

| Projection | Held-out MSE, both arms | Changed codes |
|---|---:|---:|
| Q | 0.0019364688279 | 0 |
| K | 0.0036380466614 | 0 |
| V | 0.0001168733283 | 0 |

The experiment reuses the audited F6/seed7 scalar experiment's saved real
activations: 16 calibration documents / 3767 tokens and 32 held-out documents /
6367 tokens. Calibration uses YAQA/NM source weights 1.25/1.0 through the
existing square-root row scaling convention. This scaling occurs before QQQ
activation quantization; results are specific to that convention. Held-out
inputs are unweighted. Preparation checks exact weights against the original
dense safetensors, token-row dimensions, finite activations, calibration-source
hashes and token-selection hash, then binds all inputs and implementation files.

Both arms use the actual QQQ quantizer, config serialization/reload, producer
packer, strict native state reload and W4A8 native forward. Native outputs are
compared against QQQTorchLinear on identical packed tensors and FP16 inputs,
including activation quantization. All 192 cases pass the independent mean
<=0.002 and max <=0.046875 gates. Worst mean drift is 7.9473e-8 and worst maximum
drift is 0.00390625. These are localized kernel checks, not propagated logits.

The calibration objective is independently recomputed from the decoded packed
INT8 grid and explicit activation residuals, without the optimizer's Hessian
factor. It matches the GSQ diagnostics within max(1e-8, abs(score)*1e-4).
The maximum relative difference is approximately 1.1e-5. The objective omits a
constant asymmetric residual term; held-out MSE includes the full residual and
native output casting against the original FP32 projection teacher.

Execution used an exclusive lease on physical GPU 0 / PCI DE:00.0, SM80,
124 SMs, Torch 2.15.0.dev20260817+cu130. Three idle preflight samples passed.
The run completed and released its lease. An earlier attempt failed before
quantization on an incorrect config import; its log is preserved separately.

Reproduction from the repository root:

```sh
python -m scripts.validate_gsq_qqq_layers --prepare \
  --inputs artifacts/gsq-scalar/gptq-w4-seed7-v2 --output NEW_OUTPUT
python -m gpu_allocator.cli run -n 1 --style uuid -- \
  python -m scripts.validate_gsq_qqq_layers \
  --inputs artifacts/gsq-scalar/gptq-w4-seed7-v2 --output NEW_OUTPUT
```

[Report and provenance](../../artifacts/gsq-qqq/real-qkv-seed7-v3/report.json),
[independent payload audit](../../artifacts/gsq-qqq/real-qkv-seed7-v3/payload-audit.json),
and the executed runner/native logs are retained in that experiment directory.
Large packed tensors remain workspace artifacts. This selected-projection run
is not a complete model snapshot and is not published under the model snapshot
root. Defaults remain disabled and the PR remains draft.

## Final-logit propagation with native QQQ activation quantization

A separate run regenerated dense FP32 teacher logits, installed the existing
F6 snapshot through its canonical FP32 QVQ reference operators, then evaluated
the native QQQ Q/K/V payloads for both arms. QQQ activation quantization and
output casting remain active. Each arm uses the same 32 held-out documents /
6367 tokens. All 32 baseline/GSQ final-logit SHA256s match exactly.

| Arm | Teacher-to-candidate KL | Logit MSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| F6 control | 0.1081267801 | 0.4404727656 | 0.8548767080 | 0.8303125491 | 0.8292602482 |
| F6 + baseline QQQ QKV | 0.1116840904 | 0.4584124394 | 0.8503219727 | 0.8276425318 | 0.8272655882 |
| F6 + fixed GSQ QQQ QKV | 0.1116840904 | 0.4584124394 | 0.8503219727 | 0.8276425318 | 0.8272655882 |

GSQ-versus-baseline paired document deltas are exactly zero for every metric;
there is no recovery gain in this run. The difference from the F6 control is
the effect of substituting QQQ W4A8 for those three projections, not a GSQ
effect or a controlled ranking of quantization methods. Top-5/10 are set
overlap divided by K; these agreements are not labeled task accuracy.

The [propagation report](../../artifacts/gsq-qqq/propagation-seed7-v2/report.json)
binds model inputs, payloads, source implementations, and per-document teacher
and candidate hashes. The executed runner and GPU log are archived beside it.
`scripts/validate_gsq_qqq_propagation.py --prepare --layers LAYER_OUTPUT --output NEW_OUTPUT`
prepares the run; execute the same command without `--prepare` through the GPU
allocator. This validates full-model propagation with selected native QQQ
operators, not a complete native QQQ model export. Other layers, broader
channelwise validation and the remaining compatible GSQ methods remain open.

## Channelwise real-layer validation

The same real Q/K/V inputs also pass the local lifecycle with group size -1,
using the channelwise signed-nibble packing mode. All 192 native/reference
checks pass; worst mean drift is 6.8918e-8 and worst maximum drift is 0.00390625.
The independent packed objectives match diagnostics, and all baseline/GSQ
packed tensors are exactly equal. Held-out MSE for both arms is Q=0.0036845084,
K=0.0076688737 and V=0.0002162909. No recovery gain is established.

The [channelwise report](../../artifacts/gsq-qqq/real-qkv-channelwise-seed7-v2/report.json)
and payload audit preserve these results. The runner now accepts the absent
extra group-scale tensor in this mode. The earlier harness failure on that
`None` tensor is archived alongside the successful run. Channelwise full-model
propagation and complete native exports remain unverified.
