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
root. Full-model KL/Top-K propagation with QQQ activation quantization,
channelwise real-model validation, other layers and remaining compatible GSQ
methods remain open. Defaults remain disabled and the PR remains draft.
