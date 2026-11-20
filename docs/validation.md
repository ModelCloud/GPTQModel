# Validation: measuring what quantization did to the math

`model.validate()` compares a quantized model against its reference on the same
token stream and reports four divergence metrics. It answers **"did
quantization change the computation"** in seconds - not "is the model good".
End-to-end quality is a benchmark's job (Evalution); this is the fast check that
catches a kernel, packing or calibration change moving the numbers.

## Usage

```python
from transformers import AutoModelForCausalLM
from gptqmodel import GPTQModel
from gptqmodel.validate import KLDConfig, PPLConfig, RMSEConfig, TOP_NConfig, ValidateConfig

quantized = GPTQModel.load("ModelCloud/Qwen2.5-0.5B-gptq-4bit")
reference = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="bfloat16"
).to("cuda")

report = quantized.validate(
    ValidateConfig(
        metrics=(PPLConfig(window=512, stride=256), KLDConfig(positions=64),
                 TOP_NConfig(top_k=(1, 5)), RMSEConfig()),
        dataset=texts,          # strings, HF dataset rows, or token-id lists
        reference=reference,
        max_batches=4,
        thresholds={"KLD": 0.2},  # optional; without it the run only reports
    )
)

print(report.markdown())
print(report.values["KLD"], report.values["TOP_N"][5])
```

```console
| metric | value | reference | threshold | ok |
| --- | --- | --- | --- | --- |
| PPL | 22.85 | 22.22 | - | - |
| KLD | 0.1183 | - | - | - |
| RMSE | 0.6487 | - | - | - |
| TOP_N@1 | 0.8242 | - | 0.2 | - |
| TOP_N@5 | 0.8352 | - | 0.2 | - |
result: no thresholds configured
```

## The metrics

| name | long form | reads |
| --- | --- | --- |
| `PPL` | perplexity | `exp(mean NLL)` over the token stream, both models |
| `KLD` | mean KL divergence | `mean_pos KL(reference ‖ quantized)` on the next-token distribution |
| `TOP_N` | top-k agreement | `|top_k(reference) ∩ top_k(quantized)| / k`, averaged over positions |
| `RMSE` | logit RMS error | root-mean-square logit difference at sampled positions |

Notes that matter when reading them:

- **`KLD` and `TOP_N` are blind to a constant logit shift.** Adding the same
  constant to every logit changes neither the softmax nor the argmax. `RMSE`
  exists for exactly that case; a report where `TOP_N` is perfect and `RMSE`
  grew is not a contradiction.
- **`TOP_N` uses set overlap, not ordered equality.** Near-tied logits flip order
  under rounding that does not move the model; for `k=1` this is plain top-1
  accuracy.
- **The reference pass keeps only sampled logits** (fp16, `positions` per batch)
  plus the PPL sum, so validating a 7B model costs tens of megabytes, not a
  second copy of the weights.

## What "good" looks like

There is no universal budget. Measured on Qwen2.5-0.5B over out-of-distribution
docs text, same calibration and evaluation rows:

| setting | PPL | KLD | TOP-1 | RMSE |
| --- | --- | --- | --- | --- |
| fp16 reference | 22.22 | - | - | - |
| 8-bit g128 | 22.14 | 0.0024 | 97.3% | 0.094 |
| 4-bit g128 | 22.85 | 0.1183 | 82.4% | 0.649 |

A 0.5B model at 4-bit is much lossier than a 7B at the same setting, so a
threshold that fails the 4-bit row above would be wrong as a default. The
useful reading is **relative**: two settings on the same model, or the same
setting before and after a change to the quantization path.

`BACKEND.TORCH` and `BACKEND.MARLIN` produce the same divergence for the same
checkpoint, which is the point: the metric measures the quantization, not the
kernel.

## Guarantees

- **Identity is exact.** Comparing a model with itself reports `KLD 0`,
  `RMSE 0`, `TOP_N {1: 1.0}` - not "close to zero". A harness that cannot do
  that is measuring drifted positions rather than models.
- **Report-only by default.** `report.passed` is `None` when no thresholds were
  configured, so "validated" can never be read as "passed".
- **`strict=True` raises** `ValidationError` when a threshold breaks, for CI use.

## Not in this version

- `quantize(validate=...)`, which would capture reference logits before the
  first module is replaced (no second model needed).
- A `python -m gptqmodel.validate` CLI.
- Task benchmarks - use [Evalution](https://github.com/modelcloud/Evalution) for
  end-to-end quality.
