# Measurement: what quantization did to the math

`model.measure()` compares a quantized model against its reference on the same
token stream and reports four divergence metrics. It answers **"did quantization
change the computation"** in seconds - not "is the model good". End-to-end
quality is a benchmark's job (Evalution); this is the fast check that catches a
kernel, packing or calibration change moving the numbers.

## Usage

```python
from transformers import AutoModelForCausalLM
from gptqmodel import GPTQModel
from gptqmodel.measure import (
    GradedThresholdConfig, KLDConfig, MeasureConfig, MetricGrades,
    PPLConfig, RMSEConfig, TOP_NConfig,
)

quantized = GPTQModel.load("ModelCloud/Qwen2.5-0.5B-gptq-4bit")
reference = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="bfloat16"
).to("cuda")

report = quantized.measure(MeasureConfig(
    metrics=(
        PPLConfig(window=512, stride=256),
        KLDConfig(positions=64, threshold=GradedThresholdConfig(
            grades={MetricGrades.PASS: "<=0.05", MetricGrades.WARN: "<=0.15"},
        )),
        TOP_NConfig(top_k=(1, 5), threshold=GradedThresholdConfig(
            grades={MetricGrades.PASS: ">=0.95", MetricGrades.WARN: ">=0.90"},
        )),
        RMSEConfig(),
    ),
    dataset=texts,          # strings, HF dataset rows, or token-id lists
    reference=reference,
    max_batches=4,
))

print(report.markdown())
print(report.overall, report.passed)
```

```console
| metric | value | reference | grade |
| --- | --- | --- | --- |
| PPL | 22.85 | 22.22 | - |
| KLD | 0.1183 | - | WARN |
| RMSE | 0.6487 | - | - |
| TOP_N@1 | 0.8242 | - | FAIL |
| TOP_N@5 | 0.8352 | - | FAIL |
result: fail
```

`MeasureConfig` also accepts a single metric (`metrics=KLDConfig()`), metric
names (`metrics=Metric.PPL`), or a mix.

## The metrics

| name | long form | reads |
| --- | --- | --- |
| `PPL` | perplexity | `exp(mean NLL)` over the token stream, both models |
| `KLD` | mean KL divergence | `mean_pos KL(reference ‖ quantized)` on the next-token distribution |
| `TOP_N` | top-k agreement | `|top_k(reference) ∩ top_k(quantized)| / k`, averaged over positions |
| `RMSE` | logit RMS error | root-mean-square logit difference at sampled positions |

- **`KLD` and `TOP_N` are blind to a constant logit shift.** The softmax and the
  argmax do not move, so `RMSE` exists for exactly that case: a report where
  `TOP_N` is perfect and `RMSE` grew is not a contradiction.
- **`TOP_N` uses set overlap**, not ordered equality; for `k=1` it is plain
  top-1 accuracy. A metric with several widths is graded on its worst width.
- **One forward per batch serves all four.** They declare which captures they
  read (`nll`, `sampled_logits`), both produced by one forward, so the runner
  fuses them. A metric that needed a different capture - a weight-space metric,
  say - would get its own pass instead of silently sharing one.

## Grades

Thresholds live on the metric config that owns them, and are written as
expressions evaluated in grade order:

```python
GradedThresholdConfig(
    grades={MetricGrades.PASS: ">=0.02", MetricGrades.WARN: ">=0.01"},
)  # everything below 0.01 falls through to FAIL
```

- `PASS`/`WARN` never overlap: the first matching bound wins.
- `FAIL` is the fallthrough and takes no expression.
- The report keeps the worst grade seen; `report.warnings` and
  `report.violations` break it down.
- `strict=True` raises `MeasurementError` on `FAIL` only - a `WARN` never raises.
- Without grades at all, `report.overall` is `None` and `report.passed` is
  `None`: "measured" never reads as "passed".

## What "good" looks like

There is no universal budget. Measured on Qwen2.5-0.5B over out-of-distribution
docs text, same calibration and evaluation rows:

| setting | PPL | KLD | TOP-1 | RMSE |
| --- | --- | --- | --- | --- |
| fp16 reference | 22.22 | - | - | - |
| 8-bit g128 | 22.14 | 0.0024 | 97.3% | 0.094 |
| 4-bit g128 | 22.85 | 0.1183 | 82.4% | 0.649 |

A 0.5B model at 4-bit is much lossier than a 7B at the same setting, so a
threshold that fails the 4-bit row above would be wrong as a default. The useful
reading is **relative**: two settings on the same model, or the same setting
before and after a change to the quantization path.

`BACKEND.TORCH` and `BACKEND.MARLIN` produce the same divergence for the same
checkpoint, which is the point: the metric measures the quantization, not the
kernel.

## Guarantees

- **Identity is exact.** A model compared with itself reports `KLD 0`, `RMSE 0`,
  `TOP_N {1: 1.0}` - not "close to zero". A harness that cannot do that is
  measuring drifted positions rather than models.
- **One forward per batch** for any number of fused metrics (asserted by a
  counting test).
- **Report-only by default**: no grades, no verdict.

## Not in this version

- `quantize(measure=...)`, which would capture reference logits before the first
  module is replaced (no second model needed).
- A `python -m gptqmodel.measure` CLI.
- Task benchmarks - use [Evalution](https://github.com/modelcloud/Evalution) for
  end-to-end quality.
