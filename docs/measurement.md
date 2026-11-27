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
    PPLConfig, RMSEConfig, TOPKConfig,
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
        TOPKConfig(k=(1, 5), threshold=GradedThresholdConfig(
            grades={MetricGrades.PASS: ">=0.95", MetricGrades.WARN: ">=0.90"},
        )),
        RMSEConfig(),
    ),
    dataset=texts,          # strings, HF dataset rows, or token-id lists
    reference=reference,
    batch=4,                # rows are padded and batched together
    max_batches=4,          # rows read, and the forward-batch cap
))

print(report)              # aligned columns for a terminal or TUI
print(report.markdown())   # GitHub table for docs and PRs
print(report.overall, report.passed)
```

```console
metric   value   reference  grade
-------  ------  ---------  -----
PPL      22.85   22.22      -
KLD      0.1183  -          WARN
RMSE     0.6487  -          -
TOP_K@1  0.8242  -          FAIL
TOP_K@5  0.8352  -          FAIL

measured on 4 batch(es), 64 position(s)
result: fail
```

`str(report)` renders through LogBar (`gptqmodel.utils.logger.render_table`), so
the columns are sized to their content and line up in a CLI or TUI. For a
GitHub-rendered table - docs, a PR comment - use `report.markdown()`:

```console
| metric | value | reference | grade |
| --- | --- | --- | --- |
| PPL | 22.85 | 22.22 | - |
| KLD | 0.1183 | - | WARN |
| RMSE | 0.6487 | - | - |
| TOP_K@1 | 0.8242 | - | FAIL |
| TOP_K@5 | 0.8352 | - | FAIL |
result: fail
```

`MeasureConfig` also accepts a single metric (`metrics=KLDConfig()`), metric
names (`metrics=Metric.PPL`), or a mix.

## The metrics

| name | long form | reads |
| --- | --- | --- |
| `PPL` | perplexity | `exp(mean NLL)` over the token stream, both models |
| `KLD` | mean KL divergence | `mean_pos KL(reference ‖ quantized)` on the next-token distribution |
| `TOP_K` | top-k agreement | `|topk(reference) ∩ topk(quantized)| / k`, averaged over positions |
| `RMSE` | logit RMS error | root-mean-square logit difference at sampled positions |

- **`KLD` and `TOP_K` are blind to a constant logit shift.** The softmax and the
  argmax do not move, so `RMSE` exists for exactly that case: a report where
  `TOP_K` is perfect and `RMSE` grew is not a contradiction.
- **`TOP_K` uses set overlap**, not ordered equality; for `k=1` it is plain
  top-1 accuracy. A metric with several widths is graded on its worst width.
- **One forward per batch serves all four.** They declare which captures they
  read (`nll`, `sampled_logits`), both produced by one forward, so the runner
  fuses them. A metric that needed a different capture - a weight-space metric,
  say - would get its own pass instead of silently sharing one.
- **Sampled logits keep the model's dtype.** The capture stores the positions
  exactly as the model returned them; the metrics promote to FP32 when they
  compute. Narrowing the sample to FP16 rounded distinct FP32 logits together,
  hiding a top-token flip, and turned finite BF16 logits above 65504 into
  infinities, so a model compared with itself reported NaN. FP32 models now
  store a 4-byte sample instead of a 2-byte one; BF16 and FP16 models are
  unchanged.

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

## Sliding windows

PPL scores every next-token transition in a row exactly once. A row longer than
`window` is sliced into windows, and each window carries a scoring mask over its
tokens: only targets the row has not scored yet are marked, so overlap is used
as model context and never counts a transition twice.

The effective step is capped at `window - 1`. A window of `w` tokens can score
at most `w - 1` transitions, and stepping further would drop the transition at
each seam. Capping it keeps one token of context across every boundary, so
`stride=0`, `stride=window`, and overlapping strides all cover the same
transitions. `build_token_rows()` returns these `TokenWindow` objects; passing
plain `[1, tokens]` tensors to `measure_models()` still means "score every
transition in this row".

```python
from gptqmodel.measure import PPLConfig, build_token_rows

windows = build_token_rows(tokenizer, texts, window=512, stride=256)
# each window: .input_ids [1, tokens], .score_mask [1, tokens]
```

A position-local model is the regression for this: its logits do not depend on
the window, so changing `window` or `stride` must not change PPL.


## Batching and reproducibility

`batch` changes how the same rows are fed, never which transitions are sampled
or how they are weighted. Each row samples its own positions up to its real
length - a short peer no longer caps a longer row - and KLD, TOP_K and RMSE pool
per-position sums and element counts across batches before dividing once. A
ragged batch and a short final batch therefore report the same `positions` and
the same metric definition as `batch=1`.

In bfloat16 the forward itself is still sensitive to batch shape: the same row
alone and inside a batch of equally long rows differs by up to 0.84 in logits.
That residual is kernel accumulation, not sampling or padding - the padded and
unpadded float32 runs agree to 1.5e-4. `batch > 1` gives the speed (fewer
forwards), while `batch=1` gives the most reproducible forward numerics.

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
  `TOP_K {1: 1.0}` - not "close to zero". A harness that cannot do that is
  measuring drifted positions rather than models.
- **One forward per batch** for any number of fused metrics (asserted by a
  counting test). `batch` batches rows of different lengths too: they are
  padded to the widest row, and every consumer masks the padding - the attention
  mask keeps pads out of attention, the NLL ignores padded targets, and position
  sampling stays inside each row's real length. In float32 the padded and
  unpadded runs agree to 1.5e-4, and adding `attention_mask` alone changes
  nothing.
- **`batch` does not change the sample.** Every row samples its own positions,
  and KLD/TOP_K/RMSE pool per-position sums and counts across batches instead
  of averaging already-reduced batch means. Batch 1, a ragged batch and a short
  final batch produce the same `report.positions` and the same values.
- **`report.positions` is what was actually sampled**, not what was requested:
  a short row contributes its own two or three positions, not the configured 64.
- **Report-only by default**: no grades, no verdict.

## Command line

```
python -m gptqmodel.measure --model <quant> --reference <fp16> --text <file> [--text ...] \
    --ppl --kld --topk 1,5 --rmse --kld-pass "<=0.05" --topk-pass ">=0.95" --fail-on fail
```

It prints the table through the same LogBar column renderer as `print(report)`
(or JSON with `--json`, or `--json-out FILE` for a clean file when the library
logs share stdout) and exits `0` when nothing failed, `1` when a grade failed,
`2` when the measurement itself could not run. `--fail-on warn` also fails on
WARN.

Arguments are validated before any model is loaded: numeric bounds and top-k
widths in argparse, grade expressions through `GradedThresholdConfig`, and
device indices against this machine's GPU count. The device check reads the
index from the string because `torch.device("cuda:999").index` wraps to `-25`,
so `cuda:999` would otherwise be accepted.

`--reference-device cpu` keeps the fp16 copy out of VRAM: the runner already
moves each batch to whichever device each model lives on, so the reference can
run on the host while the quantized model stays on the GPU.

## Capturing a reference once

`capture_reference()` runs the reference forward once and returns a
`ReferenceCapture`; `measure_against()` then measures any number of candidates
against those artifacts:

```python
from gptqmodel.measure import KLDConfig, MeasureConfig, PPLConfig
from gptqmodel.measure.runner import build_token_rows, capture_reference, measure_against

config = MeasureConfig(
    metrics=(PPLConfig(window=2048, stride=512), KLDConfig(positions=128)),
    dataset=texts, reference=reference_model, batch=4, max_batches=16,
)
rows = build_token_rows(tokenizer, texts, window=2048, stride=512)
capture = capture_reference(reference_model, rows, config)

report_a = measure_against(candidate_a, capture, config)
report_b = measure_against(candidate_b, capture, config)
```

A capture only holds the signals its metrics asked for: asking `measure_against`
for KLD when the capture was built for PPL raises instead of comparing against
an empty sample.

## Not in this version

- `quantize(measure=...)`. It is not implementable as designed: a model loaded for
  quantization is a `LazyTurtle` shell (every parameter is `meta` until the looper
  materialises it layer by layer), so there are no dense weights to capture at the
  start - and materialising the whole model first would cost the copy the feature
  was meant to avoid. Measure after quantizing against a reference checkpoint, or
  keep the reference off the GPU with `--reference-device cpu`.
- Task benchmarks - use [Evalution](https://github.com/modelcloud/Evalution) for
  end-to-end quality.
