---
name: gptqmodel-quantization-regressions
description: Perform evidence-first GPT-QModel pre-, during-, and post-quantization error analysis to localize abnormal error to a model family, layer, module, weight group, input/output channel, embedding token row, LM-head vocabulary row, MoE route/expert, adapter, packing boundary, backend, tokenizer, or evaluator. Use before quantization to rank risky tensors, during quantization to catch loss or scale spikes, after quantization to diagnose quality regressions, or when designing model-specific mixed-precision and selective-exemption controls.
---

# GPTQModel Quantization Error Analysis

Find the earliest failing boundary and test causality before changing quantization code. Treat every score, outlier,
and correlation as a lead until an independent control reproduces or rescues the failure.

## Establish the investigation contract

1. Keep the source checkpoint, failing artifact, logs, and raw evaluation outputs immutable. Write diagnostics to a
   separate artifact directory.
2. Establish three references where available:
   - the source model under its native tensor contracts;
   - a healthy higher-bit or otherwise known-good quantized model;
   - the candidate under investigation.
3. Do not assume the source is uniformly dense. Record whether every tensor is BF16, FP16, FP8, QAT, native INT4,
   or another format. A format conversion can be the experiment even when the model is already quantized.
4. Hold the model revision, quantization config, calibration row selection/order/hash, tokenizer, rendered prompts,
   evaluation implementation, seed, backend, and software versions fixed across controls.
5. Keep calibration and verification data disjoint. Use a held-out prompt set from the intended deployment domains;
   do not validate only on the distribution used to select scales or importance.
6. Establish the non-quantized or native-source score, logits, generation, rendered prompts, and exact input IDs
   before calling a difference a quantization regression.
7. When a physical GPU is requested, follow `../gptqmodel-gpu-testing/SKILL.md`: set
   `CUDA_DEVICE_ORDER=PCI_BUS_ID`, resolve the physical ID to PCI bus and UUID, restrict visibility to that device,
   and verify the process-local mapping. Never infer hardware from a fixed index.

Record:

| Area | Required facts |
|---|---|
| Model | source path and revision, architecture, parameter count, tensor dtypes/formats, tied-weight setting |
| Quantization | method, format, bits, group size, symmetry, activation order, scale search, GAR/static groups, dynamic overrides |
| Calibration | dataset and revision, selected row/order hash, token hash/count, chat template, concat/batch settings |
| Runtime | GPT-QModel, PyTorch, CUDA, Transformers, Safetensors, evaluator, backend |
| Hardware | physical GPU ID, PCI bus, UUID, name, compute capability, SM count, memory |
| Evaluation | task/version, prompt rendering, tokenizer, seed, batch size, row count, generation parameters |

## Phase 1: inspect before quantization

### Build an architecture and tensor census

Enumerate actual modules, parameters, aliases, and serialized tensors. Do not rely only on a model-family template.
Classify at least:

- input embeddings, positional/rotary parameters, norms, and output/final norm;
- fused or separate Q/K/V, attention output, and any latent-attention projections;
- MLP gate/up/down projections and the nonlinear product they feed;
- MoE routers, shared experts, routed experts, expert counts, and observed route coverage;
- vision/audio encoders, projectors, multimodal embeddings, and other non-text towers;
- adapters, low-rank corrections, smoothers, rotations, and post-quantization processors;
- LM head and other output heads;
- tied parameters, storage aliases, ignored modules, and per-module dynamic quantization rules.

Reconcile the census with the expected target count before quantization and after reload. Record each tensor's shape,
dtype, format, byte size, quantization policy, and role. Explicitly record whether embedding and LM-head weights are
tied; name equality is not proof of shared storage.

Use GPT-QModel's weight-only scout when applicable:

```python
GPTQConfig(
    bits=4,
    group_size=128,
    preprocessors=[AnalysisConfig(top_k=64)],
)
```

Inspect `model.quantize_analysis`. It estimates RTN-style grouped relative RMSE, small-value pressure, bad-block
fraction, and max/median outlier pressure under the effective dynamic config. It is a cheap prioritization heuristic,
not an activation-aware analysis or quality prediction.

### Measure weights along both axes

For every candidate tensor, report finite/non-finite counts, zero fraction, min/max, mean/std, absolute percentiles,
L2 norm, max/median absolute ratio, row and column norm distributions, estimated grouped RTN error, saturation
incidence, and share of total model bytes.

Assert axis semantics before naming a channel:

- a standard linear weight is `[output, input]`;
- an embedding is `[vocabulary, hidden]`;
- an LM head is normally `[vocabulary, hidden]`;
- Transformers `Conv1D`, fused projections, transposed formats, and packed layouts require explicit normalization;
- quantizer-returned pre-pack scales are commonly `[output_channels, groups]`;
- saved QuantLinear scales are commonly `[groups, output_channels]`.

Calculate whole-tensor and per-output-row metrics. Also inspect per-input-column error when activation outliers can
make an apparently ordinary output row important. Never infer an output channel from the largest raw tensor index
without first proving the layout.

### Measure activation and coverage risk

On both calibration and held-out prompts, collect bounded aggregate hooks for:

- input and output RMS, norm, max absolute value, percentiles, and non-finite counts;
- per-channel RMS/max and activation frequency;
- sample/token count per module;
- MoE router probabilities, selected-expert counts, and unvisited experts;
- Hessian diagonal range, numerical rank, condition evidence, and Cholesky/eigensolver fallbacks when available;
- adapter/correction input and output norms.

Do not log full activations or per-element model data. Keep enough aggregates and bounded samples to reproduce a
candidate.

Capture teacher-forced source outputs after the embedding, each decoder block, final norm, and LM head. Within a
suspicious block, capture residual branches and compound operations such as `activation(gate) * up`; two individually
healthy projections can become a destructive nonlinear product.

### Treat endpoints as first-class modules

For embeddings, measure error per token row and report:

- frequency-weighted error on calibration and held-out tokens;
- unweighted row statistics so rare and unseen tokens are not hidden;
- special/control/tool/image token rows separately;
- tied-weight consistency when the LM head shares the tensor.

For the LM head, measure per-vocabulary-row weight error, output-logit error, probability-distribution divergence,
top-k overlap, and error versus dense logit margin. A high global weight cosine does not establish endpoint safety.

## Phase 2: instrument during quantization

Preserve an ordered record for every target module:

1. source weight identity, storage identity, dtype/device/shape, and bounded fingerprint;
2. activation coverage and Hessian/observer statistics;
3. scale/zero search result and quantizer reconstruction;
4. canonical post-quantizer `Wq` or logical codes;
5. weight after every adapter, replay, restore, smoothing, rotation, or finalization step;
6. weight and metadata immediately before packing;
7. unpacked logical codes immediately after packing.

Enable the existing diagnostics deliberately:

- `quantization_diagnostics="auto"` summarizes module loss and flags extreme loss concentration.
- `quantization_diagnostics="channel"` adds bounded dense-`W` versus canonical-GPTQ-`Wq` reconstruction metrics,
  localized output-row/input-feature error, output-channel/group scale analysis, `g_idx` invariants, and bounded
  logical-code fingerprints after quantization, before packing, and after packing.
- `GPTQMODEL_QUANTIZATION_DIAGNOSTICS=off|auto|channel` overrides one process.

Use `channel` for an investigation, not every production run: scale reduction synchronizes summary values to the
host, and reconstruction diagnostics scan both source and reconstructed weights in bounded chunks. The logical-code
fingerprint is bounded to at most 4,096 samples per module. Saved models include complete
`quantization_diagnostics.json` evidence and a matching human-readable `quantization_diagnostics.md`; neither applies
an automatic precision change.

For every module, calculate:

```text
RMSE = sqrt(mean((Wq - W)^2))
relative RMSE = ||Wq - W||_2 / max(||W||_2, eps)
cosine = dot(Wq, W) / max(||Wq||_2 * ||W||_2, eps)
SQNR_dB = 20 * log10(||W||_2 / max(||Wq - W||_2, eps))
loss share = module loss / sum(all module losses)
```

Also calculate the metrics per output row/channel and, when relevant, per input column and quantization group. Report
scale/zero percentiles, legal code range, underflow/overflow before saturation, saturation rate, `g_idx` invariants,
non-finite values, and error concentration.

Compare a failing candidate with a matched healthy snapshot:

```text
candidate/reference ratio = candidate module metric / reference module metric

normalized degradation =
    candidate/reference ratio
    / median(candidate/reference ratio over matched modules)
```

Compare like roles across depth. Use median/MAD or percentile ranks within a role rather than a universal numerical
threshold. Existing 50x-mean warnings are investigation triggers, not proof of causality. Include calibration and
held-out activation-weighted output error; a better local calibration loss can still overfit and reduce downstream
quality.

For adapters and correction processors, record numerical rank, discarded directions, condition estimate, factor
norms/maxima, reconstructed correction norm, and actual correction-output norm on held-out activations. A huge factor
can be dormant on one reference and explode on another.

For runs longer than 60 seconds, emit the complete live result table required by the GPU-testing skill at least once
per minute. Do not omit completed baselines or invent unavailable metrics.

## Phase 3: verify after quantization

Treat a clean reload as the production boundary. Freshly packed in-memory modules may remain in staging form.

1. Reload the saved model through a generic eager/reference backend.
2. Verify target counts, module classes, module-local bits/group size/symmetry, metadata, dtypes, tied storage,
   ignored dense tensors, and untouched tensor fingerprints.
3. Independently unpack logical codes and reconstruct weights one module at a time.
4. Compare manual unpack with eager dequantization.
5. Compare eager module outputs and logits with every optimized backend under investigation.
6. Run teacher-forced layerwise and endpoint comparisons on identical token IDs.
7. Run paired generation and bounded task evaluation, then expand only after the base is coherent.

Use two module-replay modes:

- Feed the same captured source input to dense and quantized copies to measure local operator error.
- Feed each model's live input to expose propagated and accumulated error.

Locate the first hidden-state divergence after embedding, block sublayers, compound nonlinear operations, residual
adds, final norm, and LM head. Compare norm ratios as well as cosine/RMSE; cosine can hide activation blow-up.

For aligned token positions, calculate dense-to-quantized KL divergence in FP32:

```text
KLD_t = sum_v p_dense[t, v] * (log_p_dense[t, v] - log_p_quant[t, v])
```

Report mean, median, p95, p99, and maximum KLD; top-1 agreement; top-k overlap; logit RMSE/cosine; margin changes; and
sequence position. Report task-answer flips as correct-to-wrong and wrong-to-correct, not only net accuracy.
Perplexity, aggregate accuracy, one prompt, one top-1 token, or whole-model cosine alone can conceal material drift.

### Separate promotion from escalation

Do not make one small or synthetic all-metric gate the only path to further investigation.

Use synthetic tensors only to verify algebra, kernels, serialization, and adversarial corner cases. Never use a
synthetic fixture to accept, reject, rank, or choose the default for a model-quality experiment. Start quality work
with real checkpoint weights and real tokenized activations from a tractable model such as Llama 3.2 1B. A limited
real-model slice is the initial screen; larger disjoint rows, deeper scope, more seeds, and task-like evaluation are
the confirmation path.

Do not use a binary all-metrics pass/fail rule when observed changes may be noise. Compare absolute and relative effect
sizes, paired bootstrap intervals or paired tests where the per-example data permit them, and practical metric
importance. A tiny guardrail loss within uncertainty may be outweighed by a reproducible propagated-loss or task gain;
a material regression is still a blocker. If the uncertainty is unresolved, escalate with a larger disjoint real-model
split and preserve the baseline instead of silently promoting or discarding the candidate.

Local reconstruction MSE/KL is not the optimization endpoint. When a candidate has higher local error but materially
lower disjoint final-logit KL and better Top-K agreement, that local miss is acceptable under the uncertainty-aware
policy: the quantizer is judged by propagated model recovery, not an isolated module proxy. Preserve local metrics
for diagnosis, but do not reject the candidate merely to improve them. The exception is a material local instability
that causes non-finite values, activation blow-up, or a clear downstream regression.

For every metric, report one of three classifications: clear positive (the uncertainty interval excludes zero in the
desired direction), noise-consistent (the interval overlaps zero or the effect is below the predeclared practical
threshold), or clear negative (a material adverse effect supported by the interval and effect size). Never reject a
candidate merely because one of five columns is numerically lower. If that column is noise-consistent while the other
gains are material, escalate the candidate and retain the baseline for rollback; only a clear negative should block
the experiment at that stage.

- **Promote** only when the predeclared locked gate passes on disjoint evidence.
- **Escalate** when a candidate has a material improvement in a primary propagated metric, remains finite and
  coherent, and only a minority of guardrails regress by small amounts that could plausibly be sampling noise or
  limited-scope behavior. Preserve the exact baseline and candidate, then expand the test before deciding.
- Expand along the dimensions that were missing: more independent rows and valid tokens, realistic model depth and
  module roles, full live propagation, multiple seeds, intended deployment domains, paired task flips, and the
  production inference backend. Prefer paired confidence intervals or repeated splits to unpaired aggregate deltas.
- Reject without escalation when a regression is material, repeats across independent splits, crosses a declared
  safety limit, produces non-finite values, or appears at an independently verified correctness boundary.

Escalation is not acceptance. Do not serialize the candidate as a default, relax the original gate after seeing the
result, or average away a critical regression. Record which metrics triggered escalation and the larger confirmation
contract before running it.

## Follow the first failing boundary

| Earliest failing evidence | Leading area |
|---|---|
| Source baseline, prompts, or IDs already disagree | loader, tokenizer, chat template, evaluator, or reference contract |
| Weight scout flags an outlier but held-out outputs remain healthy | risk marker only; do not call it causal |
| Quantizer reconstruction or pre-pack loss is bad | quantization math, grouping, Hessian/error feedback, scale search |
| Canonical `Wq` changes after a zero/no-op processor | aliased state, replay, restoration, or finalization lifecycle |
| Valid pre-pack codes differ after independent unpack | packing, serialization, shifts/masks, zero convention, layout |
| Manual reconstruction is correct but eager output is wrong | QuantLinear reconstruction/application |
| Eager output is correct but optimized output is wrong | backend selection, preprocessing, or kernel |
| Individual projections look healthy but a product/residual/router diverges | nonlinear amplification, routing, or accumulated state |
| First divergence is at the embedding | token rows, tying, device/input capture, multimodal embedding path |
| Hidden states remain healthy through final norm but logits diverge | LM head, tied-weight restoration, output dtype |
| Accuracy is stable but KLD and paired flips are large | real distribution drift hidden by aggregate cancellation |

Later mismatches may be consequences. Do not blame packing or a kernel when pre-pack reconstruction is already bad.

## Run controlled rescue experiments

Change one factor at a time, in this order:

1. Repeat the same configuration to establish determinism.
2. Compare a matched higher-bit model.
3. Compare native quantization without adapters, GAR, smoothing, or unrelated processors.
4. For a coupled quantization-plus-adapter run, test base-only, base plus its exact matching adapter, and a
   zero-correction processor arm that preserves lifecycle.
5. Compare GPTQ with independent symmetric RTN at the same bit width/group size.
6. Promote or exempt one suspicious module role, layer, or endpoint; report size cost and KLD/evaluation rescue.
7. Test embeddings and LM head separately: both dense, embedding-only quantized, head-only quantized, then both.
8. For MoE, separately test router, shared experts, routed experts, and poorly covered experts.
9. Test static/dynamic groups, GAR, activation order, and scale-search policy independently.
10. Compare CPU/GPU packers, independent manual unpack, eager execution, and optimized backends.

A post-hoc channel replacement can validate its local metric without reversing error accumulated during sequential
quantization. If it does not rescue logits, rerun with an in-loop fallback or precision override before excluding the
candidate. Likewise, suppressing a conspicuous scale outlier without recovering held-out outputs proves only that the
outlier was not sufficient.

Use dynamic precision as a tested conclusion, not a starting assumption. Rank promotion candidates by local
reconstruction error, held-out activation-weighted error, propagation/KLD impact, route/token frequency, and bytes
added. Prefer a Pareto table of quality recovery versus size over one opaque composite score.

## Run the configuration-aware pre-quantization scanner

Use `scripts/analyze_quantization_error.py` when the source model is available and the question is which modules,
output rows, input features, weight groups, embedding rows, or LM-head rows are risky under a proposed config. The
scanner simulates grouped RTN as a cheap proxy, streams LazyTurtle checkpoint weights without hydrating the meta
shell, uses bounded chunks, and emits:

- `quantization_analysis.json`: full module records, raw statistics, role-relative anomaly percentiles, limitations,
  and runtime/hardware identity;
- `quantization_analysis.md`: compact ranked module table;
- `quantization_regions.json`: bounded row, feature, and group mappings;
- `quantization_plan.json`: review-only recommendations and module-level `dynamic` overrides;
- `quantize_config.planned.json`: emitted only with `--apply-plan`.

Example using physical GPU 4 for chunked math while keeping the checkpoint-backed shell on CPU:

```bash
python scripts/analyze_quantization_error.py \
  --model /path/to/dense-model \
  --output-dir /path/to/new-analysis-artifacts \
  --method gptq \
  --bits 4 \
  --group-size 128 \
  --sym \
  --desc-act \
  --physical-gpu 4
```

The script resolves the requested physical index to PCI bus ID and UUID before importing Torch and refuses a busy
GPU by default. Use `--module-regex` and `--max-modules` for a bounded smoke run before a whole-model scan. Use
`--quant-config` to analyze a serialized config. `--apply-plan` is an explicit action: existing user-authored
`dynamic` rules win conflicts, endpoints remain review-only, and no plan is applied to a source checkpoint in place.

The default `model_definition` fusion profile expands direct risk recommendations using the module groups declared
by the loaded GPTQModel model class. This model-definition metadata is the authoritative fusibility/support contract:
modules sharing a group inherit compatible proposed precision when one member is directly flagged. For Qwen3 the
definition declares:

- `q_proj + k_proj + v_proj` in one attention-input group;
- `gate_proj + up_proj` in one MLP-input group.

Keep direct outlier evidence separate from group closure. A companion may have ordinary local error and still need a
compatible precision/layout policy because GPTQModel quantizes the declared group as one supported block. Record the
direct trigger, every companion, the model-definition class/source, and the exact expanded group. Do not infer extra
companions from module-name conventions or from one serving engine: vLLM/SGLang fusion is useful explanatory context,
but it does not override the GPTQModel definition. For dense Qwen3, `o_proj` and `down_proj` are later, separate
definition groups. Use `--fusion-profile none` only when deliberately disabling definition-group closure.

Interpret the output as **Observed, pre-quantization proxy evidence**. A high-risk group or channel is a candidate,
not a causal finding. The current packed formats and kernels apply `dynamic` settings per module; the region mapping
cannot encode arbitrary mixed bit widths inside one packed module. Roll regional evidence up to a module-level
promotion or implement and validate a new format/backend contract before claiming finer-grained remediation.

## Use the bundled snapshot analyzer

Run the cheap log comparison first:

```bash
python .agents/skills/gptqmodel-quantization-regressions/scripts/analyze_quant_regression.py \
  --snapshot healthy=/path/to/healthy \
  --snapshot failing=/path/to/failing
```

Scan saved scale tensors only after module-loss comparison identifies a suspicious area:

```bash
python .agents/skills/gptqmodel-quantization-regressions/scripts/analyze_quant_regression.py \
  --snapshot healthy=/path/to/healthy \
  --snapshot failing=/path/to/failing \
  --scan-scales \
  --json /tmp/quantization-regression.json
```

For same-bit snapshots with comparable layouts, add `--scan-codes`. It processes one packed module at a time and
reports logical-code mismatch rate/delta plus exact scale, zero, and group-index equality:

```bash
python .agents/skills/gptqmodel-quantization-regressions/scripts/analyze_quant_regression.py \
  --snapshot native=/path/to/native \
  --snapshot coupled=/path/to/coupled \
  --scan-codes \
  --json /tmp/packed-code-comparison.json
```

Do not compare logical codes across different bit widths. Equal metadata plus different codes proves different
reconstructed inputs reached packing; it does not prove the packer mutated identical inputs. Use in-process
post-quantizer/pre-pack fingerprints to distinguish those cases.

## Apply case-study lessons without copying their thresholds

Repository investigations established reusable signatures:

- A Qwen3 low-bit loss/scale spike concentrated in one `mlp.down_proj` output channel, but repairing that saved
  channel did not rescue end-to-end logits. The channel was a marker of unstable feedback, not a sufficient cause.
- Qwen3 gate/up projections looked individually reasonable before their SwiGLU product collapsed. Inspect compound
  operations and norm amplification, not only linear outputs.
- An EoRA near-null covariance direction produced extreme factors and an activation explosion. Factor magnitude
  alone was insufficient; actual correction output on live/held-out activations localized the failure.
- Embedding/LM-head post-quantization retained very high sampled weight and logit cosine while paired benchmark
  answers still flipped. Endpoint decisions require held-out distribution and task evidence.
- Finer AWQ/GPTQ local objectives did not consistently improve downstream quality. Calibration reconstruction is a
  proxy and needs an independent held-out gate.

The Unsloth Kimi K2.6 report adds a format-boundary lesson: its reference contract uses native INT4 for MoE weights
and BF16 elsewhere, and a model-specific INT4 bijection change is applied only to the native MoE tensors. Reproduce
the exact source code mapping and preserve non-target tensor precision; never transfer a `max / -7` versus
`max / -8` rule to GPTQ or another model without proving the format contract.

The Unsloth Dynamic 2.0 report adds evaluation lessons: choose precision at model/layer/tensor granularity, validate
on data independent of calibration, include instruct/chat formatting in calibration when appropriate, and use
distribution KLD plus paired flips and hard tasks alongside perplexity/accuracy. Treat these reports as hypotheses
for controlled GPT-QModel experiments, not as authority for GPT-QModel layouts or defaults.

Sources:

- <https://unsloth.ai/docs/models/kimi-k2.6>
- <https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs>
- `quantize_embed_lm_head.md`
- `gptq_scale_search.md`
- `awq_scale_search_log.md`
- `adjacent_exact.md`
- `eora_svd_algo.md`
- `eora_test_model_paths.md`

## Report evidence and confidence

Preserve raw JSON, exact commands, and a compact report containing:

| Snapshot | Phase/boundary | Module/role | Axis/index | Coverage | Weight error | Output/KLD effect | Rescue result | Status |
|---|---|---|---|---:|---:|---:|---|---|

Include separate embedding, decoder, MoE/router, adapter, final norm, and LM-head rows when present. State whether
every check ran, skipped, or was blocked.

Label conclusions:

- **Observed**: a metric or mismatch was measured.
- **Localized**: the earliest failing boundary is known.
- **Causal**: an independent intervention reproduces or rescues the downstream failure.
- **Excluded**: a boundary matches its reference under an independent implementation.

End with a ranked suspect table containing evidence for, evidence against, the cheapest decisive next experiment,
expected size/quality tradeoff, and confidence. Do not claim root cause from correlation, one diagnostic prompt, or
one outlier. A diagnosis is complete only when an independent boundary check and a controlled rescue/exclusion agree.

## Guard implementation work

This skill is fact-finding. If the user requests a fix, also read `../gptqmodel-quantization/SKILL.md` and its packing
reference when codes, scales, zeros, grouping, or serialization are involved. Put regression assertions in `tests/`,
timed investigations in `scripts/`, preserve CPU and non-target accelerator fallbacks, and recreate corrupted
artifacts instead of repairing packed checkpoints in place.
