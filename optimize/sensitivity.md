# Hierarchical linear-kernel sensitivity

Use `sweep_sensitivity.py` to measure how an actual output perturbation propagates through the unchanged model.
It reuses `calibration_coverage.find_target_groups`; the existing coverage scanner still measures calibration
coverage, and `scripts/analyze_quantization_sensitivity.py` still provides local RTN-style proxy curves.
Neither existing score is substituted for an observed final-logit difference.

## Purpose and prerequisites

This tool answers a narrow engineering question: if one or more physical linear
operators change while the rest of the model remains unchanged, how much does the
change reach the model output? It helps prioritize kernel work and identify shared
input opportunities. It is a teacher-forced propagation harness, not a quantizer,
benchmark runner, or task-accuracy evaluator.

Use a dense or quantized checkpoint with the exact weights and runtime path you intend
to study. Supply ordinary held-out text that is disjoint from calibration and task
evaluation data. Keep the baseline and candidate in the same process, with the same
token IDs, masks, dtype, device, and model state. The harness stores baseline logits on
the CPU but does not save weights or activation histories.

The command line loads a Transformers model and discovers decoder blocks from
`--layers-path`. The Python API is required for custom model loaders, custom operators,
private state reset logic, or a candidate implementation that is not expressible as a
built-in probe.

## Command-line workflow

Start with the random CPU fixture. This verifies the harness and report format only:

```bash
python optimize/sweep_sensitivity.py \
  --tiny \
  --max-batches 2 \
  --max-length 12 \
  --amplitudes 0.001,0.002 \
  --top-layers 2 \
  --output /tmp/tiny-sensitivity
```

Then run a real checkpoint with held-out JSONL data:

```bash
python optimize/sweep_sensitivity.py \
  --model /models/checkpoint \
  --data /data/held-out.jsonl \
  --layers-path model.language_model.layers \
  --probe output \
  --amplitudes 0.001,0.002,0.005,0.01 \
  --top-layers 4 \
  --max-batches 32 \
  --max-length 512 \
  --dtype bfloat16 \
  --output /tmp/model-sensitivity
```

The `--model` path requires `--data`. `--tiny` without data creates random token IDs;
`--tiny` with data accepts `input_ids` records and avoids tokenizer downloads. Text
records are tokenized with the checkpoint tokenizer. Use `--revision` to pin a model
and tokenizer revision, and `--trust-remote-code` only for checkpoints that require it.

For a physical NVIDIA GPU, identify the device with `nvidia-smi` and pass its physical
index before Torch is imported:

```bash
nvidia-smi --query-gpu=index,pci.bus_id,uuid,name --format=csv
python optimize/sweep_sensitivity.py \
  --model /models/checkpoint \
  --data /data/held-out.jsonl \
  --physical-gpu 0 \
  --output /tmp/model-sensitivity-gpu
```

The preflight requires zero GPU utilization, no foreign compute process, and at most
16 MiB of driver memory. It resolves the physical index to a UUID, sets
`CUDA_VISIBLE_DEVICES`, and the model runs on the resulting `cuda:0`. The CLI does not
shard a model across multiple GPUs; use one physical index for a single-device run.
GPU execution is a runtime check and is not covered by the CPU tests below.

## Data file formats

Each nonempty JSONL record produces one or more teacher-forced batches. Supported
forms are:

```json
{"text": "ordinary held-out text"}
{"messages": [{"role": "user", "content": "ordinary held-out text"}]}
{"input_ids": [1, 42, 17, 2], "attention_mask": [1, 1, 1, 1]}
{"input_ids": [1, 42, 17, 0], "attention_mask": [1, 1, 1, 0], "sensitivity_mask": [0, 1, 1, 0]}
```

`input_ids` must be a one-dimensional token sequence. `attention_mask` defaults to all
ones when omitted. An aligned `sensitivity_mask` selects the positions used for final
logit metrics and masked local epsilon; it may not select padding. All invocation
positions remain available in `local_all_positions`, whose maximum error is useful for
checking a kernel boundary. Keep at least one selected position per record.

The harness rejects `past_key_values`, `past_key_value`, `cache_params`, and `mems` in
input records. This prevents cached state from making repeated baseline and candidate
arms incomparable. Private recurrent or convolutional state must be cleared by the
API's `reset_state` callback.

## Selecting modules and subsets

Automatic discovery inventories `nn.Linear` and Transformers `Conv1D` modules under
each decoder block. `--layers-path` must resolve to a module container whose children
are the decoder blocks, such as `model.layers` or `model.language_model.layers`.

Use `--modules` for exact physical paths, including custom quantized operators:

```json
[
  "model.layers.0.self_attn.q_proj",
  "model.layers.0.self_attn.k_proj",
  "model.layers.0.self_attn.v_proj",
  "model.layers.1.self_attn.q_proj",
  "model.layers.1.self_attn.k_proj",
  "model.layers.1.self_attn.v_proj"
]
```

When the CLI supplies an explicit module list, include at least one target under every
layer discovered from `--layers-path`; the layer screen must have a defined target for
each layer. For a focused one-layer run, use the API with `layers=["model.layers.0"]`.

Use `--subsets` for named jointly enabled groups:

```json
{
  "attention_qkv": [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj"
  ],
  "mlp_gate_up": [
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj"
  ]
}
```

Each subset must contain at least two distinct inventoried targets from one decoder
layer. The coverage scanner's names are only hints. During each forward the sensitivity
harness verifies storage identity, view metadata, dtype, device, shape, strides, and
the in-place mutation version before reporting shared input. Equal values in separate
allocations are not shared inputs.

## Two stages and a combined control

1. Run an unchanged teacher-forced baseline and repeat it to measure baseline drift.
2. Sweep each decoder layer: enable the candidate only in that layer's selected linear modules. Every other linear,
   nonlinear operation, embedding and LM head retains its original implementation. Repeat for each amplitude.
3. Rank layers by their worst per-batch final relative-L2 error across the declared amplitudes. Nonfinite,
   unobserved or undefined cases receive investigation priority. Expand the first `--top-layers` into individual
   modules, runtime shared-input groups, and optionally their pairs. `--subsets` adds explicit definition subsets.
4. Enable all selected linears together and compare measured drift with the uncorrelated layer prediction.

The layer screen tests the aggregate effect of changing its linears, **not** arbitrary noise added at a block output.
Ranking depends on the probe, amplitude, live inputs and selected modules. It is not an intrinsic architecture score.
`--top-layers 0` runs the layer screen and combined control only; select all layers for a complete module sweep.

## Reuse and verify shared inputs

The coverage scanner suggests QKV and gate/up groups by projection names. Its census now optionally includes
nonstandard linears, matches complete leaf names, and separates incompatible input widths. The sensitivity harness
verifies actual storage/view identity, shape, strides, dtype, device and mutation version within each block invocation.
It retains views until that block returns to prevent allocator address reuse from looking like input sharing.

Equal values in separate allocations are not classified as shared input. Aliased views of the same unchanged tensor
are shared; a mutation between consumers separates them. Sharing counts are recorded for the baseline and the
candidate arm. Name hints remain separate from this evidence. Observed sharing does not grant permission to fuse.
Large automatic groups above `--max-group-size` are explicitly reported as skipped to bound the pairwise expansion.

Default census support is `nn.Linear` and Transformers `Conv1D`. Fused QKV/gate-up linears remain one physical
module; splitting logical output slices requires an adapter. Supply `--modules` or the API's `module_names` to
restrict the census to the exact optimized modules, including custom QuantLinear operators. Input/output tensor
contracts still apply. Explicit subsets must stay inside one decoder layer and may have distinct inputs.
Do not infer model-defined precision/fusion policy from the automatic diagnostic groups.

## Measurements

For each selected module, accumulate its actual post-cast error on the same live input as its eager reference:

```text
epsilon_i = ||Y_candidate_i - Y_reference_i||_F / ||Y_reference_i||_F
E         = ||logits_candidate - logits_reference||_F / ||logits_reference||_F
g_effective = E / sqrt(sum_i(epsilon_i**2))
```

For a single module this is its measured directional gain g_i. For a layer/group/all-enabled arm it is a descriptive
ratio that absorbs interactions, not an RMS of isolated gains. Norms aggregate over the same declared batches;
ranking separately uses the worst batch. No denominator is silently clamped. Zero reference norms, unresolved
local errors, nonfinite/overflowing measurements and effects within measured baseline noise yield undefined gains.

Reports retain local max-absolute/mean-absolute/RMSE/relative-L2 errors, final raw-logit errors, per-batch full-vocabulary
KL(reference || candidate), top-1 agreement and minimum reference top-token margin. Tensor metrics accumulate in
FP64 with bounded temporary chunks. KL rounds tiny negative roundoff to zero; tensor errors are never clipped.

The evaluation mask defaults to `attention_mask`. Optional `sensitivity_mask` selects a subset of valid positions.
Local epsilon uses that mask when the module output's leading dimensions match it; otherwise it uses all invocation
rows and `local_masked_calls` exposes the distinction (important for flattened/routed experts). All output positions
still contribute to `local_all_positions` and its **maximum absolute 2e-3** comparison. A diagnostic within that
tolerance is not a production kernel certification, and no tolerance or checkpoint default is changed.

For jointly enabled subsets, `uncorrelated_prediction = sqrt(sum(E_single_i**2))` is compared with actual joint E.
The combined control also reports `uncorrelated_layer_prediction`. Correlations and nonlinearities can amplify or
cancel errors; neither prediction is a safety bound or a forecast of task-accuracy percentage points.

## CPU smoke test

```bash
python optimize/sweep_sensitivity.py --tiny --max-batches 2 --max-length 12 \
  --amplitudes 0.001,0.002 --top-layers 2 --output /tmp/tiny-sensitivity
python optimize/sweep_sensitivity.py --tiny --probe shared-input --amplitudes 0.001 \
  --max-batches 1 --top-layers 1 --output /tmp/tiny-shared-sensitivity
pytest -q tests/test_sensitivity_sweep.py tests/test_calibration_coverage.py
```

Tiny mode initializes a random three-layer Llama on CPU. It tests the harness and real layer forward implementation;
it provides no evidence about a trained Qwen checkpoint's quality. CPU defaults use one thread and FP32 tensors.
The CPU suite includes independent tensor-metric checks, an analytic Q/K cancellation fixture, tiny HF Llama
inference, masking, alias/mutation detection, target coverage and cleanup on failure.

## Real checkpoint and candidate kernels

```bash
python optimize/sweep_sensitivity.py --model /models/checkpoint --data held-out.jsonl \
  --layers-path model.language_model.layers --probe output \
  --amplitudes 0.001,0.002,0.003,0.005,0.01,0.02 --top-layers 4 \
  --modules optimized-module-paths.json --subsets definition-subsets.json --output /tmp/model-sensitivity
```

Use the actual layer container from the loaded model. JSONL rows may contain `text`, `messages`, or `input_ids`
(with optional aligned masks). Text tokenization reuses the coverage scanner's helpers; an explicit `--revision`
pins the Hugging Face tokenizer as well as the model. The CLI bounds batches/sequence lengths and writes
`sensitivity.json` and `sensitivity.md`, including model/config/runtime identity and token/input hashes.
An optional `--physical-gpu` reuses the existing NVIDIA preflight before Torch import; GPU runs are not covered by
the CPU test evidence. Other hardware/model-loader arrangements can call the Python API directly.

The `output` probe uses deterministic independent output noise with the requested pre-cast relative norm.
`shared-input` uses one deterministic input perturbation for consumers of the same observed input view and recomputes
their `nn.Linear` outputs. It does not mutate the original shared input. Probe directions remain fixed across
amplitudes and target subsets; actual error is remeasured after output casting. Both built-ins are diagnostic.

For a real kernel, pass a pure callback through the API instead of treating synthetic noise as kernel evidence:

```python
from optimize.sensitivity import SensitivitySweep

def candidate(context, module, args, kwargs, reference):
    x = args[0] if args else kwargs["input"]
    return optimized_forward[module](x)  # Same packed weights and semantics; do not call module() recursively.

report = SensitivitySweep(
    model, decoder_layer_paths, module_names=optimized_module_paths,
    subsets=model_definition_subsets, candidate=candidate,
).sweep(held_out_batches, amplitudes=[1.0], top_k=4)
```

The callback owns the amplitude meaning; the example measures the actual candidate at scale one. It must preserve
shape/dtype/device and must not mutate inputs, weights, metadata or private state. Use identical reconstructed
quantized weights to isolate kernel error from quantization error. Real-model quality decisions require trained
weights and disjoint real tokenized data, followed by paired task evaluation where warranted.

The harness clones input batches, removes its hooks in `finally`, and restores module training flags and RNG state.
It disables standard caching and rejects supplied input caches. Provide `reset_state(model)` for models with private
mutable recurrent/convolution/KV caches; each forward must start fresh. Cached decode, serving-engine execution,
logical fused slices and automatic MoE route/expert coverage are not certified by this teacher-forced screen.
Unvisited modules remain unknown. Baseline logits are retained on CPU, so bound evaluation positions/batches for
large vocabularies. No weights or activation histories are saved in the report.
