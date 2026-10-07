<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GPTQ INT4 weights with FP8 or NVFP4 activations on GB10

Both policies keep the saved GPTQ checkpoint in the ordinary INT32-packed
`qweight`, `qzeros`, `scales`, and `g_idx` layout. `post_init()` reads those
tensors once and builds nonpersistent GPU operands. The weight cache is never
serialized. Selected Llama decoder layers carry encoded activations and their
scales across decoder boundaries and into every selected GEMM. Operators that
do not consume FP8/FP4 operands perform their arithmetic in wider precision,
then pack once at the next GEMM or decoder boundary. The embedding, unselected
decoder layers, and `lm_head` remain in the model dtype.

Device and model-dtype conversions preserve the hardware weight-cache formats
and bytes without rebuilding them. Repacking or loading native weight tensors
invalidates every derived cache; call `post_init()` before using the new
weights. Re-exporting a loaded W4A8 or NVFP4 checkpoint preserves the source
tensors' values and dtypes, including BF16 tensors outside the FP16 range.

| Policy | Input rounding | One-time weight operand | Group arithmetic |
| --- | --- | --- | --- |
| `w4afp8` | E4M3, one dynamic scale per token | Exact E4M3 representation of each centered INT4 code | Apply each original GPTQ group scale to its FP8 MMA partial result |
| `w4a_nvfp4` | E2M1, least-squares-refined E4M3 scale per 16 values, calibrated global scale | Two exact E2M1 planes satisfying `centered_code = low + 4 × high` | Add the two native FP4 partial results per group, then apply the original GPTQ scale |

The second path uses NVIDIA's block-scaled FP4 operand layout. It does not
round an arbitrary INT4 value to one FP4 code or approximate an arbitrary GPTQ
scale with an E4M3 hardware scale. The `w4afp8` policy follows the per-token
FP8 recipe; it is not an MXFP8 block scale format. Neither policy changes the
checkpoint's weight codebook.

For each NVFP4 activation block, the packer starts with two hardware-valid
candidates. M=6 maps the block maximum to E2M1 value 6 and provides finer
levels near zero. M=4 maps it to value 4 and provides a different set of
middle and upper-range thresholds, while value 6 remains available as
headroom. For each resulting E2M1 code assignment, the packer solves the
least-squares scalar, rounds that scalar back to E4M3, and keeps it only when
the actual 16-value squared error is lower. It performs one final refinement
from the winning assignment. This is a monotone extension of NVIDIA's
Four-Over-Six scale search: every emitted value is still E2M1 and every local
scale is still E4M3. Runtime activation storage and GEMM operands remain one
FP4 code per value plus the standard E4M3 block scales.

New NVFP4 checkpoints serialize an explicit recipe, for example
`{"mode": "w4a_nvfp4", "recipe": "least_squares"}`. `nvidia` uses
NVIDIA's max-to-6 block rule, while `four_six` and `least_squares` use the hardware-valid
block-scale extensions described above. `nvidia_headroom` and `least_squares_headroom`
add a calibration pass for NVIDIA's percentile headroom global scale before
GPTQ Hessian capture. `least_squares` is the default when a NVFP4 stream omits
`recipe`. The recipe is
propagated through calibration replay, `HookedLinear` Hessian capture, runtime
carriers, and fused packing so a saved checkpoint cannot silently switch
scale selection.

## Activation stream boundaries

Within a selected Llama decoder layer, `W4AActivation` carries codes, scales,
logical shape, and model dtype. FP8 uses E4M3 codes and one FP32 scale per
token. NVFP4 uses packed E2M1 codes, swizzled per-16 E4M3 scales, and a global
scale for that activation. W4A Linears consume these operands directly with
native FP8 or FP4 GEMM and return the model dtype. Carriers also retain a
wider reference value for residual arithmetic and normalization. `q_proj`,
`k_proj`, and `v_proj` share one encoded RMSNorm output; `gate_proj` and
`up_proj` share another. The saved INT32-packed GPTQ weights are unchanged.

| Selected Llama 3.2 decoder boundary | Activation handed to next module |
| --- | --- |
| Decoder input → RMSNorm → attention | Encoded FP8 or FP4 with scales |
| RMSNorm output → `q_proj`/`k_proj`/`v_proj` | Same encoded input object for all three |
| Q/K/V GEMMs → attention operator | GEMMs emit model dtype directly; RoPE, attention, and KV cache use wider internal values |
| Attention output → `o_proj` | Packed once and consumed directly by the GEMM |
| `o_proj` output → residual add → RMSNorm | Model-dtype branch output; residual sum packed with its wider reference retained |
| RMSNorm output → `gate_proj`/`up_proj` | Same encoded input object for both |
| Gate/up GEMMs → SiLU/multiply | GEMMs emit model dtype directly for the wider elementwise math |
| MLP product → `down_proj` | Rotated when required, packed once, and consumed directly by the GEMM |
| `down_proj` output → residual add → next selected decoder layer | Model-dtype branch output; residual sum carries codes, scales, and its wider reference directly to the next layer |

RMSNorm and residual addition use the carrier's wider reference for arithmetic;
the hardware GEMMs consume its codes and scales. Eligible fused NVFP4 norms
reuse those codes with a token multiplier; other norms pack the normalized
value. RoPE, attention, SiLU, multiplication, and the current KV cache
use BF16/FP16 inside the attention or MLP operator. Selected GEMMs return the
model dtype directly when their next consumer is one of those wider operators;
they do not quantize an output only for that consumer to decode it immediately.
The final model norm
consumes the encoded stream and emits BF16/FP16 for the unselected `lm_head`. A dense layer adjacent to a
selected block is also a model-dtype boundary. The current scope does not
provide an FP8/FP4 KV cache or fully low-precision internal attention math.

The guarded dtype audit runs a short forward pass and a cached two-token
generation on a saved checkpoint. It records selected W4A Linear, RMSNorm,
attention, MLP, and decoder boundaries. It also checks that successive
selected layers receive the exact codes and scales emitted by the previous
layer, and that fan-out projections reuse the same encoded input:

```bash
export GPTQMODEL_TEST_PYTHON=/root/gptqmodel-test-venv/bin/python
tests/models/run_w4a_gb10_safe.sh dtype-audit \
  --checkpoint /root/models/Llama-3.2-1B-Instruct-W4AFP8-stream-v2 \
  --variant w4afp8 --no-require-full-coverage \
  --output /root/models/w4a-quality/w4afp8_dtype_audit.json
tests/models/run_w4a_gb10_safe.sh dtype-audit \
  --checkpoint /root/models/Llama-3.2-1B-Instruct-W4ANVFP4-stream-v2 \
  --variant w4a_nvfp4 --no-require-full-coverage \
  --output /root/models/w4a-quality/w4a_nvfp4_dtype_audit.json
```

Those two historical fast checkpoints covered only layers 14 and 15, with 14
selected W4A Linears. Their earlier audit encoded Linear outputs as well; that
output behavior predates the current transport contract. The current audit
checks model-dtype Linear outputs, encoded GEMM inputs, shared Q/K/V and gate/up
operands, and direct codes/scales handoff between selected layers. Full decoder
coverage is required by default; the commands above explicitly opt out for
those partial-model checkpoints.

## Configuration

```python
import torch
from gptqmodel import GPTQModel
from gptqmodel.quantization.config import QuantizeConfig

qcfg = QuantizeConfig(
    bits=4, group_size=128, sym=True, desc_act=False,
    activation="w4afp8",  # or "w4a_nvfp4"
    offload_to_disk=False,
)
model = GPTQModel.load(model_path, qcfg, device="cuda", dtype=torch.bfloat16)
model.quantize(calibration_data)
model.save(output_path)
loaded = GPTQModel.load(output_path, device="cuda", dtype=torch.bfloat16)
```

The stream policy selects its own backend when `backend=AUTO` and is
stored in `quantize_config.json`. The stream currently supports
Llama decoder layers on GB10 / SM121 with GPTQ symmetric
INT4, group size 128, contiguous groups, INT32 packing, a Linear input width
divisible by 128, and every selected projection output width divisible by 128.
FP4 calibration stores
one global input scale per selected Linear as exact FP32 bits in an INT32
buffer, preserving it across BF16 model loads. The encoded stream derives a
global scale for each activation tensor and carries it with the packed codes
and per-16 scales. The stored per-Linear scale is used by the standalone
BF16/FP16 Linear compatibility path. Once a model installs the
stream, its selected W4A Linears reject plain BF16/FP16 inputs so an external
caller cannot silently bypass the activation contract.

NVFP4 calibration and replay use the same M=4/M=6 selection as inference.
The global scale reserves the E4M3 range needed to map the largest observed
block to M=4; each local block can still select M=6 when it has lower error.
The block-scale choice is represented entirely by that block's E4M3 scale.
Separately, carriers retain a wider reference for the residual path; native
GEMMs consume only the encoded operand and its scales.
The dynamic global-scale value is rounded on the source activation dtype grid
and then carried and applied as FP32. Making this conversion explicit preserves
the scored BF16 model behavior while keeping all subsequent scale arithmetic
in FP32. If rounding a positive scale to the source grid would produce zero,
the valid FP32 scale is retained so small FP16 operands remain encodable.
Calibration replay uses the same rule.

For a fresh quantization, Llama calibration replay rounds selected
Linear inputs before GPTQ Hessian capture and propagates their effect to
downstream calibration while preserving the wider residual path. Linear
outputs whose next operator uses the model dtype receive no extra QDQ.
Excluded decoder layers remain dense. Fresh GPTQ solves use these replayed
inputs; applying an activation policy to already packed GPTQ weights does not
change those weight tensors.
Calibration replay uses Torch quantize/dequantize tensors because GPTQ's
Hessian collector consumes ordinary tensors; inference passes codes and scales
directly. Replay retains decoded GEMM operands in FP32 to avoid an extra
FP16/BF16 round trip before the GEMM. Projection results return to the model
dtype, matching inference.

## Ordinary quantization compatibility

Omitting `activation` keeps the ordinary GPTQ, AWQ, or weight-only lifecycle.
The saved config omits that field, backend selection keeps its existing order,
and the W4A backends have zero AUTO-selection priority. NVFP4 statistics and
scale probes run only for an explicit NVFP4 policy. The existing kernel
discovery process may import W4A module definitions; selecting an ordinary
backend does not activate their calibration, transport, weight caches, or
export paths.

`tests/models/test_non_w4a_lifecycle.py` guards those opt-in helpers with failing
sentinels while exercising GPTQ, AWQ, and RTN quantize/save/load/re-export,
including lazy offload, FP16/BF16, GPTQ act-order, asymmetric weights, and
rotation. It checks ordinary tensor inputs and outputs, generated tokens,
saved configuration, native packed weights, and bias handling during dense
dequantization. The guarded runner accepts this suite:

```bash
tests/models/run_w4a_gb10_safe.sh tests/models/test_non_w4a_lifecycle.py
```

## Activation policy reference

`activation` is validated and normalized by `ActivationConfig` in
`gptqmodel/quantization/config.py`. A bare mode string is shorthand for that
mode:

```python
activation="w4afp8"     # {"mode": "w4afp8"}
activation="w4a_nvfp4"  # {"mode": "w4a_nvfp4", "recipe": "least_squares"}
```

The full form, showing every supported field:

```python
from gptqmodel.quantization.config import QuantizeConfig

qcfg = QuantizeConfig(
    bits=4, group_size=128, sym=True, desc_act=False, lm_head=False,
    pack_dtype="int32", rotation="hadamard", offload_to_disk=False,
    activation={
        "mode": "w4a_nvfp4",
        "recipe": "least_squares_grid",
        "attention": {"mode": "w4afp8"},
        "mlp": {"mode": "w4afp8", "layers": [11, 15]},
        "global_scales": {"model.layers.0.input": 0.0031034},
    },
)
```

### Fields

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `mode` | `str` | `"w4afp8"` | Default activation format for both projection groups. `w4afp8` or `w4a_nvfp4`. |
| `recipe` | `str \| None` | `None` | NVFP4 block-scale recipe. Only valid when `mode="w4a_nvfp4"`. |
| `attention` | `ActivationAttentionConfig \| None` | `None` | Override for the attention projection group. |
| `mlp` | `ActivationMlpConfig \| None` | `None` | Per-layer override for the MLP projection group. |
| `global_scales` | `dict[str, float] \| None` | `None` | Calibrated producer-to-scale map. Requires NVFP4. |

### Modes

| Mode | Operand | Scale layout |
| --- | --- | --- |
| `w4afp8` | E4M3 codes | One dynamic FP32 scale per token. |
| `w4a_nvfp4` | E2M1 codes | One E4M3 scale per 16-element block, plus one FP32 global scale per tensor. |

`w4afp8` is a per-token FP8 format, not MXFP8 block scaling. `w4a_nvfp4`
follows NVIDIA's block-scaled FP4 operand layout. Neither mode changes the
checkpoint weight codebook.

### Transport contract

There is one consumer-driven transport contract for each mode. An encoded
carrier is emitted only where an FP8/FP4 GEMM or the next decoder layer
actually consumes it; Linear and nonlinear branches otherwise stay in the
model dtype. The residual stream is always carried in compute precision and
only GEMM operands are rounded.

When `rotation` is configured, the rotation folds the RMSNorm weights into the
q/k/v and gate/up projections and resets the norms to unit weights. The NVFP4
stream then reuses the incoming FP4 codes and rescales only the token
multiplier instead of repacking a freshly normed operand. The stream validates
the unit-weight precondition at install time; without a rotation the norm
repacks its operand. The `nvidia_headroom` and `least_squares_headroom` recipes
also repack the normalized operand, even with fused norms: their frozen scales
are calibrated after normalization and must be applied at that same boundary
in replay and inference. Code reuse is governed by rotation, activation mode,
and the boundary's recipe.

### Recipes

`recipe` selects how the per-block E4M3 hardware scale is chosen. It applies to
`w4a_nvfp4` only.

| Recipe | Block-scale rule |
| --- | --- |
| `nvidia` | NVIDIA's max-to-6 rule. No least-squares refinement. |
| `four_six` | Two-candidate M=4/M=6 search. |
| `least_squares` | M=4/M=6 seeds plus monotone least-squares refinement. Default. |
| `least_squares_grid` | `least_squares` plus a bounded search over the positive finite E4M3 bit grid. Reproduces an exhaustive search over all 126 hardware scales. |
| `nvidia_headroom` | `nvidia` plus a calibrated percentile headroom global scale. |
| `least_squares_headroom` | `least_squares` plus a calibrated percentile headroom global scale. |

Legacy aliases `lsq`, `lsq_headroom`, and `lsq_grid` are expanded to their
`least_squares` forms on load. An NVFP4 stream without a `recipe` defaults to
`least_squares`.

### `attention`

```python
"attention": {"mode": "w4afp8"}                    # default; FP8 attention
"attention": {"mode": "w4a_nvfp4"}                 # NVFP4 attention
"attention": {"mode": "w4a_nvfp4", "recipe": "least_squares_grid"}
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `mode` | `str` | `"w4afp8"` | `w4afp8` or `w4a_nvfp4`. |
| `recipe` | `str \| None` | `None` | NVFP4 attention only; inherits the stream `recipe` when omitted. |

The sub-policy only refines an NVFP4 stream. FP8 attention must not carry a
`recipe`.

The default puts attention on FP8 and the MLP on NVFP4. That split is NVIDIA's
own `nvfp4_w4a4_mlp_fp8_attn_max` recipe, and it matches every measurement in
the [accuracy appendix](gptq-w4a-accuracy-notes.md): the 4-bit activation grid
is far more damaging on attention, which is only about a sixth of the
projection work in a Llama decoder.

### `mlp`

```python
"mlp": {"mode": "w4afp8", "layers": [11, 15]}   # promote two decoder layers
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `mode` | `str` | `"w4afp8"` | Promotion only. `w4afp8` is the sole accepted value. |
| `layers` | `list[int]` | `[]` (invalid when supplied as an override) | Non-empty, unique decoder layer indices. Sorted on load; duplicates are rejected. |

This promotes the named decoder layers' MLP boundaries from the stream default
to FP8. The saved INT4 weight tensors are unchanged; only the activation
staging of those layers moves. Promoting every layer is equivalent to an
all-FP8 MLP.

Like `attention`, this only refines an NVFP4 stream. Indices are validated
against the decoder depth at install time.

### `global_scales`

```python
"global_scales": {"model.layers.0.input": 0.0031034, "model.layers.0.mlp.down_proj.input": 0.0024}
```

A producer-to-scale map of calibrated FP32 global scales, keyed by the
producer boundary that emits the tensor. Requires NVFP4, and cannot be
combined with the `nvidia_headroom` or `least_squares_headroom` recipes,
including an attention recipe override. Activation-only calibration uses each
producer's effective recipe: the attention override for attention boundaries
and the stream recipe for NVFP4 MLP boundaries. Its report records that recipe
alongside each fitted scale; FP8 boundaries remain dynamically scaled.
Producer names must start with `model.layers.`, and every scale must be finite
and strictly positive after rounding through FP32.

### Validation rules

`ActivationConfig` rejects, with a specific message:

- an unknown top-level field, including the removed legacy `version` selector
- a missing `mode`, or a `mode` outside the two supported modes
- a `recipe` on an FP8 stream
- `attention` or `mlp` on a non-NVFP4 stream
- an unknown field inside `attention` or `mlp`
- FP8 attention carrying an NVFP4 recipe
- an `mlp.mode` other than `w4afp8`, or an empty, duplicate, or negative `layers`
- `global_scales` on a non-NVFP4 stream, or with an invalid producer name or scale

`QuantizeConfig` additionally requires the weight layout to be GPTQ INT4:
`method="gptq"`, `bits=4`, `group_size=128`, `sym=True`, `desc_act=False`,
`pack_dtype=torch.int32`, `lm_head=False`, and no per-layer weight-layout
override in `dynamic`.

### Serialization

`activation` is stored in `quantize_config.json` in its canonical normalized
form. Loading a checkpoint normalizes legacy aliases and defaults, so a
round-trip through `ActivationConfig.from_value(...).to_dict()` is stable. The
class is the single source of truth for the schema; `QuantizeConfig` stores the
normalized dict so existing serialization and downstream readers are
unchanged.

### Recommended configurations

These are the three configurations that have been measured end to end on
Llama 3.2 1B Instruct over full GSM8K Platinum. The reference is the same
packed INT4 weights with activations left at W4A16.

| Goal | `activation` | GSM8K Platinum | vs W4A16 |
| --- | --- | --- | --- |
| Maximum accuracy | `{"mode": "w4afp8"}` | 0.41853 | +0.33 pp |
| Balanced (recommended) | `{"mode": "w4a_nvfp4", "recipe": "least_squares_grid", "attention": {"mode": "w4afp8"}}` | 0.39206 | -2.32 pp |
| Per-layer promotion | the balanced config plus `"mlp": {"mode": "w4afp8", "layers": [11, 15]}` | 0.39289 | -2.23 pp |

The balanced configuration is the accepted W4A4 operating point: genuine
4-bit NVFP4 activations on the MLP, FP8 on attention, and native GPTQ INT4
weights throughout. It costs 2.32 percentage points against W4A16, which is
the measured price of the 4-bit activation grid on this model.

The per-layer promotion is the best measured point estimate but is not
statistically separable from the balanced configuration at 1,209 rows, so it
is offered as a knob rather than as a recommendation. See the
[accuracy appendix](gptq-w4a-accuracy-notes.md) for the paired statistics.

All three keep the saved weight tensors byte-identical. Switching between them
is a metadata-only change; see the `mlp-view` and `split-view` subcommands in
`tests/models/w4a_quality_regression.py`.

## Validation and performance gate

The GB10 kernel tests compare packed codes and activation rounding with
independent Torch oracles. Quantization values use `rtol=atol=1e-6`; inference
outputs use `rtol=atol=2e-3`. A generated two-layer Llama test covers
quantize, save, reload, and output shape for both policies. The separate
Llama 3.2 1B Instruct tests require an accessible checkpoint, supplied through
`GPTQMODEL_LLAMA3_2_MODEL`. A public BF16 checkpoint is available from
`unsloth/Llama-3.2-1B-Instruct`; `alpindale/Llama-3.2-1B-Instruct` is another
non-Meta source. The two downloaded weight files had the same SHA256
`1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`;
their tokenizer and generation metadata differ, so use one repository
consistently for calibration and evaluation. A local Neural Magic calibration
parquet can be selected with `GPTQMODEL_CALIBRATION_PARQUET`.

```bash
export GPTQMODEL_LLAMA3_2_MODEL=/root/models/Llama-3.2-1B-Instruct-unsloth
export GPTQMODEL_CALIBRATION_PARQUET=/root/models/nm-calibration/llm.parquet
export GPTQMODEL_TEST_PYTHON=/root/gptqmodel-test-venv/bin/python
export GPTQMODEL_W4A_TEST_PHASE=quant-save
export GPTQMODEL_W4AFP8_SAVE_PATH=/root/models/Llama-3.2-1B-Instruct-W4AFP8-stream-v2-fresh
tests/models/run_w4a_gb10_safe.sh tests/models/test_llama3_2_w4afp8.py
unset GPTQMODEL_W4A_TEST_PHASE
tests/models/run_w4a_gb10_safe.sh dtype-audit \
  --checkpoint "$GPTQMODEL_W4AFP8_SAVE_PATH" --variant w4afp8 \
  --output /root/models/w4a-quality/w4afp8_stream_v2_fresh_dtype_audit.json

export GPTQMODEL_W4A_TEST_PHASE=quant-save
export GPTQMODEL_W4A_NVFP4_SAVE_PATH=/root/models/Llama-3.2-1B-Instruct-W4ANVFP4-stream-v2-fresh
tests/models/run_w4a_gb10_safe.sh tests/models/test_llama3_2_w4a_nvfp4.py
unset GPTQMODEL_W4A_TEST_PHASE
tests/models/run_w4a_gb10_safe.sh dtype-audit \
  --checkpoint "$GPTQMODEL_W4A_NVFP4_SAVE_PATH" --variant w4a_nvfp4 \
  --output /root/models/w4a-quality/w4a_nvfp4_stream_v2_fresh_dtype_audit.json
```

Run these tests sequentially through the wrapper. It keeps 2 GiB of physical
or cgroup memory headroom, uses a lock to prevent parallel 1B W4A runs, and
puts the test process in a systemd scope capped at the current headroom minus
2 GiB. `MemoryHigh` matches that live cap and swap is disabled. There is no
unrelated 16, 24, or 32 GiB process ceiling. The tests require this scope
even when invoked directly. The wrapper also stops the scope if host
`MemAvailable` or root-cgroup headroom falls below 2 GiB. GB10 CUDA allocations
share system RAM and may be charged to an ancestor cgroup instead of the test
scope, so the scope cap alone does not guarantee host headroom. Fast ARC
evaluation uses batch size 8 to avoid an oversized automatic batch. If the
guard refuses or stops a run, inspect `free -h`, `/proc/meminfo`, and the root
and app-slice `memory.current`, `memory.stat`, and `memory.events` files before
retrying.

The `quant-save` phase exits after writing a persistent checkpoint. Run the
reload and dtype audit in a second guarded process: retaining quantization
objects while reloading the 1B model exceeded an earlier cgroup limit. GB10 post-quant layer offload stages CUDA
copies through a bounded pinned host buffer to avoid a device-to-host copy
stall observed during FP4 quantization.

The dedicated Llama W4A tests select all 16 decoder layers even in fast mode.
Their save checks require 112 packed projections. The dtype audit also requires
full decoder coverage by default; `--no-require-full-coverage` is an explicit
opt-out for historical partial-model diagnostics. The shared ModelTest harness
otherwise defaults to two layers, so increasing calibration rows alone does
not establish full coverage. A partial fresh FP8 artifact exposed this gap and
was excluded from scoring before rerunning with all layers selected.

Use `GPTQMODEL_W4A_CALIBRATION_ROWS=512` and
`GPTQMODEL_W4A_CALIBRATION_CONCAT_SIZE=2048` for the representative calibration.
The split tests exercise quantize/save, then reload, forward, and cached
generation in the dtype audit. The paired runner scores complete datasets in
separate guarded processes. GSM8K Platinum controls acceptance; ARC is a
diagnostic. A 32-record run checks activation transport and is not a production
calibration recipe.

## Paired full-row quality regression

The paired runner compares a saved W4A16 baseline with the W4A
activation stream. Both lanes read the same `model.safetensors`
file, and the runner checks that rendered prompts, targets, and row order match
before comparing scores. It evaluates all 1,172 ARC-Challenge rows and all
1,209 GSM8K Platinum rows; no `max_rows` or row selector is set.

Full GSM8K Platinum is the downstream acceptance gate because its paired
score has been materially more stable across identical-weight W4A runs. ARC
Challenge remains an optional sensitivity diagnostic for answer and logits
changes; an ARC delta by itself does not accept or reject an activation
format. The comparison command therefore records ARC's statistical result as
`statistical_verdict`, reports `verdict: diagnostic_only`, and exits
successfully for ARC. Only a full GSM8K Platinum comparison can fail the
quality-gate process. Do not replace the full GSM8K run with a row subset.

```bash
CHECKPOINT=/root/models/Llama-3.2-1B-Instruct-W4AFP8-stream-v2
RESULTS=/root/models/w4a-quality
tests/models/run_w4a_gb10_safe.sh quality-eval \
  --checkpoint "$CHECKPOINT" --variant w4afp8 \
  --task arc_challenge --output "$RESULTS/arc_stream_w4afp8.json"
"$GPTQMODEL_TEST_PYTHON" -m tests.models.w4a_quality_regression compare \
  --w4a16 "$RESULTS/arc_w4a16.json" \
  --w4a-float "$RESULTS/arc_stream_w4afp8.json" --task arc_challenge \
  --output "$RESULTS/arc_stream_fp8_paired.json"
tests/models/run_w4a_gb10_safe.sh quality-eval \
  --checkpoint "$CHECKPOINT" --variant w4afp8 \
  --task gsm8k_platinum_cot --output "$RESULTS/gsm_stream_w4afp8.json"
"$GPTQMODEL_TEST_PYTHON" -m tests.models.w4a_quality_regression compare \
  --w4a16 "$RESULTS/gsm_w4a16.json" \
  --w4a-float "$RESULTS/gsm_stream_w4afp8.json" --task gsm8k_platinum_cot \
  --output "$RESULTS/gsm_stream_fp8_paired.json"
"$GPTQMODEL_TEST_PYTHON" -m tests.models.w4a_quality_regression compare \
  --w4a16 "$RESULTS/arc_w4a16.json" \
  --w4a-float "$RESULTS/arc_stream_w4afp8.json" \
  --task arc_challenge --metric accuracy,loglikelihood_norm \
  --output "$RESULTS/arc_stream_fp8_norm_paired.json"
```

The comparison reports the paired 95% interval, counts of rows gained and
lost, and how many answers changed. For GSM8K Platinum, its provisional
regression gate passes only when the interval rules out
a loss greater than 2 percentage points; uncertain and confirmed larger losses
fail. This tests the incremental cost of activation
rounding on one fixed GPTQ checkpoint; it does not claim full-model quality
from the fast two-layer quantization configuration. A separately saved FP4
checkpoint can be evaluated the same way with `--variant w4a_nvfp4`, using a
W4A16 view prepared from that FP4 checkpoint.

Keep the W4A16 per-row JSON result as baseline data for subsequent activation
experiments. The comparison computes its score from that file; it does not use
an observed score as a fixed pass threshold. Llama's chat template inserts the
current date into each prompt, so freeze the date before comparing runs. The
dated view shares the original `model.safetensors` file and changes only the
date expression in its chat template:

```bash
"$GPTQMODEL_TEST_PYTHON" -m tests.models.w4a_quality_regression freeze-date \
  --checkpoint /root/models/Llama-3.2-1B-Instruct-W4AFP8-gptq \
  --view /root/models/Llama-3.2-1B-Instruct-W4AFP8-quality-2026-09-25 \
  --baseline "$RESULTS/gsm_w4a16.json"
```

Evaluate the dated view with `quality-eval`. Paired comparison rejects any row
whose rendered prompt or target differs from the baseline, including a changed
date.

### Encoded-stream results on migrated fast checkpoints

These runs carry codes and scales across selected decoder modules and between
layers 14 and 15. They reuse the version 1 INT4 weights and the saved W4A16
per-row baselines, so they isolate the inference-stream change on those
weights. ARC uses all 1,172 rows; GSM8K Platinum uses all 1,209 rows. The FP8
full-row runs began immediately before the version 2 metadata gate was added;
the runtime stream code and saved weight file were the same, and the version 2
view passed a separate encoded-boundary and cached-generation audit.

| Stream | Task | W4A16 | Encoded stream | Change (pp) | Answer changes | 2-point gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| FP8 | ARC-Challenge | 0.3157 | 0.3148 | −0.09 | 35 | Within budget |
| FP8 | GSM8K Platinum | 0.4748 | 0.5004 | +2.56 | 366 | Within budget |
| NVFP4 | ARC-Challenge | 0.3114 | 0.3148 | +0.34 | 161 | Within budget |
| NVFP4 | GSM8K Platinum | 0.4665 | 0.4582 | −0.83 | 604 | Inconclusive |

The FP4 GSM8K paired 95% interval is −3.20 to +1.54 points, so the 2-point
loss threshold is unresolved despite the point estimate being within it. The
normalized ARC metric is also inconclusive for FP4 (−0.68 points, interval
−2.23 to +0.87); the standard ARC metric above is within budget. Do not treat
the migrated FP4 stream as quality-accepted on these data alone.

### Full decoder coverage calibration diagnostics

Fresh version 2 checkpoints quantized all 16 Llama decoder layers, covering
112 W4A projections. The paired W4A16 lane shares each activation checkpoint's
packed INT4 tensors. ARC-Challenge uses all 1,172 rows and GSM8K Platinum all
1,209 rows. These runs expose the full-layer activation effect; they are not
quality acceptance results because the W4A16 control scores vary sharply with
calibration size and remain below the saved two-layer W4A16 reference. The
recorded checkpoints used no calibration concatenation, unlike the standard
Llama 3.2 test's 2,048-token concatenation. The W4A e2e test defaults now match
that standard; collect new scores before treating these diagnostics as the
representative recipe.

| Calibration rows | Activation | Task | W4A16 | W4A | Change (pp) | Paired 95% interval (pp) | Gate |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 32 | FP8 | ARC-Challenge | 0.3080 | 0.2995 | −0.85 | [−2.39, +0.68] | Inconclusive |
| 32 | FP8 | GSM8K Platinum | 0.3168 | 0.2415 | −7.53 | [−10.01, −5.05] | Regression |
| 32 | NVFP4 | ARC-Challenge | 0.3020 | 0.2560 | −4.61 | [−7.36, −1.85] | Inconclusive |
| 32 | NVFP4 | GSM8K Platinum | 0.3358 | 0.0116 | −32.42 | [−35.13, −29.71] | Regression |
| 256 | FP8 | ARC-Challenge | 0.2858 | 0.3003 | +1.45 | [−0.02, +2.92] | Within budget |
| 256 | FP8 | GSM8K Platinum | 0.0397 | 0.1191 | +7.94 | [+5.94, +9.94] | Within budget |

The 32-row FP8 GSM8K pair shows an activation-related regression for those
weights. The 32-row NVFP4 checkpoint regresses on both complete tasks, with a
particularly large GSM8K loss, and is rejected. With 256 calibration rows the paired FP8 lane scores higher than its
W4A16 control on both tasks, but both GSM8K scores are poor. Do not select a
calibration recipe from these pairs alone. The per-row data and paired
summaries are saved in `/root/models/w4a-quality/` with `full32` or `full256`
in their filenames. The full-coverage dtype audits are also saved there; they
verify all decoder layers and all 112 W4A Linear boundaries.

A subsequent 32-record FP8 run used the corrected 2,048-token calibration
concatenation and passed the strict full-coverage audit. It still failed the
GSM8K quality gate:

| Activation | Task | Rows | W4A16 | W4A | Change (pp) | Paired 95% interval (pp) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP8 | ARC-Challenge | 1,172 | 0.3020 | 0.2935 | −0.85 | [−2.46, +0.75] | Inconclusive |
| FP8 | GSM8K Platinum | 1,209 | 0.3242 | 0.2779 | −4.63 | [−7.10, −2.16] | Regression |

This checkpoint is rejected for quality. A representative FP8 run then used
all 512 calibration records, 188,256 non-padding calibration tokens, and the
same 2,048-token concatenation. It quantized all 16 decoder layers and passed
the strict 112-projection encoded-stream audit:

| Activation | Task | Rows | W4A16 | W4A | Change (pp) | Paired 95% interval (pp) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP8 | ARC-Challenge | 1,172 | 0.3089 | 0.3166 | +0.77 | [−0.60, +2.14] | Within budget |
| FP8 | GSM8K Platinum | 1,209 | 0.3706 | 0.3292 | −4.14 | [−6.69, −1.58] | Inconclusive |

ARC has 29 W4A16-only and 38 FP8-only correct rows. GSM8K has 150 W4A16-only
and 100 FP8-only correct rows, 250 correctness flips, and 732 extracted-answer
changes. The GSM8K drop is statistically detectable, but its confidence
interval does not establish that the true loss exceeds the 2-point gate. This
recipe therefore remains unaccepted pending a more precise or improved FP8
recipe. Results use `full512_concat2048_fp8` in their filenames under
`/root/models/w4a-quality/`; the strict audit is
`w4afp8_stream_v2_full512_concat2048_dtype_audit.json`.

The representative NVFP4 run used the same 512 records, 188,256 non-padding
tokens, 2,048-token concatenation, and full 16-layer coverage. It passed the
strict packed-FP4 stream audit but failed the quality gate:

| Activation | Task | Rows | W4A16 | W4A | Change (pp) | Paired 95% interval (pp) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| NVFP4 | ARC-Challenge | 1,172 | 0.2952 | 0.2765 | −1.88 | [−4.46, +0.70] | Inconclusive |
| NVFP4 | GSM8K Platinum | 1,209 | 0.3921 | 0.0422 | −34.99 | [−37.80, −32.17] | Regression |

ARC has 130 W4A16-only and 108 NVFP4-only correct rows. GSM8K has 436
W4A16-only and 13 NVFP4-only correct rows, 449 correctness flips, and 1,137
extracted-answer changes. This NVFP4 recipe is rejected. Results use
`full512_concat2048_nvfp4` in their filenames under
`/root/models/w4a-quality/`; the strict audit is
`w4a_nvfp4_stream_v2_full512_concat2048_dtype_audit.json`.

The M=4/M=6 per-block activation search was then evaluated on the identical
checkpoint and frozen prompts. This isolates the runtime scale-selection
change; the GPTQ INT4 tensors were not recalibrated or repacked. It improved
both tasks relative to the preceding max-scaled A4 runtime, but did not repair
the full-stream GSM8K loss:

| Activation scale selection | Task | Rows | W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| M=4/M=6 block MSE | ARC-Challenge | 1,172 | 0.2952 | 0.2850 | −1.02 | [−3.52, +1.47] | Inconclusive |
| M=4/M=6 block MSE | GSM8K Platinum | 1,209 | 0.3921 | 0.0554 | −33.66 | [−36.52, −30.81] | Regression |

Compared with the earlier max-scaled A4 runtime, ARC rose from 0.2765 to
0.2850 and GSM8K rose from 0.0422 to 0.0554. ARC had 117 W4A16-only and 105
W4A4-only correct rows. GSM8K had 427 W4A16-only and 20 W4A4-only correct
rows, with 1,123 extracted-answer changes. The M=4/M=6 experiment is therefore
rejected as a complete A4 quality solution. Its per-row and paired outputs use
`four_six` in their filenames under `/root/models/w4a-quality/`.

A fixed trace on the same checkpoint showed that the scale search still
improved numerical propagation: final-layer relative RMSE fell from 0.6193 to
0.5828 and logits relative RMSE from 0.5608 to 0.4669. The quality result shows
that this numerical improvement is too small to make repeated full-stream A4
rounding acceptable by itself. The next A4 recipe must change calibration or
precision placement while preserving the encoded-stream contract and native
GPTQ INT4 checkpoint.

The GPTQ solve was then repeated from scratch with the same M=4/M=6 activation
replay propagated through all 16 decoder layers. This used 512 calibration
records, 188,256 non-padding tokens, 2,048-token concatenation, and all 112
selected projections. The resulting checkpoint preserves the native packed
INT4 tensors and passed the strict full-stream dtype audit. Full-row paired
evaluation produced:

| Activation scale selection | Task | Rows | W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh M=4/M=6 replay | ARC-Challenge | 1,172 | 0.3063 | 0.2628 | −4.35 | [−6.89, −1.81] | Inconclusive |
| Fresh M=4/M=6 replay | GSM8K Platinum | 1,209 | 0.3763 | 0.0620 | −31.43 | [−34.26, −28.60] | Regression |

ARC has 142 W4A16-only and 91 W4A4-only correct rows, 233 correctness flips,
and 411 selected-answer changes. Its loss is statistically detectable, while
the paired interval narrowly crosses the provisional −2-point gate. GSM8K has
402 W4A16-only and 22 W4A4-only correct rows, 424 correctness flips, and 1,115
extracted-answer changes. Re-solving the GPTQ weights under exact Four-Over-Six
replay therefore does not repair end-to-end A4 quality and remains rejected as
the default recipe. The checkpoint is
`Llama-3.2-1B-Instruct-W4A-NVFP4-four-six-full512-concat2048`; result files use
`four_six_fresh_full512_concat2048` under `/root/models/w4a-quality/`.

Four-Over-Six remains useful as a common FP4 packer policy. It can be reused
for layer activation carriers, one-time shared Q/K/V and gate/up input packing,
future FP4 KV-cache blocks, and scale-calibration diagnostics. These uses emit
ordinary FP4 codes plus E4M3 block scales and require no selector metadata.
They do not apply directly to the exact GPTQ INT4 weight planes, and they must
retain independent end-to-end quality gates.

An additional matched experiment changed the dynamic global-scale denominator
from `4 * 448` to NVIDIA NVFP4's full `6 * 448` range, then repeated the
512-record GPTQ solve and both full-row tasks. The fixed-prompt logits trace
improved relative to the preceding matched checkpoint, but task quality did
not:

| Global-scale denominator | Task | Rows | W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `6 * 448` | ARC-Challenge | 1,172 | 0.3131 | 0.2637 | −4.95 | [−7.55, −2.35] | Regression |
| `6 * 448` | GSM8K Platinum | 1,209 | 0.3656 | 0.0587 | −30.69 | [−33.50, −27.87] | Regression |

ARC has 151 W4A16-only and 93 W4A4-only correct rows. GSM8K has 393
W4A16-only and 22 W4A4-only correct rows, with 1,123 extracted-answer
changes. The implementation therefore retains `4 * 448`: it keeps both M=4
and M=6 candidate local scales inside E4M3 range before comparing their block
reconstruction errors. Results for the rejected experiment use
`four_six_scale6_full512_concat2048` under `/root/models/w4a-quality/`.

### Least-squares-refined E4M3 block scales

The explicit `nvidia` recipe implements the TorchAO/NVIDIA reference formula:
one tensor scale `amax / (6 * 448)`, followed by per-16 `amax / 6` scales
rounded to E4M3 and E2M1 value rounding. It is retained as the exact format
baseline. `four_six` and `least_squares` are hardware-valid scale-selection extensions;
they still emit ordinary E2M1 values and E4M3 block scales.

The hardware-scale refinement evaluates the original M=4 and M=6 seeds,
solves the least-squares scalar for each E2M1 code assignment, rounds every
candidate back to E4M3, and accepts only a lower-SSE candidate. An independent
GB10 test verifies that block SSE never increases relative to Four-Over-Six,
and the fused Triton packer emits the same FP4 codes and E4M3 scale bytes as
the Torch oracle.

A fresh 64-record, 27,455-token GPTQ solve used the refined replay, Hadamard
rotation, all 16 decoder layers, and all 112 projections. Full GSM8K Platinum
evaluation at batch size 32 produced:

| Activation scale selection | Rows | W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E4M3 least-squares refinement | 1,209 | 0.4136 | 0.1009 | −31.27 | [−34.24, −28.29] | Regression |

The refined lane improves the preceding corrected 64-record A4 score from
0.0885 to 0.1009. It has 416 W4A16-only correct rows, 38 W4A4-only correct
rows, 454 correctness flips, and 1,095 extracted-answer changes. The local
scale improvement is retained because it is format-correct and monotone, but
it does not make repeated full-stream A4 rounding acceptable. Result files
use `gsm_lsrefine_hadamard_triton_b32_full64` under
`/root/models/w4a-quality/`.

### Single-pack rotated down-projection operand

The first refined runtime encoded the MLP product before the online Hadamard
transform, then decoded, rotated, and encoded it again inside `down_proj`.
Because that product has no other consumer, runtime and GPTQ replay now apply
Hadamard first and create one FP4 carrier for `down_proj`. The carrier records
that the projection rotation has already been applied, and the Linear rejects
that marker unless it has the matching online-rotation contract. Focused tests
verify that neither Hadamard nor activation rounding runs a second time.

A fresh replay-matched 64-record solve covered all 16 decoder layers and all
112 projections. The strict dtype audit passed. Full GSM8K Platinum at batch
size 32 produced:

| Activation boundary recipe | Rows | W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LS E4M3 scales + single-pack rotated MLP operand | 1,209 | 0.4318 | 0.1208 | −31.10 | [−33.99, −28.21] | Regression |

The A4 lane has 405 W4A16-only correct rows, 29 A4-only correct rows, 434
correctness flips, and 1,050 extracted-answer changes. Removing the duplicate
rounding improved A4 from 0.1009 to 0.1208, while the remaining loss confirms
that duplicate MLP packing was only one part of the accumulated full-stream
A4 error. Result files use
`gsm_lsrefine_singlepack_hadamard_b32_full64` under
`/root/models/w4a-quality/`.

### Consumer-driven stream

Version 2 still encoded every selected Linear output. Q/K/V were quantized and
immediately decoded for attention; gate/up were quantized and immediately
decoded for SiLU and multiplication; o/down were quantized and immediately
decoded for residual addition. These conversions did not feed another native
low-precision operator and accumulated avoidable error.

Version 3 makes the next consumer determine the output representation. Every
selected GEMM still requires and directly consumes an encoded FP8/FP4 carrier.
Q/K/V, gate/up, o, and down emit BF16 directly into operators that only support
wider arithmetic. RMSNorm outputs, attention-to-o inputs, MLP-product-to-down
inputs, residual outputs, and decoder-to-decoder handoffs remain encoded. This
is the minimum lossy boundary set that satisfies the end-to-end carrier
contract with the currently implemented operators. Version 2 remains
loadable so existing checkpoint numerics do not change silently.

On the unchanged single-pack GPTQ weights, the fixed-prompt final-logit
relative RMSE fell from 0.6672 to 0.2784, cosine rose from 0.7449 to 0.9632,
and the next token matched W4A16. Full GSM8K Platinum confirmed a large but
incomplete recovery:

| Lifecycle | Rows | Matched W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Version 3 view of version 2 weights | 1,209 | 0.4318 | 0.3052 | -12.66 | [-15.45, -9.86] | 235 / 82 | Regression |

The version 3 view recovered 223 correct answers over version 2, from 146 to
369, while sharing the exact same `model.safetensors` file. It changed 831
extracted answers relative to W4A16. This establishes redundant Linear-output
rounding as the dominant version 2 quality defect.

A nominal fresh version 3 GPTQ solve used 64 calibration records, 27,455
non-padding tokens, 2,048-token concatenation, Hadamard rotation, all 16
decoder layers, and all 112 projections. It exposed a lifecycle defect:
`HookedLinear.from_linear()` copied replay mode and recipe but not replay
version, so Hessian capture silently retained version 2 Linear-output QDQ.
The runtime checkpoint used version 3, making this result diagnostic rather
than a valid fresh version 3 solve:

| Lifecycle | Rows | Matched W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Nominal version 3; version 2 Hessian replay | 1,209 | 0.4624 | 0.3135 | -14.89 | [-17.74, -12.04] | 258 / 78 | Invalid lifecycle |

The lifecycle now propagates `activation_version` through dense Linear,
`HookedLinear`, GPTQ Hessian capture, export, and runtime. A corrected solve
used the same calibration setup. Its strict audit passed all 16 decoder
layers, 112 W4A Linears, 16 attention modules, 16 MLP modules, and 33 RMSNorm
modules, including cached generation. The checkpoint retains native
INT32-packed GPTQ INT4 weights:

| Lifecycle | Rows | Matched W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Corrected version 3 Hessian replay | 1,209 | 0.4334 | 0.3011 | -13.23 | [-16.05, -10.42] | 241 / 81 | Regression |

The corrected pair changed 824 extracted answers. On the fixed trace prompt,
both lanes selected token 18820; the final-logit Jensen-Shannon divergence was
0.0130. Correct replay therefore fixes the lifecycle contract but does not
close the activation quality gap. Result files use
`gsm_nvfp4_consumer_stream_v3` and
`gsm_nvfp4_consumer_stream_v3_fresh64` for the earlier runs and
`gsm_nvfp4_consumer_stream_v3_hookfix_fresh64` for the corrected run under
`/root/models/w4a-quality/`. Further QAD work uses version 3; the earlier
version 2 full-parameter experiments are retained only as historical evidence.

The strongest fully calibrated FP8 checkpoint was also given a metadata-only
version 3 view. The view reuses its exact packed GPTQ weight file and changes
only the activation-stream lifecycle metadata. Its strict audit passed all 16
decoder layers, 112 W4A Linears, 16 attention modules, 16 MLP modules, and 33
RMSNorm modules. A batch-size-32 W4A16 control and W4AFP8 lane then evaluated
the same 1,209 GSM8K Platinum rows with the same prompt date:

| Lifecycle | Rows | Matched W4A16 | W4A8 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A8-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Version 3 view of full-512 FP8 weights | 1,209 | 0.3656 | 0.3548 | -1.08 | [-3.47, +1.32] | 116 / 103 | Inconclusive |

The FP8 pair has 219 correctness flips and 646 extracted-answer changes.
McNemar's exact p-value is 0.4175, so the run does not detect a regression;
the interval still crosses the provisional -2-point quality boundary, so it
also cannot accept the recipe. Result files use
`gsm_fp8_consumer_stream_v3_full512_concat2048` under
`/root/models/w4a-quality/`, including the paired report and strict dtype
audit. This supersedes comparisons against the older batch-size-8 W4A16 run.

A fresh version 3 FP8 solve then used 512 calibration records, 188,256
non-padding tokens, 2,048-token concatenation, and all 112 projections across
16 decoder layers. Saved weights remain INT32-packed GPTQ INT4. Unlike the
metadata-only view above, this checkpoint uses version 3 during both Hessian
capture and inference. Its strict audit passed full coverage, shared Q/K/V
and gate/up carriers, layer handoffs, and cached generation.

| Lifecycle | Rows | Matched W4A16 | W4A8 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A8-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Fresh version 3 FP8, 512 records | 1,209 | 0.3573 | 0.3648 | +0.74 | [-1.54, +3.03] | 95 / 104 | Within budget |

The matched lanes scored 432 and 441 correct rows respectively, with 199
correctness flips and 637 extracted-answer changes. The lower confidence bound
is above the provisional -2-point limit. This passes the activation regression
gate for these weights; it does not establish a statistically significant
accuracy gain or an absolute model-quality target. Both lanes used batch size
32 and the same rendered prompts dated 27 Sep 2026.

Per-row results and comparison use
`gsm_fp8_consumer_stream_v3_fresh_full512_concat2048` under
`/root/models/w4a-quality/`. The reusable baseline manifest is
`w4afp8_stream_v3_fresh_full512_gsm_baseline_manifest.json`; it records hashes
of the packed weights, configuration, results, audit, and modified source files.
Reuse its W4A16 rows only with identical weights and evaluation settings,
including the captured prompt date. A4 recovery and full-model performance
remain separate pending requirements.

### NVIDIA activation headroom lifecycle

The headroom recipes add a calibration-only pass before each GPTQ subset. It
collects one maximum per 16-value activation block in a bounded log2
histogram, then applies NVIDIA's default policy:

`calibrated_amax = max(16384 * percentile(block_amax, 1),
percentile(block_amax, 99.99))`.

The resulting FP32 global scale is `calibrated_amax / (6 * 448)`. Each Linear
stores that scalar as exact INT32 bits. The frozen value is used in the second
pass that captures the GPTQ Hessian and at the matching runtime input
boundary. Q/K/V share one calibrated carrier, as do gate/up. The checkpoint
still stores native INT32-packed GPTQ INT4 weights; activation scale
calibration does not change the weight codebook.

Two fresh 64-record, 27,455-token solves used Hadamard rotation, all 16 layers,
all 112 projections, and the same full 1,209-row GSM8K Platinum test. ARC was
not used because it is diagnostic only. The first combines headroom with the
exact NVIDIA max-to-6 local scale. The second keeps NVIDIA's global headroom
lifecycle and replaces only local max-to-6 selection with the monotone E4M3
least-squares rule:

| Activation recipe | W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NVIDIA headroom + max-to-6 | 0.4119 | 0.0571 | -35.48 | [-38.35, -32.62] | 447 / 18 | Regression |
| NVIDIA headroom + LS E4M3 | 0.4136 | 0.0968 | -31.68 | [-34.51, -28.84] | 405 / 22 | Regression |

The least-squares local rule recovers 48 correct answers over max-to-6
(117 versus 69), but remains a confirmed material regression against its
matching 500-correct W4A16 control. It changes 1,075 extracted answers and has
McNemar exact p-value `2.32e-92`. The strict carrier audit passes all 16
decoder layers, 112 Linears, 16 attention blocks, 16 MLPs, and 33 RMSNorms,
so this result measures the required end-to-end A4 contract rather than an
input-only simulation.

The reusable baseline and comparison are pinned in
`w4a_nvfp4_lsq_headroom_full64_gsm_baseline_manifest.json` under
`/root/models/w4a-quality/`. Its dated checkpoint view freezes the prompt date
to 26 Sep 2026 and shares the same `model.safetensors` file. Later activation
experiments on these exact packed weights must reuse the saved 500/1209
W4A16 row file and run all 1,209 A4 rows. The result confirms that global and
local scale selection alone cannot remove the cumulative error from
materializing A4 carriers at projection, residual, and layer boundaries.

### Boundary error and norm adaptation

A guarded reconstruction profiler measured 209 materialized A4 boundaries on
the full 16-layer stream. RMSNorm outputs average about 2.6% relative RMSE,
while Q/K/V outputs, attention-to-O inputs, projection outputs, and MLP
product-to-down inputs are generally near 8%. NVIDIA headroom and dynamic least-squares fitting
produce essentially the same intrinsic boundary error. This explains why
changing only the local or global scale rule reaches a quality plateau: the
error is accumulated at many separate projection and nonlinear boundaries.

Two norm-only adaptations kept every `qweight`, `qzeros`, `scales`, and
`g_idx` tensor byte-identical. The first trained all 33 RMSNorm vectors with a
causal language-model objective. Full GSM8K Platinum rejected it:

| Adaptation | Rows | W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32-step causal norm QAT | 1,209 | 0.4078 | 0.1092 | -29.86 | [-32.77, -26.95] | 395 / 34 | Regression |

It changed 1,065 extracted answers and has McNemar exact p-value `4.45e-79`.
The second adaptation used the matching dequantized GPTQ model as a teacher,
disabled activation rounding only while capturing teacher states, and trained
against relative reconstruction error at every decoder layer. On 32 training
and eight held-out calibration records, held-out reconstruction loss improved
from 0.04999 to 0.04709 while the packed GPTQ digest remained unchanged. It
did not advance to the downstream gate: a fixed trace retained a different
next token and logit JS divergence 0.349. The stronger-looking causal norm
candidate had JS divergence 0.071 on that trace and still failed full GSM8K,
so another full run was not justified by the screening evidence.

ARC is excluded from both adaptation decisions. Full 1,209-row GSM8K Platinum
remains the downstream acceptance gate; layer traces and held-out calibration
loss are screening diagnostics only.

### GPTQ scale reconstruction and scale-only QAD

A one-shot scale reconstruction experiment solved new GPTQ group scales against
the activation-rounded model while preserving the centered INT4 code planes.
It failed both sides of the paired gate: W4A16 scored 376/1,209 (0.3110) and
W4A4 scored 91/1,209 (0.0753), a -23.57 point change. Reconstructing every
scale at once moved the native W4A16 function too far, so this is not a viable
post-init conversion.

The next experiment adapted NVIDIA's scale-only QAD and Dual-LSQ trust-region
idea to native GPTQ. The teacher was the unchanged W4A16 model; only relative
deltas on existing per-group GPTQ scales were trainable. A4 replay used the
real E2M1/E4M3 path with a straight-through gradient, while a separate W4A16
loss limited drift. Selection used held-out records and exported multipliers
against the original serialized FP16 scales. `qweight`, `qzeros`, and `g_idx`
remained byte-identical.

The selected step reduced held-out A4 distillation loss by 9.5% and logit KL
by 12.8%. Its fixed trace restored the teacher's next token and raised final
logit cosine from 0.7449 to 0.8215. Full GSM8K Platinum did not confirm that
local improvement:

| Adaptation | Rows | W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GPTQ scale-only QAD | 1,209 | 0.4318 | 0.1199 | -31.18 | [-34.11, -28.26] | 410 / 33 | Regression |

The W4A16 output was exactly preserved at 522 correct rows, while A4 lost one
correct row relative to the unadapted 146/1,209 baseline. Scale-only QAD is
therefore retained as a diagnostic and export-invariant test, not as the A4
recovery recipe. Reports use `gsm_scale_qad_p1024_r64_v8_s64_t128` under
`/root/models/w4a-quality/`.

### E4M3 grid oracle and full-parameter QAD

An exhaustive diagnostic evaluated all 126 positive finite E4M3 local scales
on 4,096 activation blocks from each of layers 0, 7, 14, and 15. Relative to
the coordinate-refined least-squares rule, exact block SSE fell by 10.8%, 9.8%, 14.0%,
and 12.4%, respectively. A bounded search over eight adjacent E4M3 bit steps
recovered the exhaustive optimum on all four samples. The explicit `least_squares_grid`
recipe performs that search directly on hardware scale encodings; its Torch
and Triton paths produce identical FP4 code and scale bytes. This changes
activation scales only and retains the native packed GPTQ weight representation.

The local grid search costs additional packing work. The GB10 layer benchmark
for 2,048 by 2,048 weights changed from approximately 110.07 to 150.31 us at
one token and from 128.39 to 156.80 us at 128 tokens. It is an accuracy-first
candidate, but its all-row result did not generalize the block and fixed-trace
improvements: A4 scored 145/1,209 (0.1199), one row below the default least-squares
result of 146/1,209. Against W4A16 at 522/1,209, the change is -31.18 points
with paired 95% interval [-34.16, -28.20]. `least_squares` therefore remains the default;
`least_squares_grid` is retained as an explicit diagnostic recipe. A sparse 13-candidate
set recovers 92.2% to more than 99.9% of the measured oracle gain if a future
workload shows a downstream benefit worth optimizing. The
exhaustive and neighborhood reports are
`gsm_nvfp4_exhaustive_scale_oracle.json` and
`gsm_nvfp4_e4m3_neighborhood_oracle.json` under
`/root/models/w4a-quality/`.

Scale-only methods leave the INT4 code assignment fixed. The next recovery
stage follows NVIDIA's current full-parameter QAD recipe while retaining GPTQ
storage. Its optimization view uses latent FP32 weights with an INT4
straight-through forward, true A4 replay, a matched W4A16 preservation loss,
and held-out trust-region selection. Export rounds to `[-8, 7]`, repacks only
standard INT32 `qweight`, and digest-checks `qzeros` and `g_idx`. Saved weights
remain ordinary group-128 GPTQ INT4 rather than an FP4 weight codebook.

The first bounded full-parameter trial trained only decoder layer 15. It used
16 training records, four held-out records, 64-token sequences, and selected
step 96 of 128. Export changed 30,599 INT4 values while preserving every
`qzeros` and `g_idx` tensor. Held-out A4 loss improved from 0.19591 to 0.19483,
and the fixed-prompt final-logit cosine against its matched W4A16 view improved
from 0.7449 to 0.7850. The all-row downstream result rejected that local gain:

| Adaptation | Rows | Matched W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Layer 15 weight QAD | 1,209 | 0.4351 | 0.1050 | -33.00 | [-35.84, -30.17] | 418 / 19 | Regression |

The candidate raised its matched W4A16 result from the original checkpoint's
522 correct rows to 526, but reduced A4 from 146 to 127 correct rows. A local
held-out reconstruction improvement and one fixed trace are therefore
insufficient selection criteria. The QAD tool now accepts an exact inclusive
decoder range through `--first-trainable-layer` and
`--last-trainable-layer`; subsequent experiments can adapt one early layer at
a time without exposing all 973 million weight values to the optimizer. Full
GSM8K remains mandatory before an exported stage can become the source for a
later stage. Reports use `gsm_weight_qad_l15_s96` under
`/root/models/w4a-quality/`.

The optimizer was then tightened for an exact layer-0 experiment. Its
regularizer measures movement in INT4 code-cell units rather than physical
weight units, and it stops at the first held-out W4A16 preservation breach.
The blockwise objective compares only the selected decoder output, avoiding
the noisy full-model gradient observed when one early layer is trainable. With
64 training records, 16 held-out records, 128-token sequences, and an exact
layer-0 range, step 64 changed 788 INT4 values. Held-out layer-0 A4 error fell
from 0.0186000 to 0.0183574 while W4A16 error remained inside its trust region.
One fixed prompt also improved final-logit cosine from 0.7449 to 0.8544 and
restored the reference next token. Full GSM8K again rejected the local signal:

| Adaptation | Rows | Matched W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Layer 0 blockwise weight QAD | 1,209 | 0.4293 | 0.1133 | -31.60 | [-34.55, -28.64] | 417 / 35 | Regression |

The layer-0 candidate preserved the weight-only reference at 519 correct rows,
but A4 scored 137 correct rows, nine fewer than the unadapted 146/1,209
baseline. It changed 1,064 extracted answers and produced 452 correctness
flips. This stage is not used as the source for layer 1. The two failed
full-parameter trials show that held-out block reconstruction and single-prompt
logit traces cannot select weight updates for this stream. Further recovery
work returns to activation-boundary diagnosis before changing more GPTQ codes.
Reports use `gsm_weight_qad_block_l0_s64` under
`/root/models/w4a-quality/`.

### Full-model master-weight QAD

The full-model QAD path now initializes its latent weights from the original
BF16 checkpoint after applying the same RMSNorm fusion and Hadamard rotation
as GPTQ. Each master value is projected smoothly into the interior quarter of
its existing GPTQ code cell. Step zero therefore rounds to the exact saved
INT4 code while retaining the master residual's direction. The regularizer is
centered on this projected master position. Export still changes only packed
INT32 `qweight`; `qzeros` and `g_idx` are digest checked.

The first robust master-initialized trial used 64 training records, 16 held-out
records, 128-token sequences, and all 973,078,528 quantized weight values. At
step 15 it changed 20 INT4 values, all in layer 0 `v_proj`. Held-out A4 loss
fell from 0.10897 to 0.10549 and its fixed-trace logit JS divergence improved
from 0.01298 to 0.00849. Full GSM8K Platinum did not reduce the matched gap:

| Adaptation | Rows | Matched W4A16 | W4A4 | Change (pp) | Paired 95% interval (pp) | W4A16-only / A4-only | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full-model master QAD, 20 code changes | 1,209 | 0.4351 | 0.3027 | -13.23 | [-16.07, -10.40] | 243 / 83 | Regression |

Both lanes gained two correct answers over the unadapted corrected version 3
pair, leaving the 160-row score gap unchanged. The candidate changed 825
extracted answers between its matched lanes and is rejected as a recovery
checkpoint. Results use
`gsm_nvfp4_v3_masterqad_r64_v16_s128_t24_lr5e6` under
`/root/models/w4a-quality/`.

The trainer also supports sequential gradient accumulation, linear warmup,
cosine decay, logits-only teacher distillation, optional continuation after a
W4A16 trust-region breach, and omission of unused hidden-state targets. An
eight-way accumulation run used 1,024 effective tokens per optimizer step. A
strict preservation run selected 24 code changes but worsened fixed-trace JS
divergence to 0.10957. A 96-step continuation run established that stopping
at the first discrete transition was not hiding later recovery: held-out A4
KL was best at step 40 with 14 code changes, then worsened as the count rose to
15,365. The retained step-40 trace had JS divergence 0.03105, also worse than
the unadapted checkpoint, so neither candidate advanced to full GSM8K.

These trials match NVIDIA's master-weight and logits-distillation structure as
far as this bounded local corpus permits, while adding the GPTQ INT4 export
constraint. Their sparse accepted updates and worsening later transitions
show that native GPTQ code adaptation is not closing the remaining version 3
activation gap. Further A4 work should target the activation-boundary policy
or use a substantially larger QAD corpus; these QAD checkpoints must not be
used as new baselines.

### Recorded input-only prototype results (version 1)

These earlier results used input-only activation rounding. They are retained
as a historical comparison and do not validate the version 2 activation
stream. They use all test rows but quantize only the last two of 16 decoder
layers. Each W4A16 lane uses the same packed-weight file as its activation
lane; the FP8 and FP4 checkpoints have different packed GPTQ weights.

| Activation | Task | Rows | W4A16 | W4A | Change (pp) | Correctness flips | Answer changes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 | ARC-Challenge | 1,172 | 0.3157 | 0.3166 | +0.09 | 11 | 19 |
| FP8 | GSM8K Platinum | 1,209 | 0.4748 | 0.4648 | −0.99 | 60 | 218 |
| NVFP4 | ARC-Challenge | 1,172 | 0.3114 | 0.3174 | +0.60 | 27 | 56 |
| NVFP4 | GSM8K Platinum | 1,209 | 0.4665 | 0.4665 | 0.00 | 142 | 413 |

The saved per-row JSON and baseline manifests are in
`/root/models/w4a-quality/` on the GB10 test machine. The FP8 GSM8K paired
95% interval for score change is −2.25 to +0.26 percentage points, so its
provisional 2-point gate is inconclusive. FP4 GSM8K is within that gate, yet
changes many individual answers. A repeat W4A16 run on the FP4 checkpoint
produced identical generated text on all 1,209 rows. The recorded scores are
observations, not fixed pass thresholds. ARC answer changes count selected
choices; GSM8K answer changes count extracted numbers.

Run the benchmark under the same memory guard with
`tests/models/run_w4a_gb10_safe.sh benchmark`. CUDA graph replay on GB10,
Torch 2.11+cu130, Triton 3.6, BF16 inputs and 2048×2048 weights measured
approximately:

| Tokens | W4AFP8 | W4A NVFP4 | Dense BF16 |
| ---: | ---: | ---: | ---: |
| 1 | 16.55 µs | 110.07 µs | 9.18 µs |
| 128 | 53.16 µs | 128.39 µs | 17.50 µs |

These are layer microbenchmarks, not full-model throughput. One Triton kernel
packs FP4 activation codes and swizzles E4M3 scales; another fuses both FP4
plane results with the original GPTQ group scale and final output conversion.
The two exact FP4 weight planes are stored side by side in one nonpersistent
post-init cache, allowing one native FP4 GEMM per 128-wide GPTQ group. The
speed goal is not yet met. A fused SM121 GEMM should consume that cache,
perform all group products, apply the original GPTQ scales after each group,
and write one result. Keep the independent oracle and saved INT4 checkpoint
invariants as acceptance gates.
