<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# GPT-QModel agent guidance

This file governs the whole repository in addition to any parent agent instructions.

## Product naming

Use **GPT-QModel** in user-facing prose, UI labels, documentation headings, and
release titles/notes. Keep `gptqmodel` for the package/import name and
`GPTQModel` for the Python API class. Preserve actual identifiers, filenames,
repository names, and URLs when editing display text.

## Explicit accelerator families

For device selection, model loading/placement, accelerator checks, or kernel
`SUPPORTS_*` declarations, read and follow
`.agents/skills/gptqmodel-device-families/SKILL.md` before editing.

GPTQModel uses `DEVICE.CUDA` for NVIDIA and `DEVICE.ROCM` for AMD in kernel
support declarations. PyTorch's HIP runtime also calls its AMD devices `cuda`;
that runtime spelling is not a hardware-support declaration. Keep ROCm explicit
in GPTQModel configuration and support-family decisions, resolving to PyTorch's
namespace only at execution/placement boundaries. Centralize HIP detection in
the family resolver instead of scattering `cuda` plus HIP checks across callers.
Kernels must self-declare supported hardware, including `DEVICE.ROCM` when
supported; never add ROCm implicitly because CUDA or Triton is available.
Preserve concrete ordinals separately and validate every selected GPU, including
on mixed-vendor or mixed-architecture hosts.

## Quantized inference weight storage

All inference kernels must keep weights in their quantized representation
between uses. Never retain a dense, dequantized, or expanded weight cache after
its consuming operation completes. This applies to every quantization method,
device, backend, optimized path, and fallback, including compiled/JIT paths.
An INT4-to-INT8 expansion is also an expanded weight cache, even though it is
integer-valued. Packed quantized layouts and their scales, zero points,
codebooks, and indexing metadata may persist.

Temporary dequantization is allowed for the current computation; prefer bounded
tiles and release references after use. Moving an expanded cache to another
device, marking a buffer non-persistent, or hiding it behind an opt-in flag does
not satisfy this rule. A speedup cannot justify violating quantized storage.

For inference implementation, dispatch, or weight-cache changes, read and follow
the persistent-storage validation requirements in
`.agents/skills/gptqmodel-torch-oracle-numerics/SKILL.md` as well as its numerical
requirements. Apply this rule when auditing existing inference kernels too.

## Numerical oracle validation

Fusion backend changes must preserve AUTO's native kernel selection. Add an
explicit native-layout contract and actual-device numerical tests for each
newly admitted kernel; a portable substitution or mocked dispatch test does
not qualify that backend. Cover supported activation dtypes, unequal QKV
widths, gate/up, native shape boundaries, workspace/descriptor lifecycle,
mutation, direct source calls, exact packed-weight restoration and persistent
storage. Keep the complete kernel inventory and measured device report current;
unavailable architectures/dependencies and other formats require explicit
coverage reasons. See `tests/module_tree/test_fusion_kernel_backends.py`.

For any new or changed quantization math, weight packer, or inference kernel,
read and follow `$gptqmodel-torch-oracle-numerics` in
`.agents/skills/gptqmodel-torch-oracle-numerics/SKILL.md` before editing.

For every MLX inference kernel, test FP16 and BF16 activations independently.
Assert that each supported path returns the input activation dtype, and measure
accuracy drift against an independent Torch oracle for both dtypes. Follow the
skill's output-rounding guidance and report arithmetic error separately from
the visible low-precision output error.

## Tokenizer normalization ownership

For tokenizer initialization, tokenization normalization, special-token compatibility, prompt rendering, chat-template correctness, or unexpectedly low inference/evaluation scores, read and follow `$gptqmodel-tokenizer-normalization` in `.agents/skills/gptqmodel-tokenizer-normalization/SKILL.md` before editing.

GPT-QModel depends on [ModelCloud/Tokenicer](https://github.com/ModelCloud/Tokenicer). Put reusable corrective tokenizer and chat-template normalization in Tokenicer, add relevant tests, increment its version, run its complete test suite, then commit, push, and open a Tokenicer pull request. Do not leave model-loader patches in GPT-QModel merely to avoid fixing the dependency.

Establish a non-quantized baseline before treating bad generation or evaluation scores as a quantization regression. Compare exact rendered prompts and input IDs as well as aggregate scores.

## Logic-flow change documentation

For any change that alters control flow, replaces a decision mechanism (for
example name matching replaced by declared metadata), changes resolution or
selection order, changes data or tensor lifecycle, or introduces a fail-closed
contract, read and follow `$gptqmodel-logic-flow-diagrams` in
`.agents/skills/gptqmodel-logic-flow-diagrams/SKILL.md` before opening the pull
request.

Such a change is not reviewable from the diff alone. The pull request
description must contain, at minimum, a Mermaid diagram of the new flow drawn
from the shipped code, a before/after view of the mechanism being removed with
`file:line` evidence, and an explicitly marked failure/ambiguity path. The same
diagrams must be committed as a reference page under `docs/`, titled for the
mechanism rather than the pull request, and linked from `README.md` where the
related feature is described.

Diagrams must use real symbol names and measured numbers. Do not draw intended
design, estimate test results, or include follow-up work as if implemented. Keep
diagrams in sync with the code: a later change to the same flow updates the page
in the same pull request.

## Model-definition weight coverage

For every new model type or alias in `MODEL_MAP`, and every change to a model
definition or module-plan consumer, follow
`.agents/skills/gptqmodel-model-plan-tests/SKILL.md`. New model support must add
an explicit registry plan case and a model-specific module-plan unit test in
the same PR. Cover its layout variants and method overrides. The registry
contract runs without an opt-in flag and must not skip supported definitions.

`module_tree` is the oracle for checkpoint-owned weights, including
non-quantized weights. Use `state_dict()` (parameters and persistent buffers)
for model architecture fixtures, not `named_modules()`, `named_parameters()`,
or `named_buffers()` alone. For an actual checkpoint audit, verify ownership
against checkpoint keys and declared conversion mappings: tensors created
after loading must not become definition obligations solely because they
appear in a runtime `state_dict()`.

For every checkpoint-owned decoder-layer entry:

- The deepest owning module must resolve through `ModulePlan.find()`.
- A quantized owner must be declared by `simple_layer_modules()`.
- A non-quantized owner must carry an explicit `:!` or `:?` tag.

Add real-model regression coverage for new or changed definitions. The Flash
family reference is
`tests/module_tree/test_flash_model_tree_consumption.py`; it builds the actual
meta model and checks quantizable paths, weight owners, roles, MoE hooks, and
looper consumers.
