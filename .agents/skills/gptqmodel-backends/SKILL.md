---
name: gptqmodel-backends
description: Add, tune, review, or debug GPT-QModel quantized inference backends and QuantLinear implementations, including capability declarations, automatic selection priorities, explicit backend requests, dependency checks, packing, post-initialization, and fallback behavior.
---

# GPT-QModel inference backends

Make backend selection a truthful capability contract. A backend must never be selected for a method, format, shape, dtype, device, or platform it cannot execute correctly.

Read [references/backend-contract.md](references/backend-contract.md) before adding a backend or changing selection priority.

## Define the contract first

Write down the intended matrix before coding:

- quantization `METHOD` and serialized `FORMAT`;
- bits, group sizes, symmetry, and `desc_act` states;
- input/output features and automatic-padding rules;
- pack dtype and activation dtype;
- device, platform, sharding, training, and adapter support;
- optional dependency, minimum architecture, and fallback;
- auto-selection priority versus explicit-only support.

If any row is unknown, mark it unsupported until a test establishes it.

## Follow the repository's discovery path

1. Add a public `BACKEND` value and normalization aliases in `gptqmodel/utils/backend.py` only when exposing a new backend name.
2. Derive the implementation from the semantically correct base in `gptqmodel/nn_modules/qlinear/__init__.py`: grouped, packed, weight-only, GPTQ, AWQ, or another existing contract.
3. Declare every relevant `SUPPORTS_*` field. `SUPPORTS_FORMATS` maps formats to priorities: a positive value participates in automatic selection; zero or a negative value is explicit-only.
4. Implement `validate_once`, per-request validation, availability checks, packing, `post_init`, and cleanup as the backend requires. Error messages must state the rejected capability.
5. Let `gptqmodel/utils/importer.py` discover subclasses and build support maps. Do not add a parallel hand-maintained registry or edit generated maps.
6. If native code is required, use `$gptqmodel-cuda-kernels` as well and register the extension through the existing JIT path.

Keep wrappers thin. Normalize and validate at the Python boundary, then launch the kernel on the current device and stream. Do not hide import failures that explain why an explicit backend is unavailable.

## Prove selection and execution

Add tests for:

1. hierarchy and declarative-field validation;
2. positive and negative selection cases, including explicit-only priority;
3. missing dependency, unsupported device/architecture, and fallback behavior;
4. numerical output against the dense or dequantized reference across boundary shapes;
5. pack/save/reload and post-initialization state;
6. coexistence with at least one competing backend so priority changes are intentional.

Start with `tests/kernels/test_selection.py`, `tests/kernels/test_qlinear_hierarchy.py`, and `tests/kernels/test_fallback.py`, then add method- and kernel-specific coverage. Benchmark only after correctness passes; keep timed runs in `scripts/` and include both prefill-like and decode-like shapes when relevant.
