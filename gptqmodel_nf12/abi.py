# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned upstream NF12 ABI, so this vendored copy cannot drift silently.

The ABI we consume is the one upstream *exposes*: the declarations in
``include/nf12/nf12.h``, the torch operator schemas in ``csrc/torch/ops.cpp``,
the exported symbol map, and the public Python names. This module records that
surface for ``dphnAI/nf12@97c410154`` (NF12 1.1) and
:func:`verify` re-parses the vendored copies to prove they still match.

Additions are expected and allowed -- this tree adds ``decode_fp32``,
``sub_fp32``, the ``rowmajor_*`` family and the staged-linear entry points -- but
every name upstream exposes keeps its signature, struct layout, schema and
export. ``tests/test_nf12_abi_parity.py`` enforces that; ``tools/generate_nf12_abi.py``
regenerates this file after :data:`UPSTREAM_COMMIT` moves.
"""

from __future__ import annotations

import re
from pathlib import Path


UPSTREAM_REPOSITORY = "dphnAI/nf12"
UPSTREAM_COMMIT = '97c4101541a8d92d2b3d89c910c452cfa8efc9c9'
UPSTREAM_VERSION = (1, 1)

# ``name -> "return name(args)"`` for every function upstream declares.
C_FUNCTIONS: dict[str, str] = {
    'nf12_bf16t_bytes': 'size_t nf12_bf16t_bytes(int64_t n, int64_t k)',
    'nf12_decode': 'nf12_status_t nf12_decode(const nf12_weight_t* w, void* out_bf16, int64_t ldo, cudaStream_t stream)',
    'nf12_embedding': 'nf12_status_t nf12_embedding(const nf12_weight_t* w, const int32_t* ids, int64_t num_ids, void* out_bf16, int64_t ldo, cudaStream_t stream)',
    'nf12_embedding_i64': 'nf12_status_t nf12_embedding_i64(const nf12_weight_t* w, const int64_t* ids, int64_t num_ids, void* out_bf16, int64_t ldo, cudaStream_t stream)',
    'nf12_encode': 'nf12_status_t nf12_encode(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base, void* packed_out, void* escapes_out, uint32_t escapes_capacity, void* workspace, size_t workspace_bytes, cudaStream_t stream)',
    'nf12_encode_analyze': 'nf12_status_t nf12_encode_analyze(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, void* workspace, size_t workspace_bytes, uint8_t* exp_base_out, uint32_t* num_escapes_out, cudaStream_t stream)',
    'nf12_encode_workspace_bytes': 'size_t nf12_encode_workspace_bytes(int64_t n, int64_t k)',
    'nf12_linear': 'nf12_status_t nf12_linear(const void* x_bf16, int64_t m, int64_t ldx, const nf12_weight_t* w, const void* bias_bf16, void* y_bf16, int64_t ldy, void* workspace, size_t workspace_bytes, cudaStream_t stream)',
    'nf12_linear_bf16t': 'nf12_status_t nf12_linear_bf16t(const void* x_bf16, int64_t m, int64_t ldx, const void* w_bf16t, int64_t n, int64_t k, const void* bias_bf16, void* y_bf16, int64_t ldy, void* workspace, size_t workspace_bytes, cudaStream_t stream)',
    'nf12_linear_decode_gemm_rows': 'int64_t nf12_linear_decode_gemm_rows(int64_t m, int64_t n, int64_t k, int device)',
    'nf12_linear_static_weights': 'nf12_status_t nf12_linear_static_weights(const void* x_bf16, int64_t m, int64_t ldx, const nf12_weight_t* w, const void* bias_bf16, void* y_bf16, int64_t ldy, void* workspace, size_t workspace_bytes, cudaStream_t stream)',
    'nf12_linear_workspace_bytes': 'size_t nf12_linear_workspace_bytes(int64_t m, int64_t n, int64_t k, int device)',
    'nf12_moe_align': 'nf12_status_t nf12_moe_align(const int32_t* topk_ids, int64_t num_tokens, int64_t topk, int64_t num_experts, int32_t* sorted_out, int64_t* expert_offsets_out, void* workspace, size_t workspace_bytes, cudaStream_t stream)',
    'nf12_moe_align_workspace_bytes': 'size_t nf12_moe_align_workspace_bytes(int64_t num_tokens, int64_t topk, int64_t num_experts)',
    'nf12_moe_combine': 'nf12_status_t nf12_moe_combine(const void* y_bf16, int64_t ldy, int64_t num_tokens, int64_t topk, int64_t n, void* out_bf16, int64_t ldo, cudaStream_t stream)',
    'nf12_moe_decode': 'nf12_status_t nf12_moe_decode(const nf12_moe_weight_t* w, const int64_t* expert_offsets, int64_t max_assignments, int64_t row0, int64_t rows, void* out_bf16, int64_t ldo, cudaStream_t stream)',
    'nf12_moe_grouped_operands': 'nf12_status_t nf12_moe_grouped_operands(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div, const int64_t* expert_offsets, int64_t max_assignments, int64_t num_experts, int64_t k, void* x_out_bf16, int64_t ldo, int32_t* group_ends, cudaStream_t stream)',
    'nf12_moe_linear': 'nf12_status_t nf12_moe_linear(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div, const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments, const nf12_moe_weight_t* w, const float* row_scale, void* y_bf16, int64_t ldy, void* workspace, size_t workspace_bytes, cudaStream_t stream)',
    'nf12_moe_linear_bf16t': 'nf12_status_t nf12_moe_linear_bf16t(const void* x_bf16, int64_t ldx, const int32_t* gather, int64_t gather_div, const int32_t* scatter, const int64_t* expert_offsets, int64_t max_assignments, const void* w_bf16t, int64_t num_experts, int64_t n, int64_t k, const float* row_scale, void* y_bf16, int64_t ldy, void* workspace, size_t workspace_bytes, cudaStream_t stream)',
    'nf12_moe_linear_decode_gemm_rows': 'int64_t nf12_moe_linear_decode_gemm_rows(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k, int device)',
    'nf12_moe_linear_workspace_bytes': 'size_t nf12_moe_linear_workspace_bytes(int64_t max_assignments, int64_t num_experts, int64_t n, int64_t k, int device)',
    'nf12_moe_silu_mul': 'nf12_status_t nf12_moe_silu_mul(const void* h_bf16, int64_t ldh, int64_t rows, int64_t inter, void* out_bf16, int64_t ldo, cudaStream_t stream)',
    'nf12_pack_bf16t': 'nf12_status_t nf12_pack_bf16t(const void* w_bf16, int64_t n, int64_t k, int64_t ldw, void* out, cudaStream_t stream)',
    'nf12_packed_bytes': 'size_t nf12_packed_bytes(int64_t n, int64_t k)',
    'nf12_status_string': 'const char* nf12_status_string(nf12_status_t status)',
}

# ``struct -> (field declarations in order)``; order and types are the layout.
STRUCTS: dict[str, tuple[str, ...]] = {
    'nf12_moe_weight_t': ('const void* packed', 'const void* escapes', 'const int64_t* escape_offsets', 'const uint8_t* exp_base', 'int64_t num_experts', 'int64_t n', 'int64_t k'),
    'nf12_weight_t': ('const void* packed', 'const void* escapes', 'uint32_t num_escapes', 'int64_t n', 'int64_t k', 'uint8_t exp_base'),
}

STATUS: dict[str, int] = {
    'NF12_ERROR_CUDA': 4,
    'NF12_ERROR_INSUFFICIENT_WORKSPACE': 3,
    'NF12_ERROR_INVALID_ARGUMENT': 1,
    'NF12_ERROR_UNSUPPORTED': 2,
    'NF12_OK': 0,
}

# ``namespace -> schemas`` as registered with torch.library.
TORCH_OPS: dict[str, tuple[str, ...]] = {
    "nf12": (
        'encode(Tensor w) -> (Tensor packed, Tensor escapes, int exp_base)',
        'decode(Tensor packed, Tensor escapes, int exp_base) -> Tensor',
        'linear(Tensor x, Tensor packed, Tensor escapes, int exp_base, Tensor? bias=None, *, bool static_weights=False) -> Tensor',
        'pack_bf16t(Tensor w) -> Tensor',
        'linear_bf16t(Tensor x, Tensor w_bf16t, Tensor? bias=None) -> Tensor',
        'embedding(Tensor ids, Tensor packed, Tensor escapes, int exp_base) -> Tensor',
        'encode_moe(Tensor w) -> (Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base)',
        'moe_align(Tensor topk_ids, int num_experts) -> (Tensor sorted, Tensor expert_offsets)',
        'moe_decode(Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base) -> Tensor',
        'moe_linear(Tensor x, Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base, Tensor expert_offsets, Tensor? gather=None, int gather_div=1, Tensor? scatter=None, Tensor? row_scale=None, bool zero_fill=False) -> Tensor',
        'moe_linear_bf16t(Tensor x, Tensor w_bf16t, Tensor expert_offsets, Tensor? gather=None, int gather_div=1, Tensor? scatter=None, Tensor? row_scale=None, bool zero_fill=False) -> Tensor',
        'moe_combine(Tensor y, int topk) -> Tensor',
        'moe_silu_mul(Tensor h) -> Tensor',
        '_fused_moe(Tensor x, Tensor packed13, Tensor? escapes13, Tensor? escape_offsets13, Tensor? exp_base13, Tensor packed2, Tensor? escapes2, Tensor? escape_offsets2, Tensor? exp_base2, Tensor topk_weights, Tensor topk_ids, bool zero_fill=False) -> Tensor',
    ),
}

# Glob patterns the version script keeps global; everything else is local.
EXPORT_MAP: tuple[str, ...] = ('nf12_*',)

# Public names of the upstream ``nf12`` package (its ``__all__``).
PYTHON_EXPORTS: tuple[str, ...] = (
    'library_path',
    'load',
    'NF12Weight',
    'decode',
    'embedding',
    'encode',
    'linear',
    'linear_bf16t',
    'pack_bf16t',
    'NF12MoEWeight',
    'encode_moe',
    'fused_moe',
    'moe_align',
    'moe_combine',
    'moe_linear',
    'moe_linear_bf16t',
    'moe_silu_mul',
    'pack_bf16t_moe',
    'NF12Embedding',
    'NF12Linear',
    'verify_weights',
)

# ``str(inspect.signature(...))`` for the upstream callables above.
PYTHON_SIGNATURES: dict[str, str] = {
    'NF12Embedding': "(num_embeddings: 'int', embedding_dim: 'int', device=None) -> 'None'",
    'NF12Linear': "(in_features: 'int', out_features: 'int', bias: 'bool' = True, device=None) -> 'None'",
    'NF12MoEWeight': "(packed: 'torch.Tensor', escapes: 'torch.Tensor', escape_offsets: 'torch.Tensor', exp_base: 'torch.Tensor') -> None",
    'NF12Weight': "(packed: 'torch.Tensor', escapes: 'torch.Tensor', exp_base: 'int') -> None",
    'decode': "(weight: 'NF12Weight') -> 'torch.Tensor'",
    'embedding': "(ids: 'torch.Tensor', weight: 'NF12Weight') -> 'torch.Tensor'",
    'encode': "(w: 'torch.Tensor') -> 'NF12Weight'",
    'encode_moe': "(w: 'torch.Tensor') -> 'NF12MoEWeight'",
    'fused_moe': "(x: 'torch.Tensor', w13: 'NF12MoEWeight | torch.Tensor', w2: 'NF12MoEWeight | torch.Tensor', topk_weights: 'torch.Tensor', topk_ids: 'torch.Tensor', activation: 'str' = 'silu', *, expert_map: 'torch.Tensor | None' = None) -> 'torch.Tensor'",
    'library_path': "(name: 'str | None' = None) -> 'Path'",
    'linear': "(x: 'torch.Tensor', weight: 'NF12Weight', bias: 'torch.Tensor | None' = None, *, static_weights: 'bool' = False) -> 'torch.Tensor'",
    'linear_bf16t': "(x: 'torch.Tensor', w_bf16t: 'torch.Tensor', bias: 'torch.Tensor | None' = None) -> 'torch.Tensor'",
    'load': "() -> 'None'",
    'moe_align': "(topk_ids: 'torch.Tensor', num_experts: 'int') -> 'tuple[torch.Tensor, torch.Tensor]'",
    'moe_combine': "(y: 'torch.Tensor', topk: 'int') -> 'torch.Tensor'",
    'moe_linear': "(x: 'torch.Tensor', weight: 'NF12MoEWeight | torch.Tensor', expert_offsets: 'torch.Tensor', *, gather: 'torch.Tensor | None' = None, gather_div: 'int' = 1, scatter: 'torch.Tensor | None' = None, row_scale: 'torch.Tensor | None' = None, zero_fill: 'bool' = False) -> 'torch.Tensor'",
    'moe_linear_bf16t': "(x: 'torch.Tensor', w_bf16t: 'torch.Tensor', expert_offsets: 'torch.Tensor', *, gather: 'torch.Tensor | None' = None, gather_div: 'int' = 1, scatter: 'torch.Tensor | None' = None, row_scale: 'torch.Tensor | None' = None, zero_fill: 'bool' = False) -> 'torch.Tensor'",
    'moe_silu_mul': "(h: 'torch.Tensor') -> 'torch.Tensor'",
    'pack_bf16t': "(w: 'torch.Tensor') -> 'torch.Tensor'",
    'pack_bf16t_moe': "(w: 'torch.Tensor') -> 'torch.Tensor'",
    'verify_weights': "(model) -> 'None'",
}

ABI_DIGEST = '768f1d20f90bd7d7'

_ROOT = Path(__file__).resolve().parent.parent
HEADER = "gptqmodel_ext/nf12/include/nf12/nf12.h"
TORCH_OPS_SOURCE = "gptqmodel_ext/nf12/csrc/torch/ops.cpp"
EXPORT_MAP_SOURCE = "gptqmodel_ext/nf12/csrc/nf12.map"


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def parse_header(path: Path | str):
    """Return ``(functions, structs, status, version)`` parsed from an ``nf12.h``."""
    text = Path(path).read_text()
    body = _strip_comments(text)
    functions = {}
    for match in re.finditer(
        r"\b(nf12_\w+|size_t|int64_t|const char\*)\s+(nf12_\w+)\s*\(([^;{]*?)\)\s*;", body, re.S
    ):
        functions[match.group(2)] = f"{match.group(1)} {match.group(2)}(" + " ".join(match.group(3).split()) + ")"
    structs = {}
    for match in re.finditer(r"typedef struct (\w+) \{(.*?)\} (\w+);", body, re.S):
        structs[match.group(3)] = tuple(" ".join(x.split()) for x in match.group(2).split(";") if x.strip())
    status = {key: int(value) for key, value in re.findall(r"(NF12_\w+) = (\d+)", body)}
    version = tuple(int(x) for x in re.search(r"MAJOR (\d+).*?MINOR (\d+)", text, re.S).groups())
    return functions, structs, status, version


def parse_torch_ops(path: Path | str) -> tuple[str, ...]:
    """Return the operator schemas registered by a ``TORCH_LIBRARY`` source."""
    found = []
    for match in re.finditer(r"\bm\.def\((.*?)\)\s*;", Path(path).read_text(), re.S):
        schema = " ".join("".join(re.findall(r'"([^"]*)"', match.group(1))).split())
        if schema:
            found.append(schema)
    return tuple(found)


def parse_export_map(path: Path | str) -> tuple[str, ...]:
    return tuple(re.findall(r"(nf12_\*)", Path(path).read_text()))


def digest() -> str:
    """Stable digest of the pinned surface, for release notes and CI logs."""
    return ABI_DIGEST


def verify() -> None:
    """Raise when the vendored copy stops matching the pinned upstream ABI."""
    functions, structs, status, version = parse_header(_ROOT / HEADER)
    if version != UPSTREAM_VERSION:
        raise ValueError(f"nf12 ABI: vendored header is version {version}, pinned {UPSTREAM_VERSION}")
    missing = sorted(set(C_FUNCTIONS) - set(functions))
    if missing:
        raise ValueError(f"nf12 ABI: vendored header is missing upstream functions {missing}")
    drift = [name for name, signature in C_FUNCTIONS.items() if functions[name] != signature]
    if drift:
        raise ValueError(f"nf12 ABI: signature drift in {drift}")
    for name, fields in STRUCTS.items():
        if structs.get(name) != fields:
            raise ValueError(f"nf12 ABI: struct {name} layout drift: {structs.get(name)}")
    for name, value in STATUS.items():
        if status.get(name) != value:
            raise ValueError(f"nf12 ABI: status {name} is {status.get(name)}, pinned {value}")
    missing_ops = sorted(set(TORCH_OPS["nf12"]) - set(parse_torch_ops(_ROOT / TORCH_OPS_SOURCE)))
    if missing_ops:
        raise ValueError(f"nf12 ABI: vendored torch ops are missing upstream schemas {missing_ops}")
    if parse_export_map(_ROOT / EXPORT_MAP_SOURCE) != EXPORT_MAP:
        raise ValueError("nf12 ABI: export map drifted from upstream")


def verify_python_surface(package) -> None:
    """Raise when the package drops a public name or re-signs an upstream callable."""
    import inspect

    missing = sorted(set(PYTHON_EXPORTS) - set(package.__all__))
    if missing:
        raise ValueError(f"nf12 ABI: package is missing upstream exports {missing}")
    for name, expected in PYTHON_SIGNATURES.items():
        actual = str(inspect.signature(getattr(package, name)))
        if actual != expected:
            raise ValueError(f"nf12 ABI: {name} signature {actual} != upstream {expected}")
