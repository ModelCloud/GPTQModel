# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Signed ABI manifest for the NF12 kernel surface.

The schema a kernel is registered with *is* its ABI: argument order, dtypes and
defaults are what a caller compiles against. This module records the schema of
every operator the CUDA (``torch.ops.nf12``) and ROCm (``torch.ops.nf12_hip``)
libraries register, derives a stable ``signature`` from it, and names the
MLX/Metal variant that covers the same operation, if one exists.

``tests/test_nf12_kernel_manifest.py`` re-parses both registries and fails when
a kernel's schema changes without the manifest being re-signed, so a stale
prebuilt library cannot be loaded against a newer Python side without the
mismatch being visible. The same test pins :data:`MLX_GAPS`, so an operation
cannot lose (or silently gain) an MLX variant without an explicit edit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


# Bump when the meaning of an existing signature changes (not when a new
# operator is added).
ABI_VERSION = 1

_ROOT = Path(__file__).resolve().parent.parent
CUDA_SOURCES = ("gptqmodel_ext/nf12/csrc/torch/ops.cpp",)
HIP_SOURCES = ("gptqmodel_ext/nf12_hip/nf12_torch.cpp",)


@dataclass(frozen=True)
class Kernel:
    """One registered NF12 operator."""

    namespace: str
    schema: str
    sources: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.schema.split("(", 1)[0]

    @property
    def qualified(self) -> str:
        return f"{self.namespace}.{self.name}"

    @property
    def signature(self) -> str:
        """Digest of the ABI contract, stable across implementation edits."""
        payload = f"{ABI_VERSION}\n{self.qualified}\n{self.schema}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _kernels(namespace: str, schemas: tuple[str, ...], sources: tuple[str, ...]) -> tuple[Kernel, ...]:
    return tuple(Kernel(namespace=namespace, schema=schema, sources=sources) for schema in schemas)


CUDA_SCHEMAS = (
    "encode(Tensor w) -> (Tensor packed, Tensor escapes, int exp_base)",
    "decode(Tensor packed, Tensor escapes, int exp_base) -> Tensor",
    "decode_fp32(Tensor packed, Tensor escapes, int exp_base) -> Tensor",
    "sub_fp32(Tensor packed, Tensor escapes, int exp_base, Tensor wq) -> Tensor",
    "linear(Tensor x, Tensor packed, Tensor escapes, int exp_base, Tensor? bias=None, "
    "*, bool static_weights=False) -> Tensor",
    "linear_staged_b1(Tensor x, Tensor packed, Tensor escapes, int exp_base, Tensor? bias=None) -> Tensor",
    "pack_bf16t(Tensor w) -> Tensor",
    "linear_bf16t(Tensor x, Tensor w_bf16t, Tensor? bias=None) -> Tensor",
    "embedding(Tensor ids, Tensor packed, Tensor escapes, int exp_base) -> Tensor",
    "embedding_repeat4(Tensor ids, Tensor packed, Tensor escapes, int exp_base) -> Tensor",
    "rowmajor_pack(Tensor packed, Tensor escapes, int exp_base) -> "
    "(Tensor row_packed, Tensor row_escapes, int original_escapes, int collisions)",
    "rowmajor_embedding(Tensor ids, Tensor row_packed, Tensor row_escapes, int exp_base) -> Tensor",
    "encode_moe(Tensor w) -> (Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base)",
    "moe_align(Tensor topk_ids, int num_experts) -> (Tensor sorted, Tensor expert_offsets)",
    "moe_decode(Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base) -> Tensor",
    "moe_linear(Tensor x, Tensor packed, Tensor escapes, Tensor escape_offsets, Tensor exp_base, "
    "Tensor expert_offsets, Tensor? gather=None, int gather_div=1, Tensor? scatter=None, "
    "Tensor? row_scale=None, bool zero_fill=False) -> Tensor",
    "moe_linear_bf16t(Tensor x, Tensor w_bf16t, Tensor expert_offsets, Tensor? gather=None, "
    "int gather_div=1, Tensor? scatter=None, Tensor? row_scale=None, bool zero_fill=False) -> Tensor",
    "moe_combine(Tensor y, int topk) -> Tensor",
    "moe_silu_mul(Tensor h) -> Tensor",
    "_fused_moe(Tensor x, Tensor packed13, Tensor? escapes13, Tensor? escape_offsets13, "
    "Tensor? exp_base13, Tensor packed2, Tensor? escapes2, Tensor? escape_offsets2, "
    "Tensor? exp_base2, Tensor topk_weights, Tensor topk_ids, bool zero_fill=False) -> Tensor",
)

HIP_SCHEMAS = (
    "decode(Tensor packed, Tensor escapes, int exp_base) -> Tensor",
    "decode_fp32(Tensor packed, Tensor escapes, int exp_base) -> Tensor",
    "sub_fp32(Tensor packed, Tensor escapes, int exp_base, Tensor wq) -> Tensor",
    "embedding(Tensor ids, Tensor packed, Tensor escapes, int exp_base) -> Tensor",
    "rowmajor_pack(Tensor packed, Tensor escapes, int exp_base) -> "
    "(Tensor row_packed, Tensor row_escapes, int original_escapes, int collisions)",
    "rowmajor_embedding(Tensor ids, Tensor row_packed, Tensor row_escapes, int exp_base) -> Tensor",
    "linear(Tensor x, Tensor packed, Tensor escapes, int exp_base, Tensor? bias=None) -> Tensor",
    "encode(Tensor w) -> (Tensor packed, Tensor escapes, int exp_base)",
    "selftest(Tensor ref) -> bool",
    "arch_info() -> str",
)

CUDA_KERNELS = _kernels("nf12", CUDA_SCHEMAS, CUDA_SOURCES)
HIP_KERNELS = _kernels("nf12_hip", HIP_SCHEMAS, HIP_SOURCES)
KERNELS = CUDA_KERNELS + HIP_KERNELS

# ``operation -> MLX/Metal implementation``. The value is ``module:attribute``
# so the test can resolve it; ``None`` means the Metal backend has no variant.
MLX_VARIANTS: dict[str, str | None] = {
    "encode": None,  # host-side codec in nn_modules/qlinear/nf12.py
    "decode": "gptqmodel.nn_modules.qlinear.mlx_nf12:_decode_kernel",
    "decode_fp32": "gptqmodel.nn_modules.qlinear.mlx_nf12:MlxNF12Linear",
    "sub_fp32": None,
    "linear": "gptqmodel.nn_modules.qlinear.mlx_nf12:MlxNF12Linear",
    "linear_staged_b1": "gptqmodel.nn_modules.qlinear.mlx_nf12:MlxNF12Linear",
    "pack_bf16t": None,
    "linear_bf16t": None,
    "embedding": "gptqmodel.nn_modules.qlinear.mlx_nf12:MlxNF12Embedding",
    "embedding_repeat4": None,
    "rowmajor_pack": None,
    "rowmajor_embedding": None,
    "encode_moe": None,
    "moe_align": None,
    "moe_decode": None,
    "moe_linear": None,
    "moe_linear_bf16t": None,
    "moe_combine": None,
    "moe_silu_mul": None,
    "_fused_moe": None,
    "selftest": None,
    "arch_info": None,
}

# Operations with no MLX/Metal variant yet. Pinned by the test so the gap list
# only changes on purpose.
MLX_GAPS = frozenset(name for name, variant in MLX_VARIANTS.items() if variant is None)


def signatures() -> dict[str, str]:
    """Return ``namespace.op -> signature`` for every registered kernel."""
    return {kernel.qualified: kernel.signature for kernel in KERNELS}


def verify() -> None:
    """Raise when the manifest no longer matches the checked-in registries."""
    for kernel in KERNELS:
        for source in kernel.sources:
            if not (_ROOT / source).is_file():
                raise FileNotFoundError(f"nf12: kernel source missing: {source}")
    missing = set(MLX_VARIANTS) - {kernel.name for kernel in KERNELS}
    if missing:
        raise ValueError(f"nf12: MLX variant entries for unknown kernels: {sorted(missing)}")
