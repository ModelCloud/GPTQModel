# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The NF12 ABI manifest must match the kernels that register it."""

from __future__ import annotations

import hashlib
import importlib
import re
from pathlib import Path

import pytest

from gptqmodel_nf12 import kernels


ROOT = Path(__file__).resolve().parent.parent

# Digest of the whole ``namespace.op -> signature`` table. It changes when an
# operator's schema changes, which is the point: a kernel ABI edit has to show
# up in this file, so a prebuilt library cannot drift from the Python side.
SIGNED_ABI_DIGEST = "119e36690516f1ea"

# Every operation without an MLX/Metal variant. Kept explicit so the gap list
# only moves on purpose; removing a name here is how a variant gets claimed.
MLX_GAPS = frozenset({
    "_fused_moe",
    "arch_info",
    "embedding_repeat4",
    "encode",
    "encode_moe",
    "linear_bf16t",
    "moe_align",
    "moe_combine",
    "moe_decode",
    "moe_linear",
    "moe_linear_bf16t",
    "moe_silu_mul",
    "pack_bf16t",
    "rowmajor_embedding",
    "rowmajor_pack",
    "selftest",
    "sub_fp32",
})


def _registered(path: str, namespace: str) -> list[str]:
    """Return the ``m.def`` schemas of one ``TORCH_LIBRARY`` block, in order."""
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(rf"TORCH_LIBRARY\(\s*{namespace}\s*,\s*m\s*\)\s*\{{", text)
    assert match is not None, f"{namespace} registry not found in {path}"
    body = text[match.end(): text.index("\n}", match.end())]
    schemas = []
    for definition in re.finditer(r"m\.def\(\s*(.*?)\);", body, re.S):
        schemas.append("".join(re.findall(r'"((?:[^"\\]|\\.)*)"', definition.group(1))))
    return schemas


@pytest.mark.parametrize(
    "namespace,path,expected",
    [
        ("nf12", "gptqmodel_ext/nf12/csrc/torch/ops.cpp", kernels.CUDA_SCHEMAS),
        ("nf12_hip", "gptqmodel_ext/nf12_hip/nf12_torch.cpp", kernels.HIP_SCHEMAS),
    ],
)
def test_manifest_matches_the_registered_schemas(namespace, path, expected):
    """The signed schemas are exactly what the kernels register, in order."""
    assert tuple(_registered(path, namespace)) == expected


def test_signed_abi_digest_matches():
    """Any schema change has to re-sign this manifest."""
    table = "\n".join(f"{name}={signature}" for name, signature in sorted(kernels.signatures().items()))
    digest = hashlib.sha256(table.encode()).hexdigest()[:16]
    assert digest == SIGNED_ABI_DIGEST, (
        "NF12 kernel ABI changed. Re-sign by updating SIGNED_ABI_DIGEST to "
        f"{digest!r} after reviewing the schema diff."
    )


def test_every_kernel_has_an_explicit_mlx_decision():
    """No registered operator may be missing from the MLX variant table."""
    registered = {kernel.name for kernel in kernels.KERNELS}
    assert set(kernels.MLX_VARIANTS) == registered


@pytest.mark.parametrize("operation,variant", sorted(kernels.MLX_VARIANTS.items()))
def test_mlx_variant_resolves(operation, variant):
    """A claimed MLX variant must exist; an unclaimed one must be a listed gap."""
    if variant is None:
        assert operation in MLX_GAPS
        return
    module_name, _, attribute = variant.partition(":")
    assert module_name and attribute, f"{operation}: expected 'module:attribute', got {variant!r}"
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute, None)), f"{operation}: {variant} is not callable"


def test_mlx_gap_list_is_pinned():
    """The MLX coverage gaps only change when this test is edited."""
    assert kernels.MLX_GAPS == MLX_GAPS
