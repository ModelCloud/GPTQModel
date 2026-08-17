# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native CPU inner GEMV for the planar QVQ trellis format."""

from __future__ import annotations

import platform
import threading
from collections.abc import Callable
from pathlib import Path

import torch

from ..quantization.qvq_rates import qvq_transition_bits
from .cpp import TorchOpsJitExtension, default_jit_cflags, default_torch_ops_build_root


_QVQ_CPU_OPS_NAME = "gptqmodel_qvq_cpu_ops"
_QVQ_CPU_NAMESPACE = "gptqmodel_qvq"
_QVQ_CPU_OP: Callable | None = None
_QVQ_CPU_OP_LOCK = threading.Lock()


def _qvq_cpu_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "qvq"


def _qvq_cpu_sources() -> list[str]:
    return [
        str(_qvq_cpu_root() / "qvq_gemv_cpu.cpp"),
        str(_qvq_cpu_root() / "qvq_viterbi_cpu.cpp"),
    ]


def _qvq_cpu_extra_cflags() -> list[str]:
    flags = list(default_jit_cflags(enable_bf16=True))
    if platform.system() == "Linux":
        flags.append("-fopenmp")
    return flags


def _qvq_cpu_extra_ldflags() -> list[str]:
    return ["-fopenmp"] if platform.system() == "Linux" else []


_QVQ_CPU_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_QVQ_CPU_OPS_NAME,
    namespace=_QVQ_CPU_NAMESPACE,
    required_ops=("gemv_cpu", "viterbi_cpu"),
    sources=_qvq_cpu_sources,
    build_root_env="GPTQMODEL_QVQ_CPU_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("qvq_cpu"),
    display_name="QVQ CPU kernels",
    extra_cflags=_qvq_cpu_extra_cflags,
    extra_ldflags=_qvq_cpu_extra_ldflags,
    extra_include_paths=lambda: [str(_qvq_cpu_root())],
    force_rebuild_env="GPTQMODEL_QVQ_CPU_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=False,
)


def qvq_cpu_supported() -> bool:
    return platform.machine().lower() in ("x86_64", "amd64")


def qvq_cpu_error() -> str:
    if not qvq_cpu_supported():
        return "QVQ CPU kernel requires x86-64 (AMD64)."
    return _QVQ_CPU_TORCH_OPS_EXTENSION.last_error_message()


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def _qvq_cpu_op() -> Callable:
    global _QVQ_CPU_OP
    if _QVQ_CPU_OP is None:
        with _QVQ_CPU_OP_LOCK:
            if _QVQ_CPU_OP is None:
                _QVQ_CPU_OP = _extension_api().op("qvq_cpu", "gemv_cpu")
    return _QVQ_CPU_OP


_QVQ_CPU_VITERBI_OP: Callable | None = None


def _qvq_cpu_viterbi_op() -> Callable:
    global _QVQ_CPU_VITERBI_OP
    if _QVQ_CPU_VITERBI_OP is None:
        with _QVQ_CPU_OP_LOCK:
            if _QVQ_CPU_VITERBI_OP is None:
                _QVQ_CPU_VITERBI_OP = _extension_api().op("qvq_cpu", "viterbi_cpu")
    return _QVQ_CPU_VITERBI_OP


def qvq_cpu_gemv(
    x: torch.Tensor,
    trellis: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    vector_size: int = 2,
    bank_ids: torch.Tensor | None = None,
    v2b4_p64: bool = False,
    v2b2_p32: bool = False,
    bank_alt_id: int = 0,
) -> torch.Tensor:
    """Compute x @ W for QVQ V2/V2B2-P64/V2B2-P32 on CPU.

    Args:
        x: [M, K] float tensor on CPU.
        trellis: packed planar trellis, int32, on CPU.
        bits: QVQ rate (e.g. 1.5, 2, 3.5).
        out_features: N dimension of the weight matrix.
        vector_size: 2 for V2 formats.
        bank_ids: packed V2B2/V2B4 bank selectors, uint8 [tile_count].
        v2b4_p64: use V2B4-P64 segment banking.
        v2b2_p32: use V2B2-P32 segment banking.
        bank_alt_id: 1..3 for V2B2-P32, ignored otherwise.
    """

    if not qvq_cpu_supported():
        raise RuntimeError("QVQ CPU kernel requires x86-64 (AMD64).")
    if x.device.type != "cpu" or trellis.device.type != "cpu":
        raise ValueError("qvq_cpu_gemv requires CPU tensors")
    if vector_size != 2:
        raise ValueError("QVQ CPU kernel currently supports vector_size=2")
    if v2b4_p64 and v2b2_p32:
        raise ValueError("v2b4_p64 and v2b2_p32 are mutually exclusive")

    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    x = x.contiguous()
    trellis = trellis.contiguous()
    if bank_ids is not None:
        bank_ids = bank_ids.to(torch.uint8).contiguous()

    return _qvq_cpu_op()(
        x,
        trellis,
        transition_bits,
        out_features,
        bank_ids,
        bank_alt_id,
        v2b4_p64,
        v2b2_p32,
    )


def qvq_cpu_viterbi(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    transition_bits: int,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native CPU batched Viterbi trellis quantization.

    Args:
        sequences: [batch, steps, V] float tensor on CPU.
        codebook: [state_count, V] float tensor on CPU.
        transition_bits: QVQ transition width in bits.
        overlap: optional int64 [batch] tail-biting overlap.
        step_weights: optional float [batch, steps] per-step weights.

    Returns:
        (states [batch, steps], squared_error [batch]).
    """

    if not qvq_cpu_supported():
        raise RuntimeError("QVQ CPU kernel requires x86-64 (AMD64).")
    if sequences.device.type != "cpu" or codebook.device.type != "cpu":
        raise ValueError("qvq_cpu_viterbi requires CPU tensors")
    if sequences.dim() != 3 or codebook.dim() != 2 or sequences.size(2) != codebook.size(1):
        raise ValueError("qvq_cpu_viterbi: sequence/codebook shape mismatch")
    if overlap is not None and overlap.device.type != "cpu":
        raise ValueError("qvq_cpu_viterbi: overlap must be on CPU")
    if step_weights is not None and step_weights.device.type != "cpu":
        raise ValueError("qvq_cpu_viterbi: step_weights must be on CPU")

    sequences = sequences.contiguous()
    codebook = codebook.contiguous()
    if overlap is not None:
        overlap = overlap.to(torch.int64).contiguous()
    if step_weights is not None:
        step_weights = step_weights.to(torch.float32).contiguous()

    return _qvq_cpu_viterbi_op()(
        sequences,
        codebook,
        transition_bits,
        overlap,
        step_weights,
    )
