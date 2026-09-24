"""Optional native CUDA block update for GPTQ's 4-bit, 128-column path."""

from pathlib import Path

import torch

from gptqmodel.utils.cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)


def _sources() -> list[str]:
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "gptq_block"
    return [str(root / "gptq_block.cpp"), str(root / "gptq_block.cu")]


_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_gptq_block_ops",
    namespace="gptqmodel_gptq_block",
    required_ops=("block_update",),
    sources=_sources,
    build_root_env="GPTQMODEL_GPTQ_BLOCK_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("gptq_block"),
    display_name="GPTQ block update",
    extra_cflags=default_jit_cflags(),
    extra_cuda_cflags=lambda: default_jit_cuda_cflags(include_lineinfo=True) + ["--fmad=false"],
    force_rebuild_env="GPTQMODEL_GPTQ_BLOCK_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    python_abi_dependent=False,
)


def block_update_available() -> bool:
    """Build and load on first use, returning False when CUDA compilation fails."""
    return _EXTENSION.load()


def gptq_block_update(
    work: torch.Tensor,
    hinv: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    column_group: torch.Tensor,
    quantized: torch.Tensor,
    errors: torch.Tensor,
    losses: torch.Tensor,
) -> None:
    _EXTENSION.op("block_update")(work, hinv, scale, zero, column_group, quantized, errors, losses)
