# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The AWQ CPU int4 repack must not leave operands in the checkpoint mapping.

#4785 fixed the GPTQ CPU paths: a safetensors load hands the module zero-copy
views of the shard mapping, the first-forward repack replaces ``qweight`` and
``qzeros`` with private tensors, and any *other* operand still pointing into the
file pins the whole shard so every page the loader faulted in stays resident.

The AWQ siblings had the same gap. ``TorchAtenAwqLinear`` keeps ``scales`` at the
checkpoint dtype and order by design and never touches ``bias``, and
``TorchCpuAvx512AwqLinear`` reached for ``to(torch.float32).contiguous()``, which
returns the same tensor when the dtype already matches. Measured on the parent
commit, both left ``['scales', 'bias']`` inside the mapping after the transform.

These tests bind the module's checkpoint tensors to a real ``mmap`` of a file and
assert that after ``transform_cpu`` no parameter or buffer points inside it.
"""

from __future__ import annotations

import mmap
import os

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear

from .test_awq_runtime_dtype_storage import _build_checkpoint, _copy_checkpoint


IN_FEATURES = 256
OUT_FEATURES = 128
GROUP_SIZE = 32

CHECKPOINT_TENSORS = ("qweight", "qzeros", "scales", "bias", "g_idx")


def _available(cls) -> bool:
    try:
        ok, _ = cls.cached_validate_once()
    except Exception:
        return False
    return bool(ok)


def _module(cls, *, fp32_scales: bool = False):
    codes, zeros, qweight, qzeros, scales, bias = _build_checkpoint(
        IN_FEATURES, OUT_FEATURES, GROUP_SIZE
    )
    if fp32_scales:
        # A checkpoint whose scales are already FP32 makes
        # ``scales.to(torch.float32)`` return the same tensor, so the AVX512
        # path would keep the mapping view.
        scales = scales.float()
        bias = bias.float() if bias is not None else None
    module = cls(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=True,
    )
    _copy_checkpoint(module, qweight, qzeros, scales, bias)
    return module.eval()


def _bind_file_views(module, tmp_path):
    """Point every checkpoint tensor at a real ``mmap`` of a file."""

    mappings = []
    ranges = []
    for name in CHECKPOINT_TENSORS:
        tensor = getattr(module, name, None)
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        source = tensor.detach().contiguous()
        path = tmp_path / f"{name}.bin"
        path.write_bytes(source.view(torch.uint8).reshape(-1).numpy().tobytes())

        fd = os.open(path, os.O_RDWR)
        mapping = mmap.mmap(fd, 0)
        os.close(fd)

        view = torch.frombuffer(
            mapping, dtype=source.dtype, count=source.numel()
        ).reshape(source.shape)
        delattr(module, name)
        module.register_buffer(name, view)

        mappings.append(mapping)
        ranges.append(
            (view.data_ptr(), view.data_ptr() + source.numel() * source.element_size())
        )
    return mappings, ranges


def _file_backed_names(module, ranges):
    names = []
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if tensor.numel() == 0:
            continue
        pointer = tensor.data_ptr()
        if any(low <= pointer < high for low, high in ranges):
            names.append(name)
    return names


def _run(module, tmp_path):
    mappings, ranges = _bind_file_views(module, tmp_path)
    try:
        assert _file_backed_names(module, ranges), "test setup did not alias a mapping"
        scales = module.scales.detach().clone()

        module.transform_cpu()

        assert torch.equal(module.scales, scales), "detaching changed the scales"
        pinned = _file_backed_names(module, ranges)
        assert pinned == [], (
            f"{type(module).__name__} still keeps {pinned} inside the checkpoint "
            f"mapping; the whole shard stays resident (#4785)"
        )
    finally:
        for mapping in mappings:
            mapping.close()


@pytest.mark.parametrize("fp32_scales", [False, True], ids=["ckpt_scales", "fp32_scales"])
@pytest.mark.skipif(not _available(TorchAtenAwqLinear), reason="ATen CPU int4pack ops are unavailable")
def test_aten_awq_repack_detaches_every_checkpoint_view(tmp_path, fp32_scales):
    _run(_module(TorchAtenAwqLinear, fp32_scales=fp32_scales), tmp_path)


@pytest.mark.parametrize("fp32_scales", [False, True], ids=["ckpt_scales", "fp32_scales"])
@pytest.mark.skipif(
    not _available(TorchCpuAvx512AwqLinear), reason="AVX512-BF16 CPU kernel unavailable"
)
def test_avx512_awq_repack_detaches_every_checkpoint_view(tmp_path, fp32_scales):
    _run(_module(TorchCpuAvx512AwqLinear, fp32_scales=fp32_scales), tmp_path)
