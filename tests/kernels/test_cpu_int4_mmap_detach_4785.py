# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A CPU int4 repack must not leave any operand inside the checkpoint mapping (#4785).

A CPU load keeps quantized tensors as zero-copy views of the shard's
``safetensors`` mapping. The first forward repacks ``qweight``/``qzeros`` into
private tensors, but ``g_idx`` (and ``scales`` on the ATen path) used to stay
views of that mapping. One surviving view pins the whole shard, so every page
the loader touched stays resident and an 8B model reports about 1.6x its real
working set in RSS.

The tests bind the module's checkpoint tensors to a real ``mmap`` of a file and
assert that after ``transform_cpu`` no parameter or buffer points inside those
mappings. Values are checked too, so detaching cannot change numerics.
"""

from __future__ import annotations

import mmap
import os

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.packing import hf_pack_layer


IN_FEATURES = 128
OUT_FEATURES = 64
GROUP_SIZE = 32

CHECKPOINT_TENSORS = ("qweight", "qzeros", "scales", "g_idx")


def _avx512_available() -> bool:
    try:
        ok, _ = TorchCpuAvx512Linear.cached_validate_once()
    except Exception:
        return False
    return bool(ok)


def _aten_available() -> bool:
    try:
        ok, _ = TorchAtenLinear.cached_validate_once()
    except Exception:
        return False
    return bool(ok)


def _packed_module(cls, dtype=torch.bfloat16):
    module = cls(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        format=FORMAT.GPTQ_V2,
        dtype=dtype,
    )
    torch.manual_seed(4785)
    dense = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float32)
    with torch.no_grad():
        dense.weight.normal_(mean=0.0, std=0.02)
    groups = IN_FEATURES // GROUP_SIZE
    hf_pack_layer(
        module,
        dense,
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=torch.float32),
        checkpoint_format=FORMAT.GPTQ_V2,
        post_init=True,
    )
    return module.eval()


def _bind_file_views(module, tmp_path):
    """Point every checkpoint tensor at a real ``mmap`` of a file.

    A loader hands the module tensors that alias the shard mapping; binding
    them here reproduces that without a checkpoint on disk.
    """

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
        # The repack assigns plain tensors back to these names, so they are
        # registered as buffers (as the loader leaves them), not parameters.
        delattr(module, name)
        module.register_buffer(name, view)

        mappings.append(mapping)
        ranges.append(
            (view.data_ptr(), view.data_ptr() + source.numel() * source.element_size())
        )
    return mappings, ranges


def _file_backed_names(module, ranges):
    """Parameters and buffers whose storage lies inside one of the mappings."""

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
        # Precondition: the setup really did leave operands inside the mapping,
        # otherwise the assertion below would pass for the wrong reason.
        assert _file_backed_names(module, ranges), "test setup did not alias a mapping"

        before = {
            name: getattr(module, name).detach().clone()
            for name in CHECKPOINT_TENSORS
            if isinstance(getattr(module, name, None), torch.Tensor)
        }
        module.transform_cpu()
        pinned = _file_backed_names(module, ranges)

        # Values survive the detach: it is a copy, not a recompute.
        for name, original in before.items():
            current = getattr(module, name, None)
            if name in ("qweight", "qzeros"):
                # The repack replaces the packed words with its runtime layout.
                continue
            assert torch.equal(current, original), name

        assert pinned == [], (
            f"{type(module).__name__} still keeps {pinned} inside the checkpoint "
            f"mapping; the whole shard stays resident (#4785)"
        )
    finally:
        for mapping in mappings:
            mapping.close()


@pytest.mark.skipif(not _avx512_available(), reason="AVX512-BF16 CPU kernel unavailable")
def test_avx512_repack_detaches_every_checkpoint_view(tmp_path):
    _run(_packed_module(TorchCpuAvx512Linear), tmp_path)


@pytest.mark.skipif(not _aten_available(), reason="ATen CPU int4pack ops are unavailable")
def test_aten_repack_detaches_every_checkpoint_view(tmp_path):
    _run(_packed_module(TorchAtenLinear), tmp_path)
