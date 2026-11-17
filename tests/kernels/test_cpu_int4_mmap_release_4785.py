# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Issue #4785: a CPU int4 repack must release the checkpoint file mapping.

``safetensors`` hands the loader zero-copy views of the shard mapping. The CPU
int4 transforms replace ``qweight``/``qzeros`` with anonymous runtime tensors
but kept ``g_idx`` (and the checkpoint-dtype ``scales`` on the ATen/fused
kernels) as views of the file. Each surviving view pins the whole shard, so
every page the loader faulted in stays resident and the model's quantized
weights are counted twice in RSS.

After the first forward no module tensor may point inside a ``*.safetensors``
mapping, and the shard must no longer appear in ``/proc/self/maps``.
"""

from __future__ import annotations

import gc
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear._fused_train_state import QZEROS_FORMAT_BUFFER
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.cpu_features import avx512_bf16_error, has_avx512_bf16
from tests.test_checkpoint_state_invariant import (
    IN_FEATURES,
    _as_loaded_from,
    _gptq_source,
)


pytestmark = [
    pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="reads /proc/self/maps"
    ),
]


CASES = [
    pytest.param(TorchAtenLinear, BACKEND.GPTQ_TORCH_ATEN, torch.float16, id="torch_aten"),
    pytest.param(TorchFusedLinear, BACKEND.GPTQ_TORCH_FUSED, torch.bfloat16, id="torch_fused"),
    pytest.param(
        TorchCpuAvx512Linear,
        BACKEND.GPTQ_TORCH_CPU_AVX512,
        torch.bfloat16,
        id="torch_cpu_avx512",
    ),
]


def _require_case(cls) -> None:
    if cls is TorchCpuAvx512Linear and not has_avx512_bf16():
        pytest.skip(avx512_bf16_error())
    ok, err = cls.cached_validate_once()
    if not ok:
        pytest.skip(str(err))


def _mapping_ranges(path) -> list[tuple[int, int]]:
    """Virtual-address ranges mapped from ``path`` in this process."""

    needle = str(path)
    ranges = []
    with open("/proc/self/maps", encoding="ascii", errors="replace") as maps:
        for line in maps:
            if needle not in line:
                continue
            start, end = (int(value, 16) for value in line.split()[0].split("-"))
            ranges.append((start, end))
    return ranges


def _file_mapped_ranges() -> list[tuple[int, int]]:
    """Every non-anonymous file mapping in this process."""

    ranges = []
    with open("/proc/self/maps", encoding="ascii", errors="replace") as maps:
        for line in maps:
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            path = parts[5].strip()
            if not path or path.startswith("["):
                continue
            start, end = (int(value, 16) for value in parts[0].split("-"))
            ranges.append((start, end))
    return ranges


def _storage_range(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    start = storage.data_ptr()
    return start, start + storage.nbytes()


def _mapped_module_tensors(module, ranges) -> dict[str, tuple[str, tuple[int, ...], str]]:
    """Module tensors whose storage lives inside one of ``ranges``."""

    found = {}
    for owner_name, owner in module.named_modules():
        prefix = owner_name or "<module>"
        candidates = [*owner._parameters.items(), *owner._buffers.items()]
        candidates += [
            (name, getattr(owner, name, None))
            for name in ("scales_and_zeros", "ret_idx")
        ]
        for name, tensor in candidates:
            if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                continue
            start, end = _storage_range(tensor)
            if any(lo <= start < hi or lo < end <= hi for lo, hi in ranges):
                found[f"{prefix}.{name}"] = (
                    hex(start),
                    tuple(tensor.shape),
                    str(tensor.dtype),
                )
    return found


def _write_checkpoint(module, path) -> None:
    # A real checkpoint does not carry the in-process ``qzeros_format`` marker.
    state = {
        name: tensor.contiguous()
        for name, tensor in module.state_dict().items()
        if torch.is_tensor(tensor) and not name.endswith(QZEROS_FORMAT_BUFFER)
    }
    save_file(state, str(path))


def _load_mapped(cls, backend, path):
    """Load ``path`` the way the CPU loader does: zero-copy mmap views."""

    module = _as_loaded_from(_gptq_source(cls, backend), FORMAT.GPTQ_V2)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        state = {name: handle.get_tensor(name) for name in handle.keys()}
    result = module.load_state_dict(state, assign=True)
    assert not result.missing_keys, result.missing_keys
    assert not result.unexpected_keys, result.unexpected_keys
    del state
    gc.collect()
    return module


def _load_copied(cls, backend, path):
    """Load ``path`` into anonymous buffers, the non-zero-copy control."""

    module = _as_loaded_from(_gptq_source(cls, backend), FORMAT.GPTQ_V2)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        state = {name: handle.get_tensor(name) for name in handle.keys()}
    result = module.load_state_dict(state)
    assert not result.missing_keys, result.missing_keys
    assert not result.unexpected_keys, result.unexpected_keys
    del state
    gc.collect()
    return module


@pytest.mark.parametrize(("cls", "backend", "dtype"), CASES)
def test_first_forward_releases_the_checkpoint_mapping(cls, backend, dtype, tmp_path):
    _require_case(cls)
    torch.manual_seed(4785)
    source = _as_loaded_from(_gptq_source(cls, backend), FORMAT.GPTQ_V2)
    path = tmp_path / "model.safetensors"
    _write_checkpoint(source, path)
    del source

    module = _load_mapped(cls, backend, path)
    control = _load_copied(cls, backend, path)
    before = _mapping_ranges(path)
    assert before, "setup: the checkpoint is not mapped"
    assert _mapped_module_tensors(module, before), (
        "setup: the load did not leave zero-copy mapping views"
    )
    assert not _mapped_module_tensors(control, before), (
        "setup: the copied control unexpectedly holds mapping views"
    )

    x = torch.randn(2, IN_FEATURES, dtype=dtype)
    with torch.no_grad():
        # Pre-repack numerics: the checkpoint-layout dense dequantization.
        reference = torch.matmul(x, module.dequantize_weight().to(dtype))

    with torch.inference_mode():
        output = module(x)
        control_output = control(x)

    # The repack must not change the layer output. ``reference`` is the
    # checkpoint-layout dense result; the tolerance is the one the existing CPU
    # int4 dense-vs-fused tests use. The control repacks anonymous copies of the
    # same bytes, so the two fused outputs must be bitwise identical.
    torch.testing.assert_close(output.float(), reference.float(), atol=0.2, rtol=0.01)
    assert torch.equal(output, control_output), (
        f"{cls.__name__}: detaching the mapping changed the repacked output"
    )

    still_mapped = _mapping_ranges(path)
    assert still_mapped == [], (
        f"{cls.__name__}: the shard mapping is still alive after the repack; "
        f"file-backed module tensors: {_mapped_module_tensors(module, still_mapped)}"
    )
    assert _mapped_module_tensors(module, _file_mapped_ranges()) == {}, (
        f"{cls.__name__}: a module tensor is still a view of a file mapping"
    )


@pytest.mark.parametrize(("cls", "backend", "dtype"), CASES)
def test_repacked_module_survives_deleting_the_checkpoint(cls, backend, dtype, tmp_path):
    _require_case(cls)
    torch.manual_seed(4786)
    source = _as_loaded_from(_gptq_source(cls, backend), FORMAT.GPTQ_V2)
    path = tmp_path / "model.safetensors"
    _write_checkpoint(source, path)
    del source

    module = _load_mapped(cls, backend, path)
    x = torch.randn(2, IN_FEATURES, dtype=dtype)
    with torch.inference_mode():
        first = module(x).clone()

    path.unlink()
    gc.collect()

    with torch.inference_mode():
        second = module(x)
    assert torch.equal(first, second), f"{cls.__name__}: output depends on the deleted file"

    # The repacked operands and the emitted checkpoint are still readable.
    state = module.state_dict()
    assert state
    for name, tensor in state.items():
        if torch.is_tensor(tensor) and tensor.numel() and tensor.is_floating_point():
            assert torch.isfinite(tensor.float()).all(), name
