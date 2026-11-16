# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AwqMacheteLinear must not advertise group_size=32 (#4218).

The native Machete mainloop requires ``group_size % 64 == 0``; declaring 32 let
AUTO rank AwqMacheteLinear above AwqMarlin for AWQ gs=32 checkpoints, which then
hit a TORCH_CHECK on the first forward.
"""

import pytest
import torch

import gptqmodel.nn_modules.qlinear.machete_awq as machete_awq
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.importer import clear_validation_cache, select_quant_linear


def _mock_machete_available(monkeypatch) -> None:
    """Pretend the host is an SM90 machine with a built Machete extension."""

    monkeypatch.setattr(machete_awq, "machete_extension_available", lambda: True)
    monkeypatch.setattr(
        machete_awq, "_validate_machete_device_support", lambda device=None: True
    )
    clear_validation_cache()


def _validate(group_size: int):
    return AwqMacheteLinear.validate(
        bits=4,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=512,
        out_features=256,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        dynamic=None,
        trainable=False,
        adapter=None,
    )


def _auto_candidates(group_size: int) -> list[str]:
    clear_validation_cache()
    return [
        kernel.__name__
        for kernel in select_quant_linear(
            bits=4,
            group_size=group_size,
            desc_act=False,
            sym=False,
            device=None,
            format=FORMAT.GEMM,
            quant_method=METHOD.AWQ,
            pack_dtype=torch.int32,
            dtype=torch.float16,
            multi_select=True,
        )
    ]


def test_declares_only_native_machete_group_sizes():
    assert AwqMacheteLinear.SUPPORTS_GROUP_SIZE == [-1, 64, 128]
    # Keep the AWQ and GPTQ sub-kernels of the same native backend in step.
    assert AwqMacheteLinear.SUPPORTS_GROUP_SIZE == MacheteLinear.SUPPORTS_GROUP_SIZE


@pytest.mark.parametrize("group_size", [-1, 64, 128])
def test_validate_accepts_native_group_sizes(group_size, monkeypatch):
    _mock_machete_available(monkeypatch)
    ok, error = _validate(group_size)
    assert ok, error


@pytest.mark.parametrize("group_size", [32])
def test_validate_rejects_group_size_32(group_size, monkeypatch):
    _mock_machete_available(monkeypatch)
    ok, error = _validate(group_size)
    assert not ok
    assert "group_size" in str(error)


def test_auto_selection_drops_awq_machete_for_group_size_32(monkeypatch):
    _mock_machete_available(monkeypatch)

    candidates_32 = _auto_candidates(32)
    # AwqMarlinLinear is the intended fallback, but its extension is not built
    # in this environment, so assert the broken candidate is dropped instead.
    assert "AwqMacheteLinear" not in candidates_32
    assert candidates_32, "AUTO must still return a usable AWQ kernel"

    # Positive control: the same mocked host does select Machete for gs=64, so
    # the exclusion above comes from the group size and not from the mocks.
    assert "AwqMacheteLinear" in _auto_candidates(64)
