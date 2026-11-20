# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Asymmetric GPTQ must select the Marlin zero-point kernels (#5737).

``MarlinLinear`` declared ``SUPPORTS_SYM = [True]`` and mapped only the
implicit-zero scalar types, so AUTO never offered it for ``sym=False``
checkpoints and fell through to ExLlamaV2 (FP16) or TritonV2 (BF16) on
SM80-SM89/SM120. The shipped native op already instantiates the integer
zero-point kernels (``kU4``/``kU8``) for every generated group size, so the
contract now admits ``sym=False`` while activation order keeps the fallbacks:
``generate_kernels.py`` has no ``group_blocks == 0`` zero-point instantiation.
"""

from __future__ import annotations

import pytest
import torch

import gptqmodel.models._const as model_const
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import (
    clear_validation_cache,
    select_quant_linear,
    validate_quant_linear,
)
from gptqmodel.utils.marlin_scalar_type import scalar_types


@pytest.fixture(autouse=True)
def nvidia_selector_runtime(monkeypatch):
    """Assert the NVIDIA family and keep selection runnable without a GPU.

    The device contract (compute capability >= 8.0, runtime extension present)
    has its own tests; here the selector order and the declared quant contracts
    are the subject.
    """

    monkeypatch.setattr(model_const, "IS_ROCM", False)
    monkeypatch.setattr(importer, "IS_ROCM", False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        MarlinLinear, "validate_device", classmethod(lambda cls, _device: None)
    )
    monkeypatch.setattr(
        MarlinLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )
    clear_validation_cache()
    yield
    clear_validation_cache()


def _validate(**kwargs):
    args = {
        "dtype": torch.float16,
        "pack_dtype": torch.int32,
        "in_features": 256,
        "out_features": 128,
        "device": DEVICE.CUDA,
    }
    args.update(kwargs)
    return validate_quant_linear(MarlinLinear, **args)


def _select(**kwargs):
    args = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": False,
        "device": DEVICE.CUDA,
        "backend": BACKEND.AUTO,
        "format": FORMAT.GPTQ,
        "quant_method": METHOD.GPTQ,
        "pack_dtype": torch.int32,
        "dtype": torch.float16,
        "multi_select": True,
    }
    args.update(kwargs)
    return select_quant_linear(**args)


def test_contract_declares_the_native_zero_point_types():
    assert MarlinLinear.SUPPORTS_SYM == [True, False]
    assert MarlinLinear.REQUIRES_FORMAT_V2 is True
    assert MarlinLinear.TYPE_MAP == {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
        (4, False): scalar_types.uint4,
        (8, False): scalar_types.uint8,
    }


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
def test_validate_accepts_asymmetric_non_activation_order(bits, group_size):
    ok, error = _validate(bits=bits, group_size=group_size, desc_act=False, sym=False)

    assert ok, error


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_validate_rejects_asymmetric_activation_order(group_size):
    """No ``kU4``/``kU8`` image exists for ``group_blocks == 0``."""

    ok, error = _validate(bits=4, group_size=group_size, desc_act=True, sym=False)

    assert not ok
    assert isinstance(error, NotImplementedError)
    assert "activation-order" in str(error)


def test_validate_keeps_asymmetric_channelwise_desc_act():
    """One channelwise group makes ``desc_act`` a no-op, so zeros still fit."""

    ok, error = _validate(bits=4, group_size=-1, desc_act=True, sym=False)

    assert ok, error


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sym", [True, False])
def test_auto_ranks_marlin_first_for_symmetric_and_asymmetric(dtype, sym):
    candidates = _select(sym=sym, dtype=dtype)

    assert candidates[0] is MarlinLinear, [cls.__name__ for cls in candidates]


def test_asymmetric_construction_rejects_activation_order_for_fallback():
    """Per-module AUTO falls through when the kernel cannot honor the contract."""

    with pytest.raises(NotImplementedError, match="activation-order"):
        MarlinLinear(
            bits=4,
            group_size=128,
            desc_act=True,
            sym=False,
            in_features=256,
            out_features=128,
            backend=BACKEND.AUTO,
        )


@pytest.mark.parametrize("sym", [True, False])
def test_symmetric_construction_still_normalizes_channelwise_desc_act(sym):
    layer = MarlinLinear(
        bits=4,
        group_size=256,
        desc_act=True,
        sym=sym,
        in_features=256,
        out_features=128,
        backend=BACKEND.AUTO,
    )

    assert layer.desc_act is False
