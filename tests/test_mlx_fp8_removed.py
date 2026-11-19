# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The invented MLX FP8 runtime is gone; production FP8 stays on Torch."""

import importlib

import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.quantization.config import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


def test_mlx_has_no_fp8_kernel():
    """No MLX kernel may claim FORMAT.FP8 after the Metal kernels were removed."""

    with pytest.raises(ValueError, match="Unsupported backend"):
        select_quant_linear(
            bits=8,
            group_size=-1,
            desc_act=False,
            sym=True,
            backend=BACKEND.MLX,
            format=FORMAT.FP8,
            quant_method=METHOD.FP8,
            device=DEVICE.MPS,
            pack_dtype=torch.int32,
            dtype=torch.float16,
        )


def test_mlx_fp8_holder_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("gptqmodel.nn_modules.qlinear.mlx_fp8")

    from gptqmodel.nn_modules.qlinear import mlx as mlx_holders

    assert not hasattr(mlx_holders, "FP8MlxQuantLinear")


def test_production_fp8_still_loads_on_torch():
    """Removing the MLX holder must not disturb the Torch FP8 path."""

    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    selected = select_quant_linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        backend=BACKEND.AUTO,
        format=FORMAT.FP8,
        quant_method=METHOD.FP8,
        device=DEVICE.CPU,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )
    assert selected is TorchFP8Linear


def test_mlx_checkpoint_loader_rejects_fp8():
    """The MLX loader fails fast instead of decoding production FP8 itself."""

    from gptqmodel.utils import mlx as mlx_utils

    with pytest.raises(
        ValueError,
        match="FP8 Metal kernels and their loader were removed.*Torch backend",
    ):
        mlx_utils._build_mlx_model_from_checkpoint(
            "unused",
            None,
            {"format": "fp8", "method": "fp8", "bits": 8},
            "lm_head",
        )
