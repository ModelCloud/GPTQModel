# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The removed ``GPTQModel.export()`` and MLX conversion APIs stay removed.

Exporting a checkpoint is a checkpoint-level operation, not a model-loading
one. Callers use ``gptqmodel.utils.model_dequant.dequantize_model()`` for a
dense Transformers checkpoint.

``gptqmodel.utils.mlx.convert_gptq_to_mlx_weights()`` was the replacement named when
``export()`` was removed, but it repeats the same mistake: converting weights
for another runtime is a step of that runtime's backend, not a public API. MLX
support is reached through ``GPTQModel.load(..., backend=BACKEND.MLX)``, which
runs the conversion internally. The public function is gone as well, with no
shim and no compatibility alias: a direct call fails at import.
"""

import pytest

from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseQModel


@pytest.mark.cpu
def test_export_is_not_part_of_the_public_api():
    assert "export" not in GPTQModel.__dict__
    assert not hasattr(GPTQModel, "export")
    assert "export" not in BaseQModel.__dict__


@pytest.mark.cpu
def test_export_call_raises_attribute_error(tmp_path):
    with pytest.raises(AttributeError, match="export"):
        GPTQModel.export(str(tmp_path), str(tmp_path / "out"), "hf")


@pytest.mark.cpu
def test_mlx_weight_conversion_is_not_part_of_the_public_api():
    """``convert_gptq_to_mlx_weights`` must not be importable by that name.

    The conversion still exists as an internal step of the MLX backend; only
    the public surface is removed. No deprecation warning and no forwarding
    alias, matching how ``export()`` was removed.
    """

    from gptqmodel.utils import mlx

    assert "convert_gptq_to_mlx_weights" not in mlx.__dict__
    assert not hasattr(mlx, "convert_gptq_to_mlx_weights")
    with pytest.raises(ImportError):
        from gptqmodel.utils.mlx import convert_gptq_to_mlx_weights  # noqa: F401

    # The internal conversion is what the MLX backend calls; it must stay.
    assert callable(mlx._convert_gptq_to_mlx_weights)
