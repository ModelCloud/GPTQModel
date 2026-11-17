# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The deprecated ``GPTQModel.export()`` API is removed.

Exporting a checkpoint is a checkpoint-level operation, not a model-loading
one. Callers use ``gptqmodel.utils.model_dequant.dequantize_model()`` for a
dense Transformers checkpoint and
``gptqmodel.utils.mlx.convert_gptq_to_mlx_weights()`` for MLX weights.
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
