# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4792.

`GPTQCheckpoint.restore` rebuilds an empty module template and requires the
bundle's tensors to match `module.state_dict()` exactly in names, shapes and
dtypes. Two kernels could never match:

* NF12 allocates `nf12_escapes` with zero rows and `pack()` fills as many as
  the data needs, so the template shape was `(0, 8)` for every module.
* MACaQ allocates `BH`/`pre_scale` at the run's compute dtype, which is not the
  source config's dtype the template used.

Both made every resume fail with
`CheckpointError: checkpoint tensor schema differs`.
"""

import pytest
import torch

from gptqmodel.looper.checkpoint_modules import (
    packed_module_config,
    restore_packed_module,
)
from gptqmodel.looper.checkpoint_store import CheckpointError
from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.quantization.config import MacaqConfig, NF12Config


IN_FEATURES = 128
OUT_FEATURES = 64
NAME = "model.layers.0.self_attn.q_proj"


def _schema(module) -> dict:
    return {
        key: (tuple(tensor.shape), tensor.dtype)
        for key, tensor in module.state_dict().items()
    }


def _assert_template_matches(module, template) -> None:
    expected = _schema(module)
    restored = _schema(template)
    assert set(restored) == set(expected)
    for key, (shape, dtype) in expected.items():
        assert restored[key] == (shape, dtype), f"{key}: {restored[key]} != {(shape, dtype)}"


def test_nf12_restore_template_matches_a_data_dependent_escape_table():
    config = NF12Config(bits=12)
    module = NF12Linear(
        bits=12,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        register_buffers=True,
    )
    # `pack()` fills one row per escaped weight; the count is data-dependent.
    module.nf12_escapes = torch.zeros((7, 8), dtype=torch.int16)

    module_config = packed_module_config(module, config, module_name=NAME)
    assert module_config["escape_rows"] == 7

    template = restore_packed_module(
        module_config,
        name=NAME,
        config=config,
        kernel=NF12Linear,
        lm_head_name="lm_head",
    )

    _assert_template_matches(module, template)


def test_macaq_restore_template_uses_the_run_compute_dtype():
    config = MacaqConfig(bits=4)
    module = MacaqLinear(
        bits=4,
        group_size=32,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        pack_dtype=torch.int32,
        dtype=torch.float16,
        register_buffers=True,
    )
    assert module.BH is not None and module.pre_scale is not None
    assert module.BH.dtype is torch.float16

    module_config = packed_module_config(module, config, module_name=NAME)
    assert module_config["compute_dtype"] == "torch.float16"

    template = restore_packed_module(
        module_config,
        name=NAME,
        config=config,
        kernel=MacaqLinear,
        lm_head_name="lm_head",
    )

    assert template.compute_dtype is torch.float16
    _assert_template_matches(module, template)


@pytest.mark.parametrize("escape_rows", [-1, 1.5, "3"])
def test_invalid_nf12_escape_row_count_is_rejected(escape_rows):
    config = NF12Config(bits=12)
    module_config = {
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "bits": 12,
        "group_size": -1,
        "desc_act": False,
        "sym": True,
        "bias": False,
        "pack_dtype": "int32",
        "format": "nf12",
        "escape_rows": escape_rows,
    }

    with pytest.raises(CheckpointError, match="escape row count"):
        restore_packed_module(
            module_config,
            name=NAME,
            config=config,
            kernel=NF12Linear,
            lm_head_name="lm_head",
        )
