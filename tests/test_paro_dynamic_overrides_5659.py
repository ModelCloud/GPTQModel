# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""#5659: the Paro overrides that reached the clone must reach the packed layer.

#5573 made every `dynamic` field land on a per-module config clone. For Paro
two of them stopped there: the packed `ParoLinear` was built from the top-level
`quant_linear_init_kwargs()` (so `krot`/`kernel` came from the run config) and
the fallback threshold was read from the processor instead of the module.
"""

import threading

import pytest
import torch

from gptqmodel.looper.paro_processor import ParoProcessor
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.quantization.config import FORMAT, ParoConfig, clone_config_for_module
from gptqmodel.quantization.module_config import resolve_module_constructor_config
from gptqmodel.utils.model import create_quant_module


MODULE_NAME = "model.layers.0.self_attn.q_proj"


def _fake_linear(full_name: str = MODULE_NAME, in_features: int = 256) -> torch.nn.Linear:
    module = torch.nn.Linear(in_features, in_features, bias=False)
    module.full_name = full_name
    module.name = full_name.rsplit(".", 1)[-1]
    module.layer_index = 0
    module.state = {}
    return module


def _bare_paro_processor(qcfg: ParoConfig) -> ParoProcessor:
    processor = object.__new__(ParoProcessor)
    processor.qcfg = qcfg
    processor._module_qcfg_cache = {}
    processor.tasks = {}
    processor.fallback = None
    processor.calculate_w_wq_diff = False
    processor.lock = threading.Lock()
    return processor


def test_resolve_module_constructor_config_applies_the_paro_krot():
    resolved = resolve_module_constructor_config(
        bits=4, group_size=128, desc_act=False, sym=True, pack_dtype=torch.int32,
        format=FORMAT.PARO, init_kwargs={"krot": 8}, overrides={"krot": 1},
    )
    assert resolved is not None
    assert resolved.options_dict()["krot"] == 1


def test_resolve_module_constructor_config_rejects_a_bad_paro_krot():
    for bad in (0, -1, True, "many", 1.5):
        with pytest.raises(ValueError):
            resolve_module_constructor_config(
                bits=4, group_size=128, desc_act=False, sym=True, pack_dtype=torch.int32,
                format=FORMAT.PARO, init_kwargs={"krot": 8}, overrides={"krot": bad},
            )


@pytest.mark.cpu
def test_the_packed_paro_layer_is_built_with_the_module_krot():
    """Red on main: the layer took the top-level `krot` (8) while the optimizer
    produced buffers for the module's `krot` (1), so `pairs.copy_` broadcast."""

    qcfg = ParoConfig(krot=8, dynamic={MODULE_NAME: {"krot": 1}})
    linear = _fake_linear()
    parent = torch.nn.Sequential(linear)
    qmodule = create_quant_module(
        name=MODULE_NAME,
        linear_cls=ParoLinear,
        bits=qcfg.runtime_bits,
        desc_act=qcfg.desc_act,
        dynamic=qcfg.dynamic,
        group_size=qcfg.group_size,
        module=parent,
        submodule=linear,
        sym=qcfg.sym,
        device=DEVICE.CPU,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        format=FORMAT.PARO,
        register_buffers=False,
        init_kwargs=qcfg.quant_linear_init_kwargs(),
        resolved_module_config=qcfg.resolve_module_config(MODULE_NAME),
        install=False,
    )

    assert isinstance(qmodule, ParoLinear)
    assert qmodule.krot == 1
    # The rotation buffers are sized by `krot`; a broadcast hid the mismatch.
    assert qmodule.pairs.shape[0] == 1
    assert qmodule.theta.shape[0] == 1


def test_an_explicit_none_rule_inherits_instead_of_crashing():
    """Red on main: `{'krot': None}` reached `int(None)` and raised TypeError."""

    qcfg = ParoConfig(
        krot=8,
        fallback=None,
        dynamic={MODULE_NAME: {"krot": None, "fallback": None, "bits": None}},
    )
    # `None` is not an override, so the rule keeps no fields at all.
    assert qcfg.dynamic[MODULE_NAME] == {}
    clone = clone_config_for_module(qcfg, MODULE_NAME)
    assert clone is not None
    assert clone.krot == 8
    assert clone.fallback is None
    assert clone.bits == qcfg.bits


def test_an_explicit_none_scope_key_is_not_rejected():
    """`opt_scope` is rejected per module, but `None` means inherit."""

    qcfg = ParoConfig(opt_scope="layer", dynamic={MODULE_NAME: {"opt_scope": None}})
    assert qcfg.opt_scope == "layer"
    assert qcfg.dynamic[MODULE_NAME] == {}


def test_the_fallback_threshold_comes_from_the_module_rule():
    """Red on main: the clone said 1.0%, the processor read its own 0.5%."""

    qcfg = ParoConfig(
        fallback="0.5%",
        dynamic={MODULE_NAME: {"fallback": "1.0%"}},
    )
    processor = _bare_paro_processor(qcfg)
    processor.fallback = qcfg.fallback
    processor.total_calibration_tokens = 1000
    module = _fake_linear()
    # 5 of 1000 rows: below the module's 1.0%, above the run-level 0.5%.
    processor.tasks = {module.name: {"raw_tokens": 5}}

    assert processor._module_qcfg(module).fallback.threshold == "1.0%"
    assert processor._module_fallback_requested(module) is True
    assert processor._module_fallback_enabled(module.name, _state(module)) is True


def test_a_module_without_a_rule_still_uses_the_run_level_threshold():
    qcfg = ParoConfig(fallback="0.5%", dynamic={"other.proj": {"fallback": "1.0%"}})
    processor = _bare_paro_processor(qcfg)
    processor.fallback = qcfg.fallback
    processor.total_calibration_tokens = 1000
    module = _fake_linear()
    processor.tasks = {module.name: {"raw_tokens": 5}}

    # 5/1000 is above 0.5%: no fallback for this module.
    assert processor._module_fallback_requested(module) is False


def _state(module):
    return type("_State", (), {"modules": {module.name: module}})()
