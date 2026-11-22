# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Explicit ``None`` in a ``dynamic`` rule (#5689).

``None`` means "inherit the run-level value", but the #5659 implementation
dropped every ``None`` before validating the rule, so a typo hidden behind a
``None`` value was accepted and the historical ``{"gsq": None}`` per-module
disable stopped working.  A per-module ``kernel`` rule was also shadowed by
the run-level ``_resolved_kernel``.
"""

import pytest

from gptqmodel.quantization.config import (
    GPTQConfig,
    GSQConfig,
    ParoConfig,
    clone_config_for_module,
)
from gptqmodel.utils.backend_config import BACKEND, resolve_quantize_backend
from gptqmodel.utils.kernel import KernelConfig


MODULE = "model.layers.0.mlp.down_proj"
OTHER_MODULE = "model.layers.1.mlp.down_proj"


def test_a_none_valued_typo_is_still_rejected():
    """The value being ``None`` must not hide an unknown key."""

    with pytest.raises(TypeError, match="unknown `dynamic"):
        ParoConfig(dynamic={MODULE: {"not_a_real_option": None}})


def test_a_valid_none_valued_key_still_inherits():
    """The #5659 contract is unchanged: a real key with ``None`` inherits."""

    qcfg = ParoConfig(krot=8, dynamic={MODULE: {"krot": None}})
    assert qcfg.dynamic[MODULE] == {}
    clone = clone_config_for_module(qcfg, MODULE)
    assert clone is not None
    assert clone.krot == 8


def test_gsq_none_disables_gsq_for_the_module():
    """``{"gsq": None}`` is the historical per-module disable."""

    base = GPTQConfig(
        bits=4,
        group_size=128,
        gsq=GSQConfig(enabled=True, steps=3),
        dynamic={MODULE: {"gsq": None}},
    )
    assert base.dynamic[MODULE] == {"gsq": None}

    clone = clone_config_for_module(base, MODULE)
    assert clone is not None
    assert clone.gsq is None

    # A module the rule does not match keeps the run-level GSQ.
    other = clone_config_for_module(base, OTHER_MODULE)
    assert other is not None
    assert isinstance(other.gsq, GSQConfig)
    assert other.gsq.enabled is True


def test_gsq_none_reaches_the_gptq_processor_clone():
    """The processor clone reads ``resolved_cfg.get("gsq", ...)``."""

    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    base = GPTQConfig(
        bits=4,
        group_size=128,
        gsq=GSQConfig(enabled=True, steps=3),
        dynamic={MODULE: {"gsq": None}},
    )
    clone = clone_gptq_config_for_module(base, MODULE)
    assert clone is not None
    assert clone.gsq is None

    other = clone_gptq_config_for_module(base, OTHER_MODULE)
    assert other is not None
    assert isinstance(other.gsq, GSQConfig)
    assert other.gsq.enabled is True


def test_gsq_none_survives_a_config_round_trip():
    base = GPTQConfig(
        bits=4,
        group_size=128,
        gsq=GSQConfig(enabled=True, steps=3),
        dynamic={MODULE: {"gsq": None}},
    )
    restored = GPTQConfig.from_quant_config(base.to_dict())
    assert restored.dynamic[MODULE] == {"gsq": None}
    assert clone_config_for_module(restored, MODULE).gsq is None


def test_gsq_dict_rule_still_disables():
    """The explicit ``{"enabled": False}`` form keeps working."""

    base = GPTQConfig(
        bits=4,
        group_size=128,
        gsq=GSQConfig(enabled=True, steps=3),
        dynamic={MODULE: {"gsq": {"enabled": False}}},
    )
    clone = clone_config_for_module(base, MODULE)
    assert clone is not None
    assert isinstance(clone.gsq, GSQConfig)
    assert clone.gsq.enabled is False


def test_dynamic_kernel_rule_wins_over_the_resolved_kernel():
    """Red on main: the run-level ``_resolved_kernel`` shadowed the rule."""

    qcfg = ParoConfig(
        kernel=KernelConfig(paro_cpu_avx512=False),
        dynamic={MODULE: {"kernel": {"paro_cpu_avx512": True}}},
    )
    resolve_quantize_backend(BACKEND.AUTO, qcfg)

    clone = clone_config_for_module(qcfg, MODULE)
    assert clone is not None
    assert clone.quant_linear_init_kwargs()["kernel"].paro_cpu_avx512 is True


def test_dynamic_kernel_rule_merges_onto_the_run_level_policy():
    """A partial rule only changes the fields it names."""

    qcfg = ParoConfig(
        kernel=KernelConfig(paro_cpu_avx512=False, paro_triton_autotune=False),
        dynamic={MODULE: {"kernel": {"paro_cpu_avx512": True}}},
    )
    resolve_quantize_backend(BACKEND.AUTO, qcfg)

    kernel = clone_config_for_module(qcfg, MODULE).quant_linear_init_kwargs()["kernel"]
    assert kernel.paro_cpu_avx512 is True
    assert kernel.paro_triton_autotune is False


def test_unknown_per_module_kernel_option_is_rejected():
    qcfg = ParoConfig(
        kernel=KernelConfig(),
        dynamic={MODULE: {"kernel": {"not_a_kernel_option": True}}},
    )
    resolve_quantize_backend(BACKEND.AUTO, qcfg)

    with pytest.raises(TypeError, match="unknown per-module `kernel` option"):
        clone_config_for_module(qcfg, MODULE)
