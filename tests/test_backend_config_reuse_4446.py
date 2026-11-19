# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""#4446: a resolved backend policy is never read back as a caller's ``kernel=``.

``resolve_quantize_backend`` used to store its generated policy in
``QuantizeConfig.kernel``. A later load or ``quantize()`` on the same config
then saw a caller-supplied legacy kernel: typed backends and ``runtime=``
raised "cannot be supplied together", and an enum selection inherited the
previous typed settings. The tests drive ``GPTQModel.load`` with a model
definition that re-resolves its arguments the way ``ModelLoader.from_pretrained``
does, and ``resolve_quantize_backend(..., inherit=True)`` exactly as
``BaseQModel.quantize`` calls it.
"""

from types import SimpleNamespace

import pytest

from gptqmodel import GPTQModel
from gptqmodel.backend import BACKEND, AwqTorchConfig, MarlinConfig, RuntimeConfig, TorchConfig
from gptqmodel.quantization import AWQConfig, QuantizeConfig
from gptqmodel.utils.backend_config import resolve_quantize_backend
from gptqmodel.utils.kernel import KernelConfig, get_kernel_config, reset_kernel_config, resolve_kernel_config


DEFAULTS = KernelConfig().to_dict()


@pytest.fixture
def fake_load(monkeypatch, tmp_path):
    """Run ``GPTQModel.load`` up to the model definition without a checkpoint."""

    from gptqmodel.models import auto

    calls = []

    class _Definition:
        @staticmethod
        def from_pretrained(**kwargs):
            # Same resolution ModelLoader.from_pretrained performs.
            selected, policy = resolve_quantize_backend(
                kwargs["backend"], kwargs["quantize_config"], kwargs.get("kernel"), kwargs.get("runtime")
            )
            calls.append((selected, policy))
            # Current GPTQModel.load logs the KV-cache estimate on the returned
            # model; this fake stops before a real model exists.
            return SimpleNamespace(
                quantize_config=kwargs["quantize_config"],
                log_budget=lambda: None,
            )

    monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: _Definition)
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda *args, **kwargs: False)
    monkeypatch.setattr(auto, "native_floatx_source_format", lambda *args, **kwargs: None)
    monkeypatch.setattr(auto, "_kick_quantize_prewarms", lambda *args, **kwargs: None)
    monkeypatch.setattr(auto, "_prebuild_device_thread_pool", lambda *args, **kwargs: None)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda *args: False)

    def load(quantize_config, **kwargs):
        return GPTQModel.load(str(tmp_path), quantize_config=quantize_config, **kwargs)

    load.calls = calls
    try:
        yield load
    finally:
        reset_kernel_config()


def _looper_policy(qcfg):
    """The policy a ModuleLooper / WeightOnlyLooper installs for this config."""

    return resolve_kernel_config(qcfg, use_process_default=False)


def test_typed_backend_after_runtime_config_load_and_quantize(fake_load):
    """(a) the documented flow: typed backend after a ``runtime=`` config."""

    qcfg = QuantizeConfig(bits=4, runtime=RuntimeConfig(max_packers=2))
    fake_load(qcfg)
    assert qcfg.kernel is None
    fake_load(qcfg, backend=TorchConfig(streaming=True))
    assert qcfg.kernel is None
    policy = _looper_policy(qcfg)
    assert policy.torch_streaming is True
    assert policy.max_packers == 2

    # BaseQModel.quantize(backend=TorchConfig(...)) on the loaded config.
    selected, policy = resolve_quantize_backend(TorchConfig(stream_tile=64), qcfg, inherit=True)
    assert selected is BACKEND.GPTQ_TORCH
    assert policy.torch_stream_tile == 64
    assert policy.torch_streaming is False
    assert policy.max_packers == 2
    assert qcfg.kernel is None
    assert _looper_policy(qcfg) is policy


def test_reusing_a_config_with_a_different_runtime(fake_load):
    """(b) the second ``runtime=`` replaces the first."""

    qcfg = QuantizeConfig(bits=4)
    fake_load(qcfg, runtime=RuntimeConfig(max_packers=2))
    second = RuntimeConfig(max_packers=3)
    fake_load(qcfg, runtime=second)
    assert qcfg.kernel is None
    assert qcfg.runtime is second
    assert _looper_policy(qcfg).max_packers == 3
    assert get_kernel_config().max_packers == 3


def test_enum_selection_after_a_typed_load_uses_defaults(fake_load):
    """(c) ``BACKEND.TORCH`` after ``TorchConfig(...)`` selects Torch with defaults."""

    qcfg = QuantizeConfig(bits=4)
    fake_load(qcfg, backend=TorchConfig(streaming=True, stream_tile=128))
    assert get_kernel_config().torch_streaming is True
    fake_load(qcfg, backend=BACKEND.TORCH)
    assert qcfg.kernel is None
    assert _looper_policy(qcfg).to_dict() == DEFAULTS
    assert get_kernel_config().to_dict() == DEFAULTS
    assert fake_load.calls[-1] == (BACKEND.TORCH, None)


def test_quantize_enum_after_a_typed_load_drops_the_typed_settings(fake_load):
    """(c) ``quantize(backend=BACKEND.MARLIN)`` after a typed Torch load."""

    qcfg = QuantizeConfig(bits=4)
    fake_load(qcfg, backend=TorchConfig(streaming=True))
    selected, policy = resolve_quantize_backend(BACKEND.MARLIN, qcfg, inherit=True)
    assert (selected, policy) == (BACKEND.MARLIN, None)
    assert _looper_policy(qcfg).to_dict() == DEFAULTS
    # AUTO now has no typed choice left to inherit.
    assert resolve_quantize_backend(BACKEND.AUTO, qcfg, inherit=True) == (BACKEND.AUTO, None)


def test_config_runtime_conflicts_with_a_kernel_argument(fake_load):
    """Related: ``QuantizeConfig.runtime`` + ``load(kernel=...)`` used to drop the runtime."""

    runtime = RuntimeConfig(max_packers=2, gptq_fused_block_update=False)
    qcfg = QuantizeConfig(bits=4, runtime=runtime)
    with pytest.raises(ValueError, match="QuantizeConfig.runtime and legacy kernel= cannot be supplied together"):
        fake_load(qcfg, kernel=KernelConfig(marlin_fp32=False))
    # Nothing was recorded from the rejected call.
    assert qcfg.kernel is None and qcfg.runtime is runtime


def test_config_kernel_conflicts_with_a_runtime_argument(fake_load):
    kernel = KernelConfig(marlin_fp32=False)
    qcfg = QuantizeConfig(bits=4, kernel=kernel)
    with pytest.raises(ValueError, match="runtime= and legacy QuantizeConfig.kernel cannot be supplied together"):
        fake_load(qcfg, runtime=RuntimeConfig(max_packers=2))
    assert qcfg.kernel is kernel and qcfg.runtime is None
    with pytest.raises(ValueError, match="QuantizeConfig.kernel cannot be supplied together"):
        fake_load(qcfg, backend=TorchConfig())


def test_reusing_a_config_with_a_different_legacy_kernel(fake_load):
    qcfg = QuantizeConfig(bits=4)
    fake_load(qcfg, kernel=KernelConfig(max_packers=2))
    second = KernelConfig(max_packers=3)
    fake_load(qcfg, kernel={"max_packers": 3})
    assert qcfg.kernel == second
    assert _looper_policy(qcfg).max_packers == 3


def test_quant_linear_kwargs_carry_the_resolved_policy():
    """Every config class builds its kernels from the resolved policy."""

    for qcfg in (QuantizeConfig(bits=4), AWQConfig(bits=4)):
        resolve_quantize_backend(MarlinConfig(fp32=False), qcfg, runtime=RuntimeConfig(max_packers=2))
        assert qcfg.kernel is None
        policy = qcfg.quant_linear_init_kwargs()["kernel"]
        assert policy.marlin_fp32 is False
        assert policy.max_packers == 2


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (TorchConfig(), BACKEND.GPTQ_TORCH),
        (BACKEND.GPTQ_TORCH, BACKEND.GPTQ_TORCH),
        (AwqTorchConfig(), BACKEND.AWQ_TORCH),
        (BACKEND.AWQ_TORCH, BACKEND.AWQ_TORCH),
    ],
)
def test_load_backend_selection_is_inherited_by_quantize(fake_load, selection, expected):
    """A typed config and its enum spelling must survive into quantize() alike (#4582)."""

    qcfg = AWQConfig(bits=4) if expected is BACKEND.AWQ_TORCH else QuantizeConfig(bits=4)

    fake_load(qcfg, backend=selection)
    assert fake_load.calls[-1][0] is expected

    selected, _ = resolve_quantize_backend(BACKEND.AUTO, qcfg, inherit=True)

    assert selected is expected
    assert selected is not BACKEND.AUTO


def test_explicit_enum_load_is_inherited_without_a_typed_config(fake_load):
    """The issue's exact case: enum load, then quantize(backend=AUTO)."""

    qcfg = AWQConfig(bits=4, group_size=128)
    fake_load(qcfg, backend=BACKEND.AWQ_TORCH)

    selected, policy = resolve_quantize_backend(BACKEND.AUTO, qcfg, inherit=True)

    assert selected is BACKEND.AWQ_TORCH
    assert policy is None


@pytest.mark.parametrize("invalid", ["not-a-backend", 123, object()])
def test_unknown_load_backend_raises_instead_of_falling_back_to_auto(fake_load, invalid):
    qcfg = QuantizeConfig(bits=4)

    with pytest.raises((TypeError, ValueError)):
        fake_load(qcfg, backend=invalid)

    assert getattr(qcfg, "_load_backend", None) is None
    assert resolve_quantize_backend(BACKEND.AUTO, qcfg, inherit=True) == (BACKEND.AUTO, None)
