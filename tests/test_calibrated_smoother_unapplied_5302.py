# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A calibrated config must not accept a smoother its quantizer never applies.

`#5302`: `smoother=` (and its `smooth=` alias, and a `SmootherConfig` in
`preprocessors`) is a field of `PreProcessorConfig`, so every calibrated config
accepted it. The value was validated, normalized and written into
`quantize_config.json`, but AWQ/Paro/VPTQ/MACaQ/EXL3 have no weight-smoothing
step and GPTQ/QQQ read only `fallback.smooth`:

    GPTQConfig(bits=4, smooth=SmoothMAD(k=1.0))    -> identical qweight/scales
    AWQConfig(bits=4,  smooth=SmoothMAD(k=1.0))    -> identical qweight/scales

The tests below pin the rejection at construction (before a model is loaded,
matching the protocol compiler's refusal of `weight.prepare`), the control
cases that must keep working, and the load path for a checkpoint written before
the fix. The end-to-end assertions use the model weights rather than the config
so a future regression cannot pass by only rejecting or only accepting.
"""

import pytest
import torch

from gptqmodel.quantization import config as C
from gptqmodel.quantization.config import (
    AWQConfig,
    CalibrationConfig,
    EXL3Config,
    Fallback,
    GPTQConfig,
    MacaqConfig,
    ParoConfig,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
    SmoothMAD,
    VPTQConfig,
)
from gptqmodel.quantization.rtn import RTN


CALIBRATED_UNSUPPORTED = [
    ("gptq", lambda: GPTQConfig(bits=4, group_size=128)),
    ("qqq", lambda: QQQConfig(bits=4, group_size=128)),
    ("awq", lambda: AWQConfig(bits=4, group_size=128)),
    ("paro", lambda: ParoConfig(bits=4, group_size=128)),
    ("vptq", lambda: VPTQConfig()),
    ("macaq", lambda: MacaqConfig()),
    ("exl3", lambda: EXL3Config()),
]


@pytest.mark.parametrize(
    "factory",
    [factory for _, factory in CALIBRATED_UNSUPPORTED],
    ids=[name for name, _ in CALIBRATED_UNSUPPORTED],
)
def test_calibrated_configs_reject_an_unapplied_smoother(factory):
    base = factory()
    config_cls = type(base)

    kwargs = {"bits": 4}
    if config_cls is VPTQConfig:
        kwargs = {}
    with pytest.raises(ValueError) as excinfo:
        config_cls(smoother=SmoothMAD(k=1.0), **kwargs)

    message = str(excinfo.value)
    assert config_cls.__name__ in message, message
    assert "not applied" in message, message
    # The remedy must name a surface that actually applies a smoother.
    assert "Fallback(smooth=" in message or "RTN" in message, message


@pytest.mark.parametrize(
    "factory",
    [factory for _, factory in CALIBRATED_UNSUPPORTED],
    ids=[name for name, _ in CALIBRATED_UNSUPPORTED],
)
def test_calibrated_configs_reject_the_legacy_smooth_alias(factory):
    base = factory()
    kwargs = {"bits": 4} if type(base) is not VPTQConfig else {}
    with pytest.raises(ValueError, match="not applied"):
        type(base)(smooth=SmoothMAD(k=1.0), **kwargs)


def test_smoother_preprocessor_and_dict_payload_are_rejected():
    """Every spelling normalizes into `smoother`, so all of them must fail."""

    spellings = [
        {"code": "smoother", "smooth": {"type": "mad", "k": 1.0}},
        {"type": "mad", "k": 1.0},
        {"smooth": {"type": "mad", "k": 1.0}},
        "mad",
    ]
    for payload in spellings:
        with pytest.raises(ValueError, match="not applied"):
            AWQConfig(bits=4, group_size=128, preprocessors=[payload])


def test_dynamic_rule_smoother_is_rejected_for_the_same_methods():
    """A per-module rule is applied later, but it is still never consumed."""

    for cls, kwargs in (
        (AWQConfig, {"bits": 4, "group_size": 128}),
        (GPTQConfig, {"bits": 4, "group_size": 128}),
        (EXL3Config, {}),
    ):
        with pytest.raises(ValueError) as excinfo:
            cls(
                dynamic={r".*down_proj": {"smoother": SmoothMAD(k=1.0)}},
                **kwargs,
            )
        message = str(excinfo.value)
        assert "dynamic['.*down_proj'].smoother" in message, message
        assert "not applied" in message, message


def test_awq_dynamic_fallback_smoother_is_rejected():
    """AWQ has no smoothing step anywhere, so even `fallback.smooth` is a no-op."""

    with pytest.raises(ValueError) as excinfo:
        AWQConfig(
            bits=4,
            group_size=128,
            dynamic={r".*down_proj": {"fallback": {"smooth": {"type": "mad", "k": 1.0}}}},
        )
    assert "dynamic['.*down_proj'].fallback.smooth" in str(excinfo.value)
    with pytest.raises(ValueError, match="fallback.smooth"):
        AWQConfig(bits=4, group_size=128, fallback=Fallback(smooth=SmoothMAD(k=1.0)))


# --- controls: the paths that do apply a smoother still work -----------------


def test_gptq_and_qqq_keep_the_fallback_smoother_they_apply():
    for cls in (GPTQConfig, QQQConfig):
        config = cls(bits=4, group_size=128, fallback=Fallback(smooth=SmoothMAD(k=1.0)))
        assert isinstance(config.fallback.smooth, SmoothMAD)
        assert config.fallback.smooth.k == pytest.approx(1.0)
        restored = QuantizeConfig.from_quant_config(config.to_dict())
        assert restored.fallback.smooth.k == pytest.approx(1.0)


def test_weight_only_configs_keep_the_smoother_their_packer_applies():
    rtn = RTNConfig(bits=4, group_size=128, smoother=SmoothMAD(k=1.25))
    assert rtn.smooth.k == pytest.approx(1.25)
    assert rtn.uses_weight_only_lifecycle() is True

    gguf = C.GGUFConfig(bits=4, format="q_k_m", smoother=SmoothMAD(k=1.25))
    assert gguf.smooth.k == pytest.approx(1.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required")
def test_rtn_smoother_still_changes_the_quantized_weights():
    """Control for the end-to-end claim: where it is supported, it bites."""

    torch.manual_seed(0)
    layer = torch.nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(64, 128, device="cuda", dtype=torch.float16))

    def packed(smoother):
        config = RTNConfig(
            bits=4,
            group_size=128,
            device="cuda",
            offload_to_disk=False,
            smoother=smoother,
        )
        weight, scales, zeros, *_ = RTN(layer, qcfg=config).quantize()
        return weight.clone(), scales.clone(), zeros.clone()

    plain_w, plain_s, plain_z = packed(None)
    smooth_w, smooth_s, smooth_z = packed(SmoothMAD(k=1.0))

    assert not torch.equal(plain_w, smooth_w), "the smoother must reach the packed weights"
    assert not torch.equal(plain_s, smooth_s), "the smoother must reach the scales"
    # Symmetric 4-bit stores a fixed zero point of 8 in every group, so the
    # zero tensor is not a signal here; the weights and scales are.
    assert torch.equal(plain_z, smooth_z)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required")
def test_gptq_fallback_smoother_still_changes_the_fallback_grid():
    """The GPTQ fallback is the one calibraton path that reads a smoother."""

    from gptqmodel.quantization.config import FallbackStrategy
    from gptqmodel.quantization.gptq import GPTQ

    torch.manual_seed(0)
    layer = torch.nn.Linear(128, 64, bias=False, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(64, 128, device="cuda", dtype=torch.float16))

    def fallback_grid(smooth):
        config = GPTQConfig(
            bits=4,
            group_size=32,
            device="cuda",
            offload_to_disk=False,
            fallback=Fallback(strategy=FallbackStrategy.RTN, threshold=True, smooth=smooth),
        )
        task = GPTQ(layer, qcfg=config)
        task.quantizer.configure(perchannel=True)
        return task._fallback_quantize(FallbackStrategy.RTN, 128)

    plain = fallback_grid(None)
    smoothed = fallback_grid(SmoothMAD(k=1.0, group_size_threshold=16))

    assert not torch.equal(plain[0], smoothed[0]), "the fallback smoother must reach the weights"
    assert not torch.equal(plain[1], smoothed[1]), "the fallback smoother must reach the scales"


# --- load path: a pre-fix checkpoint must still load ------------------------


def test_pre_fix_checkpoint_drops_the_unapplied_smoother_on_load():
    """The option never changed a weight, so the stored config is corrected.

    A checkpoint quantized before this fix contains plain AWQ weights while
    `meta.preprocessors` claims a smoother. Refusing the load would break an
    artifact whose tensors are valid, so the loader drops the claim instead and
    the resulting config describes the weights that were actually produced.
    """

    payload = AWQConfig(bits=4, group_size=128).to_dict()
    payload["meta"]["preprocessors"] = [
        {"code": "smoother", "smooth": {"type": "mad", "k": 1.0}}
    ]

    restored = QuantizeConfig.from_quant_config(payload)

    assert isinstance(restored, AWQConfig)
    assert restored.smoother is None
    assert not any(
        getattr(item, "code", None) == "smoother" for item in restored.preprocessors
    )
    # Re-saving writes a config that no longer claims the option.
    resaved = restored.to_dict()
    assert "preprocessors" not in resaved.get("meta", {}), resaved["meta"]


def test_pre_fix_checkpoint_drops_a_top_level_smoother_on_factory_reload():
    payload = AWQConfig(bits=4, group_size=128).to_dict()
    payload["smoother"] = {"type": "mad", "k": 1.0}
    payload["smooth"] = {"type": "mad", "k": 1.0}

    restored = QuantizeConfig(**payload)

    assert isinstance(restored, AWQConfig)
    assert restored.smoother is None
    assert restored.smooth is None


def test_pre_fix_checkpoint_keeps_the_gptq_fallback_smoother():
    """The GPTQ fallback path did apply it, so the load path must keep it."""

    payload = GPTQConfig(bits=4, group_size=128).to_dict()
    payload["meta"]["fallback"] = {
        "strategy": "rtn",
        "threshold": "5%",
        "smooth": {"type": "mad", "k": 1.0},
    }

    restored = QuantizeConfig.from_quant_config(payload)

    assert isinstance(restored, GPTQConfig)
    assert isinstance(restored.fallback.smooth, SmoothMAD)
    assert restored.fallback.smooth.k == pytest.approx(1.0)


def test_pre_fix_checkpoint_keeps_a_weight_only_smoother():
    """RTN/GGUF smoothing is applied, so its serialized smoother survives."""

    payload = C.GGUFConfig(bits=4, format="q_k_m", smoother=SmoothMAD(k=1.25)).to_dict()

    restored = QuantizeConfig.from_quant_config(payload)

    assert isinstance(restored, C.GGUFConfig)
    assert restored.smooth.k == pytest.approx(1.25)


# --- the issue's own repro, now failing at construction ---------------------


def test_issue_repro_is_refused_before_a_model_is_loaded():
    """The repro from #5302 must fail at config construction, not silently run."""

    with pytest.raises(ValueError, match="not applied by the GPTQ quantizer"):
        GPTQConfig(bits=4, group_size=128, smoother=SmoothMAD(k=1.0))
    with pytest.raises(ValueError, match="not applied by the AWQ quantizer"):
        AWQConfig(bits=4, group_size=128, smoother=SmoothMAD(k=1.0))


def test_calibration_config_is_untouched():
    """The smoother is a weight transform, not a calibration option."""

    calibration = CalibrationConfig(dataset=["a b c d"])
    assert calibration.dataset == ["a b c d"]
