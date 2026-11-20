# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A method config must not accept a shared option its quantizer never reads.

`#5583` (the `#5302` class): `desc_act`, `mse`, `gptaq` and `act_group_aware`
are fields of `PreProcessorConfig`/`GPTQConfig`, so every method config accepted
them, the writer persisted them in `quantize_config.json`, and the quantizer
never looked at them:

    AWQConfig(desc_act=True)        -> identical tensors, "desc_act": true
    AWQConfig(mse=2.0)              -> identical tensors, meta.mse = 2.0
    GGUFConfig(mse=2.0)             -> identical tensors, meta.mse = 2.0
    BitsAndBytesConfig(mse=2.0)     -> identical tensors, meta.mse = 2.0
    QQQConfig(gptaq=GPTAQConfig())  -> identical tensors, meta.gptaq = {...}
    QQQConfig(act_group_aware=True) -> identical tensors, meta.act_group_aware

The tests pin the rejection at construction, the controls that must keep
working (GPTQ/QQQ/RTN really do apply `mse`), the `dynamic` rules, the load
path for a checkpoint written before the fix, and the fact that AWQ never
persists `desc_act: true`.
"""

import pytest

from gptqmodel.quantization.config import (
    AWQConfig,
    BitsAndBytesConfig,
    GGUFConfig,
    GPTAQConfig,
    GPTQConfig,
    MSEConfig,
    QQQConfig,
    QuantizeConfig,
    RTNConfig,
)


#: `(label, factory, option, expected wording in the remedy)`. The factory
#: must fail at construction; the wording check makes sure the message tells
#: the caller what to do instead.
REJECTED = [
    (
        "awq-desc-act",
        lambda: AWQConfig(bits=4, group_size=128, desc_act=True),
        "desc_act",
        "desc_act=False",
    ),
    (
        "awq-mse",
        lambda: AWQConfig(bits=4, group_size=128, mse=2.0),
        "mse.norm",
        "clip_grid_budget",
    ),
    (
        "gguf-mse",
        lambda: GGUFConfig(bits=4, format="q_k_m", mse=2.0),
        "mse.norm",
        "remove `mse`",
    ),
    (
        "bnb-mse",
        lambda: BitsAndBytesConfig(bits=4, mse=2.0),
        "mse.norm",
        "remove `mse`",
    ),
    (
        "qqq-gptaq",
        lambda: QQQConfig(bits=4, group_size=128, gptaq=GPTAQConfig()),
        "gptaq",
        "remove `gptaq`",
    ),
    (
        "qqq-act-group-aware",
        lambda: QQQConfig(bits=4, group_size=128, act_group_aware=True),
        "act_group_aware",
        "act_group_aware=False",
    ),
]


@pytest.mark.parametrize(
    ("factory", "field", "remedy"),
    [case[1:] for case in REJECTED],
    ids=[case[0] for case in REJECTED],
)
def test_config_rejects_the_unapplied_option(factory, field, remedy):
    """The issue's own repro must fail at config construction, not silently run."""

    with pytest.raises(ValueError) as excinfo:
        factory()

    message = str(excinfo.value)
    assert f"`{field}`" in message, message
    assert "is not applied by the" in message, message
    assert "quantizer" in message, message
    assert remedy in message, message


# --- controls: the options that do apply still work -------------------------


def test_gptq_qqq_and_rtn_keep_the_mse_they_apply():
    for config in (
        GPTQConfig(bits=4, group_size=128, mse=2.0),
        QQQConfig(bits=4, group_size=128, mse=2.0),
        RTNConfig(bits=4, group_size=128, mse=2.0),
    ):
        assert isinstance(config.mse, MSEConfig)
        assert config.mse.norm == pytest.approx(2.0)


def test_awq_keeps_the_clip_budget_it_applies():
    config = AWQConfig(bits=4, group_size=128, mse={"clip_grid_budget": 0})
    assert config.mse.clip_grid_budget == 0
    assert config.mse.norm == 0.0


def test_qqq_keeps_the_defaults_it_supports():
    config = QQQConfig(bits=4, group_size=128, gptaq=None, act_group_aware=False)
    assert config.gptaq is None
    assert config.act_group_aware is False


def test_awq_never_persists_desc_act_true():
    """AWQ cannot ask for activation reordering, so the checkpoint cannot claim it."""

    payload = AWQConfig(bits=4, group_size=128).to_dict()
    assert payload["desc_act"] is False


# --- dynamic rules ----------------------------------------------------------


DYNAMIC_REJECTED = [
    (
        "awq-dynamic-desc-act",
        lambda: AWQConfig(
            bits=4, group_size=128, dynamic={".*k_proj": {"desc_act": True}}
        ),
        "dynamic['.*k_proj'].desc_act",
    ),
    (
        "awq-dynamic-mse",
        lambda: AWQConfig(
            bits=4, group_size=128, dynamic={".*k_proj": {"mse": {"norm": 2.0}}}
        ),
        "dynamic['.*k_proj'].mse.norm",
    ),
    (
        "gguf-dynamic-mse",
        lambda: GGUFConfig(
            bits=4, format="q_k_m", dynamic={".*down_proj": {"mse": 2.0}}
        ),
        "dynamic['.*down_proj'].mse.norm",
    ),
    (
        "bnb-dynamic-mse",
        lambda: BitsAndBytesConfig(bits=4, dynamic={".*down_proj": {"mse": 2.0}}),
        "dynamic['.*down_proj'].mse.norm",
    ),
    (
        "qqq-dynamic-gptaq",
        lambda: QQQConfig(
            bits=4, group_size=128, dynamic={".*o_proj": {"gptaq": {"alpha": 0.5}}}
        ),
        "dynamic['.*o_proj'].gptaq",
    ),
    (
        "qqq-dynamic-act-group-aware",
        lambda: QQQConfig(
            bits=4, group_size=128, dynamic={".*o_proj": {"act_group_aware": True}}
        ),
        "dynamic['.*o_proj'].act_group_aware",
    ),
]


@pytest.mark.parametrize(
    ("factory", "field"),
    [case[1:] for case in DYNAMIC_REJECTED],
    ids=[case[0] for case in DYNAMIC_REJECTED],
)
def test_dynamic_rule_rejects_the_unapplied_option(factory, field):
    """A per-module rule is merged later, but the option is still never consumed."""

    with pytest.raises(ValueError) as excinfo:
        factory()
    message = str(excinfo.value)
    assert f"`{field}`" in message, message
    assert "not applied by the" in message, message


def test_dynamic_controls_keep_the_options_that_apply():
    gptq = GPTQConfig(bits=4, group_size=128, dynamic={".*k_proj": {"mse": 2.0}})
    assert gptq.dynamic[".*k_proj"]["mse"] == 2.0

    awq = AWQConfig(
        bits=4, group_size=128, dynamic={".*k_proj": {"mse": {"clip_grid_budget": 0}}}
    )
    assert awq.dynamic[".*k_proj"]["mse"]["clip_grid_budget"] == 0


# --- load path: a pre-fix checkpoint must still load ------------------------


def _legacy_payload(config, **meta_and_top):
    payload = config.to_dict()
    for key, value in meta_and_top.items():
        payload[key] = value
        if isinstance(payload.get("meta"), dict):
            payload["meta"][key] = value
    return payload


def test_pre_fix_awq_checkpoint_drops_desc_act_and_mse():
    """The options never changed a weight, so the stored config is corrected."""

    payload = _legacy_payload(
        AWQConfig(bits=4, group_size=128),
        desc_act=True,
        mse=2.0,
    )
    payload["dynamic"] = {".*k_proj": {"desc_act": True, "mse": {"norm": 2.0}}}

    for restored in (
        QuantizeConfig.from_quant_config(payload),
        QuantizeConfig(**payload),
    ):
        assert isinstance(restored, AWQConfig)
        assert restored.desc_act is False
        assert restored.mse.norm == 0.0
        assert restored.dynamic == {".*k_proj": {}}
        # Re-saving writes a config that no longer claims the options.
        resaved = restored.to_dict()
        assert resaved["desc_act"] is False
        assert resaved["meta"].get("mse") is None


def test_pre_fix_gguf_and_bitsandbytes_checkpoints_drop_mse():
    for config in (
        GGUFConfig(bits=4, format="q_k_m"),
        BitsAndBytesConfig(bits=4),
    ):
        payload = _legacy_payload(config, mse=2.0)
        restored = QuantizeConfig.from_quant_config(payload)
        assert type(restored) is type(config)
        assert restored.mse.norm == 0.0
        assert restored.to_dict()["meta"].get("mse") is None


def test_pre_fix_qqq_checkpoint_drops_gptaq_and_act_group_aware():
    payload = _legacy_payload(
        QQQConfig(bits=4, group_size=128),
        gptaq={"alpha": 0.25, "device": "auto"},
        act_group_aware=True,
    )
    payload["dynamic"] = {
        ".*o_proj": {"gptaq": {"alpha": 0.5}, "act_group_aware": True}
    }

    for restored in (
        QuantizeConfig.from_quant_config(payload),
        QuantizeConfig(**payload),
    ):
        assert isinstance(restored, QQQConfig)
        assert restored.gptaq is None
        assert restored.act_group_aware is False
        assert restored.dynamic == {".*o_proj": {}}
        resaved = restored.to_dict()
        assert resaved["meta"].get("gptaq") is None
        assert resaved["meta"].get("act_group_aware") is False


def test_pre_fix_checkpoint_keeps_the_options_gptq_applies():
    """GPTQ/QQQ/RTN do apply `mse`, so their serialized value must survive."""

    payload = _legacy_payload(GPTQConfig(bits=4, group_size=128), mse=2.0)
    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, GPTQConfig)
    assert restored.mse.norm == pytest.approx(2.0)
    assert restored.to_dict()["meta"]["mse"] == pytest.approx(2.0)

    qqq = _legacy_payload(QQQConfig(bits=4, group_size=128), mse=2.0)
    restored_qqq = QuantizeConfig.from_quant_config(qqq)
    assert restored_qqq.mse.norm == pytest.approx(2.0)


def test_mse_is_a_real_weight_transform_where_it_is_accepted():
    """Control for the end-to-end claim: the option bites on RTN's CPU path."""

    import torch

    from gptqmodel.quantization.rtn import RTN

    torch.manual_seed(0)
    layer = torch.nn.Linear(128, 64, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(64, 128, dtype=torch.float32))

    def packed(mse):
        config = RTNConfig(
            bits=4,
            group_size=32,
            device="cpu",
            offload_to_disk=False,
            mse=mse,
        )
        weight, scales, zeros, *_ = RTN(layer, qcfg=config).quantize()
        return weight.clone(), scales.clone()

    plain_w, plain_s = packed(0.0)
    mse_w, mse_s = packed(2.0)

    assert not torch.equal(plain_w, mse_w), "the MSE grid must reach the packed weights"
    assert not torch.equal(plain_s, mse_s), "the MSE grid must reach the scales"
