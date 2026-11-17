# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module
from gptqmodel.quantization.config import (
    FOEMConfig,
    GPTAQConfig,
    GPTQConfig,
    QQQConfig,
    QuantizeConfig,
    VPTQConfig,
)
from gptqmodel.quantization.gptq import GPTQ


SINK_ONLY_GPTQ = "only implemented for the plain GPTQ Hessian path"


def test_vptq_rejects_exclude_attention_sinks():
    # VPTQ inherits the GPTQ field but flattens its calibration input before
    # accumulating the Hessian, so it cannot honor the mask (#4447 item 1).
    with pytest.raises(ValueError, match="Leave it False for VPTQ"):
        VPTQConfig(bits=2, exclude_attention_sinks=True)
    with pytest.raises(ValueError, match="Leave it False for VPTQ"):
        QuantizeConfig(method="vptq", exclude_attention_sinks=True)

    # The accepted default still serializes the truth: no mask ran.
    assert VPTQConfig(bits=2).to_dict()["meta"]["exclude_attention_sinks"] is False


def test_vptq_rejects_dynamic_exclude_attention_sinks_override():
    with pytest.raises(ValueError, match="dynamic overrides"):
        VPTQConfig(
            bits=2,
            dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": True}},
        )


def test_dynamic_exclude_attention_sinks_override_is_applied():
    # #4447 item 2: a per-module rule used to be accepted and serialized but
    # dropped by `apply_scalar_module_overrides`, so the clone kept False.
    config = QuantizeConfig(
        bits=4,
        group_size=2,
        dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": True}},
    )
    selected = clone_gptq_config_for_module(config, "model.layers.0.mlp.down_proj")
    unselected = clone_gptq_config_for_module(config, "model.layers.0.self_attn.q_proj")
    assert selected.exclude_attention_sinks is True
    assert unselected.exclude_attention_sinks is False
    # The serialized config keeps the per-module rule, which is what records
    # that only the matching modules ran with the mask.
    assert config.to_dict()["dynamic"][r"+:.*down_proj.*"]["exclude_attention_sinks"] is True

    # The copied field must also change what the Hessian actually sees.
    torch.manual_seed(0)
    layer = nn.Linear(4, 3, bias=False, dtype=torch.float32).eval()
    gptq = GPTQ(layer, qcfg=selected)
    gptq.quantizer.configure(perchannel=True)
    inp = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4) / 7.0
    # The processor resolves BOS flags from the calibration ids and passes
    # them down; a direct add_batch without evidence deliberately keeps every
    # row (#4526/#4568).
    gptq.add_batch(inp, None, sink_bos_mask=torch.ones(2, dtype=torch.bool))
    assert gptq.nsamples == 2 * 3  # first token of each sequence dropped


def test_dynamic_exclude_attention_sinks_override_must_be_bool():
    with pytest.raises(TypeError, match="exclude_attention_sinks.*must be a bool"):
        GPTQConfig(
            bits=4,
            group_size=2,
            dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": "yes"}},
        )


@pytest.mark.parametrize(
    "override",
    [
        {"gptaq": {"alpha": 0.25}},
        {"foem": {"alpha": 0.25, "beta": 0.5}},
    ],
)
def test_dynamic_gptaq_foem_override_with_attention_sinks_is_rejected(override):
    # #4447 item 3: the top-level pair is rejected, but a per-module GPTAQ/FOEM
    # rule bypassed the check and ran without the mask.
    with pytest.raises(ValueError, match="enables GPTAQ/FOEM"):
        GPTQConfig(
            bits=4,
            group_size=2,
            exclude_attention_sinks=True,
            dynamic={r"+:.*down_proj.*": override},
        )


def test_dynamic_attention_sinks_override_with_base_gptaq_is_rejected():
    with pytest.raises(ValueError, match="enables GPTAQ/FOEM"):
        GPTQConfig(
            bits=4,
            group_size=2,
            gptaq=GPTAQConfig(),
            dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": True}},
        )


def test_clone_rechecks_dynamic_gptaq_after_construction():
    # `dynamic` is mutable after validation; the clone path re-checks so a
    # post-construction edit cannot recreate the silent mismatch.
    config = GPTQConfig(
        bits=4,
        group_size=2,
        dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": True}},
    )
    config.dynamic[r"+:.*down_proj.*"]["gptaq"] = {"alpha": 0.25}
    with pytest.raises(ValueError, match="runs GPTAQ/FOEM"):
        clone_gptq_config_for_module(config, "model.layers.0.mlp.down_proj")


@pytest.mark.parametrize("method", ["awq", "paroquant"])
def test_factory_rejects_attention_sinks_for_methods_without_mask(method):
    # #4447 item 4: the factory filtered the flag out and the run proceeded.
    with pytest.raises(ValueError, match=SINK_ONLY_GPTQ):
        QuantizeConfig(method=method, exclude_attention_sinks=True)
    # The same method still builds when the unsupported flag is absent.
    QuantizeConfig(method=method)


@pytest.mark.parametrize("method", ["awq", "paroquant"])
def test_factory_rejects_dynamic_attention_sinks_for_methods_without_mask(method):
    with pytest.raises(ValueError, match=SINK_ONLY_GPTQ):
        QuantizeConfig(
            method=method,
            dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": True}},
        )


def test_qqq_still_rejects_and_gptq_accepts_attention_sinks():
    with pytest.raises(ValueError, match="leave it False for QQQ"):
        QuantizeConfig(method="qqq", exclude_attention_sinks=True)

    config = QuantizeConfig(method="gptq", bits=4, group_size=2, exclude_attention_sinks=True)
    assert config.to_dict()["meta"]["exclude_attention_sinks"] is True


def test_qqq_rejects_dynamic_attention_sinks_override():
    with pytest.raises(ValueError, match=SINK_ONLY_GPTQ):
        QQQConfig(
            bits=4,
            group_size=2,
            dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": True}},
        )


def test_exl3_rejects_dynamic_attention_sinks_override():
    with pytest.raises(ValueError, match="leave it False for EXL3"):
        QuantizeConfig(
            method="exl3",
            dynamic={r"+:.*down_proj.*": {"exclude_attention_sinks": True}},
        )


def test_gptaq_foem_configs_are_available_for_the_dynamic_rejections():
    # Guard the test fixtures themselves: these are the exact objects the
    # per-module override path would instantiate.
    assert GPTAQConfig(alpha=0.25).alpha == 0.25
    assert FOEMConfig(alpha=0.25, beta=0.5).beta == 0.5


@pytest.mark.parametrize(
    "method",
    [
        "awq",
        "paroquant",
        "exl3",
        "macaq",
        "fp8",
        "nf12",
        "tt-blockfloat",
        "bitsandbytes",
        "gguf",
        "vptq",
        "qqq",
    ],
)
def test_factory_rejects_attention_sinks_for_every_method_without_the_mask(method):
    # Every method that only inherits/accepts the key in the factory must say
    # so instead of filtering it out (#4447 item 4 and its sibling paths).
    with pytest.raises(ValueError, match="only implemented for"):
        QuantizeConfig(method=method, exclude_attention_sinks=True)


@pytest.mark.parametrize(
    "method,payload",
    [
        ("vptq", {"bits": 2, "group_size": 128, "sym": True, "format": "vptq"}),
        ("qqq", {"bits": 4, "group_size": 128, "sym": True, "format": "qqq"}),
        ("awq", {"bits": 4, "group_size": 128, "sym": True, "format": "gemm"}),
        ("paroquant", {"bits": 4, "group_size": 128, "sym": True, "format": "paroquant"}),
        ("exl3", {"bits": 4, "group_size": -1, "sym": True, "format": "exl3"}),
    ],
)
def test_loading_legacy_checkpoint_with_sink_metadata_true_still_works(method, payload):
    # A checkpoint written before the fix can carry `meta.exclude_attention_sinks
    # = true` even though the mask never ran. Loading must repair it to the
    # behavior that actually ran instead of making the checkpoint unloadable.
    config = QuantizeConfig.from_quant_config(
        {**payload, "method": method, "meta": {"exclude_attention_sinks": True}}
    )
    assert getattr(config, "exclude_attention_sinks", False) is False
    # A re-save must not repeat the false claim.
    assert config.to_dict()["meta"]["exclude_attention_sinks"] is False


def test_loading_legacy_gptq_dynamic_gptaq_with_sinks_is_repaired():
    # #4447 item 3: a checkpoint written by the old code can have a dynamic
    # GPTAQ rule that also claims the mask ran. Explicit construction rejects
    # it, but loading repairs the rule so the checkpoint stays usable.
    config = QuantizeConfig.from_quant_config(
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "method": "gptq",
            "format": "gptq",
            "dynamic": {
                r"+:.*down_proj.*": {
                    "exclude_attention_sinks": True,
                    "gptaq": {"alpha": 0.25},
                }
            },
        }
    )
    rule = config.to_dict()["dynamic"][r"+:.*down_proj.*"]
    assert rule["exclude_attention_sinks"] is False
    assert rule["gptaq"]["alpha"] == 0.25


def test_loading_plain_gptq_sink_metadata_true_is_preserved():
    # GPTQ implements the mask, so a true flag is real metadata and must load
    # unchanged (the repair only targets paths that ignored it).
    config = QuantizeConfig.from_quant_config(
        {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "method": "gptq",
            "format": "gptq",
            "meta": {"exclude_attention_sinks": True},
        }
    )
    assert config.exclude_attention_sinks is True
    assert config.to_dict()["meta"]["exclude_attention_sinks"] is True


@pytest.mark.parametrize(
    "override",
    [{"gptaq": {"alpha": 0.25}}, {"foem": {"alpha": 0.25, "beta": 0.5}}],
)
def test_dynamic_gptaq_foem_without_sinks_still_works(override):
    # The conflict check must not reject GPTAQ/FOEM rules on their own.
    config = GPTQConfig(
        bits=4,
        group_size=2,
        dynamic={r"+:.*down_proj.*": override},
    )
    clone = clone_gptq_config_for_module(config, "model.layers.0.mlp.down_proj")
    assert clone.exclude_attention_sinks is False
    assert (clone.gptaq is not None) or (clone.foem is not None)


def test_dynamic_gptaq_with_explicit_sinks_false_is_allowed():
    config = GPTQConfig(
        bits=4,
        group_size=2,
        exclude_attention_sinks=True,
        dynamic={
            r"+:.*down_proj.*": {
                "exclude_attention_sinks": False,
                "gptaq": {"alpha": 0.25},
            }
        },
    )
    clone = clone_gptq_config_for_module(config, "model.layers.0.mlp.down_proj")
    assert clone.exclude_attention_sinks is False
    assert clone.gptaq is not None
    plain = clone_gptq_config_for_module(config, "model.layers.0.self_attn.q_proj")
    assert plain.exclude_attention_sinks is True
    assert plain.gptaq is None

