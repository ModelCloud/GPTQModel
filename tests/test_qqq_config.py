# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel import QQQConfig, QuantizeConfig
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.qqq import QQQ


def test_qqq_config_defaults_to_no_activation_order():
    config = QQQConfig(bits=4, group_size=128, offload_to_disk=False)

    assert config.method == METHOD.QQQ
    assert config.format == FORMAT.QQQ
    assert config.desc_act is False
    assert config.to_dict()["desc_act"] is False


def test_quantize_config_factory_defaults_qqq_to_no_activation_order():
    config = QuantizeConfig(
        bits=4,
        group_size=128,
        quant_method=METHOD.QQQ,
        format=FORMAT.QQQ,
        offload_to_disk=False,
    )

    assert isinstance(config, QQQConfig)
    assert config.desc_act is False


def test_qqq_checkpoint_config_without_desc_act_defaults_to_false():
    config = QuantizeConfig.from_quant_config(
        {
            "group_size": 128,
            "quant_method": "qqq",
            "wbits": 4,
        }
    )

    assert isinstance(config, QQQConfig)
    assert config.desc_act is False


def test_qqq_config_rejects_activation_order():
    with pytest.raises(ValueError, match="desc_act=False"):
        QQQConfig(
            bits=4,
            group_size=128,
            desc_act=True,
            offload_to_disk=False,
        )


def test_qqq_config_rejects_dynamic_activation_order():
    with pytest.raises(ValueError, match="model.layers.0.self_attn.q_proj"):
        QQQConfig(
            bits=4,
            group_size=128,
            dynamic={
                "model.layers.0.self_attn.q_proj": {
                    "desc_act": True,
                }
            },
            offload_to_disk=False,
        )


def test_qqq_kernels_declare_no_activation_order_support():
    assert QQQLinear.SUPPORTS_DESC_ACT == [False]
    assert QQQTorchLinear.SUPPORTS_DESC_ACT == [False]


def test_qqq_linear_rejects_activation_order():
    with pytest.raises(ValueError, match="activation ordering"):
        QQQLinear(
            bits=4,
            group_size=128,
            desc_act=True,
            sym=True,
            in_features=128,
            out_features=128,
            bias=False,
            register_buffers=False,
        )


def test_qqq_quantize_rejects_activation_order_before_hessian_work():
    layer = torch.nn.Linear(128, 128, bias=False)
    config = QQQConfig(bits=4, group_size=128, offload_to_disk=False)
    config.desc_act = True

    task = QQQ(layer, config)
    with pytest.raises(ValueError, match="desc_act=False"):
        task.quantize()
