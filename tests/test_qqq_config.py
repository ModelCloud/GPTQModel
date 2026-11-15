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


def test_qqq_default_damp_percent_matches_gptq():
    """Issue #4074 item 4: QQQ must not damp the Hessian 10x below GPTQ.

    QQQ has no activation-group-aware calibration, so Llama's BOS massive
    activation dominates the Hessian. The old 0.005 default measured 59.5
    wikitext-2 PPL on Llama-3.2-1B QQQ4; 0.05 measures 34.4 (reference GPTQ
    without AGA).
    """
    assert QQQConfig(bits=4, group_size=128, offload_to_disk=False).damp_percent == 0.05
    assert QuantizeConfig(
        bits=4, group_size=128, quant_method=METHOD.QQQ, format=FORMAT.QQQ,
        offload_to_disk=False,
    ).damp_percent == 0.05


def test_qqq_processor_kernel_selection_follows_supports_dtypes(monkeypatch):
    """Issue #4074 item 2: the in-memory kernel must match the model dtype.

    ``QQQLinear`` declares FP16 only and the reload path rejects it for BF16.
    The quantization path must use the same ``SUPPORTS_DTYPES`` rule instead of
    always picking the CUDA kernel.
    """
    from gptqmodel.looper import qqq_processor
    from gptqmodel.utils.backend import BACKEND

    monkeypatch.setattr(qqq_processor, "IS_ROCM", False)
    processor = object.__new__(qqq_processor.QQQProcessor)
    processor.qcfg = QQQConfig(bits=4, group_size=128, offload_to_disk=False, device="cuda:0")
    processor.requested_backend = BACKEND.AUTO

    assert processor._quant_linear_kernel(dtype=torch.bfloat16) == (
        QQQTorchLinear,
        BACKEND.QQQ_TORCH,
    )
    assert processor._quant_linear_kernel(dtype=torch.float16) == (
        QQQLinear,
        BACKEND.QQQ,
    )

    # An explicit native-kernel request for a dtype it cannot run is an
    # actionable error, not a late NotImplementedError in the first forward.
    processor.requested_backend = BACKEND.QQQ
    with pytest.raises(NotImplementedError, match="QQQ_TORCH"):
        processor._quant_linear_kernel(dtype=torch.bfloat16)


def test_qqq_config_serializes_vllm_weight_bits_alias():
    payload = QQQConfig(bits=4, group_size=128, offload_to_disk=False).to_dict()

    assert payload["bits"] == 4
    assert payload["wbits"] == 4


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
