# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import QuantizeConfig


def _make_processor(qcfg: QuantizeConfig) -> GPTQProcessor:
    return GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=[{"input_ids": [0], "attention_mask": [1]}],
        prepare_dataset_func=lambda calibration_dataset, **_kwargs: calibration_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )


def test_dynamic_overrides_apply_per_module(monkeypatch):
    monkeypatch.setattr(LoopProcessor, "_init_device_smi_handles", lambda _self: {})

    qcfg = QuantizeConfig(
        dynamic={
            "model.linear": {
                "gptaq": {"alpha": 0.5, "device": "cpu"},
                "failsafe": {"strategy": "median", "threshold": "2%"},
                "hessian": {"chunk_size": 32, "chunk_bytes": 1024, "staging_dtype": "bfloat16"},
            },
        }
    )

    processor = _make_processor(qcfg)

    module = NamedModule(
        torch.nn.Linear(4, 4, bias=False),
        name="linear",
        full_name="model.linear",
        layer_index=0,
    )
    module.state[NATIVE_INPUTS_STATE_KEY] = []
    processor.preprocess(module)

    dynamic_cfg = processor.qcfg_dynamic
    assert dynamic_cfg is not None
    assert dynamic_cfg.gptaq is not None
    assert dynamic_cfg.gptaq.alpha == 0.5
    assert dynamic_cfg.gptaq.device == "cpu"
    assert dynamic_cfg.failsafe is not None
    assert dynamic_cfg.failsafe.strategy == "median"
    assert dynamic_cfg.failsafe.threshold == "2%"
    assert dynamic_cfg.hessian.chunk_size == 32
    assert dynamic_cfg.hessian.chunk_bytes == 1024
    assert dynamic_cfg.hessian.staging_dtype == torch.bfloat16

    module_other = NamedModule(
        torch.nn.Linear(4, 4, bias=False),
        name="other",
        full_name="model.other",
        layer_index=0,
    )
    processor.preprocess(module_other)

    other_cfg = processor.qcfg_dynamic
    assert other_cfg is not None
    assert other_cfg.gptaq is None
    assert other_cfg.failsafe.strategy == qcfg.failsafe.strategy
    assert other_cfg.failsafe.threshold == qcfg.failsafe.threshold
    assert other_cfg.hessian.chunk_size == qcfg.hessian.chunk_size
    assert other_cfg.hessian.chunk_bytes == qcfg.hessian.chunk_bytes
    assert other_cfg.hessian.staging_dtype == qcfg.hessian.staging_dtype
