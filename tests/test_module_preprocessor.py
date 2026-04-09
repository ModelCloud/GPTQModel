# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.looper.module_preprocessor import ModulePreProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import QuantizeConfig


def test_module_preprocessor_records_auto_module_decoder_plan():
    linear = torch.nn.Linear(8, 8, bias=False)
    named = NamedModule(linear, name="proj", full_name="model.layers.0.proj", layer_index=0)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        preprocessors=[
            {
                "code": "auto_module_decoder",
                "target_dtype": "float16",
            }
        ],
    )

    processor = ModulePreProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=[],
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    processor.preprocess(named)

    assert len(named.state["preprocessor_pipeline"]) == 1
    plan = named.state["auto_module_decoder"]
    assert plan["code"] == "auto_module_decoder"
    assert plan["source_dtype"] == "auto"
    assert plan["target_dtype"] == torch.float16


def test_module_preprocessor_clears_decoder_state_when_preprocessors_absent():
    linear = torch.nn.Linear(8, 8, bias=False)
    named = NamedModule(linear, name="proj", full_name="model.layers.0.proj", layer_index=0)
    named.state["preprocessor_pipeline"] = [{"code": "stale"}]
    named.state["auto_module_decoder"] = {"code": "stale"}

    qcfg = QuantizeConfig(bits=4, group_size=128)
    processor = ModulePreProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=[],
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    processor.preprocess(named)

    assert "preprocessor_pipeline" not in named.state
    assert "auto_module_decoder" not in named.state
