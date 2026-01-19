# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# -- end do not touch
import json  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.looper.gptq_processor import GPTQProcessor  # noqa: E402
from gptqmodel.looper.loop_processor import LoopProcessor  # noqa: E402
from gptqmodel.looper.named_module import NamedModule  # noqa: E402
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402
from gptqmodel.utils import safetensor  # noqa: E402
from gptqmodel.utils.perplexity import Perplexity  # noqa: E402


class TestDynamic(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/"  # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tmp_quant_path = None

    def calculate_avg_ppl(self, model, tokenizer):
        ppl = Perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path="wikitext",
            dataset_name="wikitext-2-raw-v1",
            split="test",
            text_column="text",
        )

        all = ppl.calculate(n_ctx=512, n_batch=512)

        # average ppl
        avg = sum(all) / len(all)

        return avg

    @classmethod
    def setUpClass(cls):
        cls.tmp_quant_path = tempfile.TemporaryDirectory()

        # support dynamic override of bits, group_size, desc_act, sym for each layer/module match
        dynamic = {
            # `.*\.` matches the layers_node prefix
            # layer index start at 0
            r".*\.up_proj.*": {"bits": 8, "group_size": 128},  # match layer 1 gate module
            r".*\.gate_proj.*": {"bits": 8, "group_size": 128},  # match layer 2 gate module
            r".*\.down_proj.*": {"bits": 4, "group_size": 32},

            # r".*\.0\..*gate.*": {"bits": 8, "group_size": 128},  # match layer 1 gate module
            # r".*\.1\..*gate.*": {"bits": 8, "group_size": 128},  # match layer 2 gate module
            # r".*\.2\..*gate.*": {"bits": 8, "group_size": 128},  # match layer 20 gate module
            # r".*\.3\..*gate.*": {"bits": 8, "group_size": 128},  # match layer 21 gate module
        }
        quantize_config = QuantizeConfig(
            bits=4,
            dynamic=dynamic,
            group_size=128,
        )
        model = GPTQModel.load(
            cls.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )

        cls.tokenizer = model.tokenizer
        cls.calibration = cls.load_dataset(cls.tokenizer, rows=128)

        model.quantize(cls.calibration, batch_size=2)

        print(f"Model: {model.model}")

        model.save(cls.tmp_quant_path.name)

        # print quant config
        with open(cls.tmp_quant_path.name + "/quantize_config.json", 'r') as file:
            config_data = json.load(file)
            print(f"quantize_config.json: {config_data}")

        safetensor.inspect_safetensors(cls.tmp_quant_path.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_quant_path.cleanup()
        assert not os.path.exists(cls.tmp_quant_path.name)

    @parameterized.expand(
        [
            # exllama v1/v2 only supports 4bit so does not support dynamic bits control
            (BACKEND.TORCH, TorchQuantLinear, 15.643),
            (BACKEND.TRITON, TritonV2QuantLinear, 15.643),
            (BACKEND.MARLIN, MarlinQuantLinear, 15.644),
        ]
    )
    def test_dynamic_bits(self, backend, backendQLinear, expected_ppl):
        model = GPTQModel.load(
            self.tmp_quant_path.name,
            backend=backend,
        )

        for _, submodule in model.named_modules():
            if isinstance(submodule, backendQLinear):
                break
        else:
            raise ValueError(f"Did not find a `{backendQLinear}` linear layer for backend: `{backend}`")

        dynamic_bits_ppl = self.calculate_avg_ppl(model, self.tokenizer)

        del model
        print(f"Backend: {backend}, PPL: {dynamic_bits_ppl}")
        tolerance = 0.05
        lower_bound = expected_ppl * (1 - tolerance)
        upper_bound = expected_ppl * (1 + tolerance)
        assert lower_bound <= dynamic_bits_ppl <= upper_bound, \
            f"PPL expected: `{expected_ppl}` ±{tolerance * 100}%, actual = `{dynamic_bits_ppl}`"

    def test_skip_module(self):
        dynamic = {
            r"-:model\.layers\.0\..*": {},  # skip 0 layers
        }
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=QuantizeConfig(dynamic=dynamic),
        )
        model.quantize(self.calibration, batch_size=2)

        for name, submodule in model.named_modules():
            if name == 'model.model.layers.0.self_attn.q_proj' and isinstance(submodule, BaseQuantLinear):  # module 0 was skipped
                raise ValueError("first layer should be native module")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            del model

            q_model = GPTQModel.load(tmp_dir)
            self.assertInference(model=q_model, tokenizer=self.tokenizer, keywords=["paris", "king"])


######## test_dynamic_overrides.py #######


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
