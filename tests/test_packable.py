# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path
import unittest
from typing import Dict

import torch
from logbar import LogBar
from parameterized import parameterized
from safetensors.torch import load_file

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.exllama import ExllamaQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.exllama_eora import ExllamaEoraQuantLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedQuantLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear  # noqa: E402
from gptqmodel.utils.model import convert_gptq_v2_to_v1_format, find_modules


log = LogBar.shared()

class TestPackable(unittest.TestCase):
    QLINEAR_DICT = {
        BACKEND.EXLLAMA_EORA: ExllamaEoraQuantLinear,
        BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
        BACKEND.TRITON: TritonV2QuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        BACKEND.TORCH_FUSED: TorchFusedQuantLinear,
        BACKEND.MARLIN: MarlinQuantLinear,
        BACKEND.MARLIN_FP16: MarlinQuantLinear,
        # BACKEND.BITBLAS: BitBLASQuantLinear,
    }

    model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    TARGET = "model.layers.6.self_attn.v_proj"

    @classmethod
    def setUpClass(cls):
        weights = load_file(os.path.join(cls.model_id, "gptq_model-4bit-128g.safetensors"))
        cls.orgi_module_qweight = weights[f"{cls.TARGET}.qweight"]
        cls.orgi_module_qzeros = weights[f"{cls.TARGET}.qzeros"]
        cls.orgi_module_scales = weights[f"{cls.TARGET}.scales"]
        cls.orgi_module_g_idx = weights[f"{cls.TARGET}.g_idx"]
        del weights

    @parameterized.expand(
        [
            (BACKEND.EXLLAMA_EORA, {"qweight": False, "qzeros": True, "scales": True, "g_idx": False}),
            (BACKEND.EXLLAMA_V1, {"qweight": True, "qzeros": True, "scales": True, "g_idx": True}),
            (BACKEND.EXLLAMA_V2, {"qweight": False, "qzeros": True, "scales": True, "g_idx": True}),
            (BACKEND.TRITON, {"qweight": True, "qzeros": True, "scales": True, "g_idx": True}),
            (BACKEND.TORCH, {"qweight": True, "qzeros": True, "scales": True, "g_idx": True}),
            (BACKEND.TORCH_FUSED, {"qweight": True, "qzeros": True, "scales": False, "g_idx": True}),
            # (BACKEND.BITBLAS, {"qweight": True, "qzeros": True, "scales": True, "g_idx": True}),
            (BACKEND.MARLIN, {"qweight": False, "qzeros": False, "scales": False, "g_idx": False}),
            (BACKEND.MARLIN_FP16, {"qweight": False, "qzeros": False, "scales": False, "g_idx": False}),
        ]
    )
    def test_post_init(self, backend: BACKEND, equal: Dict[str, bool]):
        model = GPTQModel.load(self.model_id, backend=backend, device="cpu" if backend == BACKEND.TORCH_FUSED else "cuda")
        model = convert_gptq_v2_to_v1_format(model, model.quantize_config, self.QLINEAR_DICT[backend])

        module = find_modules(model.model, [self.QLINEAR_DICT[backend]])[self.TARGET]
        state_dict = module.state_dict()
        device = module.qweight.data.device

        def check(stat, key, expect_tensor):
            r = torch.equal(stat[key], expect_tensor)

            if equal[key]:
                if not r:
                    log.error(f"Expected `{key}` to be `{expect_tensor[:10]}`, but got `{stat[key][:10]}`")
                assert r
            else:
                assert not r

        check(state_dict, "qweight", self.orgi_module_qweight.to(device))
        check(state_dict, "qzeros", self.orgi_module_qzeros.to(device))
        check(state_dict, "scales", self.orgi_module_scales.to(device))
        check(state_dict, "g_idx", self.orgi_module_g_idx.to(device))

        del state_dict
        del module
        del model
