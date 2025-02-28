import os.path
import unittest

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.dynamic_cuda import DynamicCudaQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.exllama import ExllamaQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.ipex import IPEXQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear  # noqa: E402
from gptqmodel.utils.model import find_modules
from parameterized import parameterized
from safetensors.torch import load_file


class TestPackable(unittest.TestCase):
    QLINEAR_DICT = {
        BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
        BACKEND.TRITON: TritonV2QuantLinear,
        BACKEND.CUDA: DynamicCudaQuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        # BACKEND.BITBLAS: BitBLASQuantLinear,
        BACKEND.IPEX: IPEXQuantLinear,
        BACKEND.MARLIN: MarlinQuantLinear,
        BACKEND.MARLIN_FP16: MarlinQuantLinear,
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
        list(QLINEAR_DICT.keys())
    )
    def test_post_init(self, backend: BACKEND):
        model = GPTQModel.load(self.model_id, backend=backend, device_map="auto")
        module = find_modules(model.model, [self.QLINEAR_DICT[backend]])[self.TARGET]
        # assert len(modules) == 1
        state_dict = module.state_dict()
        device = module.qweight.data.device
        assert torch.equal(state_dict["qweight"], self.orgi_module_qweight.to(device))
        # assert torch.equal(state_dict["qzeros"], self.orgi_module_qzeros.to(device))
        assert torch.equal(state_dict["scales"], self.orgi_module_scales.to(device))
        assert torch.equal(state_dict["g_idx"], self.orgi_module_g_idx.to(device))

        del state_dict
        del module
        del model
