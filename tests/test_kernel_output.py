import unittest

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora, Adapter
from gptqmodel.nn_modules.qlinear.exllama import ExllamaQuantLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear
from gptqmodel.utils.model import find_modules
from parameterized import parameterized
from safetensors import safe_open
from torch import Tensor

CUDA = torch.device("cuda:0")

class TestKernelOutput(unittest.TestCase):
    model_path = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128/"
    lora_path = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/" #adapter_model.safetensors
    target_qliner_map = {
        # BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        # # BACKEND.EXLLAMA_EORA: ExllamaEoraQuantLinear,
        # BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
        # BACKEND.TRITON: TritonV2QuantLinear,
        # BACKEND.CUDA: DynamicCudaQuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        # BACKEND.BITBLAS: BitBLASQuantLinear,
        # BACKEND.IPEX: IPEXQuantLinear,
        # BACKEND.MARLIN: MarlinQuantLinear,
        # BACKEND.MARLIN_FP16: MarlinQuantLinear,
    }

    target = 'model.layers.6.self_attn.v_proj'
    adapter = Lora(
        rank=128,
        path=lora_path)

    m = 1
    k = -1
    x = None # random X input of shape (m, k)

    def setUp(self):
        self.adapter.post_init(self.target, device=CUDA) # trigger adapter weight load from disk
        self.k = self.adapter.lora_A.shape[1]
        self.x = torch.rand((self.m, self.k), device=CUDA, dtype=torch.float16)
        Adapter.reset_loader_cache() # hack reset loader

        # TORCH as reference output
        self.torch_kernel_out_with_lora = self.forward(backend=BACKEND.TORCH, adapter=self.adapter)
        self.torch_kernel_out = self.forward(backend=BACKEND.TORCH)

    def forward(self, backend, adapter=None):
        model = GPTQModel.load(self.model_path, backend=backend, adapter=adapter)

        target_qlinear_cls = self.target_qliner_map[backend]

        modules = find_modules(model.model, layers=[target_qlinear_cls])
        result = None
        for name, module in modules.items():
            if name == self.target:
                result = module(self.x)
                break

        assert result is not None

        del module
        del model
        torch.cuda.empty_cache()

        return result

    def assert_on_mismatch(self, a: Tensor, b: Tensor, rtol=0.00005, atol=0.00005):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

    @parameterized.expand([
        (BACKEND.TORCH, 0.0000, 0.0000),
        # (BACKEND.TRITON, 0.00001, 0.00001),
        # (BACKEND.EXLLAMA_V1, 0.09, 0.0001),
        # (BACKEND.EXLLAMA_V2, 0.136, 0.0001),
        # (BACKEND.MARLIN, 0.00005, 0.00005),
        # (BACKEND.MARLIN_FP16, 0.0001, 0.0035),
        # (BACKEND.EXLLAMA_EORA)
    ])
    def test_kernel_output(self, backend: BACKEND, r_tolerance: float, a_tolerance: float):
        out = self.forward(backend=backend)

        print(f"backend: {backend} ")
        print(out[0][:100])

        # torch vs exllama v1
        self.assert_on_mismatch(self.torch_kernel_out, out, r_tolerance, a_tolerance)  # use torch as reference

    @parameterized.expand([
        (BACKEND.TORCH, 0.0000, 0.0000),
        # (BACKEND.TRITON, 0.00001, 0.00001),
        # (BACKEND.EXLLAMA_V1, 0.015, 0.0008),
        # (BACKEND.EXLLAMA_V2, 0.16, 0.0003),
        # (BACKEND.MARLIN, 0.00001, 0.00003),
        # (BACKEND.MARLIN_FP16, 0.0001, 0.0035),
        # (BACKEND.EXLLAMA_EORA)
    ])
    def test_kernel_output_with_lora(self, backend: BACKEND, r_tolerance: float, a_tolerance: float):
        out = self.forward(backend=backend, adapter=self.adapter)

        print(f"backend: {backend} with lora")
        print(out[0][:10])

        # torch vs exllama v1
        self.assert_on_mismatch(self.torch_kernel_out_with_lora, out, r_tolerance, a_tolerance)  # use torch as reference
