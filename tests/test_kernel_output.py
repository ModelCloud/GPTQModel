import unittest

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import AdapterCache, Lora
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.model import find_modules
from parameterized import parameterized
from torch import Tensor

CUDA = torch.device("cuda:0")

class TestKernelOutput(unittest.TestCase):
    model_path = "sliuau/llama3.2-1b-4bit-group128" # hf "sliuau/llama3.2-1b-4bit-group128"

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

    @classmethod
    def setUpClass(cls):
        lora_path = "sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc"  # adapter_model.safetensors
        # hf "sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/"

        cls.m = 1
        cls.k = -1
        cls.x = None  # random X input of shape (m, k)

        cls.adapter = Lora(
            rank=128,
            path=lora_path)

        cls.adapter.post_init(cls.target, device=CUDA) # trigger adapter weight load from disk
        cls.k = cls.adapter.lora_A.shape[0]

        cls.x = torch.rand((cls.m, cls.k), device=CUDA, dtype=torch.float16)
        AdapterCache.reset() # allow next load to complete since we are hacking to get consume only 1 lora module

        # TORCH as reference output
        cls.torch_kernel_out = cls.forward(cls, backend=BACKEND.TORCH)
        cls.torch_kernel_out_with_lora = cls.forward(cls, backend=BACKEND.TORCH, adapter=cls.adapter)

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
