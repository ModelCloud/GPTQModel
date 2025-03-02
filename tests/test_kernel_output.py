import unittest

import torch
from parameterized import parameterized
from safetensors import safe_open
from torch import Tensor

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear.dynamic_cuda import DynamicCudaQuantLinear
from gptqmodel.nn_modules.qlinear.exllama import ExllamaQuantLinear
from gptqmodel.nn_modules.qlinear.exllama_eora import ExllamaEoraQuantLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear
from gptqmodel.utils.model import find_modules


class TestKernelOutput(unittest.TestCase):
    model_path = "/mnt/home/shihyangl/llama3.2-1b-4bit-group128/"
    lora_path = "/mnt/home/shihyangl/llama3.2-1b-4bit-group128-eora-rank128-arc/adapter_model.safetensors"
    target_qliner_map = {
        BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        BACKEND.EXLLAMA_EORA: ExllamaEoraQuantLinear,
        BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
        BACKEND.TRITON: TritonV2QuantLinear,
        BACKEND.CUDA: DynamicCudaQuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        # BACKEND.BITBLAS: BitBLASQuantLinear,
        # BACKEND.IPEX: IPEXQuantLinear,
        BACKEND.MARLIN: MarlinQuantLinear,
    }

    @classmethod
    def setUpClass(cls):
        cls.target = 'model.layers.6.self_attn.v_proj'
        eora_tensors = {}
        # with safe_open("/home/shihyangl/llama3.2-1b-4bit-group128-eora-rank128-arc-v2/adapter_model.safetensors",
        with safe_open(
                cls.lora_path,
                framework="pt", device=0) as f:
            for k in f.keys():
                # print(k)
                if cls.target in k:
                    eora_tensors[k] = f.get_tensor(k)

        m = 1
        k = eora_tensors[f'{cls.target}.lora_A.weight'].shape[1]
        n = eora_tensors[f'{cls.target}.lora_B.weight'].shape[0]
        r = 128

        bit = 4

        cls.x = torch.rand((m, k), device='cuda', dtype=torch.float16)
        cls.eora_a = eora_tensors[f'{cls.target}.lora_A.weight'].to('cuda:0').T
        cls.eora_b = eora_tensors[f'{cls.target}.lora_B.weight'].to('cuda:0').T

        cls.adapter = Lora(path=cls.lora_path, rank=128)

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

    def assert_on_mismatch(self, a: Tensor, b: Tensor, rtol=0.05, atol=0.5):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

    @parameterized.expand([
        (BACKEND.TORCH),
        (BACKEND.CUDA),
        (BACKEND.TRITON),
        (BACKEND.EXLLAMA_V1),
        (BACKEND.EXLLAMA_V2),
        (BACKEND.MARLIN),
        (BACKEND.EXLLAMA_EORA)
    ])
    def test_kernel_output(self, backend: BACKEND):
        out = self.forward(backend=backend)

        print(f"backend: {backend} ")
        print(out[0][:10])

        # torch vs exllama v1
        self.assert_on_mismatch(self.torch_kernel_out, out)  # use torch as reference

    @parameterized.expand([
        (BACKEND.TORCH),
        (BACKEND.CUDA),
        (BACKEND.TRITON),
        (BACKEND.EXLLAMA_V1),
        (BACKEND.EXLLAMA_V2),
        (BACKEND.MARLIN),
        (BACKEND.EXLLAMA_EORA)
    ])
    def test_kernel_output_with_lora(self, backend: BACKEND):
        out = self.forward(backend=backend, adapter=self.adapter)

        print(f"torch output with lora {self.torch_kernel_out_with_lora[0][:10]}")
        print(f"backend: {backend} with lora")
        print(out[0][:10])

        # torch vs exllama v1
        self.assert_on_mismatch(self.torch_kernel_out_with_lora, out)  # use torch as reference
