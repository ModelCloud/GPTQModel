import unittest

import torch
from safetensors import safe_open

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.exllama import ExllamaQuantLinear
from gptqmodel.nn_modules.qlinear.exllama_eora import ExllamaEoraQuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.model import find_modules


class TestKernelOutput(unittest.TestCase):
    model_path = "/monster/data/model/sliuau-llama3.2-1b-4bit-group128/"
    target_qliner_map = {
        BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        BACKEND.EXLLAMA_EORA: ExllamaEoraQuantLinear,
        # BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
        # BACKEND.TRITON: TritonV2QuantLinear,
        # BACKEND.CUDA: DynamicCudaQuantLinear,
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
                "/monster/data/model/sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/adapter_model.safetensors",
                framework="pt", device=0) as f:
            for k in f.keys():
                if cls.target in k:
                    eora_tensors[k] = f.get_tensor(k)

        m = 1
        k = eora_tensors[f'{cls.target}.lora_A.weight'].shape[1]
        n = eora_tensors[f'{cls.target}.lora_B.weight'].shape[0]
        r = 128

        bit = 4
        use_exllama = True

        cls.x = torch.rand((m, k), device='cuda', dtype=torch.float16)
        cls.eora_a = eora_tensors[f'{cls.target}.lora_A.weight'].to('cuda:0').T
        cls.eora_b = eora_tensors[f'{cls.target}.lora_B.weight'].to('cuda:0').T

    def forward(self, backend):
        model = GPTQModel.load(self.model_path, backend=backend)

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

    def test_kernel_output(self):
        torch_kernel_out = self.forward(BACKEND.TORCH)
        exllama_kernel_out = self.forward(BACKEND.EXLLAMA_V1)
        marlin_kernel_out = self.forward(BACKEND.MARLIN)
        exllama_v2v_out = self.forward(BACKEND.EXLLAMA_EORA)

        # print("gptq exllama kernel out: ")
        # print(exllama_out[0][:10])
        print("torch_kernel_out: ")
        print(torch_kernel_out[0][:10])

        print("exllama_kernel_out: ")
        print(exllama_kernel_out[0][:10])

        print("marlin_kernel_out: ")
        print(marlin_kernel_out[0][:10])

        print("exllama_v2v_out: ")
        print(exllama_v2v_out[0][:10])
