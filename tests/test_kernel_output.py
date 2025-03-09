import unittest

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Adapter, AdapterCache, Lora
from gptqmodel.nn_modules.qlinear.exllama import ExllamaQuantLinear
from gptqmodel.nn_modules.qlinear.exllama_eora import ExllamaEoraQuantLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear
from gptqmodel.utils.model import find_modules
from logbar import LogBar
from parameterized import parameterized
from torch import Tensor

log = LogBar.shared()

CUDA = torch.device("cuda:0")

class TestKernelOutput(unittest.TestCase):
    model_path = "sliuau/llama3.2-1b-4bit-group128" # hf "sliuau/llama3.2-1b-4bit-group128"

    target_qliner_map = {
        BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        BACKEND.EXLLAMA_EORA: ExllamaEoraQuantLinear,
        BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
        BACKEND.TRITON: TritonV2QuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        # BACKEND.BITBLAS: BitBLASQuantLinear,
        # BACKEND.IPEX: IPEXQuantLinear,
        BACKEND.MARLIN: MarlinQuantLinear,
        BACKEND.MARLIN_FP16: MarlinQuantLinear,
    }

    target = 'model.layers.6.self_attn.v_proj'
    random_input_samples = 1000

    @classmethod
    def setUpClass(cls):
        lora_path = "sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc"  # adapter_model.safetensors
        # hf "sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/"

        cls.m = 1
        cls.k = -1


        cls.x = []  # random X input of shape (m, k)

        cls.adapter = Lora(
            rank=128,
            path=lora_path)

        cls.adapter.post_init(cls.target, device=CUDA) # trigger adapter weight load from disk
        cls.k = cls.adapter.lora_A.shape[0]

        for _ in log.pb(range(cls.random_input_samples)).title("Generate Random Inputs"):
            cls.x.append(torch.rand((cls.m, cls.k), device=CUDA, dtype=torch.float16))

        AdapterCache.reset() # allow next load to complete since we are hacking to get consume only 1 lora module

        # TORCH as reference output
        cls.torch_kernel_out = cls.forward(cls, backend=BACKEND.TORCH)
        cls.torch_kernel_out_with_lora = cls.forward(cls, backend=BACKEND.TORCH, adapter=cls.adapter)

    def forward(self, backend: BACKEND, adapter: Adapter = None):
        model = GPTQModel.load(self.model_path, backend=backend, adapter=adapter)

        target_qlinear_cls = self.target_qliner_map[backend]

        modules = find_modules(model.model, layers=[target_qlinear_cls])
        result = []
        for name, module in modules.items():
            if name == self.target:
                for i in log.pb(range(self.random_input_samples)).title("Forward Pass on Random Input"):
                    result.append(module(self.x[i]))
                break

        assert result is not None

        del module
        del model
        torch.cuda.empty_cache()

        return result

    def assert_on_mismatch(self, a: Tensor, b: Tensor, atol):
        torch.testing.assert_close(a, b, rtol=0.15, atol=atol)
        #torch.allclose(a, b, rtol=0.15, atol=atol)

    @parameterized.expand([
        (BACKEND.TORCH, 0.0000),
        (BACKEND.TRITON, 0.00001),
        (BACKEND.EXLLAMA_V1, 0.0015),
        (BACKEND.EXLLAMA_V2, 0.00075),
        (BACKEND.MARLIN, 0.00005),
        (BACKEND.MARLIN_FP16, 0.0035),
        (BACKEND.EXLLAMA_EORA, 0.0011)
    ])
    def test_kernel_output(self, backend: BACKEND, a_tolerance: float):
        out = self.forward(backend=backend)

        # torch as ref
        pb = log.pb(range(len(out))).title("Actual Kernel Output With Lora").manual()
        for i in pb:
            pb.subtitle(f"backed = `{backend}`").draw()
            try:
                self.assert_on_mismatch(self.torch_kernel_out[i], out[i],
                                        a_tolerance)  # use torch as reference
            except AssertionError:
                log.error(
                    f"Torch with Lora output: backed = `{backend}`, i = `{i}`, {self.torch_kernel_out[i][:10]}")
                raise AssertionError

    @parameterized.expand([
        (BACKEND.TORCH, 0.0000),
        (BACKEND.TRITON, 0.00001),
        (BACKEND.EXLLAMA_V1, 0.0015),
        (BACKEND.EXLLAMA_V2, 0.00086),
        (BACKEND.MARLIN, 0.00003),
        (BACKEND.MARLIN_FP16, 0.0035),
        (BACKEND.EXLLAMA_EORA, 0.0014)
    ])
    def test_kernel_output_with_lora(self, backend: BACKEND, a_tolerance: float):
        out = self.forward(backend=backend, adapter=self.adapter)

        # torch as ref
        pb = log.pb(range(len(out))).title("Actual Kernel Output With Lora").manual()
        for i in pb:
            pb.subtitle(f"backed = `{backend}`").draw()
            try:
                self.assert_on_mismatch(self.torch_kernel_out_with_lora[i], out[i], a_tolerance)  # use torch as reference
            except AssertionError:
                log.error(f"Torch with Lora output: backed = `{backend}`, i = `{i}`, {self.torch_kernel_out_with_lora[i][:10]}")
                raise AssertionError
