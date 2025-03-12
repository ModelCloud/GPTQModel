import unittest

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Adapter
from gptqmodel.nn_modules.qlinear.ipex import IPEXQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.model import find_modules
from logbar import LogBar
from parameterized import parameterized
from torch import Tensor

log = LogBar.shared()

XPU = torch.device("xpu")

class TestKernelOutput(unittest.TestCase):
    model_path = "sliuau/llama3.2-1b-4bit-group128" # hf "sliuau/llama3.2-1b-4bit-group128"

    target_qliner_map = {
        BACKEND.TORCH: TorchQuantLinear,
        BACKEND.IPEX: IPEXQuantLinear,
    }

    target = 'model.layers.6.self_attn.v_proj'

    @classmethod
    def setUpClass(cls):

        cls.m = 1
        cls.k = 2048
        cls.x = torch.rand((cls.m, cls.k), device=XPU, dtype=torch.float16)

        # TORCH as reference output
        cls.torch_kernel_out = cls.forward(cls, backend=BACKEND.TORCH)


    def forward(self, backend: BACKEND, adapter: Adapter = None):
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
        (BACKEND.TORCH, 0.0000, 0.00025),
        (BACKEND.TRITON,  0.0000, 0.00025),
        (BACKEND.IPEX,  0.0000, 0.0005),
    ])
    def test_kernel_output(self, backend: BACKEND, r_tolerance: float, a_tolerance: float):
        out = self.forward(backend=backend)

        log.info(f"backend: {backend} ")
        log.info(out[0][:100])

        self.assert_on_mismatch(self.torch_kernel_out, out, r_tolerance, a_tolerance)  # use torch as reference
