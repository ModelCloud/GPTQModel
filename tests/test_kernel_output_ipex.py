import unittest

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.ipex import IPEXQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.utils.model import find_modules
from logbar import LogBar
from parameterized import parameterized
from torch import Tensor

log = LogBar.shared()


class TestKernelOutput(unittest.TestCase):
    model_path = "sliuau/llama3.2-1b-4bit-group128" # hf "sliuau/llama3.2-1b-4bit-group128"
    target_qliner_map = {
        BACKEND.TORCH: TorchQuantLinear,
        BACKEND.IPEX: IPEXQuantLinear,
    }
    target = 'model.layers.6.self_attn.v_proj'
    device_map = "cpu"
    m = 8
    k = 2048
    torch_dtype = torch.float16
    r_tolerance = 0.0
    a_tolerance = 0.01
    random_input_samples = 100

    @classmethod
    def setUp(self):
        self.torch_model = GPTQModel.load(self.model_path, backend=BACKEND.TORCH, device_map=self.device_map, torch_dtype=self.torch_dtype)
        self.x = []
        self.torch_kernel_outs = []
        for i in range(self.random_input_samples):
            self.x.append(torch.rand((self.m, self.k), dtype=self.torch_dtype))
            self.torch_kernel_outs.append(self.forward(self, self.torch_model, self.x[i], backend=BACKEND.TORCH))

    def forward(self, model, x, backend: BACKEND):
        target_qlinear_cls = self.target_qliner_map[backend]

        modules = find_modules(model.model, layers=[target_qlinear_cls])
        result = None
        for name, module in modules.items():
            if name == self.target:
                result = module(x.to(model.device))
                break

        assert result is not None

        return result

    def assert_on_mismatch(self, a: Tensor, b: Tensor, rtol=0.00005, atol=0.00005):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

    @parameterized.expand([
        (BACKEND.TORCH, 0.0000, 0.00025),
        # (BACKEND.TRITON,  0.0000, 0.00025),
        (BACKEND.IPEX,  r_tolerance, a_tolerance),
    ])
    def test_kernel_output(self, backend: BACKEND, r_tolerance: float, a_tolerance: float):
        model = GPTQModel.load(self.model_path, backend=backend, device_map=self.device_map, torch_dtype=self.torch_dtype)
        log.info(f"device_map: {self.device_map} ")
        log.info(f"backend: {backend} ")
        for i in range(self.random_input_samples):
            out = self.forward(model, self.x[i], backend=backend)
            self.assert_on_mismatch(self.torch_kernel_outs[i], out, r_tolerance, a_tolerance)  # use torch as reference


class TestKernelOutputBFloat16(TestKernelOutput):
    torch_dtype = torch.bfloat16


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), reason="Test requires XPU")
class TestKernelOutputXPU(TestKernelOutput):
    device_map = "xpu:0"
    a_tolerance = 0.0005


class TestKernelOutputXPUBFloat16(TestKernelOutputXPU):
    torch_dtype = torch.bfloat16