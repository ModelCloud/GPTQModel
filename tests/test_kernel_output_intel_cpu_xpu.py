# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import unittest

import torch
from logbar import LogBar
from parameterized import parameterized
from tabulate import tabulate
from torch import Tensor

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.gemm_hf_kernel import HFKernelLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedQuantLinear
from gptqmodel.utils.model import find_modules


log = LogBar.shared()


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


class TestKernelOutput(unittest.TestCase):
    model_path = "sliuau/llama3.2-1b-4bit-group128" # hf "sliuau/llama3.2-1b-4bit-group128"
    target_qliner_map = {
        BACKEND.TORCH: TorchQuantLinear,
        BACKEND.TORCH_FUSED: TorchFusedQuantLinear,
        BACKEND.HF_KERNEL: HFKernelLinear,
    }
    target = 'model.layers.6.self_attn.v_proj'
    device = "cpu"
    m = [1, 16, 64, 256, 1024]
    k = 2048
    dtype = torch.float16
    r_tolerance = 0.0076
    a_tolerance = 0.016
    input_samples_each_size = 20 # final size == input_samples_each_size * len(m)

    @classmethod
    def setUp(self):
        self.torch_model = GPTQModel.load(self.model_path, backend=BACKEND.TORCH, device=self.device, dtype=self.dtype)
        self.x = []
        self.torch_kernel_outs = []
        for dim_0 in self.m:
            for _ in range(self.input_samples_each_size):
                inputs = torch.rand((dim_0, self.k), dtype=self.dtype)
                self.x.append(inputs)
                self.torch_kernel_outs.append(self.forward(self, self.torch_model, inputs, backend=BACKEND.TORCH))

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
        (BACKEND.TORCH, 0.0000, 0.0005),
        # (BACKEND.TRITON,  0.0000, 0.0005),
        (BACKEND.TORCH_FUSED,  r_tolerance, a_tolerance),
        (BACKEND.HF_KERNEL,  r_tolerance, a_tolerance),
    ])
    def test_kernel_output(self, backend: BACKEND, r_tolerance: float, a_tolerance: float):
        model = GPTQModel.load(self.model_path, backend=backend, device=self.device, dtype=self.dtype)
        log.info(f"device: {self.device} ")
        log.info(f"backend: {backend} ")
        for i in range(len(self.x)):
            out = self.forward(model, self.x[i], backend=backend)
            self.assert_on_mismatch(self.torch_kernel_outs[i], out, r_tolerance, a_tolerance)  # use torch as reference

class TestKernelOutputWithBias(TestKernelOutput):
    model_path = "/monster/data/model/bloom-560m-gptqmodel-4bit"
    target = 'transformer.h.6.self_attention.query_key_value'
    k = 1024

class TestKernelOutputBFloat16(TestKernelOutput):
    dtype = torch.bfloat16


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), reason="Test requires XPU")
class TestKernelOutputXPU(TestKernelOutput):
    device = "xpu:0"
    a_tolerance = 0.0005


class TestKernelOutputXPUBFloat16(TestKernelOutputXPU):
    dtype = torch.bfloat16


class TestTorchFusedAndHFKernelDevices(unittest.TestCase):
    model_path = TestKernelOutput.model_path
    target_qliner_map = TestKernelOutput.target_qliner_map
    target = TestKernelOutput.target
    dtype = torch.float16
    m = [1, 16, 64, 256]
    k = 2048
    input_samples_each_size = 5
    r_tolerance = 0.0076
    a_tolerance = 0.016
    reference_backend = BACKEND.TORCH
    reference_device = "cpu"

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.inputs = []
        for dim_0 in cls.m:
            for _ in range(cls.input_samples_each_size):
                cls.inputs.append(torch.rand((dim_0, cls.k), dtype=cls.dtype))

        reference_model = GPTQModel.load(
            cls.model_path,
            backend=cls.reference_backend,
            device=cls.reference_device,
            dtype=cls.dtype,
        )
        cls.reference_outputs = [
            cls.forward(reference_model, sample, cls.reference_backend)
            for sample in cls.inputs
        ]
        del reference_model

    @classmethod
    def forward(cls, model, x, backend: BACKEND):
        target_qlinear_cls = cls.target_qliner_map[backend]
        modules = find_modules(model.model, layers=[target_qlinear_cls])
        result = None
        for name, module in modules.items():
            if name == cls.target:
                result = module(x.to(model.device))
                break

        assert result is not None
        return result

    def assert_on_mismatch(self, a: Tensor, b: Tensor, rtol=0.00005, atol=0.00005):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)

    @parameterized.expand([
        ("cpu", "cpu", BACKEND.TORCH_FUSED),
        ("cpu", "cpu", BACKEND.HF_KERNEL),
        ("xpu", "xpu:0", BACKEND.TORCH_FUSED),
    ])
    def test_backends_matches_cpu_reference(self, _name: str, device: str, backend: BACKEND):
        if device.startswith("xpu") and not _xpu_available():
            self.skipTest("Test requires XPU")

        model = GPTQModel.load(
            self.model_path,
            backend=backend,
            device=device,
            dtype=self.dtype,
        )
        failures = []
        for idx, sample in enumerate(self.inputs):
            model_input = sample.to(model.device)
            fused_out = self.forward(model, model_input, backend)
            reference = self.reference_outputs[idx]
            try:
                self.assert_on_mismatch(
                    reference.to("cpu"),
                    fused_out.to("cpu"),
                    self.r_tolerance,
                    self.a_tolerance,
                )
            except AssertionError as exc:
                failures.append(f"Sample {idx}: {str(exc).splitlines()[0]}")

        status = "PASS" if not failures else "FAIL"
        table = tabulate(
            [
                [
                    backend.name,
                    str(self.dtype),
                    device,
                    len(self.inputs),
                    f"{self.r_tolerance:.2e}",
                    f"{self.a_tolerance:.2e}",
                    status,
                    len(failures),
                    "\n\n".join(failures) if failures else "-",
                ]
            ],
            headers=[
                "Backend",
                "DType",
                "Device",
                "Samples",
                "RTol",
                "ATol",
                "Status",
                "Failures",
                "Details",
            ],
            tablefmt="github",
        )
        log.info("\nTorch Fused vs CPU Reference\n" + table)

        if failures:
            raise AssertionError(f"{len(failures)} mismatched samples on device {device}")

class TestTorchFusedAndHFKernelDevicesWithBias(TestTorchFusedAndHFKernelDevices):
    model_path = "/monster/data/model/bloom-560m-gptqmodel-4bit"
    target = 'transformer.h.6.self_attention.query_key_value'
    k = 1024
