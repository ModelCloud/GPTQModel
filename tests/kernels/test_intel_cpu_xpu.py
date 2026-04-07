# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import unittest

import torch
from logbar import LogBar
from parameterized import parameterized
from torch import Tensor

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.utils.model import find_modules


os.environ.setdefault("GPTQMODEL_DISABLE_BITBLAS", "1")

log = LogBar.shared()
device_compare_cols = log.columns(
    cols=[
        {"label": "Backend", "width": "fit"},
        {"label": "DType", "width": "fit"},
        {"label": "Device", "width": "fit"},
        {"label": "Samples", "width": "fit"},
        {"label": "RTol", "width": "fit"},
        {"label": "ATol", "width": "fit"},
        {"label": "Status", "width": "fit"},
        {"label": "Failures", "width": "fit"},
        {"label": "Details", "width": "fit"},
    ],
    padding=1,
)


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _ensure_model_path_available(path: str):
    if os.path.isabs(path) and not os.path.isdir(path):
        raise unittest.SkipTest(f"Local model path missing: {path}")


def _summarize_failures(failures):
    if not failures:
        return "-"
    preview = "; ".join(failures[:2])
    if len(failures) > 2:
        preview += f"; +{len(failures) - 2} more"
    return preview


class TestKernelOutput(unittest.TestCase):
    model_path = "sliuau/llama3.2-1b-4bit-group128" # hf "sliuau/llama3.2-1b-4bit-group128"
    target_qliner_map = {
        BACKEND.TORCH: TorchLinear,
        BACKEND.TORCH_FUSED: TorchFusedLinear,
        BACKEND.TORCH_INT8: TorchInt8Linear,
        BACKEND.GPTQ_TORCH_ATEN: TorchAtenLinear,
    }
    target = 'model.layers.6.self_attn.v_proj'
    device = "cpu"
    m = [1, 16, 64, 256, 1024]
    k = 2048
    dtype = torch.float16
    r_tolerance = 0.0076
    a_tolerance = 0.016
    int8_r_tolerance = 0.02
    # Keep headroom ~+0.001-0.002 over observed max abs diff in this suite.
    int8_a_tolerance = 0.025
    input_samples_each_size = 20 # final size == input_samples_each_size * len(m)

    @classmethod
    def setUp(self):
        _ensure_model_path_available(self.model_path)
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
        # Int4->float->int8 re-quantization in TorchInt8 introduces extra approximation noise.
        (BACKEND.TORCH_INT8,  int8_r_tolerance, int8_a_tolerance),
        (BACKEND.GPTQ_TORCH_ATEN,  r_tolerance, a_tolerance),
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


class TestTorchFusedAndTorchAtenDevices(unittest.TestCase):
    model_path = TestKernelOutput.model_path
    target_qliner_map = TestKernelOutput.target_qliner_map
    target = TestKernelOutput.target
    dtype = torch.float16
    m = [1, 16, 64, 256]
    k = 2048
    input_samples_each_size = 5
    r_tolerance = 0.0076
    a_tolerance = 0.016
    int8_r_tolerance = TestKernelOutput.int8_r_tolerance
    int8_a_tolerance = TestKernelOutput.int8_a_tolerance
    reference_backend = BACKEND.TORCH
    reference_device = "cpu"
    backend_tolerances = {
        BACKEND.TORCH_FUSED: (r_tolerance, a_tolerance),
        BACKEND.TORCH_INT8: (int8_r_tolerance, int8_a_tolerance),
        BACKEND.GPTQ_TORCH_ATEN: (r_tolerance, a_tolerance),
    }

    @classmethod
    def setUpClass(cls):
        _ensure_model_path_available(cls.model_path)
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
        ("cpu", "cpu", BACKEND.TORCH_INT8),
        ("cpu", "cpu", BACKEND.GPTQ_TORCH_ATEN),
        ("xpu", "xpu:0", BACKEND.TORCH_FUSED),
    ])
    def test_backends_matches_cpu_reference(self, _name: str, device: str, backend: BACKEND):
        if device.startswith("xpu") and not _xpu_available():
            self.skipTest("Test requires XPU")
        r_tolerance, a_tolerance = self.backend_tolerances.get(backend, (self.r_tolerance, self.a_tolerance))

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
                    r_tolerance,
                    a_tolerance,
                )
            except AssertionError as exc:
                failures.append(f"Sample {idx}: {str(exc).splitlines()[0]}")

        status = "PASS" if not failures else "FAIL"
        log.info("\nBackend vs CPU Reference")
        device_compare_cols.info.header()
        device_compare_cols.info(
            backend.name,
            str(self.dtype),
            device,
            str(len(self.inputs)),
            f"{r_tolerance:.2e}",
            f"{a_tolerance:.2e}",
            status,
            str(len(failures)),
            _summarize_failures(failures),
        )

        if failures:
            raise AssertionError(f"{len(failures)} mismatched samples on device {device}")

class TestTorchFusedAndTorchAtenDevicesWithBias(TestTorchFusedAndTorchAtenDevices):
    model_path = "/monster/data/model/bloom-560m-gptqmodel-4bit"
    target = 'transformer.h.6.self_attention.query_key_value'
    k = 1024
