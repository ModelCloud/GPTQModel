# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import time
import unittest
from dataclasses import dataclass
from typing import List, Tuple

import torch
from logbar import LogBar
from parameterized import parameterized

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Adapter, AdapterCache, Lora
from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.utils.logger import render_table
from gptqmodel.utils.machete import (
    machete_runtime_available,
    machete_runtime_error,
)
from gptqmodel.utils.model import find_modules


log = LogBar.shared()

DEVICE = torch.device("cuda:0")

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

os.environ.setdefault("BITBLAS_ENABLE_TUNING", "0")
os.environ.setdefault("BITBLAS_ENABLE_TENSORCORE", "0")


def _bitblas_supports_gptq_case(dtype: torch.dtype) -> bool:
    valid, _ = BitblasLinear.validate(
        bits=4,
        group_size=128,
        desc_act=True,
        sym=True,
        in_features=3072,
        out_features=1024,
        pack_dtype=torch.int32,
        dtype=dtype,
    )
    return valid


class Data:
    def __init__(self):
        self.m = 1
        self.k = -1
        self.x = []  # random X input of shape (m, k)


@dataclass
class ForwardResult:
    outputs: List[torch.Tensor]
    total_ms: float
    mean_ms: float

class TestKernelOutput(unittest.TestCase):
    # model_path = "sliuau/llama3.2-1b-4bit-group128" # hf "sliuau/llama3.2-1b-4bit-group128"
    model_path = "sliuau/Llama-3.2-3B_4bits_128group_size"
    target_qliner_map = {
        BACKEND.TORCH: TorchLinear,
        BACKEND.MACHETE: MacheteLinear,
        BACKEND.EXLLAMA_V2: ExllamaV2Linear,
        BACKEND.TRITON: TritonV2Linear,
        BACKEND.BITBLAS: BitblasLinear,
        BACKEND.MARLIN: MarlinLinear,
    }

    target = 'model.layers.6.self_attn.v_proj'
    m: List[Tuple[int, int]] = []
    random_input_sample_size = 0


    @classmethod
    def setUpClass(cls):
        # lora_path = "sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc"  # adapter_model.safetensors
        lora_path = "sliuau/llama-3.2-3b_4bits_128group_size_eora_rank64_mmlu_c4"
        # hf "sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/"

        test_dtypes = [torch.float16, torch.bfloat16]
        cls.data = {} # key is dtype, v is Data()

        def _parse_shapes(expr: str) -> List[Tuple[int, int]]:
            shapes: List[Tuple[int, int]] = []
            for part in expr.split(","):
                part = part.strip()
                if not part:
                    continue
                dim_str, samples_str = part.split(":", 1)
                shapes.append((int(dim_str), int(samples_str)))
            return shapes

        large_shapes = [(1, 256), (16, 128), (32, 64), (64, 32), (128, 16)]
        medium_shapes = [(1, 128), (16, 64), (32, 32), (64, 16)]
        small_shapes = [(1, 64), (8, 32), (16, 16)]

        env_shapes = os.getenv("GPTQMODEL_KERNEL_TEST_SHAPES")
        if env_shapes:
            cls.m = _parse_shapes(env_shapes)
        else:
            total_mem_gb = 0.0
            if torch.cuda.is_available():
                device_index = DEVICE.index if DEVICE.index is not None else 0
                try:
                    if torch.cuda.device_count() > device_index:
                        props = torch.cuda.get_device_properties(device_index)
                        total_mem_gb = props.total_memory / (1024 ** 3)
                except Exception:
                    total_mem_gb = 0.0

            if os.getenv("GPTQMODEL_FAST_TESTS", "0") == "1":
                cls.m = small_shapes
            elif total_mem_gb >= 80:
                cls.m = large_shapes
            elif total_mem_gb >= 48:
                cls.m = medium_shapes
            else:
                cls.m = small_shapes

        cls.random_input_sample_size = sum(t[1] for t in cls.m)

        for dtype in test_dtypes:
            data = Data()

            # map data to dtype
            cls.data[dtype] = data

            data.adapter = Lora(
                rank=64,
                path=lora_path)

            data.adapter.post_init(cls.target, device=DEVICE) # trigger adapter weight load from disk
            data.k = data.adapter.lora_A.shape[0]

            for _ in log.pb(cls.random_input_sample_size).title("Generate Random Inputs"):
                for dim_0, samples in cls.m:

                    for _ in range(samples):
                        inputs = torch.rand((dim_0, data.k), device=DEVICE, dtype=dtype)
                        data.x.append(inputs)

            AdapterCache.reset() # allow next load to complete since we are hacking to get consume only 1 lora module

            # TORCH as reference output
            data.torch_kernel = cls.forward(cls, backend=BACKEND.TORCH, dtype=dtype)
            data.torch_kernel_out = data.torch_kernel.outputs
            data.torch_kernel_with_lora = cls.forward(cls, backend=BACKEND.TORCH, dtype=dtype, adapter=data.adapter)
            data.torch_kernel_out_with_lora = data.torch_kernel_with_lora.outputs


    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def forward(self, backend: BACKEND, dtype: torch.dtype, adapter: Adapter = None) -> ForwardResult:
        model = GPTQModel.load(self.model_path, backend=backend, adapter=adapter, dtype=dtype, device=DEVICE)

        target_qlinear_cls = self.target_qliner_map[backend]

        modules = find_modules(model.model, layers=[target_qlinear_cls])
        result = []
        total_s = 0.0
        for name, module in modules.items():
            if name == self.target:
                data = self.data[dtype]
                module_device = self._module_device(module)
                if data.x:
                    warmup = data.x[0]
                    if module_device is not None and warmup.device != module_device:
                        warmup = warmup.to(module_device)
                    self._synchronize(DEVICE)
                    module(warmup)
                    self._synchronize(DEVICE)
                for i in log.pb(self.random_input_sample_size).title("Forward Pass on Random Input"):
                    sample = data.x[i]
                    assert sample.dtype == dtype

                    # Direct layer calls bypass accelerate's cross-device routing,
                    # so align the synthetic input with the module's local device.
                    if module_device is not None and sample.device != module_device:
                        sample = sample.to(module_device)

                    self._synchronize(DEVICE)
                    started = time.perf_counter()
                    result.append(module(sample).detach().to(device="cpu", dtype=torch.float32))
                    self._synchronize(DEVICE)
                    total_s += time.perf_counter() - started
                break

        del module
        del model
        torch.cuda.empty_cache()

        total_ms = total_s * 1000.0
        mean_ms = total_ms / len(result) if result else 0.0
        return ForwardResult(outputs=result, total_ms=total_ms, mean_ms=mean_ms)

    @staticmethod
    def _module_device(module):
        for tensor in module.parameters(recurse=False):
            if tensor is not None and not tensor.is_meta:
                return tensor.device
        for tensor in module.buffers(recurse=False):
            if tensor is not None and not tensor.is_meta:
                return tensor.device
        return None

    def _summarize_results(
        self,
        reference_outputs,
        actual_outputs,
        backend: BACKEND,
        dtype: torch.dtype,
        atol: float,
        title: str,
        reference_label: str,
        reference_mean_ms: float,
        actual_mean_ms: float,
    ):
        failures = []
        total = len(actual_outputs)
        max_abs_diff = 0.0
        mean_abs_diff = 0.0

        for i in range(total):
            reference = reference_outputs[i]
            actual = actual_outputs[i]
            reference_fp32 = reference.to(torch.float32)
            actual_fp32 = actual.to(torch.float32)
            diff = torch.abs(reference_fp32 - actual_fp32)
            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
            mean_abs_diff += float(diff.mean().item())

            is_close_tensor = torch.isclose(reference_fp32, actual_fp32, rtol=0.15, atol=atol)
            passed = bool(torch.all(is_close_tensor))

            if not passed:
                failures.append(
                    "Sample {idx}:\nExpected ({ref_label}) = {expected}\nActual = {actual_val}".format(
                        idx=i,
                        ref_label=reference_label,
                        expected=reference_fp32.detach().cpu().tolist(),
                        actual_val=actual_fp32.detach().cpu().tolist(),
                    )
                )

        status = f"{GREEN}PASS{RESET}" if not failures else f"{RED}FAIL{RESET}"
        avg_abs_diff = mean_abs_diff / total if total else 0.0
        speedup = reference_mean_ms / actual_mean_ms if actual_mean_ms else 0.0
        details = "\n\n".join(str(detail) for detail in failures) if failures else "-"

        table = render_table(
            [
                [
                    backend.name,
                    str(dtype),
                    total,
                    f"{actual_mean_ms:.4f}",
                    f"{speedup:.2f}x",
                    f"{max_abs_diff:.6f}",
                    f"{avg_abs_diff:.6f}",
                    status,
                    len(failures),
                    details,
                ]
            ],
            headers=[
                "Backend",
                "DType",
                "Samples",
                "MeanLatencyMs",
                "SpeedupVsRef",
                "MaxAbsDiff",
                "MeanAbsDiff",
                "Status",
                "Failures",
                "Expected vs Actual",
            ],
            tablefmt="github",
        )
        log.info("\n" + title + "\n" + table)

        if failures:
            raise AssertionError(
                f"{len(failures)} mismatched outputs for backend `{backend}` and dtype `{dtype}`"
            )

    def _maybe_skip_backend(self, backend: BACKEND):
        if backend == BACKEND.BITBLAS and os.getenv("RUN_BITBLAS_TESTS", "0") != "1":
            self.skipTest("BitBLAS disabled (set RUN_BITBLAS_TESTS=1 to enable)")

        if backend == BACKEND.MACHETE:
            if not machete_runtime_available():
                self.skipTest(f"Machete kernel unavailable: {machete_runtime_error()}")

    # Updated CUDA kernel tolerances below were re-baselined from full
    # torch-vs-kernel validation on H200.
    float16_cases = [
        (BACKEND.TORCH, torch.float16, 0.0000),
        (BACKEND.TRITON, torch.float16, 0.00001),
        (BACKEND.EXLLAMA_V2, torch.float16, 0.0068),
        (BACKEND.MACHETE, torch.float16, 0.0010),
        (BACKEND.MARLIN, torch.float16, 0.0010),
    ]
    if _bitblas_supports_gptq_case(torch.float16):
        float16_cases.append((BACKEND.BITBLAS, torch.float16, 0.0035))

    @parameterized.expand(float16_cases)
    def test_kernel_float16(self, backend: BACKEND,  dtype: torch.dtype, a_tolerance: float):
        self._maybe_skip_backend(backend)

        data = self.data[dtype]
        out = self.forward(backend=backend, dtype=dtype)

        self._summarize_results(
            reference_outputs=data.torch_kernel.outputs,
            actual_outputs=out.outputs,
            backend=backend,
            dtype=dtype,
            atol=a_tolerance,
            title=f"Kernel Output {dtype}",
            reference_label="Torch output",
            reference_mean_ms=data.torch_kernel.mean_ms,
            actual_mean_ms=out.mean_ms,
        )

    bfloat16_cases = [
        (BACKEND.TORCH, torch.bfloat16, 0.0000),
        (BACKEND.TRITON, torch.bfloat16, 0.00001),
        (BACKEND.EXLLAMA_V2, torch.bfloat16, 0.0080),
        (BACKEND.MACHETE, torch.bfloat16, 0.0080),
        (BACKEND.MARLIN, torch.bfloat16, 0.0080),
    ]
    if _bitblas_supports_gptq_case(torch.bfloat16):
        bfloat16_cases.append((BACKEND.BITBLAS, torch.bfloat16, 0.0031))

    @parameterized.expand(bfloat16_cases)
    def test_kernel_bfloat16(self, backend: BACKEND, dtype: torch.dtype, a_tolerance: float):
        self._maybe_skip_backend(backend)

        data = self.data[dtype]
        out = self.forward(backend=backend, dtype=dtype)

        self._summarize_results(
            reference_outputs=data.torch_kernel.outputs,
            actual_outputs=out.outputs,
            backend=backend,
            dtype=dtype,
            atol=a_tolerance,
            title=f"Kernel Output {dtype}",
            reference_label="Torch output",
            reference_mean_ms=data.torch_kernel.mean_ms,
            actual_mean_ms=out.mean_ms,
        )

    float16_lora_cases = [
        (BACKEND.TORCH, torch.float16, 0.0000),
        (BACKEND.TRITON, torch.float16, 0.00001),
        (BACKEND.EXLLAMA_V2, torch.float16, 0.0065),
        (BACKEND.MACHETE, torch.float16, 0.0010),
        (BACKEND.MARLIN, torch.float16, 0.0020),
    ]
    if _bitblas_supports_gptq_case(torch.float16):
        float16_lora_cases.append((BACKEND.BITBLAS, torch.float16, 0.00035))

    @parameterized.expand(float16_lora_cases)
    def test_kernel_float16_with_lora(self, backend: BACKEND, dtype: torch.dtype, a_tolerance: float):
        self._maybe_skip_backend(backend)

        data = self.data[dtype]
        out = self.forward(backend=backend, dtype=dtype, adapter=data.adapter)
        self._summarize_results(
            reference_outputs=data.torch_kernel_with_lora.outputs,
            actual_outputs=out.outputs,
            backend=backend,
            dtype=dtype,
            atol=a_tolerance,
            title=f"Kernel Output With Lora {dtype}",
            reference_label="Torch with Lora output",
            reference_mean_ms=data.torch_kernel_with_lora.mean_ms,
            actual_mean_ms=out.mean_ms,
        )

    bfloat16_lora_cases = [
        (BACKEND.TORCH, torch.bfloat16, 0.0000),
        (BACKEND.TRITON, torch.bfloat16, 0.00001),
        (BACKEND.EXLLAMA_V2, torch.bfloat16, 0.0160),
        (BACKEND.MACHETE, torch.bfloat16, 0.0080),
        (BACKEND.MARLIN, torch.bfloat16, 0.0080),
    ]
    if _bitblas_supports_gptq_case(torch.bfloat16):
        bfloat16_lora_cases.append((BACKEND.BITBLAS, torch.bfloat16, 0.0033))

    @parameterized.expand(bfloat16_lora_cases)
    def test_kernel_bfloat16_with_lora(self, backend: BACKEND, dtype: torch.dtype, a_tolerance: float):
        self._maybe_skip_backend(backend)

        data = self.data[dtype]
        out = self.forward(backend=backend, dtype=dtype, adapter=data.adapter)
        self._summarize_results(
            reference_outputs=data.torch_kernel_with_lora.outputs,
            actual_outputs=out.outputs,
            backend=backend,
            dtype=dtype,
            atol=a_tolerance,
            title=f"Kernel Output With Lora {dtype}",
            reference_label="Torch with Lora output",
            reference_mean_ms=data.torch_kernel_with_lora.mean_ms,
            actual_mean_ms=out.mean_ms,
        )
