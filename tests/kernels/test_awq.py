# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from logbar import LogBar
from parameterized import parameterized
from safetensors.torch import safe_open

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
from gptqmodel.nn_modules.qlinear.bitblas_awq import AWQBitBlasKernel
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import (
    AwqMarlinLinear,
    marlin_import_exception,
)
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.utils.logger import render_table
from gptqmodel.utils.marlin import marlin_make_workspace_new


try:
    from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear

    awq_triton_import_exception: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - triton import may fail in CI
    AwqGEMMTritonLinear = None  # type: ignore[assignment]
    awq_triton_import_exception = exc

from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
from gptqmodel.utils.exllamav2 import ScratchSpace
from gptqmodel.utils.machete import _validate_machete_device_support, machete_runtime_error


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

log = LogBar.shared()

DEVICE = torch.device("cuda:0")
CPU_DEVICE = torch.device("cpu")

AWQ_MARLIN_FP16_ATOL = 0.006
AWQ_MARLIN_BF16_ATOL = 0.02

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


@dataclass
class ForwardResult:
    outputs: List[torch.Tensor]
    total_ms: float
    mean_ms: float


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


class TestAwqKernelOutput(unittest.TestCase):
    MODEL_PATH = Path("/monster/data/model/deepseek-r1-distill-qwen-7b-awq")
    TARGET = "model.layers.20.self_attn.v_proj"
    BITS = 4
    GROUP_SIZE = 128
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    baseline_backend = BACKEND.TORCH_AWQ
    backend_cases = [
        (baseline_backend, torch.float16, 0.0),
        (baseline_backend, torch.bfloat16, 0.0),
        (BACKEND.GEMM, torch.float16, 0.004),
        (BACKEND.BITBLAS_AWQ, torch.float16, 0.004),
        # (BACKEND.GEMM, torch.bfloat16, 0.05),
        (BACKEND.TRITON, torch.float16, 0.004),
        (BACKEND.MACHETE, torch.float16, 0.006),
        (BACKEND.MARLIN, torch.float16, AWQ_MARLIN_FP16_ATOL),
        (BACKEND.TORCH_FUSED_AWQ, torch.float16, 0.004),
        # (BACKEND.MARLIN, torch.bfloat16, AWQ_MARLIN_BF16_ATOL),
        (BACKEND.EXLLAMA_V2, torch.float16, 0.0068),
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda_available = torch.cuda.is_available()
        cls.device = DEVICE if cls.cuda_available else CPU_DEVICE
        cls.log = log
        cls._weight_map = cls._load_weight_map()
        cls.backend_skip_reason: Dict[BACKEND, str] = {}
        if not cls.cuda_available:
            cls.backend_skip_reason[BACKEND.GEMM] = "CUDA is required for GEMM backend."
            cls.backend_skip_reason[BACKEND.BITBLAS_AWQ] = "CUDA is required for BitBLAS AWQ backend."
            cls.backend_skip_reason[BACKEND.TRITON] = "CUDA is required for AWQ Triton backend."
            cls.backend_skip_reason[BACKEND.MACHETE] = "CUDA is required for AWQ Machete kernel."
            cls.backend_skip_reason[BACKEND.MARLIN] = "CUDA is required for AWQ Marlin kernel."
            cls.backend_skip_reason[BACKEND.EXLLAMA_V2] = "CUDA is required for ExLlama v2 AWQ kernel."
        elif not _validate_machete_device_support():
            cls.backend_skip_reason[BACKEND.MACHETE] = machete_runtime_error()
        elif os.getenv("RUN_BITBLAS_TESTS", "0") != "1":
            cls.backend_skip_reason[BACKEND.BITBLAS_AWQ] = "BitBLAS disabled (set RUN_BITBLAS_TESTS=1 to enable)."
        elif not BITBLAS_AVAILABLE:
            cls.backend_skip_reason[BACKEND.BITBLAS_AWQ] = BITBLAS_INSTALL_HINT
        if awq_triton_import_exception is not None:
            cls.backend_skip_reason[BACKEND.TRITON] = (
                f"AWQ Triton kernel unavailable: {awq_triton_import_exception}"
            )

        try:
            tensors = cls._load_awq_tensors(cls.TARGET)
        except Exception as exc:  # pragma: no cover - skip if model unavailable
            raise unittest.SkipTest(f"Unable to load AWQ tensors: {exc}") from exc

        (
            qweight_cpu,
            qzeros_cpu,
            scales_cpu,
            bias_cpu,
        ) = tensors
        cls.qweight_cpu = qweight_cpu
        cls.qzeros_cpu = qzeros_cpu
        cls.scales_cpu = scales_cpu
        cls.bias_cpu = bias_cpu

        cls.in_features = qweight_cpu.shape[0]
        cls.out_features = qweight_cpu.shape[1] * (32 // cls.BITS)

        cls.modules: Dict[BACKEND, Optional[torch.nn.Module]] = {}

        cls.modules[cls.baseline_backend] = cls._build_torch_awq_module(
            qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu
        )

        cls.modules[BACKEND.GEMM] = (
            cls._build_gemm_module(qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu)
            if cls.cuda_available
            else None
        )

        try:
            cls.modules[BACKEND.BITBLAS_AWQ] = (
                cls._build_bitblas_module(qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu)
                if BACKEND.BITBLAS_AWQ not in cls.backend_skip_reason
                else None
            )
        except Exception as exc:
            cls.backend_skip_reason[BACKEND.BITBLAS_AWQ] = f"AWQ BitBLAS kernel unavailable: {exc}"
            cls.modules[BACKEND.BITBLAS_AWQ] = None

        try:
            cls.modules[BACKEND.MACHETE] = (
                cls._build_machete_module(qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu)
                if BACKEND.MACHETE not in cls.backend_skip_reason
                else None
            )
        except Exception as exc:
            cls.backend_skip_reason[BACKEND.MACHETE] = f"AWQ Machete kernel unavailable: {exc}"
            cls.modules[BACKEND.MACHETE] = None

        try:
            cls.modules[BACKEND.TRITON] = (
                cls._build_gemm_triton_module(qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu)
                if cls.cuda_available
                else None
            )
        except Exception as exc:
            cls.backend_skip_reason[BACKEND.TRITON] = f"AWQ Triton kernel unavailable: {exc}"
            cls.modules[BACKEND.TRITON] = None

        cls.modules[BACKEND.MARLIN] = (
            cls._build_marlin_module(qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu)
            if cls.cuda_available
            else None
        )

        cls.modules[BACKEND.EXLLAMA_V2] = (
            cls._build_exllama_v2_module(qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu)
            if cls.cuda_available
            else None
        )

        try:
            cls.modules[BACKEND.TORCH_FUSED_AWQ] = cls._build_torch_fused_awq_module(
                qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu
            )
        except Exception as exc:
            cls.backend_skip_reason[BACKEND.TORCH_FUSED_AWQ] = (
                f"Torch fused AWQ kernel unavailable: {exc}"
            )
            cls.modules[BACKEND.TORCH_FUSED_AWQ] = None

        base_inputs = cls._generate_inputs()
        cls.inputs: Dict[torch.dtype, List[torch.Tensor]] = {}
        cls.reference_results: Dict[torch.dtype, ForwardResult] = {}

        for dtype in cls.SUPPORTED_DTYPES:
            converted_inputs = [
                tensor.to(dtype=dtype) if tensor.dtype != dtype else tensor.clone()
                for tensor in base_inputs
            ]
            cls.inputs[dtype] = converted_inputs
            torch_module = cls.modules.get(cls.baseline_backend)
            if torch_module is None:
                raise unittest.SkipTest("Torch AWQ kernel unavailable for baseline.")

            cls.reference_results[dtype] = cls._forward(
                torch_module,
                converted_inputs,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        for module in getattr(cls, "modules", {}).values():
            if module is not None:
                del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def _load_weight_map(cls) -> Dict[str, str]:
        index_path = cls.MODEL_PATH / "model.safetensors.index.json"
        if not index_path.is_file():
            raise unittest.SkipTest(f"AWQ checkpoint not available at {index_path}")
        with open(index_path, "r") as handle:
            index = json.load(handle)
        return index["weight_map"]

    @classmethod
    def _load_tensor(cls, key: str) -> torch.Tensor:
        if key not in cls._weight_map:
            raise KeyError(f"Tensor `{key}` not found in weight map.")
        filename = cls.MODEL_PATH / cls._weight_map[key]
        with safe_open(filename, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    @classmethod
    def _load_awq_tensors(cls, target: str) -> Tuple[torch.Tensor, ...]:
        qweight = cls._load_tensor(f"{target}.qweight").contiguous()
        qzeros = cls._load_tensor(f"{target}.qzeros").contiguous()
        scales = cls._load_tensor(f"{target}.scales").contiguous()
        bias = cls._load_tensor(f"{target}.bias").contiguous()
        return qweight, qzeros, scales, bias

    @classmethod
    def _build_gemm_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
    ) -> AwqGEMMLinear:
        module = AwqGEMMLinear(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            adapter=None,
            register_buffers=True,
        ).to(cls.device)

        module.qweight.copy_(qweight_cpu.to(cls.device))
        module.qzeros.copy_(qzeros_cpu.to(cls.device))
        module.scales.copy_(scales_cpu.to(cls.device))
        module.bias.copy_(bias_cpu.to(cls.device))

        module.eval()
        module.post_init()
        return module

    @classmethod
    def _build_bitblas_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
    ) -> AWQBitBlasKernel:
        module = AWQBitBlasKernel(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            adapter=None,
        ).to(cls.device)

        source_module = SimpleNamespace(
            qweight=qweight_cpu.to(cls.device),
            qzeros=qzeros_cpu.to(cls.device),
            scales=scales_cpu.to(torch.float16).to(cls.device),
            bias=bias_cpu.to(torch.float16).to(cls.device),
        )
        module.repack_from_awq(source_module)
        module.eval()
        return module

    @classmethod
    def _build_gemm_triton_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
    ) -> AwqGEMMTritonLinear:
        if AwqGEMMTritonLinear is None:
            raise RuntimeError("AWQ Triton kernel not available.")
        module = AwqGEMMTritonLinear(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            adapter=None,
            register_buffers=True,
        ).to(cls.device)

        module.qweight.copy_(qweight_cpu.to(cls.device))
        module.qzeros.copy_(qzeros_cpu.to(cls.device))
        module.scales.copy_(scales_cpu.to(torch.float16).to(cls.device))
        module.bias.copy_(bias_cpu.to(torch.float16).to(cls.device))

        module.eval()
        module.post_init()
        return module

    @classmethod
    def _build_marlin_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> Optional[AwqMarlinLinear]:
        if marlin_import_exception is not None:
            cls.backend_skip_reason[BACKEND.MARLIN] = f"AWQ Marlin kernel unavailable: {marlin_import_exception}"
            return None

        try:
            workspace = marlin_make_workspace_new(cls.device)
            del workspace
            torch.cuda.empty_cache()
        except Exception as exc:
            cls.backend_skip_reason[BACKEND.MARLIN] = f"Unable to allocate Marlin workspace: {exc}"
            return None

        module = AwqMarlinLinear(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=False,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            dtype=dtype,
            adapter=None,
            register_buffers=True,
        ).to(cls.device)

        module.qweight.data.copy_(qweight_cpu.to(cls.device))
        module.qzeros.data.copy_(qzeros_cpu.to(cls.device))
        module.scales.data.copy_(scales_cpu.to(dtype).to(cls.device))
        module.bias.data.copy_(bias_cpu.to(dtype).to(cls.device))

        module.eval()
        module.post_init()
        return module

    @classmethod
    def _build_machete_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
    ) -> Optional[AwqMacheteLinear]:
        module = AwqMacheteLinear(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=False,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            adapter=None,
            register_buffers=True,
        ).to(cls.device)

        module.qweight.data.copy_(qweight_cpu.to(cls.device))
        module.qzeros.data.copy_(qzeros_cpu.to(cls.device))
        module.scales.data.copy_(scales_cpu.to(torch.float16).to(cls.device))
        module.bias.data.copy_(bias_cpu.to(torch.float16).to(cls.device))

        module.eval()
        module.post_init()
        return module

    @classmethod
    def _build_torch_awq_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
    ) -> AwqTorchLinear:
        module = AwqTorchLinear(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            adapter=None,
            register_buffers=True,
        ).to(cls.device)

        module.qweight.copy_(qweight_cpu.to(cls.device))
        module.qzeros.copy_(qzeros_cpu.to(cls.device))
        module.scales.copy_(scales_cpu.to(cls.device))
        module.bias.copy_(bias_cpu.to(cls.device))

        module.eval()
        module.post_init()
        return module

    @classmethod
    def _build_exllama_v2_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
    ) -> Optional[AwqExllamaV2Linear]:
        try:
            module = AwqExllamaV2Linear(
                bits=cls.BITS,
                group_size=cls.GROUP_SIZE,
                sym=True,
                desc_act=False,
                in_features=cls.in_features,
                out_features=cls.out_features,
                bias=True,
                adapter=None,
                register_buffers=True,
            ).to(cls.device)

            module.qweight.copy_(qweight_cpu.to(cls.device))
            module.qzeros.copy_(qzeros_cpu.to(cls.device))
            module.scales.copy_(scales_cpu.to(torch.float16).to(cls.device))
            module.bias.copy_(bias_cpu.to(torch.float16).to(cls.device))

            module.eval()
            scratch_bytes = module.temp_dq_size()
            scratch = ScratchSpace(scratch_bytes, cls.device)
            module.post_init(scratch)
            return module
        except Exception as exc:
            cls.backend_skip_reason[BACKEND.EXLLAMA_V2] = (
                f"ExLlama v2 AWQ kernel unavailable: {exc}"
            )
            return None

    @classmethod
    def _build_torch_fused_awq_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
        *,
        device: torch.device = CPU_DEVICE,
    ) -> TorchFusedAwqLinear:
        module = TorchFusedAwqLinear(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            adapter=None,
            register_buffers=True,
        ).to(device)

        module.qweight.copy_(qweight_cpu.to(device))
        module.qzeros.copy_(qzeros_cpu.to(device))
        module.scales.copy_(scales_cpu.to(torch.float16).to(device))
        module.bias.copy_(bias_cpu.to(torch.float16).to(device))

        module.eval()
        module.post_init()
        return module

    @classmethod
    def _generate_inputs(cls) -> List[torch.Tensor]:
        large_shapes = [(4, 32), (2, 64), (1, 96)]
        medium_shapes = [(2, 32), (1, 48), (1, 32)]
        small_shapes = [(1, 32), (1, 24), (1, 16)]

        total_mem_gb = 0.0
        if cls.device.type == "cuda":
            try:
                total_mem_gb = (
                    torch.cuda.get_device_properties(cls.device).total_memory
                    / (1024 ** 3)
                )
            except Exception:  # pragma: no cover
                total_mem_gb = 0.0

        if os.getenv("GPTQMODEL_FAST_TESTS", "0") == "1":
            shapes = small_shapes
        elif total_mem_gb >= 80:
            shapes = large_shapes
        elif total_mem_gb >= 48:
            shapes = medium_shapes
        else:
            shapes = small_shapes

        inputs: List[torch.Tensor] = []
        for batch, tokens in shapes:
            tensor = torch.rand(
                (batch, tokens, cls.in_features),
                device=cls.device,
                dtype=torch.float16,
            )
            inputs.append(tensor)
        return inputs

    @classmethod
    def _forward(
        cls,
        module: torch.nn.Module,
        inputs: Iterable[torch.Tensor],
        *,
        compute_dtype: Optional[torch.dtype] = None,
        output_dtype: Optional[torch.dtype] = None,
        target_device: Optional[torch.device] = None,
    ) -> ForwardResult:
        if target_device is None:
            target_device = cls._infer_module_device(module)
        prepared_inputs = list(inputs)
        outputs: List[torch.Tensor] = []
        total_s = 0.0
        with torch.inference_mode():
            if prepared_inputs:
                warmup_tensor = prepared_inputs[0]
                if warmup_tensor.device != target_device:
                    warmup_tensor = warmup_tensor.to(device=target_device)
                if compute_dtype is not None and warmup_tensor.dtype != compute_dtype:
                    warmup_tensor = warmup_tensor.to(dtype=compute_dtype)
                cls._synchronize(target_device)
                module(warmup_tensor)
                cls._synchronize(target_device)
            for tensor in prepared_inputs:
                local_tensor = tensor
                if local_tensor.device != target_device:
                    local_tensor = local_tensor.to(device=target_device)
                if compute_dtype is not None and local_tensor.dtype != compute_dtype:
                    local_tensor = local_tensor.to(dtype=compute_dtype)
                cls._synchronize(target_device)
                started = time.perf_counter()
                result = module(local_tensor)
                cls._synchronize(target_device)
                total_s += time.perf_counter() - started
                if output_dtype is not None and result.dtype != output_dtype:
                    result = result.to(dtype=output_dtype)
                outputs.append(result.detach().cpu())
        total_ms = total_s * 1000.0
        mean_ms = total_ms / len(outputs) if outputs else 0.0
        return ForwardResult(outputs=outputs, total_ms=total_ms, mean_ms=mean_ms)

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "xpu" and _xpu_available():
            torch.xpu.synchronize()

    @staticmethod
    def _infer_module_device(module: torch.nn.Module) -> torch.device:
        try:
            tensor = next(module.parameters())
            return tensor.device
        except StopIteration:
            pass
        try:
            tensor = next(module.buffers())
            return tensor.device
        except StopIteration:
            return torch.device("cpu")

    def _maybe_skip_backend(self, backend: BACKEND) -> None:
        reason = self.backend_skip_reason.get(backend)
        if reason:
            self.skipTest(reason)

    def _summarize_results(
        self,
        reference_outputs: List[torch.Tensor],
        actual_outputs: List[torch.Tensor],
        backend: BACKEND,
        dtype: torch.dtype,
        atol: float,
        title: str,
        reference_label: str,
        device: Optional[torch.device] = None,
        reference_mean_ms: float = 0.0,
        actual_mean_ms: float = 0.0,
    ) -> None:
        failures = []
        total = len(actual_outputs)
        max_abs_diff = 0.0
        mean_abs_diff = 0.0

        for idx, (reference, actual) in enumerate(zip(reference_outputs, actual_outputs)):
            reference_fp32 = reference.to(torch.float32)
            actual_fp32 = actual.to(torch.float32)
            diff = torch.abs(reference_fp32 - actual_fp32)
            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
            mean_abs_diff += float(diff.mean().item())
            is_close_tensor = torch.isclose(reference_fp32, actual_fp32, rtol=0.15, atol=atol)
            if not bool(torch.all(is_close_tensor)):
                failures.append(
                    "Sample {idx}:\nExpected ({ref_label}) = {expected}\nActual = {actual_val}".format(
                        idx=idx,
                        ref_label=reference_label,
                        expected=reference_fp32.detach().cpu().tolist(),
                        actual_val=actual_fp32.detach().cpu().tolist(),
                    )
                )

        status = f"{GREEN}PASS{RESET}" if not failures else f"{RED}FAIL{RESET}"
        avg_abs_diff = mean_abs_diff / total if total else 0.0
        speedup = reference_mean_ms / actual_mean_ms if actual_mean_ms else 0.0
        details = "\n\n".join(str(detail) for detail in failures) if failures else "-"
        device_label = str(device) if device is not None else "-"

        table = render_table(
            [
                [
                    backend.name,
                    str(dtype),
                    device_label,
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
                "Device",
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
        self.log.info("\n" + title + "\n" + table)

        if failures:
            raise AssertionError(
                f"{len(failures)} mismatched outputs for backend `{backend}`"
            )

    @parameterized.expand(backend_cases)
    def test_awq_kernel_outputs(self, backend: BACKEND, dtype: torch.dtype, atol: float) -> None:
        self._maybe_skip_backend(backend)

        module = self.modules.get(backend)
        if module is None:
            self.skipTest(f"Backend `{backend}` module unavailable.")

        inputs = self.inputs[dtype]
        reference_result = self.reference_results[dtype]
        if backend == self.baseline_backend:
            actual_result = reference_result
        else:
            actual_result = self._forward(module, inputs)
        self._summarize_results(
            reference_outputs=reference_result.outputs,
            actual_outputs=actual_result.outputs,
            backend=backend,
            dtype=dtype,
            atol=atol,
            title=f"AWQ Kernel Output {dtype}",
            reference_label="Torch AWQ output",
            reference_mean_ms=reference_result.mean_ms,
            actual_mean_ms=actual_result.mean_ms,
        )

    def test_awq_marlin_bfloat16_outputs(self) -> None:
        self._maybe_skip_backend(BACKEND.MARLIN)

        if not self.cuda_available:
            self.skipTest("CUDA is required for AWQ Marlin kernel.")
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA bfloat16 not supported on this device.")

        module = self._build_marlin_module(
            self.qweight_cpu,
            self.qzeros_cpu,
            self.scales_cpu,
            self.bias_cpu,
            dtype=torch.bfloat16,
        )
        if module is None:
            self.skipTest("AWQ Marlin bf16 module unavailable.")

        try:
            reference_result = self.reference_results[torch.bfloat16]
            actual_result = self._forward(module, self.inputs[torch.bfloat16])
            self._summarize_results(
                reference_outputs=reference_result.outputs,
                actual_outputs=actual_result.outputs,
                backend=BACKEND.MARLIN,
                dtype=torch.bfloat16,
                atol=AWQ_MARLIN_BF16_ATOL,
                title="AWQ Kernel Output torch.bfloat16",
                reference_label="Torch AWQ output",
                reference_mean_ms=reference_result.mean_ms,
                actual_mean_ms=actual_result.mean_ms,
            )
        finally:
            del module
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @parameterized.expand(
        [
            ("cpu", "cpu"),
            ("xpu", "xpu:0"),
        ]
    )
    def test_torch_fused_awq_devices(self, _label: str, device_str: str) -> None:
        self._maybe_skip_backend(BACKEND.TORCH_FUSED_AWQ)
        if device_str.startswith("xpu") and not _xpu_available():
            self.skipTest("Torch fused AWQ XPU test requires Intel XPU runtime.")

        device = torch.device(device_str)
        module = self._build_torch_fused_awq_module(
            self.qweight_cpu,
            self.qzeros_cpu,
            self.scales_cpu,
            self.bias_cpu,
            device=device,
        )

        try:
            actual_result = self._forward(
                module,
                self.inputs[torch.float16],
                target_device=device,
            )
            self._summarize_results(
                reference_outputs=self.reference_results[torch.float16].outputs,
                actual_outputs=actual_result.outputs,
                backend=BACKEND.TORCH_FUSED_AWQ,
                dtype=torch.float16,
                atol=0.004,
                title=f"Torch Fused AWQ Device {device_str}",
                reference_label="Torch AWQ output",
                device=device,
                reference_mean_ms=self.reference_results[torch.float16].mean_ms,
                actual_mean_ms=actual_result.mean_ms,
            )
        finally:
            del module
