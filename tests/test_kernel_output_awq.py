# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
import unittest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from logbar import LogBar
from parameterized import parameterized
from safetensors.torch import safe_open
from tabulate import tabulate

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.awq_gemm import AwqGEMMQuantLinear
from gptqmodel.nn_modules.qlinear.awq_marlin import (
    AwqMarlinQuantLinear,
    marlin_import_exception,
)
from gptqmodel.nn_modules.qlinear.awq_torch import AwqTorchQuantLinear
from gptqmodel.utils.marlin import marlin_make_workspace_new


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

log = LogBar.shared()

DEVICE = torch.device("cuda:0")

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def _reorder_packed_to_awq_order(packed: torch.Tensor, bits: int) -> torch.Tensor:
    if bits != 4:
        return packed
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    mask = (1 << bits) - 1
    result = torch.zeros_like(packed)
    for dst, src in enumerate(order):
        nib = (packed >> (src * bits)) & mask
        result |= nib << (dst * bits)
    return result


class TestAwqKernelOutput(unittest.TestCase):
    MODEL_PATH = Path("/monster/data/model/deepseek-r1-distill-qwen-7b-awq")
    TARGET = "model.layers.20.self_attn.v_proj"
    BITS = 4
    GROUP_SIZE = 128
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    baseline_backend = BACKEND.TORCH_AWQ
    backend_cases = [
        ("torch_awq_fp16", baseline_backend, torch.float16, 0.0),
        ("torch_awq_bf16", baseline_backend, torch.bfloat16, 0.0),
        ("gemm_fp16", BACKEND.GEMM, torch.float16, 0.001),
        ("gemm_bf16", BACKEND.GEMM, torch.bfloat16, 0.05),
        ("marlin_fp16", BACKEND.MARLIN, torch.float16, 0.02),
        ("marlin_bf16", BACKEND.MARLIN, torch.bfloat16, 0.05),
    ]

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for AWQ kernel output checks.")

        cls.device = DEVICE
        cls.log = log
        cls._weight_map = cls._load_weight_map()
        cls.backend_skip_reason: Dict[BACKEND, str] = {}
        cls._forward_kwargs: Dict[torch.dtype, Dict[str, torch.dtype]] = {}

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

        cls.in_features = qweight_cpu.shape[0]
        cls.out_features = qweight_cpu.shape[1] * (32 // cls.BITS)

        cls.modules: Dict[BACKEND, Optional[torch.nn.Module]] = {}

        cls.modules[cls.baseline_backend] = cls._build_torch_awq_module(
            qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu
        )

        cls.modules[BACKEND.GEMM] = cls._build_gemm_module(
            qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu
        )

        cls.modules[BACKEND.MARLIN] = cls._build_marlin_module(
            qweight_cpu, qzeros_cpu, scales_cpu, bias_cpu
        )

        base_inputs = cls._generate_inputs()
        cls.inputs: Dict[torch.dtype, List[torch.Tensor]] = {}
        cls.reference_outputs: Dict[torch.dtype, List[torch.Tensor]] = {}

        for dtype in cls.SUPPORTED_DTYPES:
            converted_inputs = [
                tensor.to(dtype=dtype) if tensor.dtype != dtype else tensor.clone()
                for tensor in base_inputs
            ]
            cls.inputs[dtype] = converted_inputs
            torch_module = cls.modules.get(cls.baseline_backend)
            if torch_module is None:
                raise unittest.SkipTest("Torch AWQ kernel unavailable for baseline.")

            forward_kwargs = {}
            if dtype == torch.bfloat16:
                forward_kwargs = {
                    "compute_dtype": torch.float16,
                    "output_dtype": dtype,
                }
            cls._forward_kwargs[dtype] = forward_kwargs
            cls.reference_outputs[dtype] = cls._forward(
                torch_module,
                converted_inputs,
                **forward_kwargs,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        for module in getattr(cls, "modules", {}).values():
            if module is not None:
                del module
        torch.cuda.empty_cache()

    @classmethod
    def _load_weight_map(cls) -> Dict[str, str]:
        index_path = cls.MODEL_PATH / "model.safetensors.index.json"
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
    ) -> AwqGEMMQuantLinear:
        module = AwqGEMMQuantLinear(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            adapter=None,
            register_buffers=False,
        ).to(cls.device)

        module.load_legacy_tensors(
            qweight_cpu.to(cls.device),
            qzeros_cpu.to(cls.device),
            scales_cpu.to(cls.device),
            bias_cpu.to(cls.device),
        )

        module.eval()
        return module

    @classmethod
    def _build_marlin_module(
        cls,
        qweight_cpu: torch.Tensor,
        qzeros_cpu: torch.Tensor,
        scales_cpu: torch.Tensor,
        bias_cpu: torch.Tensor,
    ) -> Optional[AwqMarlinQuantLinear]:
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

        module = AwqMarlinQuantLinear(
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

        qweight_reordered = _reorder_packed_to_awq_order(qweight_cpu, cls.BITS)
        qzeros_reordered = _reorder_packed_to_awq_order(qzeros_cpu, cls.BITS)

        module.qweight.data.copy_(qweight_reordered.to(cls.device))
        module.qzeros.data.copy_(qzeros_reordered.to(cls.device))
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
    ) -> AwqTorchQuantLinear:
        module = AwqTorchQuantLinear(
            bits=cls.BITS,
            group_size=cls.GROUP_SIZE,
            sym=True,
            desc_act=False,
            in_features=cls.in_features,
            out_features=cls.out_features,
            bias=True,
            adapter=None,
            register_buffers=False,
        ).to(cls.device)

        module.load_legacy_tensors(
            qweight_cpu.to(cls.device),
            qzeros_cpu.to(cls.device),
            scales_cpu.to(cls.device),
            bias_cpu.to(cls.device),
        )

        module.eval()
        return module

    @classmethod
    def _parse_shapes(cls, expr: str) -> List[Tuple[int, int]]:
        shapes: List[Tuple[int, int]] = []
        for part in expr.split(","):
            part = part.strip()
            if not part:
                continue
            dim_str, samples_str = part.split(":", 1)
            shapes.append((int(dim_str), int(samples_str)))
        return shapes

    @classmethod
    def _generate_inputs(cls) -> List[torch.Tensor]:
        large_shapes = [(1, 256), (16, 128), (32, 64), (64, 32), (128, 16)]
        medium_shapes = [(1, 128), (16, 64), (32, 32), (64, 16)]
        small_shapes = [(1, 64), (8, 32), (16, 16)]

        env_shapes = os.getenv("GPTQMODEL_KERNEL_TEST_SHAPES")
        if env_shapes:
            shapes = cls._parse_shapes(env_shapes)
        else:
            total_mem_gb = 0.0
            if torch.cuda.is_available():
                device_index = cls.device.index if cls.device.index is not None else 0
                try:  # pragma: no cover - hardware dependent
                    if torch.cuda.device_count() > device_index:
                        props = torch.cuda.get_device_properties(device_index)
                        total_mem_gb = props.total_memory / (1024 ** 3)
                except Exception:  # pragma: no cover - fall back to smallest shapes
                    total_mem_gb = 0.0

            if os.getenv("GPTQMODEL_FAST_TESTS", "0") == "1":
                shapes = small_shapes
            elif total_mem_gb >= 80:
                shapes = large_shapes
            elif total_mem_gb >= 48:
                shapes = medium_shapes
            else:
                shapes = small_shapes

        cls._shape_plan = shapes
        cls._random_input_sample_size = sum(samples for _, samples in shapes)

        inputs: List[torch.Tensor] = []
        for leading_dim, samples in shapes:
            for _ in range(samples):
                tensor = torch.rand(
                    (leading_dim, cls.in_features),
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
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        with torch.inference_mode():
            for tensor in inputs:
                local_tensor = tensor
                if compute_dtype is not None and tensor.dtype != compute_dtype:
                    local_tensor = tensor.to(dtype=compute_dtype)
                result = module(local_tensor)
                if output_dtype is not None and result.dtype != output_dtype:
                    result = result.to(dtype=output_dtype)
                outputs.append(result.detach().cpu())
        return outputs

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
            is_close_tensor = torch.isclose(reference, actual, rtol=0.15, atol=atol)
            if not bool(torch.all(is_close_tensor)):
                sample_max = float(diff.max().item())
                sample_mean = float(diff.mean().item())
                failures.append(
                    "Sample {idx}: max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}, "
                    "rtol=0.15, atol={atol:.6f}".format(
                        idx=idx,
                        max_diff=sample_max,
                        mean_diff=sample_mean,
                        atol=atol,
                    ),
                )

        status = f"{GREEN}PASS{RESET}" if not failures else f"{RED}FAIL{RESET}"
        avg_abs_diff = mean_abs_diff / total if total else 0.0
        details = "\n\n".join(str(detail) for detail in failures) if failures else "-"

        table = tabulate(
            [
                [
                    backend.name,
                    str(dtype),
                    total,
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
            preview = "\n".join(failures[:5])
            if len(failures) > 5:
                preview += f"\n... ({len(failures) - 5} additional mismatches)"
            raise AssertionError(
                f"{len(failures)} mismatched outputs for backend `{backend}` "
                f"(rtol=0.15, atol={atol:.6f})\n{preview}"
            )

    @parameterized.expand(backend_cases)
    def test_awq_kernel_outputs(
        self, _case_name: str, backend: BACKEND, dtype: torch.dtype, atol: float
    ) -> None:
        self._maybe_skip_backend(backend)

        module = self.modules.get(backend)
        if module is None:
            self.skipTest(f"Backend `{backend}` module unavailable.")

        inputs = self.inputs[dtype]
        reference_outputs = self.reference_outputs[dtype]
        if backend == self.baseline_backend:
            actual_outputs = reference_outputs
        else:
            forward_kwargs = self._forward_kwargs.get(dtype, {})
            actual_outputs = self._forward(module, inputs, **forward_kwargs)
        self._summarize_results(
            reference_outputs=reference_outputs,
            actual_outputs=actual_outputs,
            backend=backend,
            dtype=dtype,
            atol=atol,
            title=f"AWQ Kernel Output {dtype}",
            reference_label="Torch AWQ output",
        )
