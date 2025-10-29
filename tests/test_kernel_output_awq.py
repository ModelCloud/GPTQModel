# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import tempfile
import unittest
from typing import List

import torch
from datasets import load_dataset
from logbar import LogBar
from parameterized import parameterized
from tabulate import tabulate
from transformers import AutoTokenizer

from gptqmodel import BACKEND, FORMAT, GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.awq_gemm import AwqGEMMQuantLinear
from gptqmodel.nn_modules.qlinear.awq_gemv_fast import (
    AwqGEMVFastQuantLinear,
    awq_v2_ext,
    msg as awq_v2_msg,
)
from gptqmodel.nn_modules.qlinear.awq_marlin import (
    AwqMarlinQuantLinear,
    marlin_import_exception,
)
from gptqmodel.quantization import METHOD
from gptqmodel.utils.marlin import marlin_make_workspace_new
from gptqmodel.utils.model import find_modules


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

log = LogBar.shared()

DEVICE = torch.device("cuda:0")

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


class Data:
    def __init__(self) -> None:
        self.inputs: List[torch.Tensor] = []
        self.reference_outputs: List[torch.Tensor] = []


class TestAwqKernelOutput(unittest.TestCase):
    pretrained_model_id = "/monster/data/model/Llama-3.2-1B"
    dataset_path = "/monster/data/model/dataset/c4-train.00000-of-01024.json.gz"
    target = "model.layers.6.self_attn.v_proj"
    group_size = 128
    calibration_concat_size = 0

    target_qliner_map = {
        BACKEND.GEMM: AwqGEMMQuantLinear,
        BACKEND.GEMV_FAST: AwqGEMVFastQuantLinear,
        BACKEND.MARLIN: AwqMarlinQuantLinear,
    }

    backend_to_format = {
        BACKEND.GEMM: FORMAT.GEMM,
        BACKEND.MARLIN: FORMAT.GEMM,
        BACKEND.GEMV_FAST: FORMAT.GEMV_FAST,
    }

    float16_cases = [
        (BACKEND.GEMM, torch.float16, 0.0),
        (BACKEND.GEMV_FAST, torch.float16, 0.0005),
        (BACKEND.MARLIN, torch.float16, 0.0005),
    ]

    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for AWQ kernel output checks.")

        cls.test_dtypes = [torch.float16]
        cls.quantized_tempdirs = {}
        cls.quantized_model_paths = {}
        cls.data = {}

        try:
            cls._prepare_calibration_dataset()
            cls._quantize_models()
            cls._prepare_random_inputs()
        except unittest.SkipTest:
            raise
        except Exception as exc:  # pragma: no cover - defensive skip for CI env mismatches
            raise unittest.SkipTest(f"Skipping AWQ kernel output tests: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        for tmp_dir in getattr(cls, "quantized_tempdirs", {}).values():
            tmp_dir.cleanup()

    @classmethod
    def _prepare_calibration_dataset(cls) -> None:
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model_id, use_fast=True)
        except Exception as exc:
            raise unittest.SkipTest(f"Tokenizer unavailable for AWQ tests: {exc}") from exc

        requested_samples = os.getenv("GPTQMODEL_AWQ_KERNEL_SAMPLES")
        if requested_samples is not None:
            sample_count = max(8, int(requested_samples))
        else:
            try:
                total_mem_gb = (
                    torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                    / (1024 ** 3)
                )
            except Exception:  # pragma: no cover - fallback on inspect failure
                total_mem_gb = 0.0

            if total_mem_gb >= 80:
                sample_count = 256
            elif total_mem_gb >= 48:
                sample_count = 128
            else:
                sample_count = 48

        try:
            dataset = load_dataset("json", data_files=cls.dataset_path, split="train")
        except Exception as exc:
            raise unittest.SkipTest(f"Calibration dataset unavailable for AWQ tests: {exc}") from exc

        if len(dataset) < sample_count:
            raise unittest.SkipTest(
                f"Calibration dataset too small ({len(dataset)} < {sample_count})."
            )

        cls.calibration_dataset = dataset.select(range(sample_count))

    @classmethod
    def _quantize_models(cls) -> None:
        quantize_targets = [
            (FORMAT.GEMM, cls.group_size),
            (FORMAT.GEMV_FAST, cls.group_size),
        ]

        for checkpoint_format, group_size in quantize_targets:
            quantize_config = QuantizeConfig(
                bits=4,
                group_size=group_size,
                quant_method=METHOD.AWQ,
                format=checkpoint_format,
            )

            model = GPTQModel.load(
                cls.pretrained_model_id,
                quantize_config=quantize_config,
            )

            model.quantize(cls.calibration_dataset, batch_size=1, calibration_concat_size=cls.calibration_concat_size)

            tmp_dir = tempfile.TemporaryDirectory()
            model.save(tmp_dir.name)

            cls.quantized_tempdirs[(checkpoint_format, group_size)] = tmp_dir
            cls.quantized_model_paths[(checkpoint_format, group_size)] = tmp_dir.name

            del model
            torch.cuda.empty_cache()

    @classmethod
    def _prepare_random_inputs(cls) -> None:
        model_path = cls.quantized_model_paths[(FORMAT.GEMM, cls.group_size)]
        model = GPTQModel.load(model_path, backend=BACKEND.GEMM, dtype=torch.float16)

        modules = find_modules(model.model, layers=[AwqGEMMQuantLinear])
        if cls.target not in modules:
            raise unittest.SkipTest(f"Target layer `{cls.target}` missing in quantized model.")

        module = modules[cls.target]
        in_features = module.in_features

        large_shapes = [(1, 128), (1, 64), (1, 48)]
        medium_shapes = [(1, 64), (1, 48), (1, 32)]
        small_shapes = [(1, 32), (1, 24), (1, 16)]

        try:
            total_mem_gb = (
                torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
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

        for dtype in cls.test_dtypes:
            data = Data()
            cls.data[dtype] = data

            with torch.inference_mode():
                for batch_tokens, seq_len in shapes:
                    inputs = torch.rand(
                        (batch_tokens, seq_len, in_features),
                        device=DEVICE,
                        dtype=dtype,
                    )
                    data.inputs.append(inputs)

                reference_outputs = cls._forward(
                    model_path=model_path,
                    backend=BACKEND.GEMM,
                    dtype=dtype,
                    inputs=data.inputs,
                )
                data.reference_outputs.extend(reference_outputs)

        del module
        del model
        torch.cuda.empty_cache()

    @classmethod
    def _forward(
        cls,
        model_path: str,
        backend: BACKEND,
        dtype: torch.dtype,
        inputs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        model = GPTQModel.load(model_path, backend=backend, dtype=dtype)

        target_qlinear_cls = cls.target_qliner_map[backend]
        modules = find_modules(model.model, layers=[target_qlinear_cls])
        if cls.target not in modules:
            raise unittest.SkipTest(f"Target layer `{cls.target}` missing for backend `{backend}`.")

        module = modules[cls.target]

        outputs: List[torch.Tensor] = []
        with torch.inference_mode():
            for tensor in inputs:
                outputs.append(module(tensor))

        del module
        del model
        torch.cuda.empty_cache()

        return outputs

    def _maybe_skip_backend(self, backend: BACKEND) -> None:
        if backend == BACKEND.GEMV_FAST and awq_v2_ext is None:
            self.skipTest(f"AWQ GEMV_FAST kernel unavailable: {awq_v2_msg}")

        if backend == BACKEND.MARLIN:
            if marlin_import_exception is not None:
                self.skipTest(f"AWQ Marlin kernel unavailable: {marlin_import_exception}")

            # Validate CUDA capability for Marlin kernels.
            try:
                workspace = marlin_make_workspace_new(DEVICE)
                del workspace
                torch.cuda.empty_cache()
            except Exception as exc:
                self.skipTest(f"Unable to allocate Marlin workspace: {exc}")

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

        for idx, (reference, actual) in enumerate(zip(reference_outputs, actual_outputs)):
            is_close_tensor = torch.isclose(reference, actual, rtol=0.15, atol=atol)
            if not bool(torch.all(is_close_tensor)):
                failures.append(
                    "Sample {idx}:\nExpected ({ref_label}) = {expected}\nActual = {actual_val}".format(
                        idx=idx,
                        ref_label=reference_label,
                        expected=reference.detach().cpu().tolist(),
                        actual_val=actual.detach().cpu().tolist(),
                    )
                )

        status = f"{GREEN}PASS{RESET}" if not failures else f"{RED}FAIL{RESET}"
        details = "\n\n".join(str(detail) for detail in failures) if failures else "-"

        table = tabulate(
            [
                [
                    backend.name,
                    str(dtype),
                    total,
                    status,
                    len(failures),
                    details,
                ]
            ],
            headers=[
                "Backend",
                "DType",
                "Samples",
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

    @parameterized.expand(float16_cases)
    def test_awq_kernel_outputs(self, backend: BACKEND, dtype: torch.dtype, atol: float) -> None:
        self._maybe_skip_backend(backend)

        quant_format = self.backend_to_format[backend]
        model_path = self.quantized_model_paths[(quant_format, self.group_size)]

        data = self.data[dtype]
        actual_outputs = self._forward(
            model_path=model_path,
            backend=backend,
            dtype=dtype,
            inputs=data.inputs,
        )

        self._summarize_results(
            reference_outputs=data.reference_outputs,
            actual_outputs=actual_outputs,
            backend=backend,
            dtype=dtype,
            atol=atol,
            title=f"AWQ Kernel Output {dtype}",
            reference_label="AWQ GEMM output",
        )
