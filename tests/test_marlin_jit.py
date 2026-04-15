# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from shutil import copy2, which

import pytest
import torch

import gptqmodel.nn_modules.qlinear.marlin as marlin_qlinear_module
import gptqmodel.nn_modules.qlinear.marlin_awq as marlin_awq_qlinear_module
import gptqmodel.utils.marlin as marlin_utils
from gptqmodel import extension as extension_api
from gptqmodel.utils import cpp as cpp_module
from gptqmodel.utils.marlin_scalar_type import scalar_types


class _FakeLoader:
    def __init__(self, *, should_load: bool = True, last_error: str = ""):
        self.should_load = should_load
        self._last_error = last_error
        self.ops: dict[str, object] = {}
        self.load_calls = 0
        self.op_calls: list[str] = []

    def load(self) -> bool:
        self.load_calls += 1
        return self.should_load

    def op(self, op_name: str):
        self.op_calls.append(op_name)
        return self.ops[op_name]

    def last_error_message(self) -> str:
        return self._last_error

    def clear_cache(self) -> None:
        return None


class _FakeExtensionApi:
    def __init__(self, *, available: bool = False, error_text: str = ""):
        self.available = available
        self.error_text = error_text
        self.is_available_calls: list[str] = []
        self.error_calls: list[str] = []

    def is_available(self, extension_name: str) -> bool:
        self.is_available_calls.append(extension_name)
        return self.available

    def error(self, extension_name: str) -> str:
        self.error_calls.append(extension_name)
        return self.error_text


def _jit_scratch_root(tmp_path: Path, suffix: str) -> Path:
    base = Path("/dev/shm") if Path("/dev/shm").is_dir() else tmp_path
    root = base / "gptqmodel-jit-tests" / suffix
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_gptq_marlin_gemm_dispatches_fp16_to_torch_ops(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_gemm(*args):
        captured["dtype"] = args[0].dtype
        captured["shape"] = (args[12], args[13])
        return torch.full((args[12], args[13]), 3.0, dtype=args[0].dtype)

    fp16_loader.ops["gptq_marlin_gemm_fp16"] = fake_gemm

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.gptq_marlin_gemm(
        a=torch.ones((2, 128), dtype=torch.float16),
        c=None,
        b_q_weight=torch.zeros((32, 64), dtype=torch.int32),
        b_bias=None,
        b_scales=torch.ones((1, 64), dtype=torch.float16),
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=torch.zeros(1, dtype=torch.int32),
        b_q_type=scalar_types.uint4b8,
        size_m=2,
        size_n=64,
        size_k=128,
    )

    assert fp16_loader.op_calls == ["gptq_marlin_gemm_fp16"]
    assert bf16_loader.op_calls == []
    assert captured == {"dtype": torch.float16, "shape": (2, 64)}
    assert out.shape == (2, 64)
    assert out.dtype == torch.float16


def test_gptq_marlin_gemm_dispatches_bf16_to_torch_ops(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_gemm(*args):
        captured["dtype"] = args[0].dtype
        return torch.full((args[12], args[13]), 5.0, dtype=args[0].dtype)

    bf16_loader.ops["gptq_marlin_gemm_bf16"] = fake_gemm

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.gptq_marlin_gemm(
        a=torch.ones((1, 64), dtype=torch.bfloat16),
        c=None,
        b_q_weight=torch.zeros((16, 64), dtype=torch.int32),
        b_bias=None,
        b_scales=torch.ones((1, 64), dtype=torch.bfloat16),
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=torch.zeros(1, dtype=torch.int32),
        b_q_type=scalar_types.uint8b128,
        size_m=1,
        size_n=64,
        size_k=64,
    )

    assert bf16_loader.op_calls == ["gptq_marlin_gemm_bf16"]
    assert fp16_loader.op_calls == []
    assert captured == {"dtype": torch.bfloat16}
    assert out.shape == (1, 64)
    assert out.dtype == torch.bfloat16


def test_gptq_marlin_gemm_passes_float_global_scale_to_torch_ops(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_gemm(*args):
        captured["global_scale_dtype"] = args[6].dtype
        captured["global_scale_shape"] = tuple(args[6].shape)
        return torch.zeros((args[12], args[13]), dtype=args[0].dtype)

    fp16_loader.ops["gptq_marlin_gemm_fp16"] = fake_gemm

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.gptq_marlin_gemm(
        a=torch.ones((1, 64), dtype=torch.float16),
        c=None,
        b_q_weight=torch.zeros((16, 64), dtype=torch.int32),
        b_bias=None,
        b_scales=torch.ones((4, 64), dtype=torch.float16),
        global_scale=torch.tensor([1.0], dtype=torch.float32),
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=torch.zeros(1, dtype=torch.int32),
        b_q_type=scalar_types.float4_e2m1f,
        size_m=1,
        size_n=64,
        size_k=64,
    )

    assert fp16_loader.op_calls == ["gptq_marlin_gemm_fp16"]
    assert bf16_loader.op_calls == []
    assert captured == {"global_scale_dtype": torch.float32, "global_scale_shape": (1,)}
    assert out.shape == (1, 64)
    assert out.dtype == torch.float16


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_gptq_marlin_gemm_uses_scale_dtype_to_pick_runtime_for_fp8_inputs(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_gemm(*args):
        captured["input_dtype"] = args[0].dtype
        captured["a_scales_dtype"] = args[5].dtype
        return torch.zeros((args[12], args[13]), dtype=torch.bfloat16)

    bf16_loader.ops["gptq_marlin_gemm_bf16"] = fake_gemm

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.gptq_marlin_gemm(
        a=torch.ones((1, 64), dtype=torch.float8_e4m3fn),
        c=None,
        b_q_weight=torch.zeros((16, 64), dtype=torch.int32),
        b_bias=None,
        b_scales=torch.ones((1, 64), dtype=torch.bfloat16),
        global_scale=None,
        b_zeros=torch.zeros((1, 16), dtype=torch.int32),
        g_idx=None,
        perm=None,
        workspace=torch.zeros(1, dtype=torch.int32),
        b_q_type=scalar_types.uint4,
        size_m=1,
        size_n=64,
        size_k=64,
        a_scales=torch.ones((1,), dtype=torch.float32),
    )

    assert bf16_loader.op_calls == ["gptq_marlin_gemm_bf16"]
    assert fp16_loader.op_calls == []
    assert captured == {
        "input_dtype": torch.float8_e4m3fn,
        "a_scales_dtype": torch.float32,
    }
    assert out.dtype == torch.bfloat16


def test_nvfp4_global_scale_contract_is_float_in_marlin_sources():
    marlin_root = marlin_utils._marlin_root()
    kernel_h = (marlin_root / "kernel.h").read_text(encoding="utf-8")
    gemm_cu = (marlin_root / "gptq_marlin.cu").read_text(encoding="utf-8")
    template_h = (marlin_root / "marlin_template.h").read_text(encoding="utf-8")

    assert "const float *__restrict__ global_scale_ptr" in kernel_h
    assert 'global_scale = torch::empty({0}, options_fp32);' in gemm_cu
    assert 'global_scale.scalar_type() == at::ScalarType::Float' in gemm_cu
    assert "global_scale.data_ptr()" in gemm_cu
    assert "float global_scale_f32 = 1.0f;" in template_h
    assert "c0 *= global_scale_f32;" in template_h
    assert "c1 *= global_scale_f32;" in template_h


def test_marlin_capability_checks_allow_sm75_but_reject_sm70(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (7, 5))

    assert marlin_utils._marlin_capability_supported(7, 5) is True
    assert marlin_utils._marlin_environment_error() == ""
    assert marlin_utils._validate_marlin_device_support() is True

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (7, 0))

    assert marlin_utils._marlin_capability_supported(7, 0) is False
    assert "compute capability >= 7.5" in marlin_utils._marlin_environment_error()
    assert marlin_utils._validate_marlin_device_support() is False


def test_marlin_quant_linear_validate_device_allows_sm75(monkeypatch):
    monkeypatch.setattr(marlin_qlinear_module, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (7, 5))

    marlin_qlinear_module.MarlinLinear.validate_device(marlin_qlinear_module.DEVICE.CUDA)


def test_marlin_quant_linear_validate_device_rejects_pre_turing(monkeypatch):
    monkeypatch.setattr(marlin_qlinear_module, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (7, 0))

    with pytest.raises(NotImplementedError, match="compute capability >= 7.5"):
        marlin_qlinear_module.MarlinLinear.validate_device(marlin_qlinear_module.DEVICE.CUDA)


def test_sm75_turing_contract_is_present_in_marlin_sources():
    marlin_root = marlin_utils._marlin_root()
    gemm_cu = (marlin_root / "gptq_marlin.cu").read_text(encoding="utf-8")
    generator_py = (marlin_root / "generate_kernels.py").read_text(encoding="utf-8")
    template_h = (marlin_root / "marlin_template.h").read_text(encoding="utf-8")
    mma_h = (marlin_root / "marlin_mma.h").read_text(encoding="utf-8")
    loader_py = (Path(marlin_utils.__file__).resolve().parents[1] / "models" / "loader.py").read_text(
        encoding="utf-8"
    )

    assert "requires CUDA_ARCH >= 7.5" in gemm_cu
    assert "major_capability == 7 && minor_capability == 5" in gemm_cu
    assert "stages = 2;" in gemm_cu
    assert "support_sm75 = 75 in archs" in generator_py
    assert 'config_sm75["stages"] = 2' in generator_py
    assert "constexpr bool use_fp16_accum" in template_h
    assert "__CUDA_ARCH__ == 750" in mma_h
    assert "m16n8k8.row.col.f16.f16.f16.f16" in mma_h
    assert "compute capability >= 7.5" in loader_py
    assert "GPTQ Marlin on Turing (compute capability 7.5)" in loader_py
    assert "dtype=torch.float16 only." in loader_py


def test_stage2_dense_four_bit_tiles_stay_in_sync_between_selector_and_codegen(monkeypatch, tmp_path):
    source_root = marlin_utils._marlin_root()
    generator_py = (source_root / "generate_kernels.py").read_text(encoding="utf-8")
    test_root = tmp_path / "marlin"
    test_root.mkdir()
    copy2(source_root / "generate_kernels.py", test_root / "generate_kernels.py")

    monkeypatch.setattr(marlin_utils, "_marlin_root", lambda: test_root)
    monkeypatch.setattr(marlin_utils, "_marlin_generator_arch_list", lambda: "7.5,8.0")

    marlin_utils._ensure_generated_marlin_kernels()
    kernel_u4 = (test_root / "sm75_kernel_float16_u4_float16.cu").read_text(encoding="utf-8")
    kernel_u4b8 = (test_root / "sm75_kernel_float16_u4b8_float16.cu").read_text(encoding="utf-8")
    kernel_nvfp4 = (test_root / "sm75_kernel_float16_fe2m1f_float16.cu").read_text(encoding="utf-8")

    assert "support_sm75 = 75 in archs" in generator_py
    assert 'config_sm75["stages"] = 2' in generator_py

    invalid_stage2_tile = ", 256, 4, 16, 4, false, 2,"
    valid_stage2_tile = ", 256, 2, 16, 4, false, 2,"

    assert invalid_stage2_tile not in kernel_u4
    assert invalid_stage2_tile not in kernel_u4b8
    assert invalid_stage2_tile not in kernel_nvfp4
    assert valid_stage2_tile in kernel_u4
    assert valid_stage2_tile in kernel_u4b8
    assert valid_stage2_tile in kernel_nvfp4


def test_mxfp8_contract_is_present_in_marlin_sources():
    marlin_root = marlin_utils._marlin_root()
    generator_py = (marlin_root / "generate_kernels.py").read_text(encoding="utf-8")
    template_h = (marlin_root / "marlin_template.h").read_text(encoding="utf-8")

    assert generator_py.count('"s_type": "kFE8M0fnu"') >= 2
    assert '"a_type": ["kBFloat16"]' in generator_py
    assert "MXFP8: FP8 weights with e8m0 microscaling block scales" in template_h
    assert "if constexpr (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu)" in template_h


def test_fp8_activation_contract_is_present_in_marlin_sources():
    marlin_root = marlin_utils._marlin_root()
    gemm_cu = (marlin_root / "gptq_marlin.cu").read_text(encoding="utf-8")
    kernel_h = (marlin_root / "kernel.h").read_text(encoding="utf-8")
    repack_cu = (marlin_root / "awq_marlin_repack.cu").read_text(encoding="utf-8")
    preprocess_cu = (marlin_root / "marlin_int4_fp8_preprocess.cu").read_text(encoding="utf-8")
    dtypes_cuh = (marlin_root / "marlin_dtypes.cuh").read_text(encoding="utf-8")

    assert "const float *__restrict__ a_scales_ptr" in kernel_h
    assert "a_scales can only be used for 8bit activation." in gemm_cu
    assert "the a_scales parameter must be passed for 8bit activation." in gemm_cu
    assert "a.scalar_type() == at::ScalarType::Float8_e4m3fn" in gemm_cu
    assert "a_type.size_bits() == 8" in gemm_cu
    assert "bool is_a_8bit" in repack_cu
    assert "marlin_int4_fp8_preprocess_kernel_awq" in preprocess_cu
    assert "class MarlinScalarType<vllm::kFE4M3fn.id()>" in dtypes_cuh


def test_ensure_generated_marlin_kernels_repairs_stale_generated_sources(monkeypatch, tmp_path):
    source_root = marlin_utils._marlin_root()
    test_root = tmp_path / "marlin"
    test_root.mkdir()
    copy2(source_root / "generate_kernels.py", test_root / "generate_kernels.py")

    monkeypatch.setattr(marlin_utils, "_marlin_root", lambda: test_root)
    monkeypatch.setattr(marlin_utils, "_marlin_generator_arch_list", lambda: "8.0,8.9")

    assert marlin_utils._ensure_generated_marlin_kernels() == test_root

    kernel_path = test_root / "sm80_kernel_bfloat16_fe4m3fn_bfloat16.cu"
    original_text = kernel_path.read_text(encoding="utf-8")
    assert "vllm::kFE8M0fnu.id()" in original_text
    assert (test_root / "kernel_selector.h").exists()

    stale_text = "\n".join(
        line for line in original_text.splitlines() if "vllm::kFE8M0fnu.id()" not in line
    ) + "\n"
    kernel_path.write_text(stale_text, encoding="utf-8")
    assert "vllm::kFE8M0fnu.id()" not in kernel_path.read_text(encoding="utf-8")

    assert marlin_utils._ensure_generated_marlin_kernels() == test_root
    assert kernel_path.read_text(encoding="utf-8") == original_text


def test_gptq_marlin_repack_prefers_requested_dtype_extension(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit):
        captured["dtype"] = torch.bfloat16
        captured["shape"] = tuple(b_q_weight.shape)
        captured["is_a_8bit"] = is_a_8bit
        return b_q_weight + 1

    bf16_loader.ops["gptq_marlin_repack"] = fake_repack

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.gptq_marlin_repack(
        torch.zeros((32, 64), dtype=torch.int32),
        torch.arange(32, dtype=torch.int32),
        128,
        64,
        4,
        dtype=torch.bfloat16,
    )

    assert bf16_loader.op_calls == ["gptq_marlin_repack"]
    assert fp16_loader.op_calls == []
    assert captured == {"dtype": torch.bfloat16, "shape": (32, 64), "is_a_8bit": False}
    assert torch.equal(out, torch.ones((32, 64), dtype=torch.int32))


def test_awq_marlin_repack_raises_when_requested_jit_extension_is_unavailable(monkeypatch):
    fp16_loader = _FakeLoader(should_load=False, last_error="fp16 unavailable")
    bf16_loader = _FakeLoader(should_load=False, last_error="bf16 unavailable")

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    with pytest.raises(RuntimeError, match="bf16 unavailable"):
        marlin_utils.awq_marlin_repack(
            torch.zeros((64, 16), dtype=torch.int32),
            64,
            128,
            4,
            dtype=torch.bfloat16,
        )

    assert fp16_loader.op_calls == []
    assert bf16_loader.op_calls == []


def test_marlin_quant_linear_post_init_uses_compute_dtype_for_repack(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(1, dtype=torch.int32, device=device),
    )
    monkeypatch.setattr(
        marlin_qlinear_module,
        "gptq_marlin_repack",
        lambda b_q_weight, perm, size_k, size_n, num_bits, dtype=None: (
            captured.update({"dtype": dtype, "shape": tuple(b_q_weight.shape)}) or b_q_weight
        ),
    )
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_permute_scales",
        lambda scales, size_k, size_n, group_size: scales,
    )
    monkeypatch.setattr(marlin_qlinear_module, "marlin_permute_bias", lambda bias: bias)

    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        bias=False,
        dtype=torch.bfloat16,
    )
    module.post_init()

    assert captured == {"dtype": torch.bfloat16, "shape": tuple(module.qweight.shape)}


def test_marlin_quant_linear_registers_runtime_buffers_in_compute_dtype(monkeypatch):
    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)

    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        bias=True,
        dtype=torch.bfloat16,
    )

    assert module.scales.dtype == torch.bfloat16
    assert module.bias.dtype == torch.bfloat16


def test_marlin_quant_linear_forward_promotes_bias_to_input_dtype(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(1, dtype=torch.int32, device=device),
    )
    monkeypatch.setattr(
        marlin_qlinear_module,
        "gptq_marlin_repack",
        lambda b_q_weight, perm, size_k, size_n, num_bits, dtype=None: b_q_weight,
    )
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_permute_scales",
        lambda scales, size_k, size_n, group_size: scales,
    )
    monkeypatch.setattr(marlin_qlinear_module, "marlin_permute_bias", lambda bias: bias)
    monkeypatch.setattr(
        marlin_qlinear_module,
        "apply_gptq_marlin_linear",
        lambda **kwargs: (
            captured.update(
                {
                    "input_dtype": kwargs["input"].dtype,
                    "scale_dtype": kwargs["weight_scale"].dtype,
                    "bias_dtype": kwargs["bias"].dtype,
                }
            )
            or torch.zeros(
                (kwargs["input"].shape[0], kwargs["output_size_per_partition"]),
                dtype=kwargs["input"].dtype,
            )
        ),
    )

    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        bias=True,
        dtype=torch.float16,
    )
    module.post_init()

    out = module(torch.randn(2, 128, dtype=torch.bfloat16))

    assert captured == {
        "input_dtype": torch.bfloat16,
        "scale_dtype": torch.bfloat16,
        "bias_dtype": torch.bfloat16,
    }
    assert module.bias.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16


def test_awq_marlin_quant_linear_registers_runtime_buffers_in_compute_dtype(monkeypatch):
    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_import_exception", None)

    module = marlin_awq_qlinear_module.AwqMarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=64,
        bias=True,
        dtype=torch.bfloat16,
        register_buffers=True,
    )

    assert torch.bfloat16 in marlin_awq_qlinear_module.AwqMarlinLinear.SUPPORTS_DTYPES
    assert module.scales.dtype == torch.bfloat16
    assert module.bias.dtype == torch.bfloat16


def test_marlin_int4_fp8_preprocess_dispatches_to_requested_extension(monkeypatch):
    fp16_loader = _FakeLoader()
    bf16_loader = _FakeLoader()
    captured = {}

    def fake_preprocess(qweight, qzeros, inplace):
        captured["dtype"] = torch.bfloat16
        captured["shape"] = tuple(qweight.shape)
        captured["has_qzeros"] = qzeros is not None
        captured["inplace"] = inplace
        return qweight + 7

    bf16_loader.ops["marlin_int4_fp8_preprocess"] = fake_preprocess

    monkeypatch.setattr(marlin_utils, "_MARLIN_FP16_TORCH_OPS_EXTENSION", fp16_loader)
    monkeypatch.setattr(marlin_utils, "_MARLIN_BF16_TORCH_OPS_EXTENSION", bf16_loader)

    out = marlin_utils.marlin_int4_fp8_preprocess(
        torch.zeros((64, 16), dtype=torch.int32),
        torch.zeros((4, 16), dtype=torch.int32),
        inplace=True,
        dtype=torch.bfloat16,
    )

    assert bf16_loader.op_calls == ["marlin_int4_fp8_preprocess"]
    assert fp16_loader.op_calls == []
    assert captured == {
        "dtype": torch.bfloat16,
        "shape": (64, 16),
        "has_qzeros": True,
        "inplace": True,
    }
    assert torch.equal(out, torch.full((64, 16), 7, dtype=torch.int32))


def test_awq_marlin_post_init_prepares_dynamic_fp8_runtime(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_supports_fp8_input", lambda device: True)
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(1, dtype=torch.int32, device=device),
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "marlin_int4_fp8_preprocess",
        lambda qweight, qzeros, inplace=False, dtype=None: (
            captured.update({"preprocess_dtype": dtype, "preprocess_inplace": inplace, "preprocess_shape": tuple(qweight.shape)})
            or qweight
        ),
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "awq_marlin_repack",
        lambda b_q_weight, size_k, size_n, num_bits, is_a_8bit=False, dtype=None: (
            captured.update({"repack_is_a_8bit": is_a_8bit, "repack_dtype": dtype}) or b_q_weight
        ),
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "marlin_permute_scales",
        lambda scales, size_k, size_n, group_size, is_a_8bit=False: (
            captured.update({"scale_is_a_8bit": is_a_8bit}) or scales
        ),
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "awq_to_marlin_zero_points",
        lambda qzeros, size_k, size_n, num_bits, is_a_8bit=False: (
            captured.update({"zp_is_a_8bit": is_a_8bit}) or qzeros
        ),
    )
    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_permute_bias", lambda bias: bias)

    module = marlin_awq_qlinear_module.AwqMarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=64,
        bias=False,
        dtype=torch.float16,
        register_buffers=True,
        input_activations={
            "type": "float",
            "bits": 8,
            "format": "float8_e4m3fn",
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
        },
    )
    original_scales = module.scales.detach().clone()

    module.post_init()

    assert module.marlin_input_dtype is torch.float8_e4m3fn
    assert captured == {
        "preprocess_dtype": torch.float16,
        "preprocess_inplace": True,
        "preprocess_shape": tuple(module.qweight.shape),
        "repack_is_a_8bit": True,
        "repack_dtype": torch.float16,
        "scale_is_a_8bit": True,
        "zp_is_a_8bit": True,
    }
    torch.testing.assert_close(module.scales, original_scales * 512)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_awq_marlin_forward_uses_real_fp8_inputs_for_dynamic_path(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "apply_awq_marlin_linear",
        lambda **kwargs: (
            captured.update(
                {
                    "input_dtype": kwargs["input"].dtype,
                    "a_scales_dtype": kwargs["a_scales"].dtype,
                    "a_scales_shape": tuple(kwargs["a_scales"].shape),
                    "weight_scale_dtype": kwargs["weight_scale"].dtype,
                }
            )
            or torch.zeros((kwargs["input"].shape[0], kwargs["output_size_per_partition"]), dtype=kwargs["weight_scale"].dtype)
        ),
    )

    module = marlin_awq_qlinear_module.AwqMarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=False,
        in_features=128,
        out_features=64,
        bias=False,
        dtype=torch.float16,
        register_buffers=True,
        input_activations={
            "type": "float",
            "bits": 8,
            "format": "float8_e4m3fn",
            "strategy": "token",
            "dynamic": True,
            "symmetric": True,
        },
    )
    module.workspace = torch.zeros(1, dtype=torch.int32)
    module.g_idx = torch.zeros(0, dtype=torch.int32)
    module.g_idx_sort_indices = torch.zeros(0, dtype=torch.int32)
    module.marlin_input_dtype = torch.float8_e4m3fn

    out = module(torch.randn(3, 128, dtype=torch.float16))

    assert captured["input_dtype"] is torch.float8_e4m3fn
    assert captured["a_scales_dtype"] is torch.float32
    assert captured["a_scales_shape"] == (3, 1)
    assert captured["weight_scale_dtype"] is torch.float16
    assert out.dtype is torch.float16


def test_marlin_runtime_error_appends_cuda_extra_install_hint_for_missing_headers(monkeypatch):
    fake_extension_api = _FakeExtensionApi(
        error_text=(
            "Marlin fp16: failed to build torch.ops JIT extension: "
            "fatal error: cusparse.h: No such file or directory"
        ),
    )

    monkeypatch.setattr(marlin_utils, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_utils, "_extension_api", lambda: fake_extension_api)
    monkeypatch.setattr(marlin_utils, "detected_cuda_wheel_include_paths", lambda: [])
    monkeypatch.setattr(marlin_utils, "which", lambda name: "/usr/local/cuda/bin/nvcc")
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising=False)

    error_text = marlin_utils.marlin_runtime_error(torch.float16)

    assert fake_extension_api.is_available_calls == ["marlin_fp16"]
    assert fake_extension_api.error_calls == ["marlin_fp16"]
    assert "cusparse.h" in error_text
    assert 'pip install "gptqmodel[marlin-cuda]"' in error_text
    assert "A local `nvcc` on PATH is still required for Marlin JIT." in error_text


def test_marlin_runtime_error_skips_install_hint_when_cuda_wheel_headers_are_detected(monkeypatch):
    fake_extension_api = _FakeExtensionApi(
        error_text=(
            "Marlin bf16: failed to build torch.ops JIT extension: "
            "fatal error: cublas_v2.h: No such file or directory"
        ),
    )

    monkeypatch.setattr(marlin_utils, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_utils, "_extension_api", lambda: fake_extension_api)
    monkeypatch.setattr(marlin_utils, "detected_cuda_wheel_include_paths", lambda: ["/tmp/nvidia/cu13/include"])
    monkeypatch.setattr(torch.version, "cuda", "13.0", raising=False)

    marlin_utils.marlin_runtime_error(torch.bfloat16)

    assert fake_extension_api.is_available_calls == ["marlin_bf16"]
    assert fake_extension_api.error_calls == ["marlin_bf16"]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_marlin_cuda_smoke_build_and_forward(monkeypatch, tmp_path):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7 or (capability[0] == 7 and capability[1] < 5):
        pytest.skip("Marlin requires compute capability >= 7.5")
    if which("ninja") is None:
        pytest.skip("Marlin JIT smoke test requires ninja.")

    scratch_root = _jit_scratch_root(tmp_path, "marlin")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("GPTQMODEL_MARLIN_FP16_BUILD_ROOT", str(scratch_root / "marlin_fp16"))
    monkeypatch.setenv("GPTQMODEL_MARLIN_BF16_BUILD_ROOT", str(scratch_root / "marlin_bf16"))
    monkeypatch.setenv("GPTQMODEL_MARLIN_FORCE_REBUILD", "1")

    assert extension_api.load(name="marlin_fp16", use_cache=False) == {
        "marlin_fp16": True,
    }
    if capability[0] >= 8:
        assert extension_api.load(name="marlin_bf16", use_cache=False) == {
            "marlin_bf16": True,
        }

    device = torch.device("cuda:0")
    dtypes = (torch.float16, torch.bfloat16) if capability[0] >= 8 else (torch.float16,)
    for dtype in dtypes:
        module = marlin_qlinear_module.MarlinLinear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=128,
            out_features=64,
            bias=False,
            dtype=dtype,
        ).to(device)
        with torch.no_grad():
            module.qweight.copy_(torch.randint(0, 16, module.qweight.shape, device=device, dtype=torch.int32))
            module.g_idx.copy_(torch.arange(module.in_features, device=device, dtype=torch.int32))
            module.scales.copy_(torch.ones_like(module.scales, device=device))
            module.qzeros.copy_(torch.zeros_like(module.qzeros, device=device))
        module.post_init()

        out = module(torch.randn(4, 128, device=device, dtype=dtype))
        torch.cuda.synchronize(device)

        assert out.shape == (4, 64)
        assert out.dtype == dtype


def test_marlin_include_paths_use_wheel_headers_when_local_cuda_is_incomplete(monkeypatch, tmp_path):
    root = tmp_path / "marlin"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    for header_name in marlin_utils._MARLIN_REQUIRED_CUDA_HEADERS:
        (wheel_cuda_include / header_name).write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(marlin_utils, "_marlin_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = marlin_utils._marlin_include_paths()

    assert include_paths[0] == str(root)
    assert str(wheel_cuda_include) in include_paths


def test_marlin_include_paths_skip_wheel_headers_when_local_cuda_has_required_headers(monkeypatch, tmp_path):
    root = tmp_path / "marlin"
    local_cuda_include = tmp_path / "local_cuda_include"
    wheel_cuda_include = tmp_path / "wheel_cuda_include"
    root.mkdir()
    local_cuda_include.mkdir()
    wheel_cuda_include.mkdir()
    for header_name in marlin_utils._MARLIN_REQUIRED_CUDA_HEADERS:
        (local_cuda_include / header_name).write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(marlin_utils, "_marlin_root", lambda: root)
    monkeypatch.setattr(cpp_module, "detected_local_cuda_include_paths", lambda: [str(local_cuda_include)])
    monkeypatch.setattr(cpp_module, "detected_cuda_wheel_include_paths", lambda: [str(wheel_cuda_include)])

    include_paths = marlin_utils._marlin_include_paths()

    assert include_paths == [str(root)]
