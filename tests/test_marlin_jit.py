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
from gptqmodel.utils.backend import BACKEND
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
        captured["shape"] = (args[11], args[12])
        return torch.full((args[11], args[12]), 3.0, dtype=args[0].dtype)

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
        return torch.full((args[11], args[12]), 5.0, dtype=args[0].dtype)

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
        captured["global_scale_dtype"] = args[5].dtype
        captured["global_scale_shape"] = tuple(args[5].shape)
        return torch.zeros((args[11], args[12]), dtype=args[0].dtype)

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


def test_nvfp4_global_scale_contract_is_float_in_marlin_sources():
    marlin_root = marlin_utils._marlin_root()
    marlin_cuh = (marlin_root / "marlin.cuh").read_text(encoding="utf-8")
    kernel_h = (marlin_root / "kernel.h").read_text(encoding="utf-8")
    gemm_cu = (marlin_root / "gptq_marlin.cu").read_text(encoding="utf-8")
    template_h = (marlin_root / "marlin_template.h").read_text(encoding="utf-8")

    assert "#include <torch/all.h>" not in marlin_cuh
    assert "#include <torch/extension.h>" not in marlin_cuh
    assert "const float *__restrict__ global_scale_ptr" in kernel_h
    assert 'global_scale = torch::empty({0}, options_fp32);' in gemm_cu
    assert 'global_scale.scalar_type() == at::ScalarType::Float' in gemm_cu
    assert "global_scale.data_ptr<float>()" in gemm_cu
    assert "float global_scale_f32 = 1.0f;" in template_h
    assert "c0 *= global_scale_f32;" in template_h
    assert "c1 *= global_scale_f32;" in template_h


def test_marlin_extra_cuda_cflags_enable_static_global_template_stub_when_nvcc_is_compatible(monkeypatch):
    monkeypatch.setattr(marlin_utils, "is_nvcc_compatible", lambda: True)

    flags = marlin_utils._marlin_extra_cuda_cflags()

    assert flags[0] == "-static-global-template-stub=false"
    assert "-static-global-template-stub=true" not in flags


def test_marlin_extra_cuda_cflags_skip_static_global_template_stub_when_nvcc_is_incompatible(monkeypatch):
    monkeypatch.setattr(marlin_utils, "is_nvcc_compatible", lambda: False)

    flags = marlin_utils._marlin_extra_cuda_cflags()

    assert "-static-global-template-stub=false" not in flags
    assert "-static-global-template-stub=true" not in flags


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


@pytest.mark.parametrize(
    "shape,group_size,expected",
    [
        ((64, 128), -1, (64, 128)),
        ((128, 64), -1, (128, 64)),
        ((200, 288), 32, (256, 320)),
        ((256, 208), -1, (256, 256)),
        ((200, 384), 128, (256, 384)),
    ],
)
def test_marlin_padded_nk_selects_minimal_thread_tile(shape, group_size, expected):
    size_n, size_k = shape

    padded_n, padded_k = marlin_utils.marlin_padded_nk(
        size_n, size_k, group_size
    )

    assert (padded_n, padded_k) == expected
    assert marlin_utils.marlin_is_tile_aligned(padded_n, padded_k)
    if group_size > 0:
        assert padded_k % group_size == 0


def test_marlin_tile_padding_helpers_preserve_values_and_shapes():
    size_n, size_k, group_size = 200, 288, 32
    padded_n, padded_k = marlin_utils.marlin_padded_nk(
        size_n, size_k, group_size
    )

    qweight = torch.ones((size_k // 8, size_n), dtype=torch.int32)
    padded_qweight = marlin_utils.marlin_pad_qweight(
        qweight, size_n, size_k, padded_n, padded_k
    )
    assert padded_qweight.shape == (padded_k // 8, padded_n)
    assert torch.equal(padded_qweight[: qweight.size(0), :size_n], qweight)
    assert torch.count_nonzero(padded_qweight[:, size_n:]) == 0
    assert torch.count_nonzero(padded_qweight[qweight.size(0) :, :]) == 0

    scales = torch.ones((size_k // group_size, size_n))
    padded_scales = marlin_utils.marlin_pad_scales(
        scales, size_n, size_k, padded_n, padded_k, group_size
    )
    assert padded_scales.shape == (padded_k // group_size, padded_n)
    assert torch.equal(padded_scales[: scales.size(0), :size_n], scales)
    assert torch.count_nonzero(padded_scales[:, size_n:]) == 0


def test_marlin_quant_linear_validation_limits_tile_padding_to_non_act_order(monkeypatch):
    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    common = {
        "bits": 4,
        "group_size": 32,
        "sym": True,
        "in_features": 288,
        "out_features": 200,
        "pack_dtype": torch.int32,
        "dtype": torch.float16,
        "dynamic": None,
        "device": None,
        "trainable": False,
        "adapter": None,
    }

    ok, err = marlin_qlinear_module.MarlinLinear._validate(
        **common, desc_act=False
    )
    assert ok is True
    assert err is None

    ok, err = marlin_qlinear_module.MarlinLinear._validate(
        **common, desc_act=True
    )
    assert ok is False
    assert "activation-order" in str(err)

    channelwise = dict(common, group_size=-1)
    ok, err = marlin_qlinear_module.MarlinLinear._validate(
        **channelwise, desc_act=True
    )
    assert ok is True
    assert err is None

    explicit_channelwise = dict(common, group_size=common["in_features"])
    ok, err = marlin_qlinear_module.MarlinLinear._validate(
        **explicit_channelwise, desc_act=True
    )
    assert ok is True
    assert err is None

    aligned = dict(common, in_features=64, out_features=128)
    ok, err = marlin_qlinear_module.MarlinLinear._validate(
        **aligned, desc_act=True
    )
    assert ok is True
    assert err is None


def test_marlin_auto_selection_keeps_tile_padding_opt_in(monkeypatch):
    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    kwargs = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "in_features": 288,
        "out_features": 200,
        "bias": False,
        "dtype": torch.float16,
    }

    with pytest.raises(NotImplementedError, match="request GPTQ_MARLIN explicitly"):
        marlin_qlinear_module.MarlinLinear(**kwargs, backend=BACKEND.AUTO)

    explicit = marlin_qlinear_module.MarlinLinear(
        **kwargs, backend=BACKEND.GPTQ_MARLIN
    )
    assert explicit.in_features == 288
    assert explicit.out_features == 200

    aligned = marlin_qlinear_module.MarlinLinear(
        **dict(kwargs, in_features=128, out_features=64),
        backend=BACKEND.AUTO,
    )
    assert aligned.in_features == 128
    assert aligned.out_features == 64


@pytest.mark.parametrize("group_size", [32, -1])
def test_awq_marlin_tile_padding_helpers_preserve_packed_values(group_size):
    size_n, size_k, bits = 200, 288, 4
    padded_n, padded_k = marlin_utils.marlin_padded_nk(
        size_n, size_k, group_size
    )
    pack_factor = 32 // bits
    groups = size_k // group_size if group_size > 0 else 1
    padded_groups = padded_k // group_size if group_size > 0 else 1

    qweight = torch.ones((size_k, size_n // pack_factor), dtype=torch.int32)
    padded_qweight = marlin_utils.marlin_pad_awq_qweight(
        qweight, size_n, size_k, padded_n, padded_k, bits
    )
    assert padded_qweight.shape == (padded_k, padded_n // pack_factor)
    assert torch.equal(padded_qweight[:size_k, : qweight.size(1)], qweight)
    assert torch.count_nonzero(padded_qweight[:, qweight.size(1) :]) == 0
    assert torch.count_nonzero(padded_qweight[size_k:, :]) == 0

    qzeros = torch.ones((groups, size_n // pack_factor), dtype=torch.int32)
    padded_qzeros = marlin_utils.marlin_pad_awq_qzeros(
        qzeros,
        size_n,
        size_k,
        padded_n,
        padded_k,
        group_size,
        bits,
    )
    assert padded_qzeros.shape == (padded_groups, padded_n // pack_factor)
    assert torch.equal(padded_qzeros[:groups, : qzeros.size(1)], qzeros)
    assert torch.count_nonzero(padded_qzeros[:, qzeros.size(1) :]) == 0
    assert torch.count_nonzero(padded_qzeros[groups:, :]) == 0


def test_awq_marlin_quant_linear_validation_accepts_packable_tile_tails(monkeypatch):
    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_import_exception", None)
    common = {
        "bits": 4,
        "group_size": 32,
        "sym": False,
        "desc_act": False,
        "in_features": 288,
        "out_features": 200,
        "pack_dtype": torch.int32,
        "dtype": torch.float16,
        "dynamic": None,
        "device": None,
        "trainable": False,
        "adapter": None,
    }

    ok, err = marlin_awq_qlinear_module.AwqMarlinLinear._validate(**common)
    assert ok is True
    assert err is None

    ok, err = marlin_awq_qlinear_module.AwqMarlinLinear._validate(
        **dict(common, out_features=202)
    )
    assert ok is False
    assert "pack_factor=8" in str(err)

    ok, err = marlin_awq_qlinear_module.AwqMarlinLinear._validate(
        **dict(common, in_features=208, out_features=256, group_size=208)
    )
    assert ok is True
    assert err is None

    ok, err = marlin_awq_qlinear_module.AwqMarlinLinear._validate(
        **dict(common, bits=8)
    )
    assert ok is False
    assert "enabled only for 4-bit weights" in str(err)


def test_awq_marlin_auto_selection_keeps_tile_padding_opt_in(monkeypatch):
    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_import_exception", None)
    kwargs = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": False,
        "in_features": 288,
        "out_features": 200,
        "bias": False,
        "dtype": torch.float16,
    }

    with pytest.raises(NotImplementedError, match="request AWQ_MARLIN explicitly"):
        marlin_awq_qlinear_module.AwqMarlinLinear(
            **kwargs, backend=BACKEND.AUTO
        )

    explicit = marlin_awq_qlinear_module.AwqMarlinLinear(
        **kwargs, backend=BACKEND.AWQ_MARLIN
    )
    assert explicit.in_features == 288
    assert explicit.out_features == 200

    aligned = marlin_awq_qlinear_module.AwqMarlinLinear(
        **dict(kwargs, in_features=256, out_features=128),
        backend=BACKEND.AUTO,
    )
    assert aligned.in_features == 256
    assert aligned.out_features == 128


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
    assert "Turing only supports float16 dense Marlin kernels." in gemm_cu
    assert 'stage_values.insert(0, 2)' in generator_py
    assert "constexpr bool use_fp16_accum" in template_h
    assert "__CUDA_ARCH__ == 750" in mma_h
    assert "m16n8k8.row.col.f16.f16.f16.f16" in mma_h
    assert "compute capability >= 7.5" in loader_py
    assert "GPTQ Marlin on Turing (compute capability 7.5)" in loader_py
    assert "dtype=torch.float16 only." in loader_py


def test_stage2_dense_four_bit_tiles_stay_in_sync_between_selector_and_codegen():
    marlin_root = marlin_utils._ensure_generated_marlin_kernels()
    gemm_cu = (marlin_root / "gptq_marlin.cu").read_text(encoding="utf-8")
    generator_py = (marlin_root / "generate_kernels.py").read_text(encoding="utf-8")
    kernel_u4 = (marlin_root / "kernel_fp16_ku4.cu").read_text(encoding="utf-8")
    kernel_u4b8 = (marlin_root / "kernel_fp16_ku4b8.cu").read_text(encoding="utf-8")
    kernel_nvfp4 = (marlin_root / "kernel_fp16_kfe2m1f.cu").read_text(encoding="utf-8")

    assert "kIsStage2FourBitTile" in gemm_cu
    assert "THREAD_M_BLOCKS * 2 <= THREAD_K_BLOCKS" in gemm_cu
    assert "stages == 2 && num_bits == 4" in gemm_cu
    assert "thread_m_blocks * 2 > th_config.thread_k / 16" in gemm_cu
    assert "_is_4bit_weight" in generator_py
    assert "stage_value == 2" in generator_py

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
    gemm_cu = (marlin_root / "gptq_marlin.cu").read_text(encoding="utf-8")
    generator_py = (marlin_root / "generate_kernels.py").read_text(encoding="utf-8")
    template_h = (marlin_root / "marlin_template.h").read_text(encoding="utf-8")

    assert 'scalar_type == "vllm::kFE4M3fn" and group_blocks not in [-1, 2, 8]' in generator_py
    assert 'scalar_type == "vllm::kFE4M3fn" and group_blocks == 2' in generator_py
    assert 'MXFP8 is only supported with bf16 compute.' in generator_py
    assert "MXFP8_GET_IF(vllm::kFE4M3fn, pipe_stages)" in gemm_cu
    assert "W_TYPE == vllm::kFE4M3fn && GROUP_BLOCKS == 2" in gemm_cu
    assert "Float8_e8m0fnu" in gemm_cu
    assert "float8_e4m3fn with float8_e8m0fnu scales requires " in gemm_cu
    assert "float8_e4m3fn only supports group_size == 32 (MXFP8)" in gemm_cu
    assert "// MXFP8: FP8 weights with e8m0 microscaling block scales." in template_h
    assert "w_type == vllm::kFE4M3fn && !(s_type == vllm::kFE8M0fnu)" in template_h
    assert "if constexpr (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu)" in template_h


def test_ensure_generated_marlin_kernels_repairs_stale_generated_sources(monkeypatch, tmp_path):
    source_root = marlin_utils._marlin_root()
    test_root = tmp_path / "marlin"
    test_root.mkdir()
    copy2(source_root / "generate_kernels.py", test_root / "generate_kernels.py")

    monkeypatch.setattr(marlin_utils, "_marlin_root", lambda: test_root)

    assert marlin_utils._ensure_generated_marlin_kernels() == test_root

    kernel_path = test_root / "kernel_bf16_kfe4m3fn.cu"
    original_text = kernel_path.read_text(encoding="utf-8")
    assert "vllm::kFE8M0fnu.id()" in original_text

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

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
        captured["dtype"] = torch.bfloat16
        captured["shape"] = tuple(b_q_weight.shape)
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
    assert captured == {"dtype": torch.bfloat16, "shape": (32, 64)}
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
    assert module._marlin_tile_padding is None


def test_marlin_quant_linear_post_init_pads_weight_scales_and_bias(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(128, dtype=torch.int32, device=device),
    )

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits, dtype=None):
        captured["qweight"] = (tuple(b_q_weight.shape), size_k, size_n, dtype)
        pack_factor = 32 // num_bits
        return torch.zeros(
            (size_k // 16, size_n * 16 // pack_factor),
            dtype=torch.int32,
            device=b_q_weight.device,
        )

    def fake_permute_scales(scales, size_k, size_n, group_size):
        captured["scales"] = (
            tuple(scales.shape),
            size_k,
            size_n,
            group_size,
        )
        return scales

    monkeypatch.setattr(marlin_qlinear_module, "gptq_marlin_repack", fake_repack)
    monkeypatch.setattr(
        marlin_qlinear_module, "marlin_permute_scales", fake_permute_scales
    )
    monkeypatch.setattr(marlin_qlinear_module, "marlin_permute_bias", lambda bias: bias)

    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=288,
        out_features=200,
        bias=True,
        dtype=torch.float16,
    )
    module.post_init()

    assert module.in_features == 288
    assert module.out_features == 200
    assert module.qweight.shape == (20, 512)
    assert module.scales.shape == (10, 256)
    assert module.bias.shape == (256,)
    assert module._marlin_tile_padding == (256, 320)
    assert captured == {
        "qweight": ((40, 256), 320, 256, torch.float16),
        "scales": ((10, 256), 320, 256, 32),
    }


def test_apply_gptq_marlin_linear_pads_input_and_slices_output(monkeypatch):
    captured = {}

    def fake_gemm(a, _c, _weight, bias, _scales, _global_scale,
                  _weight_zp, _g_idx, _sort_indices, _workspace, _wtype,
                  **kwargs):
        captured.update(
            {
                "input_shape": tuple(a.shape),
                "bias_shape": tuple(bias.shape),
                "size_m": kwargs["size_m"],
                "size_n": kwargs["size_n"],
                "size_k": kwargs["size_k"],
            }
        )
        return torch.ones(
            (kwargs["size_m"], kwargs["size_n"]), dtype=a.dtype
        )

    monkeypatch.setattr(marlin_utils, "gptq_marlin_gemm", fake_gemm)

    output = marlin_utils.apply_gptq_marlin_linear_padded(
        input=torch.randn(2, 3, 288, dtype=torch.float16),
        weight=torch.zeros((20, 512), dtype=torch.int32),
        weight_scale=torch.ones((10, 256), dtype=torch.float16),
        weight_zp=torch.empty(0, dtype=torch.int32),
        g_idx=torch.empty(0, dtype=torch.int32),
        g_idx_sort_indices=torch.empty(0, dtype=torch.int32),
        workspace=torch.zeros(128, dtype=torch.int32),
        wtype=scalar_types.uint4b8,
        output_size_per_partition=200,
        input_size_per_partition=288,
        is_k_full=True,
        bias=torch.zeros(256, dtype=torch.float16),
        tile_padding=(256, 320),
    )

    assert captured == {
        "input_shape": (6, 320),
        "bias_shape": (256,),
        "size_m": 6,
        "size_n": 256,
        "size_k": 320,
    }
    assert output.shape == (2, 3, 200)
    assert output.is_contiguous()


def test_awq_marlin_quant_linear_post_init_pads_packed_tensors(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(
        marlin_awq_qlinear_module, "marlin_runtime_available", lambda dtype: True
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module, "marlin_runtime_error", lambda dtype: ""
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(128, dtype=torch.int32, device=device),
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "marlin_make_empty_g_idx",
        lambda device: torch.empty(0, dtype=torch.int32, device=device),
    )

    def fake_repack(qweight, size_k, size_n, num_bits, dtype=None):
        captured["qweight"] = (
            tuple(qweight.shape),
            size_k,
            size_n,
            num_bits,
            dtype,
        )
        pack_factor = 32 // num_bits
        return torch.zeros(
            (size_k // 16, size_n * 16 // pack_factor),
            dtype=torch.int32,
            device=qweight.device,
        )

    def fake_permute_scales(scales, size_k, size_n, group_size):
        captured["scales"] = (
            tuple(scales.shape),
            size_k,
            size_n,
            group_size,
        )
        return scales

    def fake_zero_points(qzeros, size_k, size_n, num_bits):
        captured["qzeros"] = (
            tuple(qzeros.shape),
            size_k,
            size_n,
            num_bits,
        )
        return qzeros

    monkeypatch.setattr(
        marlin_awq_qlinear_module, "awq_marlin_repack", fake_repack
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module, "marlin_permute_scales", fake_permute_scales
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "awq_to_marlin_zero_points",
        fake_zero_points,
    )
    monkeypatch.setattr(
        marlin_awq_qlinear_module, "marlin_permute_bias", lambda bias: bias
    )

    module = marlin_awq_qlinear_module.AwqMarlinLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=False,
        in_features=288,
        out_features=200,
        bias=True,
        dtype=torch.float16,
        register_buffers=True,
    )
    module.post_init()

    assert module.in_features == 288
    assert module.out_features == 200
    assert module.qweight.shape == (20, 512)
    assert module.scales.shape == (10, 256)
    assert module.qzeros.shape == (10, 32)
    assert module.bias.shape == (256,)
    assert module._marlin_tile_padding == (256, 320)
    assert captured == {
        "qweight": ((320, 32), 320, 256, 4, torch.float16),
        "scales": ((10, 256), 320, 256, 32),
        "qzeros": ((10, 32), 10, 256, 4),
    }


def test_apply_awq_marlin_linear_pads_input_and_slices_output(monkeypatch):
    captured = {}

    def fake_gemm(a, _c, _weight, bias, _scales, _global_scale,
                  _weight_zp, _g_idx, _sort_indices, _workspace, _wtype,
                  **kwargs):
        captured.update(
            {
                "input_shape": tuple(a.shape),
                "bias_shape": tuple(bias.shape),
                "size_m": kwargs["size_m"],
                "size_n": kwargs["size_n"],
                "size_k": kwargs["size_k"],
            }
        )
        return torch.ones(
            (kwargs["size_m"], kwargs["size_n"]), dtype=a.dtype
        )

    monkeypatch.setattr(marlin_utils, "gptq_marlin_gemm", fake_gemm)

    output = marlin_utils.apply_awq_marlin_linear_padded(
        input=torch.randn(2, 3, 288, dtype=torch.float16),
        weight=torch.zeros((20, 512), dtype=torch.int32),
        weight_scale=torch.ones((10, 256), dtype=torch.float16),
        weight_zp=torch.zeros((10, 32), dtype=torch.int32),
        g_idx=torch.empty(0, dtype=torch.int32),
        g_idx_sort_indices=torch.empty(0, dtype=torch.int32),
        workspace=torch.zeros(128, dtype=torch.int32),
        quant_type=scalar_types.uint4,
        output_size_per_partition=200,
        input_size_per_partition=288,
        bias=torch.zeros(256, dtype=torch.float16),
        tile_padding=(256, 320),
    )

    assert captured == {
        "input_shape": (6, 320),
        "bias_shape": (256,),
        "size_m": 6,
        "size_n": 256,
        "size_k": 320,
    }
    assert output.shape == (2, 3, 200)
    assert output.is_contiguous()


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


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "in_features,out_features,group_size",
    [
        (256, 200, 32),  # N tail
        (208, 256, -1),  # K tail with channelwise scales
        (208, 256, 208),  # Explicit K-sized channelwise group
        (288, 200, 32),  # N and K tails
    ],
)
def test_marlin_cuda_padded_shape_matches_dequantized_reference(
    dtype, bits, in_features, out_features, group_size
):
    capability = torch.cuda.get_device_capability()
    if capability[0] < 8 and dtype == torch.bfloat16:
        pytest.skip("Marlin BF16 requires compute capability >= 8.0")
    if not marlin_utils.marlin_runtime_available(dtype):
        pytest.skip(marlin_utils.marlin_runtime_error(dtype))

    torch.manual_seed(17)
    device = torch.device("cuda:0")
    pack_factor = 32 // bits
    num_groups = 1 if group_size == -1 else in_features // group_size
    scale_group_size = in_features if group_size == -1 else group_size

    codes = torch.randint(
        1,
        1 << bits,
        (in_features, out_features),
        dtype=torch.int32,
        device=device,
    )
    qweight = torch.zeros(
        (in_features // pack_factor, out_features),
        dtype=torch.int32,
        device=device,
    )
    for lane in range(pack_factor):
        qweight.bitwise_or_(codes[lane::pack_factor] << (lane * bits))

    scales = (
        torch.rand(
            (num_groups, out_features),
            device=device,
            dtype=torch.float32,
        )
        * 0.02
        + 0.002
    ).to(dtype)
    bias = (torch.randn(out_features, device=device) * 0.01).to(dtype)

    module = marlin_qlinear_module.MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        dtype=dtype,
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(qweight)
        module.scales.copy_(scales)
        module.g_idx.copy_(
            torch.arange(in_features, device=device, dtype=torch.int32)
            // scale_group_size
        )
        module.qzeros.zero_()
        module.bias.copy_(bias)
    module.post_init()

    x = torch.randn((8, in_features), device=device, dtype=dtype) / in_features**0.5
    dense_weight = (codes.to(dtype) - (1 << (bits - 1))) * scales.repeat_interleave(
        scale_group_size, dim=0
    )
    expected = x @ dense_weight + bias
    with torch.inference_mode():
        actual = module(x)
        repeated = module(x)
    torch.cuda.synchronize(device)

    assert actual.shape == (8, out_features)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(repeated, expected, rtol=5e-2, atol=5e-2)


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
