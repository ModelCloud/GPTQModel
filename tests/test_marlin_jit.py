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
from gptqmodel.adapter.adapter import Lora
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
        captured["packed_prefill"] = args[-2:]
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
        use_packed_prefill=True,
        packed_prefill_config=2,
    )

    assert fp16_loader.op_calls == ["gptq_marlin_gemm_fp16"]
    assert bf16_loader.op_calls == []
    assert captured == {
        "dtype": torch.float16,
        "shape": (2, 64),
        "packed_prefill": (True, 2),
    }
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
    assert "if (global_scale.defined())" in gemm_cu
    assert 'global_scale.scalar_type() == at::ScalarType::Float' in gemm_cu
    assert "global_scale.defined() ? global_scale.data_ptr<float>() : nullptr" in gemm_cu
    assert "float global_scale_f32 = 1.0f;" in template_h
    assert "c0 *= global_scale_f32;" in template_h
    assert "c1 *= global_scale_f32;" in template_h


def test_integrated_lora_contract_is_present_in_marlin_extensions():
    marlin_root = marlin_utils._marlin_root()
    lora_kernel = marlin_root.parent / "marlin_lora" / "marlin_lora_kernel.cu"
    gemm_cu = (marlin_root / "gptq_marlin.cu").read_text(encoding="utf-8")
    template_h = (marlin_root / "marlin_template.h").read_text(encoding="utf-8")

    assert str(lora_kernel) in marlin_utils._marlin_sources("fp16")
    assert str(lora_kernel) in marlin_utils._marlin_sources("bf16")
    assert "gptq_marlin_gemm_lora_fp16" in marlin_utils._MARLIN_FP16_TORCH_OPS_EXTENSION.required_ops
    assert "gptq_marlin_gemm_lora_bf16" in marlin_utils._MARLIN_BF16_TORCH_OPS_EXTENSION.required_ops
    assert "gptq_marlin_gemm_lora_prepared_fp16" in marlin_utils._MARLIN_FP16_TORCH_OPS_EXTENSION.required_ops
    assert "gptq_marlin_gemm_lora_prepared_bf16" in marlin_utils._MARLIN_BF16_TORCH_OPS_EXTENSION.required_ops

    for dtype_tag in ("fp16", "bf16"):
        source = (marlin_root / f"marlin_torch_{dtype_tag}.cpp").read_text(encoding="utf-8")
        assert f"gptq_marlin_gemm_lora_{dtype_tag}" in source
        assert f"gptq_marlin_gemm_lora_prepared_{dtype_tag}" in source
        assert "marlin_lora_fused_add_prepared_cuda" in source
        marlin_sources = marlin_utils._marlin_sources(dtype_tag)
        for rank in (32, 64, 96, 128, 192, 256):
            suffix = "" if rank == 128 else f"_r{rank}"
            rank_kernel = marlin_root / f"kernel_{dtype_tag}_lora{suffix}_ku4b8.cu"
            assert str(rank_kernel) in marlin_sources
            assert f"MarlinLoraRank{rank}" in rank_kernel.read_text(encoding="utf-8")

    assert "launch_marlin_lora_attention" in gemm_cu
    assert "prob_n != 4096 || prob_k != 4096" in gemm_cu
    assert "lora_rank != 32 && lora_rank != 64 && lora_rank != 96" in gemm_cu
    assert "lora_rank != 128 && lora_rank != 192 && lora_rank != 256" in gemm_cu
    assert "device_info.sms != 124" in gemm_cu
    assert "lora_down_ready_lock" in template_h
    assert "ld.global.acquire.gpu.b32" in template_h
    assert "atomicAdd(&locks[lora_up_arrivals_lock], 1)" in template_h


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


def test_awq_marlin_validate_device_uses_torch_visible_ordinals(monkeypatch):
    queried_devices = []

    monkeypatch.setattr(marlin_awq_qlinear_module, "IS_ROCM", False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second,MIG-third")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def get_device_capability(device):
        queried_devices.append(device)
        return (8, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)

    marlin_awq_qlinear_module.AwqMarlinLinear.validate_device(
        marlin_awq_qlinear_module.DEVICE.CUDA
    )

    assert queried_devices == [0, 1]


def test_awq_marlin_validate_device_rejects_no_visible_cuda_device(monkeypatch):
    monkeypatch.setattr(marlin_awq_qlinear_module, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(NotImplementedError, match="compute capability >= 8.0"):
        marlin_awq_qlinear_module.AwqMarlinLinear.validate_device(
            marlin_awq_qlinear_module.DEVICE.CUDA
        )


def test_awq_marlin_validate_device_rejects_pre_ampere(monkeypatch):
    monkeypatch.setattr(marlin_awq_qlinear_module, "IS_ROCM", False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

    with pytest.raises(NotImplementedError, match="compute capability >= 8.0"):
        marlin_awq_qlinear_module.AwqMarlinLinear.validate_device(
            marlin_awq_qlinear_module.DEVICE.CUDA
        )


def test_marlin_runtime_device_validation_queries_explicit_device(monkeypatch):
    queried_devices = []

    monkeypatch.setattr(marlin_utils, "IS_ROCM", False)

    def get_device_capability(device):
        queried_devices.append(device)
        return (8, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)

    capability = marlin_utils.marlin_validate_runtime_device(
        torch.device("cuda:3"),
        min_capability=(8, 0),
        backend_name="AWQ Marlin",
    )

    assert capability == (8, 0)
    assert queried_devices == [torch.device("cuda:3")]


def test_marlin_runtime_device_validation_rejects_cpu(monkeypatch):
    monkeypatch.setattr(marlin_utils, "IS_ROCM", False)

    with pytest.raises(ValueError, match="requires CUDA tensors"):
        marlin_utils.marlin_validate_runtime_device(
            torch.device("cpu"),
            min_capability=(8, 0),
            backend_name="AWQ Marlin",
        )


def test_sm75_turing_contract_is_present_in_marlin_sources():
    marlin_root = marlin_utils._marlin_root()
    gemm_cu = (marlin_root / "gptq_marlin.cu").read_text(encoding="utf-8")
    generator_py = (marlin_root / "generate_kernels.py").read_text(encoding="utf-8")
    template_h = (marlin_root / "marlin_template.h").read_text(encoding="utf-8")
    mma_h = (marlin_root / "marlin_mma.h").read_text(encoding="utf-8")
    qlinear_py = (Path(marlin_utils.__file__).resolve().parents[1] / "nn_modules" / "qlinear" / "marlin.py").read_text(
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
    assert "compute capability >= 7.5" in qlinear_py
    assert "GPTQ Marlin on compute capability 7.5" in qlinear_py
    assert "requires dtype=torch.float16." in qlinear_py


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
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_validate_runtime_device",
        lambda *args, **kwargs: (8, 0),
    )
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_make_workspace_new",
        lambda device, **kwargs: torch.zeros(1, dtype=torch.int32, device=device),
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
    assert module.workspace.numel() == 1
    assert module.g_idx_sort_indices.numel() == 0

    module.to("meta")

    assert module.workspace.device.type == "meta"
    assert module.g_idx_sort_indices.device.type == "meta"


def test_marlin_quant_linear_post_init_pads_weight_scales_and_bias(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_validate_runtime_device",
        lambda *args, **kwargs: (8, 0),
    )
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

    def validate_runtime_device(device, **kwargs):
        captured["runtime_device"] = device
        return (8, 0)

    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(
        marlin_awq_qlinear_module,
        "marlin_validate_runtime_device",
        validate_runtime_device,
    )
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
        "runtime_device": torch.device("cpu"),
    }
    assert {"workspace", "g_idx", "g_idx_sort_indices"} <= module._buffers.keys()

    module.to("meta")

    assert module.workspace.device.type == "meta"
    assert module.g_idx.device.type == "meta"
    assert module.g_idx_sort_indices.device.type == "meta"


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


def test_marlin_quant_linear_registers_nonpersistent_runtime_state(monkeypatch):
    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)

    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=128,
        out_features=64,
        bias=False,
        dtype=torch.float16,
    )

    assert {"workspace", "g_idx_sort_indices"} <= module._buffers.keys()
    assert {"workspace", "g_idx_sort_indices"} <= module._non_persistent_buffers_set
    assert "workspace" not in module.state_dict()
    assert "g_idx_sort_indices" not in module.state_dict()


def test_awq_marlin_quant_linear_registers_nonpersistent_runtime_state(monkeypatch):
    monkeypatch.setattr(marlin_awq_qlinear_module, "marlin_import_exception", None)

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
    )

    runtime_names = {"workspace", "g_idx", "g_idx_sort_indices"}
    assert runtime_names <= module._buffers.keys()
    assert runtime_names <= module._non_persistent_buffers_set
    assert runtime_names.isdisjoint(module.state_dict())


def test_marlin_quant_linear_forward_promotes_bias_to_input_dtype(monkeypatch):
    captured = {}

    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_validate_runtime_device",
        lambda *args, **kwargs: (8, 0),
    )
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_make_workspace_new",
        lambda device, **kwargs: torch.zeros(1, dtype=torch.int32, device=device),
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


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "expected_rows", "expected_packed_prefill"),
    [
        ((256,), (64,), 1, False),
        ((2, 256), (2, 64), 2, True),
        ((1, 2, 256), (1, 2, 64), 2, True),
        ((1, 1, 256), (1, 1, 64), 1, False),
    ],
)
def test_marlin_quant_linear_uses_one_integrated_lora_dispatch(
    monkeypatch,
    input_shape,
    output_shape,
    expected_rows,
    expected_packed_prefill,
):
    captured = {}
    ordinary_calls = 0

    def integrated_op(*args):
        captured["rows"] = args[0].numel() // args[0].shape[-1]
        captured["out_features"] = args[5].shape[1]
        captured["lora_a"] = args[4]
        captured["lora_b"] = args[5]
        captured["use_packed_prefill"] = args[-2]
        return torch.full(args[0].shape[:-1] + (args[5].shape[1],), 7.0, dtype=args[0].dtype)

    def ordinary_marlin(**kwargs):
        nonlocal ordinary_calls
        ordinary_calls += 1
        return torch.zeros(
            (kwargs["input"].shape[0], kwargs["output_size_per_partition"]),
            dtype=kwargs["input"].dtype,
        )

    monkeypatch.setattr(marlin_qlinear_module, "marlin_import_exception", None)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(marlin_qlinear_module, "marlin_runtime_error", lambda dtype: "")
    monkeypatch.setattr(
        marlin_qlinear_module,
        "marlin_make_workspace_new",
        lambda device, **kwargs: torch.zeros(1, dtype=torch.int32, device=device),
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
    monkeypatch.setattr(marlin_qlinear_module, "apply_gptq_marlin_linear", ordinary_marlin)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def prepare_integrated_op(adapter, **kwargs):
        captured["use_prepared_marlin"] = kwargs["use_prepared_marlin"]
        return integrated_op, adapter.lora_A, adapter.lora_B, None, 16, True

    monkeypatch.setattr(
        marlin_qlinear_module,
        "prepare_marlin_fused_lora",
        prepare_integrated_op,
    )

    adapter = Lora(
        rank=8,
        lora_A=torch.randn(256, 8, dtype=torch.float16),
        lora_B=torch.randn(8, 64, dtype=torch.float16),
    )
    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=256,
        out_features=64,
        bias=False,
        dtype=torch.float16,
        adapter=adapter,
    )
    with torch.no_grad():
        module.lora_A.copy_(adapter.lora_A)
        module.lora_B.copy_(adapter.lora_B)
    module.post_init()
    module.packed_prefill = True
    module.packed_prefill_min_rows = 1
    module.packed_prefill_config = 1
    module.adapter.apply = lambda **kwargs: pytest.fail("adapter fallback should not run")

    out = module(torch.randn(input_shape, dtype=torch.float16))

    assert ordinary_calls == 0
    assert captured == {
        "rows": expected_rows,
        "out_features": 64,
        "lora_a": module.adapter.lora_A,
        "lora_b": module.adapter.lora_B,
        "use_packed_prefill": expected_packed_prefill,
        "use_prepared_marlin": True,
    }
    assert out.shape == output_shape
    assert torch.equal(out, torch.full_like(out, 7.0))
    assert None not in module.list_buffers()


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
def test_marlin_live_row_fp32_scratch_matches_fp16_reduction(dtype):
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8 and dtype == torch.bfloat16:
        pytest.skip("Marlin BF16 requires compute capability >= 8.0")

    generator = torch.Generator(device=device)
    generator.manual_seed(31)
    in_features = 4096
    out_features = 256
    marlin_linear = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
        dtype=dtype,
    ).to(device)
    with torch.no_grad():
        marlin_linear.qweight.copy_(
            torch.randint(
                -(2**31),
                2**31 - 1,
                marlin_linear.qweight.shape,
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
        )
        marlin_linear.scales.copy_(
            torch.rand(
                marlin_linear.scales.shape,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            * 0.01
            + 0.01
        )
        marlin_linear.g_idx.zero_()
        marlin_linear.qzeros.zero_()
    marlin_linear.post_init()

    inputs = torch.randn((8, in_features), device=device, dtype=dtype, generator=generator)
    with torch.inference_mode():
        for rows in range(1, 9):
            marlin_linear.fp32 = False
            expected = marlin_linear(inputs[:rows]).float()
            marlin_linear.fp32 = True
            actual = marlin_linear(inputs[:rows]).float()
            atol = 1.25e-1 if dtype == torch.bfloat16 else 5e-2
            torch.testing.assert_close(actual, expected, rtol=5e-2, atol=atol)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [32, 64, 96, 128, 192, 256])
def test_marlin_lora_attention_mega_kernel_matches_dense_update_and_releases_locks(
    dtype, rank, monkeypatch
):
    device_index = 1 if dtype == torch.bfloat16 and torch.cuda.device_count() > 1 else 0
    device = torch.device(f"cuda:{device_index}")
    if torch.cuda.get_device_capability(device) != (8, 0):
        pytest.skip("The cooperative Marlin+LoRA mega-kernel is enabled only for sm_80")

    monkeypatch.setenv("GPTQMODEL_MARLIN_LORA_COOPERATIVE", "1")
    generator = torch.Generator(device=device)
    generator.manual_seed(27)
    features = 4096
    x = torch.randn((1, features), dtype=dtype, device=device, generator=generator)
    lora_a = torch.randn((features, rank), dtype=dtype, device=device, generator=generator) * 0.002
    lora_b = torch.randn((rank, features), dtype=dtype, device=device, generator=generator) * 0.002
    adapter = Lora(rank=rank, lora_A=lora_a, lora_B=lora_b)
    module = marlin_qlinear_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=features,
        out_features=features,
        bias=False,
        dtype=dtype,
        adapter=adapter,
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(
                -(2**31),
                2**31 - 1,
                module.qweight.shape,
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
        )
        module.scales.copy_(
            torch.rand(module.scales.shape, dtype=dtype, device=device, generator=generator) * 0.01 + 0.01
        )
        module.qzeros.zero_()
        module.g_idx.zero_()
        module.lora_A.copy_(lora_a)
        module.lora_B.copy_(lora_b)
    module.eval()
    module.post_init()
    assert module.lora_cooperative_state is not None
    assert module.lora_cooperative_state[-1] is True

    module_adapter = module.adapter
    module.adapter = None
    with torch.inference_mode():
        base = module(x)
    module.adapter = module_adapter
    expected = base.float() + (x.float() @ lora_a.float()) @ lora_b.float()

    with torch.inference_mode():
        actual = module(x)
        repeated = module(x)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            streamed = module(x)
        stream.synchronize()

    assert actual.shape == base.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(repeated.float(), expected, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(streamed.float(), expected, rtol=5e-2, atol=5e-2)
    mega_workspace_words = 128 + rank // 2
    assert module.workspace.numel() >= mega_workspace_words
    assert torch.count_nonzero(module.workspace[:128]).item() == 0

    # The M=1 mega-kernel shares this persistent buffer with ordinary Marlin.
    # Its adapter payload must remain outside the first 128 lock words before a
    # later M=12 launch reuses that prefix for global reduction.
    x_m12 = torch.randn((12, features), dtype=dtype, device=device, generator=generator)
    module.adapter = None
    with torch.inference_mode():
        base_m12 = module(x_m12)
    module.adapter = module_adapter
    expected_m12 = base_m12.float() + (x_m12.float() @ lora_a.float()) @ lora_b.float()

    with torch.inference_mode():
        actual_m12 = module(x_m12)

    assert actual_m12.shape == base_m12.shape
    assert actual_m12.dtype == dtype
    torch.testing.assert_close(actual_m12.float(), expected_m12, rtol=5e-2, atol=5e-2)

    # A workspace below the rank-specific payload requirement remains enough
    # for ordinary Marlin and must select the established two-launch fallback.
    module.workspace = torch.zeros(128, dtype=torch.int32, device=device)
    with torch.inference_mode():
        legacy_workspace_actual = module(x)
    torch.testing.assert_close(legacy_workspace_actual.float(), expected, rtol=5e-2, atol=5e-2)


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
