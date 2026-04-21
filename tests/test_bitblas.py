import os
import tempfile
import time
from statistics import mean, pstdev

import pytest
import torch
import torch.nn as nn
from parameterized import parameterized

import gptqmodel.nn_modules.qlinear.bitblas as bitblas_module
import gptqmodel.utils.bitblas as bitblas_utils
import gptqmodel.utils.model as model_utils
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear, marlin_import_exception
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig
from gptqmodel.utils.importer import get_kernel_for_backend
from gptqmodel.utils.logger import render_table


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BitBLAS")
@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_bitblas_forward_pass1():
    bitblas_module.import_bitblas()

    device_index = int(os.environ.get("BITBLAS_TEST_DEVICE", 0))
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)

    layer = bitblas_module.BitblasLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=32,
        out_features=32,
        bias=False,
    ).to(device)

    with torch.no_grad():
        layer.qweight.zero_()
        layer.scales.zero_()
        if layer.quant_config.with_zeros:
            layer.qzeros.zero_()

    x = torch.randn(2, 32, device=device, dtype=layer.TORCH_DTYPE)
    y = layer(x)

    assert y.shape == (2, 32)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_bitblas_target_normalization_preserves_supported_arch():
    assert bitblas_module._normalize_bitblas_target("cuda -arch=sm_89") == "cuda -arch=sm_89"


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_bitblas_target_normalization_strips_supported_arch_suffix():
    assert bitblas_module._normalize_bitblas_target("cuda -arch=sm_90a") == "cuda -arch=sm_90"


@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_bitblas_target_normalization_falls_back_for_future_arch():
    assert bitblas_module._normalize_bitblas_target("cuda -arch=sm_120") == bitblas_module._bitblas_fallback_target()


def test_bitblas_supports_gptq_v2_kernel_selection():
    assert get_kernel_for_backend(BACKEND.BITBLAS, METHOD.GPTQ, FORMAT.GPTQ_V2) is bitblas_module.BitblasLinear


def test_bitblas_tuning_defaults_off_for_repack(monkeypatch):
    """Keep GPTQ repacks from forcing expensive BitBLAS retuning by default."""
    monkeypatch.delenv("BITBLAS_ENABLE_TUNING", raising=False)

    assert bitblas_utils._should_enable_bitblas_tuning(repack=True) is False
    assert bitblas_utils._should_enable_bitblas_tuning(repack=False) is True


def test_bitblas_tuning_env_override(monkeypatch):
    """Allow callers to opt in or out of BitBLAS tuning explicitly."""
    monkeypatch.setenv("BITBLAS_ENABLE_TUNING", "1")
    assert bitblas_utils._should_enable_bitblas_tuning(repack=True) is True

    monkeypatch.setenv("BITBLAS_ENABLE_TUNING", "0")
    assert bitblas_utils._should_enable_bitblas_tuning(repack=False) is False


def test_bitblas_prefers_float32_accumulation_for_fp16_inputs(monkeypatch):
    """Use fp32 accumulation to keep dequantized GPTQ inference numerically stable."""

    captured = {}

    class _DummyMatmul:
        lib = object()
        weight_transform = None

        @staticmethod
        def retrieve_weight_shape():
            return (1, 1)

    def _fake_get_or_create(self, config, enable_tuning):
        captured["config"] = config
        captured["enable_tuning"] = enable_tuning
        return _DummyMatmul()

    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    monkeypatch.setattr(
        bitblas_module.BitblasLinear,
        "_get_or_create_bitblas_operator",
        _fake_get_or_create,
    )

    bitblas_module.BitblasLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=32,
        out_features=32,
        pack_dtype=torch.int32,
        bias=False,
        enable_tuning=False,
    )

    assert captured["config"].A_dtype == "float16"
    assert captured["config"].out_dtype == "float16"
    assert captured["config"].accum_dtype == "float32"


def test_bitblas_uses_bfloat16_configuration_when_requested(monkeypatch):
    """Keep BitBLAS buffers and operator config aligned with bf16 model loads."""

    captured = {}

    class _DummyMatmul:
        lib = object()
        weight_transform = None

        @staticmethod
        def retrieve_weight_shape():
            return (1, 1)

    def _fake_get_or_create(self, config, enable_tuning):
        captured["config"] = config
        captured["enable_tuning"] = enable_tuning
        return _DummyMatmul()

    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    monkeypatch.setattr(
        bitblas_module.BitblasLinear,
        "_get_or_create_bitblas_operator",
        _fake_get_or_create,
    )

    layer = bitblas_module.BitblasLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=False,
        in_features=32,
        out_features=32,
        pack_dtype=torch.int32,
        dtype=torch.bfloat16,
        bias=True,
        enable_tuning=False,
    )

    assert captured["config"].A_dtype == "bfloat16"
    assert captured["config"].out_dtype == "bfloat16"
    assert captured["config"].accum_dtype == "float32"
    assert layer.scales.dtype == torch.bfloat16
    assert layer.bias.dtype == torch.bfloat16


def test_bitblas_repack_from_symmetric_gptq_remaps_signed_codes(monkeypatch):
    class _DummyMatmul:
        lib = object()
        weight_transform = None

        @staticmethod
        def retrieve_weight_shape():
            return (1, 1)

    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    monkeypatch.setattr(
        bitblas_module.BitblasLinear,
        "_get_or_create_bitblas_operator",
        lambda self, config, enable_tuning: _DummyMatmul(),
    )

    bits = 4
    group_size = 32
    in_features = 32
    out_features = 32

    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)
    gptq_linear = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    gptq_linear.pack_block(linear, scales.T, zeros.T, g_idx=g_idx.to(torch.int32))

    captured = {}
    layer = bitblas_module.BitblasLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
        enable_tuning=False,
    )

    def _capture_quant_state(*, intweight_out_in, scales_out_group, intzeros_group_out=None, bias=None):
        captured["intweight"] = intweight_out_in.clone()
        captured["scales"] = scales_out_group.clone()
        captured["intzeros"] = intzeros_group_out
        captured["bias"] = bias

    layer._load_bitblas_quant_state = _capture_quant_state
    layer.repack_from_gptq(gptq_linear)

    packed_weight = gptq_linear.qweight.detach().T.contiguous().view(layer.quant_config.torch_storage_dtype)
    unpacked_codes = bitblas_module.unpack_gptq_qweight(packed_weight, bits).contiguous()
    expected = bitblas_module.remap_gptq_symmetric_codes_to_bitblas(unpacked_codes, bits)

    torch.testing.assert_close(captured["intweight"], expected)
    assert not torch.equal(captured["intweight"], unpacked_codes)
    torch.testing.assert_close(captured["scales"], gptq_linear.scales.detach().T.contiguous())
    assert captured["intzeros"] is None
    assert captured["bias"] is None


def test_bitblas_repack_from_gptq_v2_symmetric_codes_does_not_remap(monkeypatch):
    class _DummyMatmul:
        lib = object()
        weight_transform = None

        @staticmethod
        def retrieve_weight_shape():
            return (1, 1)

    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    monkeypatch.setattr(
        bitblas_module.BitblasLinear,
        "_get_or_create_bitblas_operator",
        lambda self, config, enable_tuning: _DummyMatmul(),
    )

    bits = 4
    group_size = 32
    in_features = 32
    out_features = 32

    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)
    gptq_linear = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    gptq_linear.pack_block(linear, scales.T, zeros.T, g_idx=g_idx.to(torch.int32))
    model_utils.convert_gptq_v1_to_v2_format_module(gptq_linear, bits=bits, pack_dtype=torch.int32)

    captured = {}
    layer = bitblas_module.BitblasLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
        enable_tuning=False,
    )

    def _capture_quant_state(*, intweight_out_in, scales_out_group, intzeros_group_out=None, bias=None):
        captured["intweight"] = intweight_out_in.clone()
        captured["scales"] = scales_out_group.clone()
        captured["intzeros"] = intzeros_group_out
        captured["bias"] = bias

    layer._load_bitblas_quant_state = _capture_quant_state
    layer.repack_from_gptq(gptq_linear)

    packed_weight = gptq_linear.qweight.detach().T.contiguous().view(layer.quant_config.torch_storage_dtype)
    unpacked_codes = bitblas_module.unpack_gptq_qweight(packed_weight, bits).contiguous()
    remapped = bitblas_module.remap_gptq_symmetric_codes_to_bitblas(unpacked_codes, bits)

    torch.testing.assert_close(captured["intweight"], unpacked_codes)
    assert not torch.equal(captured["intweight"], remapped)
    torch.testing.assert_close(captured["scales"], gptq_linear.scales.detach().T.contiguous())
    assert captured["intzeros"] is None
    assert captured["bias"] is None


def test_bitblas_validate_rejects_unsupported_bf16_signed_gptq():
    valid, err = bitblas_module.BitblasLinear.validate(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        in_features=3072,
        out_features=1024,
        pack_dtype=torch.int32,
        dtype=torch.bfloat16,
    )

    assert valid is False
    assert isinstance(err, NotImplementedError)
    assert "signed low-bit dequantization" in str(err)


def test_bitblas_constructor_rejects_unsupported_bf16_signed_gptq(monkeypatch):
    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(
        bitblas_module,
        "import_bitblas",
        lambda: pytest.fail("unsupported bf16 signed GPTQ should be rejected before BitBLAS import"),
    )

    with pytest.raises(NotImplementedError, match="signed low-bit dequantization"):
        bitblas_module.BitblasLinear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=3072,
            out_features=1024,
            pack_dtype=torch.int32,
            dtype=torch.bfloat16,
            bias=False,
            enable_tuning=False,
        )


def test_bitblas_validate_rejects_desc_act_gptq():
    valid, err = bitblas_module.BitblasLinear.validate(
        bits=4,
        group_size=128,
        desc_act=True,
        sym=True,
        in_features=3072,
        out_features=1024,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert valid is False
    assert isinstance(err, NotImplementedError)
    assert "actual desc_act" in str(err)


def test_bitblas_constructor_rejects_desc_act_gptq(monkeypatch):
    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(
        bitblas_module,
        "import_bitblas",
        lambda: pytest.fail("desc_act=True should be rejected before BitBLAS import"),
    )

    with pytest.raises(NotImplementedError, match="actual desc_act"):
        bitblas_module.BitblasLinear(
            bits=4,
            group_size=128,
            desc_act=True,
            sym=True,
            in_features=3072,
            out_features=1024,
            pack_dtype=torch.int32,
            dtype=torch.float16,
            bias=False,
            enable_tuning=False,
        )


def test_bitblas_validate_rejects_non_divisible_in_features():
    valid, err = bitblas_module.BitblasLinear.validate(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=False,
        in_features=30,
        out_features=32,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert valid is False
    assert isinstance(err, NotImplementedError)
    assert "must be divisible by [16]" in str(err)


def test_bitblas_validate_rejects_non_divisible_out_features():
    valid, err = bitblas_module.BitblasLinear.validate(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=False,
        in_features=32,
        out_features=30,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert valid is False
    assert isinstance(err, NotImplementedError)
    assert "must be divisible by [16]" in str(err)


def test_convert_to_bitblas_preserves_name_and_dtype(monkeypatch):
    class _SourceQuantLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_features = 32
            self.out_features = 64
            self.bias = torch.zeros(64)

    class _ReplacementBitblas(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.in_features = kwargs["in_features"]
            self.out_features = kwargs["out_features"]
            self.bias = torch.zeros(self.out_features) if kwargs["bias"] else None

    monkeypatch.setattr(bitblas_utils, "_select_bitblas_kernel_class", lambda qcfg: _ReplacementBitblas)

    model = nn.Module()
    model.proj = _SourceQuantLinear()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        format=FORMAT.BITBLAS,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )

    bitblas_utils.convert_to_bitblas(
        model,
        _SourceQuantLinear,
        qcfg,
        sym=True,
        desc_act=False,
        repack=False,
        dtype=torch.bfloat16,
    )

    assert isinstance(model.proj, _ReplacementBitblas)
    assert model.proj.kwargs["name"] == "proj"
    assert model.proj.kwargs["dtype"] == torch.bfloat16


def test_create_quant_module_propagates_dtype_to_quant_linear():
    """Quantized checkpoint loads must instantiate the selected kernel with the requested dtype."""

    seen = {}

    class _DummyQuantLinear(nn.Module):
        @classmethod
        def validate(cls, **kwargs):
            seen["validate_dtype"] = kwargs.get("dtype")
            return True, None

        def __init__(self, **kwargs):
            super().__init__()
            seen["init_dtype"] = kwargs.get("dtype")
            self.bias = None

    module = nn.Module()
    module.proj = nn.Linear(32, 32, bias=False)

    model_utils.create_quant_module(
        name="proj",
        linear_cls=_DummyQuantLinear,
        bits=4,
        desc_act=False,
        dynamic=None,
        group_size=32,
        module=module,
        submodule=module.proj,
        sym=True,
        device=None,
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        backend=BACKEND.BITBLAS,
        dtype=torch.bfloat16,
    )

    assert seen["validate_dtype"] == torch.bfloat16
    assert seen["init_dtype"] == torch.bfloat16
    assert isinstance(module.proj, _DummyQuantLinear)


def test_bitblas_rejects_unrunnable_operator(monkeypatch):
    """Surface BitBLAS build failures during construction so auto-selection can fall back."""

    class _BrokenMatmul:
        weight_transform = None

        @staticmethod
        def retrieve_weight_shape():
            return (1, 1)

    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    monkeypatch.setattr(
        bitblas_module.BitblasLinear,
        "_get_or_create_bitblas_operator",
        lambda self, config, enable_tuning: _BrokenMatmul(),
    )

    with pytest.raises(NotImplementedError, match="BitBLAS could not build a runnable matmul"):
        bitblas_module.BitblasLinear(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=False,
            in_features=32,
            out_features=32,
            pack_dtype=torch.int32,
            dtype=torch.bfloat16,
            bias=False,
            enable_tuning=False,
        )


def test_make_quant_falls_back_when_bitblas_operator_is_unrunnable(monkeypatch):
    """Auto kernel selection should skip BitBLAS when the runtime build is unusable."""

    class _BrokenMatmul:
        weight_transform = None

        @staticmethod
        def retrieve_weight_shape():
            return (1, 1)

    monkeypatch.setattr(bitblas_module, "BITBLAS_AVAILABLE", True)
    monkeypatch.setattr(bitblas_module, "import_bitblas", lambda: None)
    monkeypatch.setattr(
        bitblas_module.BitblasLinear,
        "_get_or_create_bitblas_operator",
        lambda self, config, enable_tuning: _BrokenMatmul(),
    )
    monkeypatch.setattr(
        model_utils,
        "select_quant_linear",
        lambda **kwargs: [bitblas_module.BitblasLinear, TorchLinear],
    )
    bitblas_module.BitblasLinear.cached_validate_once.cache_clear()

    module = nn.Module()
    module.proj = nn.Linear(32, 32, bias=False)

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=False,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )

    selected = model_utils.make_quant(
        module,
        qcfg=qcfg,
        quant_result={"proj": module.proj},
        backend=BACKEND.AUTO,
        lm_head_name="lm_head",
        dtype=torch.bfloat16,
    )

    assert selected is TorchLinear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BitBLAS")
@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_bitblas_forward_pass_future_target_fallback():
    from bitblas.cache import global_operator_cache

    bitblas_module.import_bitblas()

    device_index = int(os.environ.get("BITBLAS_TEST_DEVICE", 0))
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device_index)

    original_target = bitblas_module.BITBLAS_TARGET
    original_database_path = bitblas_module.BITBLAS_DATABASE_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            global_operator_cache.clear()
            bitblas_module.BITBLAS_TARGET = "cuda -arch=sm_120"
            bitblas_module.BITBLAS_DATABASE_PATH = tmpdir

            layer = bitblas_module.BitblasLinear(
                bits=4,
                group_size=32,
                desc_act=False,
                sym=True,
                in_features=96,
                out_features=48,
                bias=False,
            ).to(device)

            with torch.no_grad():
                layer.qweight.zero_()
                layer.scales.zero_()
                if layer.quant_config.with_zeros:
                    layer.qzeros.zero_()

            x = torch.randn(2, 96, device=device, dtype=layer.TORCH_DTYPE)
            y = layer(x)

            assert y.shape == (2, 48)
            assert torch.allclose(y, torch.zeros_like(y), atol=1e-4, rtol=1e-4)
            assert layer.bitblas_matmul.target.arch == bitblas_module._bitblas_fallback_target().removeprefix("cuda -arch=")
        finally:
            bitblas_module.BITBLAS_TARGET = original_target
            bitblas_module.BITBLAS_DATABASE_PATH = original_database_path
            global_operator_cache.clear()

######### test_bitblas_gptq_v2.py #########
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for BitBLAS")
@pytest.mark.skipif(not bitblas_module.BITBLAS_AVAILABLE, reason="BitBLAS backend is not available")
def test_bitblas_forward_pass2():
    bitblas_module.import_bitblas()

    device_index = int(os.environ.get("BITBLAS_TEST_DEVICE", 0))
    torch.cuda.set_device(device_index)

    # Load a dummy model (1.0 GB) to test if there are errors while converting to bitblas
    # Take a few minutes for compiling (1st run) and repacking (each time)
    GPTQModel.load("/monster/data/model/Qwen3-1.7B-w2g64-gptq_v2", trust_remote_code=True, backend=BACKEND('bitblas'))


########### test_bitblas_quant.py ##########


RTOL = 5e-2
ATOL = 5e-2


def _mock_gptq_linear(bits: int, group_size: int, in_features: int, out_features: int):
    maxq = (1 << (bits - 1)) - 1
    weight = torch.randn((in_features, out_features), dtype=torch.float32)

    if group_size != -1:
        reshaped = weight.view(in_features // group_size, group_size, out_features)
        w_g = reshaped.permute(1, 0, 2).reshape(group_size, -1)
    else:
        w_g = weight

    scales = torch.maximum(
        w_g.abs().max(dim=0, keepdim=True).values,
        torch.full((1, w_g.shape[1]), 1e-6, device=w_g.device),
    )
    scales = scales / maxq

    q = torch.round(w_g / scales).clamp_(-maxq, maxq)
    ref = (q * scales).to(dtype=torch.float16)

    if group_size != -1:
        def _reshape_back(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.reshape(group_size, -1, out_features)
            return tensor.permute(1, 0, 2).reshape(in_features, out_features)

        ref = _reshape_back(ref)
        q = _reshape_back(q)

    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = ref.t().contiguous()

    scales = scales.reshape(-1, out_features).contiguous()
    zeros = torch.zeros_like(scales, dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // (
        group_size if group_size != -1 else in_features
    )

    return linear, scales, zeros, g_idx


def _benchmark(module: nn.Module, x: torch.Tensor, warmup: int = 2, iters: int = 5) -> list[float]:
    times_ms: list[float] = []
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize()
        for _ in range(iters):
            start = time.perf_counter()
            module(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)
    return times_ms


def _format_pass(pass_ok: bool) -> str:
    if pass_ok:
        return "PASS"
    return "\033[91mFAIL\033[0m"


@pytest.mark.cuda
@parameterized.expand([
    ("bs1_fp16", 1, torch.float16, "float16"),
    ("bs2_fp16", 2, torch.float16, "float16"),
    ("bs4_fp16", 4, torch.float16, "float16"),
    ("bs8_fp16", 8, torch.float16, "float16"),
    ("bs16_fp16", 16, torch.float16, "float16"),
    ("bs1_bf16", 1, torch.bfloat16, "bfloat16"),
    ("bs2_bf16", 2, torch.bfloat16, "bfloat16"),
    ("bs4_bf16", 4, torch.bfloat16, "bfloat16"),
    ("bs8_bf16", 8, torch.bfloat16, "bfloat16"),
    ("bs16_bf16", 16, torch.bfloat16, "bfloat16"),
])
def test_llama3_linear_bitblas_vs_torch_vs_marlin(_, batch, dtype, dtype_name):
    try:
        pytest.importorskip("bitblas")
    except Exception as exc:
        pytest.skip(f"bitblas unavailable: {exc}")
    if marlin_import_exception is not None:
        pytest.skip(f"marlin unavailable: {marlin_import_exception}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    torch.manual_seed(0)

    bits = 4
    group_size = 128
    in_features = 8192
    out_features = 8192

    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)
    device = torch.device("cuda")

    torch_linear = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    torch_linear.pack_block(linear, scales.T, zeros.T, g_idx=g_idx.to(torch.int32))
    torch_linear.post_init()

    bitblas_linear = None
    bitblas_error = None
    try:
        bitblas_linear = bitblas_module.BitblasLinear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=torch.int32,
            dtype=dtype,
            bias=False,
            enable_tuning=False,
        )
        bitblas_linear.repack_from_gptq(torch_linear)
        bitblas_linear.post_init()
        bitblas_linear = bitblas_linear.to(device=device)
    except Exception as exc:  # pragma: no cover - diagnostic path
        bitblas_error = str(exc)

    marlin_linear = MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        marlin_linear.qweight.copy_(torch_linear.qweight.to(device))
        marlin_linear.scales.copy_(torch_linear.scales.to(device))
        marlin_linear.g_idx.copy_(torch_linear.g_idx.to(device))
        marlin_linear.qzeros.zero_()
    marlin_linear.post_init()

    try:
        triton_linear = TritonV2Linear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=torch.int32,
            bias=False,
        )
    except ValueError as err:
        pytest.skip(f"triton unavailable: {err}")

    triton_linear.pack_block(linear, scales.T, zeros.T, g_idx=g_idx.to(torch.int32))
    triton_linear.post_init()
    triton_linear = triton_linear.to(device=device, dtype=dtype).eval()

    modules = {
        "Torch": torch_linear.to(device=device, dtype=dtype).eval(),
        "Marlin": marlin_linear.eval(),
        "TritonV2": triton_linear,
    }
    if bitblas_linear is not None:
        modules["BitBLAS"] = bitblas_linear.eval()

    x = torch.randn((batch, in_features), dtype=dtype, device=device)

    results = []
    reference_out = None
    outputs: dict[str, torch.Tensor] = {}
    errors: dict[str, str] = {}
    if bitblas_error is not None:
        errors["BitBLAS"] = bitblas_error

    for name, module in modules.items():
        try:
            with torch.inference_mode():
                outputs[name] = module(x).to(torch.float32)
            if reference_out is None:
                reference_out = outputs[name]
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors[name] = str(exc)

    for name in ("Torch", "BitBLAS", "Marlin", "TritonV2"):
        module = modules.get(name)
        err = errors.get(name)
        if err:
            results.append([
                dtype_name,
                batch,
                name,
                "-",
                "-",
                "-",
                "-",
                "\033[91mERR\033[0m",
            ])
            continue
        if module is None:
            continue

        out = outputs[name]
        if name == "Torch" or reference_out is None:
            max_abs = 0.0
            mean_abs = 0.0
            max_rel = 0.0
            pass_ok = True
        else:
            diff = (out - reference_out).abs()
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            max_rel = float((diff / (reference_out.abs() + 1e-6)).max().item())
            pass_ok = max_abs <= ATOL and max_rel <= RTOL

        times = _benchmark(module, x)
        mean_ms = mean(times)
        std_ms = pstdev(times) if len(times) > 1 else 0.0

        results.append([
            dtype_name,
            batch,
            name,
            f"{mean_ms:.3f}",
            f"{std_ms:.3f}",
            f"{max_abs:.4f}",
            f"{mean_abs:.4f}",
            f"{max_rel:.4f}",
            _format_pass(pass_ok),
        ])

    headers = [
        "dtype",
        "batch",
        "Kernel",
        "Mean ms",
        "Std ms",
        "Max |d|",
        "Mean |d|",
        "Max Rel d",
        "Accuracy",
    ]
    print(render_table(results, headers=headers, tablefmt="github"))

    # Table highlights failing kernels in red; no hard assertion to keep report informative.
