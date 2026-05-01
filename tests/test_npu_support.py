import copy
import os
import sys
import warnings

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.looper.qqq_processor import QQQProcessor
from gptqmodel.looper.weight_only_processor import WeightOnlyProcessor
from gptqmodel.models._const import DEVICE, normalize_device
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.komodo import AwqKomodoLinear, KomodoLinear, _native_int4_enabled
import gptqmodel.nn_modules.qlinear.komodo_cann as komodo_cann_module
from gptqmodel.nn_modules.qlinear.komodo_cann import (
    AwqKomodoCannLinear,
    KomodoCannLinear,
    _komodo_cann_tiling_plan,
)
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, _right_shift_unpack
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.awq.utils.packing_utils import unpack_awq
from gptqmodel.quantization.config import AWQConfig, GGUFConfig, ParoConfig, QQQConfig, QuantizeConfig
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import auto_select_device, get_kernel_for_backend, select_quant_linear
from gptqmodel.utils.torch import HAS_NPU, last_npu_device_by_pci_bus_order


def _default_npu_test_device() -> str:
    selected = last_npu_device_by_pci_bus_order()
    return str(selected) if selected is not None else "npu:0"


NPU_TEST_DEVICE = os.environ.get("GPTQMODEL_TEST_NPU_DEVICE", _default_npu_test_device())
NPU_CPU_FALLBACK_MARKERS = (
    "not currently supported on the NPU backend",
    "fall back to run on the CPU",
)


def test_komodo_native_int4_default_enabled(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_KOMODO_NATIVE_INT4", raising=False)
    assert _native_int4_enabled()
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "0")
    assert not _native_int4_enabled()


def test_komodo_cann_tiling_plan_uses_split_k_for_decode_large_k(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_ACTIVE_CORES", "24")
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_SPLIT_K", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_PREFETCH", raising=False)
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED_OP", "missing_namespace.missing_op")
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_V3", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_STAGED_DEQUANT", raising=False)
    plan = _komodo_cann_tiling_plan(
        rows=1,
        in_features=8192,
        out_features=1024,
        group_size=32,
        device=torch.device("cpu"),
    )

    assert plan.split_k > 1
    assert plan.base_k == 64
    assert plan.split_k_shard_k == plan.in_features // plan.split_k
    assert plan.int4_values_per_int32 == 8
    assert plan.packed_int4_tile_bytes == plan.base_k * plan.base_n // 2
    assert plan.dequant_fp16_tile_bytes == plan.base_k * plan.base_n * 2
    assert plan.vector_dequant_tasks == (plan.out_features // plan.base_n) * plan.split_k * plan.k_tiles_per_split
    assert plan.active_cores == min(plan.cube_cores, plan.split_k * 4)
    assert plan.staged_dequant is False
    assert plan.staging_workspace_bytes == 0
    assert plan.strategy == "planned_split_k_aiv_dequant_aic_matmul"
    assert not plan.prefetch_enabled
    assert plan.fused_enabled
    assert plan.fused_supported
    assert not plan.fused_available
    assert plan.fused_reason == "op_not_registered"
    assert plan.inner_precise == 0
    assert plan.zero_offsets is False


def test_komodo_cann_tiling_plan_records_zero_offsets_flag(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_ACTIVE_CORES", "24")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED_OP", "missing_namespace.missing_op")
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_V3", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_STAGED_DEQUANT", raising=False)

    plan = _komodo_cann_tiling_plan(
        rows=8,
        in_features=256,
        out_features=256,
        group_size=32,
        device=torch.device("cpu"),
        zero_offsets=True,
    )

    assert plan.zero_offsets is True


def test_komodo_cann_staged_dequant_plan_is_opt_in_and_bounded(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_ACTIVE_CORES", "24")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED_OP", "missing_namespace.missing_op")
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_V3", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_STAGED_DEQUANT", raising=False)

    default_plan = _komodo_cann_tiling_plan(
        rows=8,
        in_features=8192,
        out_features=1024,
        group_size=32,
        device=torch.device("cpu"),
    )

    assert default_plan.staged_dequant is False
    assert default_plan.staging_slots == 0
    assert default_plan.staging_blocks == 0
    assert default_plan.staging_tile_bytes == 0
    assert default_plan.staging_workspace_bytes == 0
    assert default_plan.staging_workspace_offset == 0
    assert default_plan.cube_consumer is False
    assert default_plan.cube_workspace_bytes == 0
    assert default_plan.custom_workspace_bytes == 0

    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_STAGED_DEQUANT", "1")
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_CUBE_CONSUMER", raising=False)
    staged_plan = _komodo_cann_tiling_plan(
        rows=8,
        in_features=8192,
        out_features=1024,
        group_size=32,
        device=torch.device("cpu"),
    )

    assert staged_plan.staged_dequant is True
    assert staged_plan.base_k == 64
    assert staged_plan.staging_slots == 2
    assert staged_plan.staging_blocks == min(
        staged_plan.vector_cores,
        max(1, staged_plan.out_features // 8),
        8,
        (staged_plan.out_features // staged_plan.base_n) * staged_plan.split_k,
    )
    assert staged_plan.staging_tile_bytes == staged_plan.base_k * staged_plan.base_n * 2
    assert staged_plan.staging_workspace_bytes == (
        staged_plan.staging_slots * staged_plan.staging_blocks * staged_plan.staging_tile_bytes
    )
    assert staged_plan.staging_workspace_offset == 0
    assert staged_plan.cube_consumer is False
    assert staged_plan.cube_workspace_bytes == 0
    assert staged_plan.custom_workspace_bytes == staged_plan.staging_workspace_bytes
    assert staged_plan.staging_workspace_bytes < staged_plan.in_features * staged_plan.out_features * 2
    assert staged_plan.strategy == "planned_staged_dequant_aic_matmul"

    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_CUBE_CONSUMER", "1")
    cube_plan = _komodo_cann_tiling_plan(
        rows=8,
        in_features=8192,
        out_features=1024,
        group_size=32,
        device=torch.device("cpu"),
    )

    assert cube_plan.staged_dequant is True
    assert cube_plan.cube_consumer is True
    assert cube_plan.base_k == 128
    assert cube_plan.cube_workspace_bytes == 16 * 1024 * 1024
    assert cube_plan.staging_workspace_offset == 0
    assert cube_plan.custom_workspace_bytes == cube_plan.staging_workspace_bytes
    assert cube_plan.custom_workspace_bytes < cube_plan.in_features * cube_plan.out_features * 2

    n256_cube_plan = _komodo_cann_tiling_plan(
        rows=8,
        in_features=1024,
        out_features=256,
        group_size=32,
        device=torch.device("cpu"),
    )
    assert n256_cube_plan.base_n == 128

    n512_k512_cube_plan = _komodo_cann_tiling_plan(
        rows=8,
        in_features=512,
        out_features=512,
        group_size=32,
        device=torch.device("cpu"),
    )
    assert n512_k512_cube_plan.base_n == 128

    n512_k1024_cube_plan = _komodo_cann_tiling_plan(
        rows=8,
        in_features=1024,
        out_features=512,
        group_size=32,
        device=torch.device("cpu"),
    )
    assert n512_k1024_cube_plan.base_n == 256

    large_row_cube_plan = _komodo_cann_tiling_plan(
        rows=48,
        in_features=8192,
        out_features=1024,
        group_size=32,
        device=torch.device("cpu"),
    )

    assert large_row_cube_plan.base_m == 48
    assert large_row_cube_plan.base_k == 128

    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_BASE_K", "64")
    override_plan = _komodo_cann_tiling_plan(
        rows=8,
        in_features=8192,
        out_features=1024,
        group_size=32,
        device=torch.device("cpu"),
    )

    assert override_plan.base_k == 64


def test_komodo_cann_tiling_plan_inner_precise_auto_shape_policy(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_ACTIVE_CORES", "24")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED_OP", "missing_namespace.missing_op")
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_INNER_PRECISE", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_STAGED_DEQUANT", raising=False)

    liked = _komodo_cann_tiling_plan(
        rows=1,
        in_features=5120,
        out_features=6144,
        group_size=32,
        device=torch.device("cpu"),
    )
    narrow_decode = _komodo_cann_tiling_plan(
        rows=1,
        in_features=5120,
        out_features=1024,
        group_size=32,
        device=torch.device("cpu"),
    )
    balanced = _komodo_cann_tiling_plan(
        rows=1,
        in_features=4096,
        out_features=4096,
        group_size=128,
        device=torch.device("cpu"),
    )

    assert liked.inner_precise == 1
    assert narrow_decode.inner_precise == 0
    assert balanced.inner_precise == 0


def test_komodo_cann_default_fused_op_names_include_msopgen_snake_case(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_FUSED_OP", raising=False)
    names = komodo_cann_module._komodo_cann_fused_op_names()

    assert "gptqmodel_komodo_cann.komodo_cann_w4_a16_matmul" in names
    assert "npu.komodo_cann_w4_a16_matmul" in names


def test_komodo_cann_tiling_plan_requires_registered_fused_op(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_ACTIVE_CORES", "24")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED_OP", "missing_namespace.missing_op")
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_V3", raising=False)

    with pytest.raises(RuntimeError, match="fused W4A16 op"):
        _komodo_cann_tiling_plan(
            rows=1,
            in_features=8192,
            out_features=1024,
            group_size=32,
            device=torch.device("cpu"),
        )


def test_komodo_cann_tiling_plan_reports_v3_autoload_failure(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_ACTIVE_CORES", "24")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED", "1")
    monkeypatch.delenv("GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE", raising=False)
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_FUSED_OP", "missing_namespace.missing_op")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_V3", "1")
    monkeypatch.setattr(komodo_cann_module, "_try_load_komodo_cann_v3", lambda: False)
    monkeypatch.setattr(komodo_cann_module, "_FUSED_OP_CACHE", komodo_cann_module._FUSED_OP_UNSET)
    monkeypatch.setattr(komodo_cann_module, "_FUSED_OP_CACHE_KEY", None)

    plan = _komodo_cann_tiling_plan(
        rows=1,
        in_features=8192,
        out_features=1024,
        group_size=32,
        device=torch.device("cpu"),
    )

    assert plan.fused_enabled
    assert plan.fused_supported
    assert not plan.fused_available
    assert plan.fused_reason == "v3_extension_unavailable"


def _test_npu_device() -> torch.device:
    device = torch.device(NPU_TEST_DEVICE)
    if HAS_NPU:
        torch.npu.set_device(device)
    return device


def _assert_empty_source_buffers(module: nn.Module, names: tuple[str, ...]) -> None:
    for name in names:
        tensor = getattr(module, name)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.numel() == 0


def _assert_no_npu_cpu_fallback(caught: list[warnings.WarningMessage]) -> None:
    fallback_warnings = [
        str(warning.message)
        for warning in caught
        if any(marker in str(warning.message) for marker in NPU_CPU_FALLBACK_MARKERS)
    ]
    assert fallback_warnings == []


def _assert_npu_forward_matches_cpu(
    module: nn.Module,
    x_cpu: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    npu_module = copy.deepcopy(module).to(_test_npu_device()).eval()
    module = module.eval()

    with torch.inference_mode():
        y_cpu = module(x_cpu)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.inference_mode():
            y_npu = npu_module(x_cpu.to(_test_npu_device()))
            y_npu_cpu = y_npu.to("cpu", dtype=torch.float32)

    assert y_npu.device.type == "npu"
    _assert_no_npu_cpu_fallback(caught)
    torch.testing.assert_close(y_npu_cpu, y_cpu.to(torch.float32), atol=atol, rtol=rtol)


def _copy_matching_buffers(dst: nn.Module, src: nn.Module) -> None:
    src_buffers = dict(src.named_buffers())
    with torch.no_grad():
        for name, dst_tensor in dst.named_buffers():
            src_tensor = src_buffers.get(name)
            if src_tensor is not None and src_tensor.shape == dst_tensor.shape:
                dst_tensor.copy_(src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype))


def _set_supported_act_order_g_idx(module: TorchLinear) -> None:
    group_size = module.requested_group_size
    groups = module.in_features // group_size
    natural = torch.arange(module.in_features, dtype=torch.int32) // group_size
    act_order = torch.arange(module.in_features).reshape(groups, group_size).t().reshape(-1)
    with torch.no_grad():
        module.g_idx.copy_(natural[act_order].to(dtype=module.g_idx.dtype, device=module.g_idx.device))
    module.desc_act = True
    module._stream_reset_cache()


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros((unpacked.shape[0], unpacked.shape[1] // pack_factor), dtype=torch.int32)
    for col in range(unpacked.shape[1] // pack_factor):
        for lane, order in enumerate(order_map):
            packed[:, col] |= unpacked[:, col * pack_factor + order].to(torch.int32) << (lane * bits)
    return packed


def _make_awq_like_module(cls, dtype: torch.dtype, *, seed: int = 300, **kwargs):
    in_features = kwargs.pop("in_features", 64)
    out_features = kwargs.pop("out_features", 64)
    group_size = kwargs.pop("group_size", 16)
    bias = kwargs.pop("bias", True)
    bits = 4

    torch.manual_seed(seed)
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = ((torch.rand(groups, out_features, dtype=torch.float32) * 2.0) + 0.25).to(dtype)

    module = cls(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        dtype=dtype,
        register_buffers=True,
        **kwargs,
    )
    module.qweight.copy_(_pack_awq_tensor(int_weight, bits))
    module.qzeros.copy_(_pack_awq_tensor(zero_points, bits))
    module.scales.copy_(scales.to(module.scales.dtype))
    if bias:
        module.bias.copy_(torch.randn(out_features, dtype=dtype).to(module.bias.dtype))
    return module


def _make_gptq_module(
    bits: int,
    dtype: torch.dtype,
    *,
    in_features: int = 64,
    out_features: int = 64,
    group_size: int = 16,
) -> TorchLinear:

    torch.manual_seed(100 + bits)
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight.data.normal_(0, 0.12)
    linear.bias.data.normal_(0, 0.03)

    maxq = (1 << bits) - 1
    groups = in_features // group_size
    scales = torch.rand(out_features, groups, dtype=torch.float32) * 0.04 + 0.01
    zeros = torch.randint(0, maxq + 1, (out_features, groups), dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
    )
    module.pack_block(linear, scales, zeros, g_idx)
    module.optimized = True
    module.post_init()
    return module.to(dtype=dtype).eval()


def _make_awq_module(dtype: torch.dtype) -> AwqTorchLinear:
    module = _make_awq_like_module(AwqTorchLinear, dtype)
    module.post_init()
    return module.eval()


def _make_paro_module(dtype: torch.dtype) -> ParoLinear:
    module = _make_awq_like_module(ParoLinear, dtype, seed=350, krot=1)
    theta = torch.linspace(-0.15, 0.15, module.in_features // 2, dtype=module.theta.dtype).view_as(module.theta)
    channel_scales = torch.linspace(0.95, 1.05, module.in_features, dtype=module.channel_scales.dtype).view_as(
        module.channel_scales
    )
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    return module.eval()


def _make_gguf_module(bits: str, dtype: torch.dtype) -> GGUFTorchLinear:
    in_features = 256
    out_features = 32

    torch.manual_seed(500 + sum(ord(ch) for ch in bits))
    linear = nn.Linear(in_features, out_features, bias=True)
    linear.weight.data.normal_(0, 0.12)
    linear.bias.data.normal_(0, 0.03)

    module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
    )
    module.pack_original(linear, scales=None, zeros=None, g_idx=None)
    module.post_init()
    return module.to(dtype=dtype).eval()


def _make_qqq_module(dtype: torch.dtype, group_size: int) -> QQQTorchLinear:
    in_features = 256
    out_features = 128

    torch.manual_seed(600 + group_size)
    linear = nn.Linear(in_features, out_features, bias=True).to(dtype=torch.float16)
    linear.weight.data.normal_(0, 0.08)
    linear.bias.data.normal_(0, 0.02)

    module = QQQTorchLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
    )

    if group_size == -1:
        scales = torch.rand(out_features, 1, dtype=torch.float16) * 0.04 + 0.01
        s_extra = None
    else:
        groups = in_features // group_size
        scales = torch.rand(out_features, groups, dtype=torch.float16) * 0.04 + 0.01
        s_extra = torch.rand(out_features, dtype=torch.float32) * 0.5 + 0.75

    module.pack(linear, scales, s_extra)
    module.post_init()
    return module.eval()


def _make_exllamav3_torch_module(*, device: torch.device | str = "cpu") -> ExllamaV3TorchLinear:
    in_features = 128
    out_features = 128
    bits = 2
    target = torch.device(device)

    generator = torch.Generator(device="cpu").manual_seed(700)
    tensors = {
        "trellis": torch.randint(
            -32768,
            32767,
            (in_features // 16, out_features // 16, bits * 16),
            dtype=torch.int16,
            generator=generator,
        ),
        "suh": torch.randint(0, 2, (in_features,), dtype=torch.int8, generator=generator)
        .to(torch.float16)
        .mul_(2)
        .sub_(1),
        "svh": torch.randint(0, 2, (out_features,), dtype=torch.int8, generator=generator)
        .to(torch.float16)
        .mul_(2)
        .sub_(1),
        "bias": torch.randn(out_features, dtype=torch.float16, generator=generator) * 0.01,
    }
    tensors = {
        key: value.to(target) if target.type != "cpu" else value
        for key, value in tensors.items()
    }
    return ExllamaV3TorchLinear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name="npu_exl3_torch",
        tensors=tensors,
    ).eval()


class _NpuProcessorModelStub:
    def __init__(self, qlinear_kernel=None):
        self.qlinear_kernel = qlinear_kernel
        self.rotary_embedding = None
        self.lm_head = "lm_head"
        self.model = nn.Sequential()


def _processor_common_kwargs(qcfg):
    return {
        "tokenizer": None,
        "qcfg": qcfg,
        "calibration": None,
        "prepare_dataset_func": None,
        "calibration_concat_size": None,
        "calibration_sort": None,
        "batch_size": 1,
    }


def _npu_select_quant_linear(qcfg, *, method: METHOD, fmt: FORMAT):
    return select_quant_linear(
        bits=qcfg.runtime_bits,
        group_size=qcfg.group_size,
        desc_act=qcfg.desc_act,
        sym=qcfg.sym,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=method,
        pack_dtype=qcfg.pack_dtype,
    )


def test_last_npu_device_by_pci_bus_order_uses_visible_logical_order(monkeypatch):
    try:
        torch.device("npu:0")
    except (RuntimeError, ValueError):
        pytest.skip("This PyTorch build does not register the npu device type")

    class _FakeNpu:
        @staticmethod
        def device_count():
            return 3

    torch_utils = sys.modules[last_npu_device_by_pci_bus_order.__module__]
    monkeypatch.setattr(torch_utils, "HAS_NPU", True)
    monkeypatch.setattr(torch_utils.torch, "npu", _FakeNpu())

    assert str(last_npu_device_by_pci_bus_order()) == "npu:2"


def test_npu_device_normalization():
    assert normalize_device("npu") is DEVICE.NPU
    assert normalize_device("npu:3") is DEVICE.NPU
    assert DEVICE.NPU.type == "npu"
    try:
        expected = torch.device("npu:0")
    except (RuntimeError, ValueError):
        pytest.skip("This PyTorch build does not register the npu device type")
    assert DEVICE.NPU.to_torch_device() == expected


def test_auto_select_device_uses_npu_when_available(monkeypatch):
    monkeypatch.setattr(importer, "HAS_CUDA", False)
    monkeypatch.setattr(importer, "HAS_XPU", False)
    monkeypatch.setattr(importer, "HAS_NPU", True)
    monkeypatch.setattr(importer, "HAS_MPS", False)

    assert auto_select_device(None, BACKEND.AUTO) is DEVICE.NPU


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
def test_npu_auto_selects_komodo_gptq(fmt):
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=fmt,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is KomodoLinear


def test_npu_auto_selects_komodo_awq_for_gemm():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is AwqKomodoLinear


def test_npu_explicit_komodo_selects_gptq_and_awq():
    gptq_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.KOMODO,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )
    awq_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.KOMODO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert gptq_cls is KomodoLinear
    assert awq_cls is AwqKomodoLinear


def test_npu_explicit_komodo_cann_selects_gptq_and_awq():
    gptq_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.KOMODO_CANN,
        format=FORMAT.GPTQ,
        quant_method=METHOD.GPTQ,
        pack_dtype=torch.int32,
    )
    awq_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.KOMODO_CANN,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
    )

    assert gptq_cls is KomodoCannLinear
    assert awq_cls is AwqKomodoCannLinear


@pytest.mark.parametrize(
    ("quant_method", "fmt"),
    [(METHOD.GPTQ, FORMAT.GPTQ), (METHOD.AWQ, FORMAT.GEMM)],
)
def test_npu_explicit_komodo_rejects_bfloat16_dtype(quant_method, fmt):
    with pytest.raises(ValueError, match="only supports"):
        select_quant_linear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            device=DEVICE.NPU,
            backend=BACKEND.KOMODO,
            format=fmt,
            quant_method=quant_method,
            pack_dtype=torch.int32,
            dtype=torch.bfloat16,
        )


def test_npu_auto_selects_paroquant_torch_dense_fallback():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.PAROQUANT,
        quant_method=METHOD.PARO,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is ParoLinear


def test_npu_auto_selects_gguf_torch():
    qlinear_cls = select_quant_linear(
        bits="q4_k_m",
        group_size=-1,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.GGUF,
        quant_method=METHOD.GGUF,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is GGUFTorchLinear


def test_npu_auto_selects_qqq_torch():
    qlinear_cls = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.NPU,
        backend=BACKEND.AUTO,
        format=FORMAT.QQQ,
        quant_method=METHOD.QQQ,
        pack_dtype=torch.int32,
    )

    assert qlinear_cls is QQQTorchLinear


def test_qqq_torch_backend_selects_torch_kernel():
    assert get_kernel_for_backend(BACKEND.QQQ_TORCH, METHOD.QQQ, FORMAT.QQQ) is QQQTorchLinear


def test_npu_gptq_processor_has_torch_runtime_kernel():
    qcfg = QuantizeConfig(bits=4, group_size=128, device=DEVICE.NPU, offload_to_disk=False)
    processor = GPTQProcessor(**_processor_common_kwargs(qcfg))

    assert processor.name() == "gptq"
    assert processor.execution_config.require_fwd is True
    assert _npu_select_quant_linear(qcfg, method=METHOD.GPTQ, fmt=FORMAT.GPTQ) is TorchLinear


def test_npu_awq_processor_selects_torch_runtime_kernel():
    qcfg = AWQConfig(bits=4, group_size=128, device=DEVICE.NPU, offload_to_disk=False)
    model_stub = _NpuProcessorModelStub()
    processor = AWQProcessor(
        **_processor_common_kwargs(qcfg),
        gptq_model=model_stub,
        model=model_stub.model,
    )

    assert processor.name() == "awq"
    assert processor.execution_config.enable_activation_capture is True
    assert processor.qlinear_kernel is AwqTorchLinear
    assert _npu_select_quant_linear(qcfg, method=METHOD.AWQ, fmt=FORMAT.GEMM) is AwqTorchLinear


def test_npu_paroquant_processor_has_torch_runtime_kernel():
    qcfg = ParoConfig(
        bits=4,
        group_size=128,
        device=DEVICE.NPU,
        opt_rotation_epochs=1,
        opt_finetune_epochs=1,
        offload_to_disk=False,
    )
    model_stub = _NpuProcessorModelStub()
    processor = ParoQuantProcessor(
        **_processor_common_kwargs(qcfg),
        gptq_model=model_stub,
        model=model_stub.model,
    )

    assert processor.name() == "paroquant"
    assert processor.execution_config.enable_activation_capture is True
    assert processor.qlinear_kernel is ParoLinear
    assert _npu_select_quant_linear(qcfg, method=METHOD.PARO, fmt=FORMAT.PAROQUANT) is ParoLinear


def test_npu_qqq_processor_selects_torch_runtime_kernel():
    qcfg = QQQConfig(bits=4, group_size=128, device=DEVICE.NPU, offload_to_disk=False)
    processor = QQQProcessor(**_processor_common_kwargs(qcfg))
    qlinear_cls, backend = processor._quant_linear_kernel()

    assert processor.name() == "qqq"
    assert qlinear_cls is QQQTorchLinear
    assert backend is BACKEND.QQQ_TORCH
    assert _npu_select_quant_linear(qcfg, method=METHOD.QQQ, fmt=FORMAT.QQQ) is QQQTorchLinear


def test_npu_gguf_weight_only_processor_has_torch_runtime_kernel():
    qcfg = GGUFConfig(bits="q4_0", device=DEVICE.NPU, offload_to_disk=False)
    processor = WeightOnlyProcessor(tokenizer=None, qcfg=qcfg)

    assert processor.name() == "weight_only_gguf"
    assert processor.execution_config.require_fwd is False
    assert _npu_select_quant_linear(qcfg, method=METHOD.GGUF, fmt=FORMAT.GGUF) is GGUFTorchLinear


def test_npu_supported_quant_methods_have_torch_runnable_kernel():
    cases = [
        (METHOD.GPTQ, FORMAT.GPTQ, 4, 128, TorchLinear),
        (METHOD.AWQ, FORMAT.GEMM, 4, 128, AwqTorchLinear),
        (METHOD.PARO, FORMAT.PAROQUANT, 4, 128, ParoLinear),
        (METHOD.GGUF, FORMAT.GGUF, "q4_0", -1, GGUFTorchLinear),
        (METHOD.QQQ, FORMAT.QQQ, 4, 128, QQQTorchLinear),
    ]
    for method, fmt, bits, group_size, expected_cls in cases:
        qlinear_cls = select_quant_linear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=True,
            device=DEVICE.NPU,
            backend=BACKEND.AUTO,
            format=fmt,
            quant_method=method,
            pack_dtype=torch.int32,
        )
        assert qlinear_cls is expected_cls
        assert DEVICE.ALL in qlinear_cls.SUPPORTS_DEVICES or DEVICE.NPU in qlinear_cls.SUPPORTS_DEVICES


def test_npu_exl3_has_torch_runtime_kernel():
    module = _make_exllamav3_torch_module()

    assert isinstance(module, ExllamaV3TorchLinear)
    assert module.QUANT_TYPE == "exl3"


def test_npu_does_not_advertise_fp8_torch_until_cann_supports_float8():
    assert DEVICE.ALL not in TorchFP8Linear.SUPPORTS_DEVICES
    assert DEVICE.NPU not in TorchFP8Linear.SUPPORTS_DEVICES


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
def test_npu_awq_unpack_preserves_pack_dimension():
    device = _test_npu_device()
    qweight_cpu = torch.tensor(
        [[0, 1, -1], [-2147483648, 2147483647, -123456789]],
        dtype=torch.int32,
    )
    qzeros_cpu = torch.tensor(
        [[-1, 0, 123456789], [2147483647, -2147483648, 7]],
        dtype=torch.int32,
    )
    qweight = qweight_cpu.to(device)
    qzeros = qzeros_cpu.to(device)

    iweight, izeros = unpack_awq(qweight, qzeros, bits=4)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    expected_iweight = (qweight_cpu[:, :, None] >> shifts[None, None, :]).to(torch.int8).view(2, 24)
    expected_izeros = (qzeros_cpu[:, :, None] >> shifts[None, None, :]).to(torch.int8).view(2, 24)

    assert iweight.shape == (2, 24)
    assert izeros.shape == (2, 24)
    assert iweight.device.type == "npu"
    assert izeros.device.type == "npu"
    assert torch.equal(iweight.cpu(), expected_iweight)
    assert torch.equal(izeros.cpu(), expected_izeros)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
def test_npu_torch_gptq_unpack_preserves_pack_dimension():
    device = _test_npu_device()
    qweight_cpu = torch.tensor(
        [
            [0, 1, -1],
            [-2147483648, 2147483647, -123456789],
            [12345, -98765, 42],
            [-42, 98765, -12345],
        ],
        dtype=torch.int32,
    )
    qweight = qweight_cpu.to(device)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=device).view(1, 8, 1)

    unpacked = _right_shift_unpack(
        qweight.unsqueeze(1).expand(-1, 8, -1),
        shifts,
        torch.int8,
    )

    assert unpacked.shape == (4, 8, 3)
    assert unpacked.device.type == "npu"
    expected = (
        qweight_cpu.unsqueeze(1).expand(-1, 8, -1)
        >> torch.arange(0, 32, 4, dtype=torch.int32).view(1, 8, 1)
    ).to(torch.int8)
    assert torch.equal(unpacked.cpu(), expected)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_gptq_forward_matches_cpu(bits, dtype):
    module = _make_gptq_module(bits, dtype)
    x_cpu = torch.randn(2, 3, module.in_features, dtype=dtype)
    _assert_npu_forward_matches_cpu(module, x_cpu, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_gptq_matches_torch_baseline(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "0")
    baseline_cpu = _make_gptq_module(bits=4, dtype=dtype).eval()
    candidate = KomodoLinear(
        bits=4,
        group_size=baseline_cpu.requested_group_size,
        sym=baseline_cpu.sym,
        desc_act=baseline_cpu.desc_act,
        in_features=baseline_cpu.in_features,
        out_features=baseline_cpu.out_features,
        bias=baseline_cpu.bias is not None,
        pack_dtype=baseline_cpu.pack_dtype,
        register_buffers=True,
    )
    _copy_matching_buffers(candidate, baseline_cpu)
    candidate.optimized = True
    candidate.post_init()
    baseline = baseline_cpu.to(_test_npu_device()).eval()
    candidate = candidate.to(_test_npu_device(), dtype=dtype).eval()

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert candidate._cached_weights == {}


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("desc_act", [False, True])
def test_npu_komodo_gptq_group16_uses_packed_native_without_dense_cache_by_default(dtype, desc_act, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
    monkeypatch.delenv("GPTQMODEL_KOMODO_NATIVE_GROUP16", raising=False)
    monkeypatch.delenv("GPTQMODEL_KOMODO_NATIVE_FALLBACK_CACHE", raising=False)
    monkeypatch.setenv("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "0")
    baseline_cpu = _make_gptq_module(bits=4, dtype=dtype, group_size=16).eval()
    if desc_act:
        _set_supported_act_order_g_idx(baseline_cpu)
    candidate = KomodoLinear(
        bits=4,
        group_size=baseline_cpu.requested_group_size,
        sym=baseline_cpu.sym,
        desc_act=baseline_cpu.desc_act,
        in_features=baseline_cpu.in_features,
        out_features=baseline_cpu.out_features,
        bias=baseline_cpu.bias is not None,
        pack_dtype=baseline_cpu.pack_dtype,
        register_buffers=True,
    )
    _copy_matching_buffers(candidate, baseline_cpu)
    candidate.optimized = True
    candidate.post_init()
    baseline = baseline_cpu.to(_test_npu_device()).eval()
    candidate = candidate.to(_test_npu_device(), dtype=dtype).eval()
    assert candidate.native_plan_prepacked(device=_test_npu_device(), dtype=dtype)

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=2e-2, rtol=2e-2)
    x_decode = torch.randn(1, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected_decode = baseline(x_decode)
        actual_decode = candidate(x_decode)
        torch.npu.synchronize()
    torch.testing.assert_close(actual_decode.cpu(), expected_decode.cpu(), atol=2e-2, rtol=2e-2)
    assert candidate._native_plan_cache == {}
    assert (x.device, dtype) in candidate._native_group16_plan_cache
    assert candidate._cached_weights == {}

    candidate.clear_native_cache()
    assert candidate.prefetch_native_plan(device=x.device, dtype=dtype)
    assert (x.device, dtype) in candidate._native_group16_plan_pending
    assert candidate.native_plan_prepacked(device=x.device, dtype=dtype)
    with torch.inference_mode():
        prefetched = candidate(x)
        torch.npu.synchronize()
    torch.testing.assert_close(prefetched.cpu(), expected.cpu(), atol=2e-2, rtol=2e-2)
    assert candidate._native_group16_plan_pending == {}
    assert (x.device, dtype) in candidate._native_group16_plan_cache
    assert candidate._cached_weights == {}


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_gptq_group16_uses_exact_cached_fallback_when_enabled(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_GROUP16", "0")
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_FALLBACK_CACHE", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "0")
    baseline_cpu = _make_gptq_module(bits=4, dtype=dtype, group_size=16).eval()
    candidate = KomodoLinear(
        bits=4,
        group_size=baseline_cpu.requested_group_size,
        sym=baseline_cpu.sym,
        desc_act=baseline_cpu.desc_act,
        in_features=baseline_cpu.in_features,
        out_features=baseline_cpu.out_features,
        bias=baseline_cpu.bias is not None,
        pack_dtype=baseline_cpu.pack_dtype,
        register_buffers=True,
    )
    _copy_matching_buffers(candidate, baseline_cpu)
    candidate.optimized = True
    candidate.post_init()
    baseline = baseline_cpu.to(_test_npu_device()).eval()
    candidate = candidate.to(_test_npu_device(), dtype=dtype).eval()
    assert not candidate.native_plan_prepacked(device=_test_npu_device(), dtype=dtype)

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert candidate._native_plan_cache == {}
    assert dtype in candidate._cached_weights


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_gptq_native_int4_matches_torch_baseline(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "0")
    monkeypatch.setenv("GPTQMODEL_KOMODO_PREPACK_TILE_N", "16")
    baseline_cpu = _make_gptq_module(bits=4, dtype=dtype, group_size=32).eval()
    candidate = KomodoLinear(
        bits=4,
        group_size=baseline_cpu.requested_group_size,
        sym=baseline_cpu.sym,
        desc_act=baseline_cpu.desc_act,
        in_features=baseline_cpu.in_features,
        out_features=baseline_cpu.out_features,
        bias=baseline_cpu.bias is not None,
        pack_dtype=baseline_cpu.pack_dtype,
        register_buffers=True,
    )
    _copy_matching_buffers(candidate, baseline_cpu)
    candidate.optimized = True
    candidate.post_init()
    baseline = baseline_cpu.to(_test_npu_device()).eval()
    candidate = candidate.to(_test_npu_device(), dtype=dtype).eval()
    assert candidate.native_plan_prepacked(device=_test_npu_device(), dtype=dtype)
    assert not candidate._native_source_dropped

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert (x.device, dtype) in candidate._native_plan_cache

    candidate.clear_native_cache()
    assert candidate.prefetch_native_plan(device=x.device, dtype=dtype)
    assert (x.device, dtype) in candidate._native_plan_pending
    assert candidate.native_plan_prepacked(device=x.device, dtype=dtype)
    assert not candidate.prefetch_native_plan(device=x.device, dtype=dtype)
    with torch.inference_mode():
        prefetched = candidate(x)
        torch.npu.synchronize()
    torch.testing.assert_close(prefetched.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert (x.device, dtype) in candidate._native_plan_cache
    assert candidate._native_plan_pending == {}
    assert not candidate.prefetch_native_plan(device=x.device, dtype=dtype)

    next_candidate = KomodoLinear(
        bits=4,
        group_size=baseline_cpu.requested_group_size,
        sym=baseline_cpu.sym,
        desc_act=baseline_cpu.desc_act,
        in_features=baseline_cpu.in_features,
        out_features=baseline_cpu.out_features,
        bias=baseline_cpu.bias is not None,
        pack_dtype=baseline_cpu.pack_dtype,
        register_buffers=True,
    )
    _copy_matching_buffers(next_candidate, baseline_cpu)
    next_candidate.optimized = True
    next_candidate.post_init()
    next_candidate = next_candidate.to(_test_npu_device(), dtype=dtype).eval()
    candidate.enable_lookahead(True).set_lookahead_next(next_candidate)
    next_candidate.enable_lookahead(True)
    next_candidate.clear_native_cache()
    with torch.inference_mode():
        candidate(x)
    assert (x.device, dtype) in next_candidate._native_plan_pending or (x.device, dtype) in next_candidate._native_plan_cache
    with torch.inference_mode():
        lookahead = next_candidate(x)
        torch.npu.synchronize()
    torch.testing.assert_close(lookahead.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_cann_gptq_native_int4_matches_torch_baseline(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "0")
    monkeypatch.setenv("GPTQMODEL_KOMODO_PREPACK_TILE_N", "16")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_PREFETCH", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_CANN_PREFETCH_MIN_BYTES", "0")
    baseline_cpu = _make_gptq_module(bits=4, dtype=dtype, group_size=32).eval()
    candidate = KomodoCannLinear(
        bits=4,
        group_size=baseline_cpu.requested_group_size,
        sym=baseline_cpu.sym,
        desc_act=baseline_cpu.desc_act,
        in_features=baseline_cpu.in_features,
        out_features=baseline_cpu.out_features,
        bias=baseline_cpu.bias is not None,
        pack_dtype=baseline_cpu.pack_dtype,
        register_buffers=True,
    )
    _copy_matching_buffers(candidate, baseline_cpu)
    candidate.optimized = True
    candidate.post_init()
    baseline = baseline_cpu.to(_test_npu_device()).eval()
    candidate = candidate.to(_test_npu_device(), dtype=dtype).eval()
    assert candidate.backend == BACKEND.GPTQ_KOMODO_CANN
    assert candidate.native_plan_prepacked(device=_test_npu_device(), dtype=dtype)

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert (x.device, dtype) in candidate._native_plan_cache
    assert candidate._last_cann_plan is not None
    assert candidate._last_cann_plan.prefetch_enabled


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_gptq_native_int4_act_order_matches_torch_baseline(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "0")
    monkeypatch.setenv("GPTQMODEL_KOMODO_PREPACK_TILE_N", "16")
    baseline_cpu = _make_gptq_module(bits=4, dtype=dtype, group_size=32).eval()
    _set_supported_act_order_g_idx(baseline_cpu)
    candidate = KomodoLinear(
        bits=4,
        group_size=baseline_cpu.requested_group_size,
        sym=baseline_cpu.sym,
        desc_act=baseline_cpu.desc_act,
        in_features=baseline_cpu.in_features,
        out_features=baseline_cpu.out_features,
        bias=baseline_cpu.bias is not None,
        pack_dtype=baseline_cpu.pack_dtype,
        register_buffers=True,
    )
    _copy_matching_buffers(candidate, baseline_cpu)
    candidate.optimized = True
    candidate.post_init()
    baseline = baseline_cpu.to(_test_npu_device()).eval()
    candidate = candidate.to(_test_npu_device(), dtype=dtype).eval()
    assert candidate.native_plan_prepacked(device=_test_npu_device(), dtype=dtype)
    assert not candidate._native_source_dropped

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    native_plan = candidate._native_plan_cache[(x.device, dtype)]
    assert native_plan[4] is not None
    expected_g_idx = torch.arange(candidate.in_features, dtype=torch.int32) // candidate.requested_group_size
    sorted_g_idx = candidate.g_idx.detach().cpu()[native_plan[4].cpu()]
    assert torch.equal(sorted_g_idx, expected_g_idx)

    candidate.clear_native_cache()
    assert candidate.prefetch_native_plan(device=x.device, dtype=dtype)
    assert candidate.native_plan_prepacked(device=x.device, dtype=dtype)
    with torch.inference_mode():
        prefetched = candidate(x)
        torch.npu.synchronize()
    torch.testing.assert_close(prefetched.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert candidate._native_plan_cache[(x.device, dtype)][4] is not None
    assert candidate._native_plan_pending == {}


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_gptq_drops_source_after_native_pack(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_PREPACK_TILE_N", "16")
    baseline_cpu = _make_gptq_module(bits=4, dtype=dtype, group_size=32).eval()
    candidate = KomodoLinear(
        bits=4,
        group_size=baseline_cpu.requested_group_size,
        sym=baseline_cpu.sym,
        desc_act=baseline_cpu.desc_act,
        in_features=baseline_cpu.in_features,
        out_features=baseline_cpu.out_features,
        bias=baseline_cpu.bias is not None,
        pack_dtype=baseline_cpu.pack_dtype,
        register_buffers=True,
    )
    _copy_matching_buffers(candidate, baseline_cpu)
    candidate.optimized = True
    candidate.post_init()
    baseline = baseline_cpu.to(_test_npu_device()).eval()
    candidate = candidate.to(_test_npu_device(), dtype=dtype).eval()
    assert candidate._native_source_dropped
    assert candidate.native_plan_prepacked(device=_test_npu_device(), dtype=dtype)
    _assert_empty_source_buffers(
        candidate,
        ("qweight", "qzeros", "scales", "g_idx", "wf_unsqueeze_zero", "wf_unsqueeze_neg_one"),
    )

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert candidate._native_source_dropped
    assert (x.device, dtype) in candidate._native_plan_cache
    _assert_empty_source_buffers(
        candidate,
        ("qweight", "qzeros", "scales", "g_idx", "wf_unsqueeze_zero", "wf_unsqueeze_neg_one"),
    )

    candidate.clear_native_cache()
    assert (x.device, dtype) in candidate._native_plan_cache
    assert candidate.native_plan_prepacked(device=x.device, dtype=dtype)
    assert not candidate.prefetch_native_plan(device=x.device, dtype=dtype)
    with torch.inference_mode():
        after_clear = candidate(x)
        torch.npu.synchronize()
    torch.testing.assert_close(after_clear.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    with pytest.raises(RuntimeError, match="source quant weights are dropped"):
        candidate.train()


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_awq_forward_matches_cpu(dtype):
    module = _make_awq_module(dtype)
    x_cpu = torch.randn(2, 3, module.in_features, dtype=dtype)
    _assert_npu_forward_matches_cpu(module, x_cpu, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_awq_matches_torch_baseline(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "0")
    baseline = _make_awq_module(dtype).to(_test_npu_device()).eval()
    candidate = _make_awq_like_module(AwqKomodoLinear, dtype).to(_test_npu_device()).eval()
    _copy_matching_buffers(candidate, baseline)
    candidate.post_init()

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert candidate._cached_weights == {}


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_awq_native_int4_matches_torch_baseline(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "0")
    monkeypatch.setenv("GPTQMODEL_KOMODO_PREPACK_TILE_N", "16")
    baseline = _make_awq_like_module(AwqTorchLinear, dtype, group_size=32).to(_test_npu_device()).eval()
    baseline.post_init()
    candidate = _make_awq_like_module(AwqKomodoLinear, dtype, group_size=32).to(_test_npu_device()).eval()
    _copy_matching_buffers(candidate, baseline)
    candidate.post_init()
    assert candidate.native_plan_prepacked(device=_test_npu_device(), dtype=dtype)
    assert not candidate._native_source_dropped

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert (x.device, dtype) in candidate._native_plan_cache

    candidate.clear_native_cache()
    assert candidate.prefetch_native_plan(device=x.device, dtype=dtype)
    assert (x.device, dtype) in candidate._native_plan_pending
    assert candidate.native_plan_prepacked(device=x.device, dtype=dtype)
    assert not candidate.prefetch_native_plan(device=x.device, dtype=dtype)
    with torch.inference_mode():
        prefetched = candidate(x)
        torch.npu.synchronize()
    torch.testing.assert_close(prefetched.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert (x.device, dtype) in candidate._native_plan_cache
    assert candidate._native_plan_pending == {}
    assert not candidate.prefetch_native_plan(device=x.device, dtype=dtype)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16])
def test_npu_komodo_awq_drops_source_after_native_pack(dtype, monkeypatch):
    monkeypatch.setenv("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "1")
    monkeypatch.setenv("GPTQMODEL_KOMODO_PREPACK_TILE_N", "16")
    baseline = _make_awq_like_module(AwqTorchLinear, dtype, group_size=32).to(_test_npu_device()).eval()
    baseline.post_init()
    candidate = _make_awq_like_module(AwqKomodoLinear, dtype, group_size=32).to(_test_npu_device()).eval()
    _copy_matching_buffers(candidate, baseline)
    candidate.post_init()
    assert candidate._native_source_dropped
    assert candidate.native_plan_prepacked(device=_test_npu_device(), dtype=dtype)
    _assert_empty_source_buffers(candidate, ("qweight", "qzeros", "scales"))

    x = torch.randn(2, 3, baseline.in_features, dtype=dtype, device=_test_npu_device())
    with torch.inference_mode():
        expected = baseline(x)
        assert not candidate.prefetch_native_plan(device=x.device, dtype=dtype)
        actual = candidate(x)
        repeat = candidate(x)
        torch.npu.synchronize()

    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(repeat.cpu(), expected.cpu(), atol=5e-3, rtol=5e-3)
    assert candidate._native_source_dropped
    assert (x.device, dtype) in candidate._native_plan_cache
    assert candidate._native_plan_pending == {}
    assert candidate.native_plan_prepacked(device=x.device, dtype=dtype)
    assert not candidate.prefetch_native_plan(device=x.device, dtype=dtype)
    _assert_empty_source_buffers(candidate, ("qweight", "qzeros", "scales"))


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("cls", [KomodoLinear, AwqKomodoLinear])
def test_npu_komodo_rejects_bfloat16_inference(cls):
    if cls is KomodoLinear:
        baseline_cpu = _make_gptq_module(bits=4, dtype=torch.float16).eval()
        candidate = KomodoLinear(
            bits=4,
            group_size=baseline_cpu.requested_group_size,
            sym=baseline_cpu.sym,
            desc_act=baseline_cpu.desc_act,
            in_features=baseline_cpu.in_features,
            out_features=baseline_cpu.out_features,
            bias=baseline_cpu.bias is not None,
            pack_dtype=baseline_cpu.pack_dtype,
            register_buffers=True,
        )
        _copy_matching_buffers(candidate, baseline_cpu)
        candidate.optimized = True
        candidate.post_init()
    else:
        candidate = _make_awq_like_module(AwqKomodoLinear, torch.float16)
        candidate.post_init()

    candidate = candidate.to(_test_npu_device(), dtype=torch.bfloat16).eval()
    x = torch.randn(2, 3, candidate.in_features, dtype=torch.bfloat16, device=_test_npu_device())
    with pytest.raises(RuntimeError, match="supports only torch.float16 inference"):
        with torch.inference_mode():
            candidate(x)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_paro_forward_matches_cpu(dtype):
    module = _make_paro_module(dtype)
    x_cpu = torch.randn(2, 3, module.in_features, dtype=dtype)
    _assert_npu_forward_matches_cpu(module, x_cpu, atol=6e-3, rtol=6e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("bits", ["q1_0", "q4_0", "q4_k_m", "q5_k_m", "q6_k", "q8_0"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_gguf_forward_matches_cpu_without_fallback(bits, dtype):
    module = _make_gguf_module(bits, dtype)
    x_cpu = torch.randn(2, 3, module.in_features, dtype=dtype)
    _assert_npu_forward_matches_cpu(module, x_cpu, atol=8e-3, rtol=8e-3)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("group_size", [-1, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_torch_qqq_forward_matches_cpu_without_fallback(group_size, dtype, capfd):
    module_cpu = _make_qqq_module(dtype, group_size)
    module_npu = copy.deepcopy(module_cpu).to(_test_npu_device()).eval()
    x_cpu = torch.randn(2, 3, module_cpu.in_features, dtype=dtype)

    with torch.inference_mode():
        y_cpu = module_cpu(x_cpu)

    capfd.readouterr()
    with torch.inference_mode():
        y_npu = module_npu(x_cpu.to(_test_npu_device()))
        torch.npu.synchronize()

    captured = capfd.readouterr()
    combined_output = captured.out + captured.err
    assert "AiCpu" not in combined_output
    assert not any(marker in combined_output for marker in NPU_CPU_FALLBACK_MARKERS)
    assert y_npu.device.type == "npu"
    torch.testing.assert_close(y_npu.cpu(), y_cpu, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
def test_npu_exllamav3_torch_forward_matches_cpu_without_aicpu_sort(capfd):
    module_cpu = _make_exllamav3_torch_module()
    module_npu = _make_exllamav3_torch_module(device=_test_npu_device())
    x_cpu = torch.randn(2, 3, module_cpu.in_features, dtype=torch.float16)

    with torch.inference_mode():
        y_cpu = module_cpu(x_cpu)

    capfd.readouterr()
    with torch.inference_mode():
        y_npu = module_npu(x_cpu.to(_test_npu_device()))
        torch.npu.synchronize()

    captured = capfd.readouterr()
    combined_output = captured.out + captured.err
    assert "ArgSort" not in combined_output
    assert "AiCpu" not in combined_output
    assert not any(marker in combined_output for marker in NPU_CPU_FALLBACK_MARKERS)
    torch.testing.assert_close(y_npu.cpu(), y_cpu, atol=5e-2, rtol=5e-2)
