# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""ROCm backend gating for AWQ scheduling, GGUF, VPTQ, and ParoQuant.

These tests are pure CPU: HIP/CUDA runtimes are simulated with monkeypatched
``torch`` symbols so the architecture-dispatch decisions can be asserted on a
host without a GPU.  Nothing here claims that a kernel was executed on ROCm.
"""

from types import SimpleNamespace

import pytest
import torch

import gptqmodel.quantization.awq.modules.triton.scheduler as awq_scheduler
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear import fp8 as fp8_module
from gptqmodel.nn_modules.qlinear import gguf_triton as gguf_triton_module
from gptqmodel.nn_modules.qlinear import qqq as qqq_module
from gptqmodel.nn_modules.qlinear import vptq_cuda as vptq_cuda_module
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.nn_modules.qlinear.vptq_cuda import VPTQCudaLinear
from gptqmodel.utils import arch as arch_utils


def _fake_props(*, arch: str | None = None, warp_size: int = 64):
    return SimpleNamespace(
        gcnArchName=arch,
        warp_size=warp_size,
        multi_processor_count=120,
        total_memory=25_745_473_536,
    )


def _install_fake_rocm(monkeypatch, *, arch: str = "gfx1201:sramecc+:xnack-"):
    """Simulate a HIP runtime that reports ``arch`` for every device."""

    monkeypatch.setattr(torch.version, "hip", "7.16.26385")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: "AMD Radeon AI PRO R9700")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index=None: _fake_props(arch=arch))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    arch_utils.clear_arch_cache()

    def _forbidden_capability(index=None):
        raise AssertionError("get_device_capability must not be queried on ROCm")

    monkeypatch.setattr(torch.cuda, "get_device_capability", _forbidden_capability)


def teardown_function(_function):
    arch_utils.clear_arch_cache()
    awq_scheduler.clear_awq_triton_plan_cache()
    awq_scheduler.clear_awq_triton_rules()



def test_awq_device_identity_uses_rocm_arch_and_never_capability(monkeypatch):
    _install_fake_rocm(monkeypatch)

    identity = awq_scheduler.device_identity("cuda:0")

    assert identity[0] == "cuda"
    assert identity[1] == 0
    assert identity[3] is None, "capability must stay unset on ROCm"
    assert identity[6] == "gfx1201", "normalized gcnArchName without feature suffixes"

def test_awq_device_identity_keeps_cuda_capability(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=None: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index=None: _fake_props())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (8, 9))

    identity = awq_scheduler.device_identity("cuda:0")

    assert identity[3] == (8, 9)
    assert identity[6] is None, "CUDA identities must not carry a HIP arch"

def test_awq_builtin_rules_reject_rocm_identity():
    # A ROCm device whose name/capability fields happened to look like the
    # measured 4090 profile must still miss the NVIDIA-only rule table.
    identity = ("cuda", 0, "NVIDIA GeForce RTX 4090", (8, 9), 128, 50_950_569_984, "gfx1201")

    plan = awq_scheduler._builtin_plan(
        identity,
        M=64, N=8192, K=2048, G=128,
        input_dtype=torch.float16,
        compute_dtype=torch.float16,
        output_dtype=torch.float16,
        fp32_accum=True,
    )

    assert plan is None
    cuda_identity = identity[:6] + (None,)
    assert awq_scheduler._builtin_plan(
        cuda_identity,
        M=64, N=8192, K=2048, G=128,
        input_dtype=torch.float16,
        compute_dtype=torch.float16,
        output_dtype=torch.float16,
        fp32_accum=True,
    ) is not None

def test_gguf_capability_probe_short_circuits_on_rocm(monkeypatch):
    monkeypatch.setattr(gguf_triton_module, "IS_ROCM", True)
    gguf_triton_module._CUDA_DEVICE_CAPABILITY_CACHE.clear()

    assert gguf_triton_module._cuda_device_capability(torch.device("cuda:0")) is None
    assert not gguf_triton_module._CUDA_DEVICE_CAPABILITY_CACHE, "no capability may be cached on ROCm"

def test_gguf_triton_is_cuda_only_and_refuses_rocm(monkeypatch):
    assert gguf_triton_module.GGUFTritonKernel.SUPPORTS_DEVICES == [DEVICE.CUDA]
    monkeypatch.setattr(gguf_triton_module, "IS_ROCM", True)
    GGUFTritonKernel = gguf_triton_module.GGUFTritonKernel
    GGUFTritonKernel.cached_validate_once.cache_clear()

    ok, error = GGUFTritonKernel.validate_once()

    assert ok is False
    assert isinstance(error, NotImplementedError)
    assert "GGUF_TORCH" in str(error)

def test_vptq_cuda_declares_cuda_only():
    assert VPTQCudaLinear.SUPPORTS_DEVICES == [DEVICE.CUDA]

def test_vptq_cuda_rejects_rocm_with_actionable_error(monkeypatch):
    monkeypatch.setattr(vptq_cuda_module, "IS_ROCM", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    VPTQCudaLinear.cached_validate_once.cache_clear()

    ok, error = VPTQCudaLinear.validate_once()

    assert ok is False
    assert isinstance(error, NotImplementedError)
    assert "VPTQ_TORCH" in str(error)

    # The AUTO selector consults validate(); the ROCm refusal must surface as a
    # plain (False, exception) result so the Torch kernel is chosen instead.
    ok_validate, error_validate = VPTQCudaLinear.validate(bits=4, device=DEVICE.ROCM)
    assert ok_validate is False
    assert isinstance(error_validate, NotImplementedError)

def test_paroquant_triton_supports_rocm_explicitly():
    assert DEVICE.ROCM in ParoQuantTritonLinear.SUPPORTS_DEVICES
    # Priority 0 keeps this kernel out of AUTO selection on every device.
    assert ParoQuantTritonLinear.SUPPORTS_FORMATS[fp8_module.FORMAT.PAROQUANT] == 0


def test_qqq_marlin_kernel_declares_cuda_only():
    # The Marlin-derived W4A8 extension is CUDA inline PTX that hipcc cannot
    # compile, so QQQLinear must not advertise the ROCm family at all.
    assert qqq_module.QQQLinear.SUPPORTS_DEVICES == [DEVICE.CUDA]
    assert DEVICE.ROCM not in qqq_module.QQQLinear.SUPPORTS_DEVICES
    # The portable kernel is the ROCm/CPU replacement and stays device-agnostic.
    assert qqq_module.QQQTorchLinear.SUPPORTS_DEVICES == [DEVICE.ALL]


def test_qqq_linear_rejects_rocm_with_actionable_error(monkeypatch):
    monkeypatch.setattr(qqq_module, "IS_ROCM", True)
    qqq_module.QQQLinear.cached_validate_once.cache_clear()

    ok, error = qqq_module.QQQLinear.validate_once()

    assert ok is False
    assert isinstance(error, NotImplementedError)
    assert "QQQ_TORCH" in str(error)

    # The AUTO selector calls validate(); the refusal must surface as a plain
    # (False, exception) result so it moves on to QQQTorchLinear instead of
    # paying the ~18 s failed JIT build and then raising at pack time.
    ok_validate, error_validate = qqq_module.QQQLinear.validate(
        bits=4, group_size=128, desc_act=False, sym=True, device=DEVICE.ROCM
    )
    assert ok_validate is False
    assert isinstance(error_validate, NotImplementedError)

    # validate_device() must not infer CUDA capability from a HIP device.
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("get_device_capability must not be queried on ROCm")
        ),
    )
    with pytest.raises(NotImplementedError, match="QQQ_TORCH"):
        qqq_module.QQQLinear.validate_device(torch.device("cuda:0"))

    qqq_module.QQQLinear.cached_validate_once.cache_clear()


def test_qqq_processor_selects_torch_kernel_on_rocm(monkeypatch):
    """QQQ quantization must not build the CUDA-only kernel on gfx120*."""

    from gptqmodel.looper import qqq_processor
    from gptqmodel.quantization.config import QQQConfig
    from gptqmodel.utils.backend import BACKEND

    monkeypatch.setattr(qqq_processor, "IS_ROCM", True)
    processor = object.__new__(qqq_processor.QQQProcessor)
    processor.qcfg = QQQConfig(offload_to_disk=False)
    processor.qcfg.device = DEVICE.ROCM

    processor.requested_backend = BACKEND.AUTO
    assert processor._quant_linear_kernel() == (qqq_module.QQQTorchLinear, BACKEND.QQQ_TORCH)

    # An explicit CUDA-kernel request on ROCm is an actionable error, not a
    # late failure inside pack() or the first forward.
    processor.requested_backend = BACKEND.QQQ
    with pytest.raises(NotImplementedError, match="QQQ_TORCH"):
        processor._quant_linear_kernel()


def test_qqq_preprocess_does_not_skip_when_dynamic_is_unset():
    """A missing `dynamic` rule must not be read as "skip every module".

    ``dynamic_get`` returns None when no rule matches, so a truthiness test in
    ``preprocess`` silently quantized zero modules; the quantized AND path then
    saved a dense weight and reloaded into uninitialized packed buffers.
    """

    from gptqmodel.looper.qqq_processor import QQQProcessor
    from gptqmodel.quantization.config import QQQConfig

    processor = object.__new__(QQQProcessor)
    processor.qcfg = QQQConfig(offload_to_disk=False)
    assert processor.qcfg.dynamic is None
    assert processor.qcfg.dynamic_get(layer_name="model.layers.0.self_attn.q_proj") is None
    # Mirror exactly the guard the processor runs.
    assert (processor.qcfg.dynamic_get(layer_name="model.layers.0.self_attn.q_proj") is False) is False

    # An explicit negative rule must still skip.
    processor.qcfg.dynamic = {"-:model.layers.0.self_attn.q_proj": {}}
    assert processor.qcfg.dynamic_get(layer_name="model.layers.0.self_attn.q_proj") is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or not __import__("gptqmodel.utils.rocm", fromlist=["IS_ROCM"]).IS_ROCM,
    reason="requires a real ROCm device",
)
def test_qqq_quantize_actually_packs_modules_on_rocm(tmp_path):
    """Regression: QQQ must pack real modules, not silently quantize zero layers.

    With `dynamic=None`, `dynamic_get()` returns None and the old truthiness
    guard skipped every module.  That left the tree as HookedLinear, so `save()`
    wrote a dense `.weight` and `load()` rebuilt uninitialized packed buffers.
    """

    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
    from gptqmodel.utils.backend import BACKEND

    fixture = "/tmp/tiny-llama-fixture"
    if not __import__("os").path.isdir(fixture):
        pytest.skip("tiny local fixture missing")

    qcfg = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        format="qqq", backend=BACKEND.QQQ_TORCH, device="cuda:0", offload_to_disk=False,
    )
    calib = [("the quick brown fox jumps over the lazy dog quantize weights")] * 8
    model = GPTQModel.load(fixture, qcfg)
    model.quantize(calib, batch_size=1)

    packed = [m for _, m in model.model.named_modules() if isinstance(m, QQQTorchLinear)]
    assert packed, "QQQ packed no modules"
    assert all(torch.isfinite(m.s_group).all() for m in packed)
    assert all(int(m.B.abs().max()) > 0 for m in packed)

    ref = model.model.model.layers[0].self_attn.q_proj
    torch.manual_seed(0)
    x = torch.randn(2, 256, device="cuda:0", dtype=torch.float16)
    with torch.no_grad():
        y_ref = ref(x)

    model.save(str(tmp_path / "qqq"))
    reloaded = GPTQModel.load(str(tmp_path / "qqq"), device="cuda:0").model
    reloaded_layer = reloaded.model.layers[0].self_attn.q_proj
    assert isinstance(reloaded_layer, QQQTorchLinear)
    assert torch.equal(reloaded_layer.B.cpu(), ref.B.cpu())
    assert torch.allclose(
        reloaded_layer.s_channel.float().cpu(), ref.s_channel.float().cpu(), rtol=1e-6, atol=1e-6
    )
    with torch.no_grad():
        y_new = reloaded_layer(x)
    # fp16 output granularity: allow one ULP of the output magnitude.
    assert torch.allclose(y_ref.float(), y_new.float(), rtol=1e-3, atol=1e-3)
