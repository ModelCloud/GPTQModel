# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A trainable kernel must return a tensor that owns its storage (#4974).

PEFT's AdaLoRA wrapper for GPTQ accumulates in place
(``peft/tuners/adalora/gptq.py``: ``result += output``). Autograd refuses an
in-place update on a tensor that is a view created inside a custom
``autograd.Function``, because the update would override the Function's own
backward. The FP32-reduction epilogue used to end in
``output.to(dtype).reshape(shape)``; ``reshape`` returns a view even when the
shape does not change, so every CUDA GPTQ-family kernel rejected AdaLoRA's
accumulation with ``RuntimeError: Output 0 of View is a view and is being
modified inplace``.
"""

import contextlib

import pytest
import torch

from gptqmodel.nn_modules.qlinear._numerics import (
    linear_fp32_reduce,
    matmul_fp32_reduce,
)
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.quantization import FORMAT


CUDA = torch.cuda.is_available()
IN_FEATURES = OUT_FEATURES = 64
GROUP_SIZE = 32


def _gptq(cls, device, dtype, *, bias=True):
    k, n = IN_FEATURES, OUT_FEATURES
    groups = k // GROUP_SIZE
    linear = torch.nn.Linear(k, n, bias=bias)
    linear.weight.data.normal_(0, 0.2, generator=torch.Generator().manual_seed(7))
    if bias:
        linear.bias.data.normal_(0, 0.1, generator=torch.Generator().manual_seed(8))
    module = cls(
        bits=4, group_size=GROUP_SIZE, sym=False, desc_act=False,
        in_features=k, out_features=n, bias=bias, format=FORMAT.GPTQ_V2,
        register_buffers=True,
    )
    module.pack_block(
        linear,
        torch.full((n, groups), 0.25),
        torch.full((n, groups), 8),
        torch.zeros(k, dtype=torch.int32),
    )
    module._init_wf_unsqueeze_buffers()
    return module.to(device=device, dtype=dtype)


def _awq(device, dtype):
    k, n = IN_FEATURES, OUT_FEATURES
    module = AwqTorchLinear(
        bits=4, group_size=16, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=True, dtype=dtype,
        register_buffers=True,
    )
    generator = torch.Generator().manual_seed(11)
    module.qweight.copy_(
        torch.randint(-2 ** 31, 2 ** 31 - 1, module.qweight.shape, generator=generator, dtype=torch.int32)
    )
    module.qzeros.copy_(
        torch.randint(0, 2 ** 31 - 1, module.qzeros.shape, generator=generator, dtype=torch.int32)
    )
    module.scales.copy_(torch.rand(module.scales.shape, generator=generator) * 0.05)
    module.bias.copy_(torch.randn(module.bias.shape, generator=generator) * 0.1)
    module.post_init()
    return module.to(device=device, dtype=dtype)


def _gguf(device, dtype):
    k, n = IN_FEATURES, OUT_FEATURES
    module = GGUFTorchLinear(
        bits="q4_0", group_size=-1, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=True, register_buffers=True,
    )
    linear = torch.nn.Linear(k, n, bias=True)
    linear.weight.data.normal_(0, 0.2, generator=torch.Generator().manual_seed(3))
    linear.bias.data.normal_(0, 0.1, generator=torch.Generator().manual_seed(4))
    module.pack_original(linear, scales=None, zeros=None)
    return module.to(device=device, dtype=dtype)


def _fp8(device, dtype):
    k, n = IN_FEATURES, OUT_FEATURES
    module = TorchFP8Linear(
        bits=8, group_size=-1, desc_act=False, sym=True,
        in_features=k, out_features=n, bias=True, format="float8_e4m3fn",
        weight_scale_method="row", activation_quant_method="none",
    )
    linear = torch.nn.Linear(k, n, bias=True)
    linear.weight.data.normal_(0, 0.2, generator=torch.Generator().manual_seed(5))
    linear.bias.data.normal_(0, 0.1, generator=torch.Generator().manual_seed(6))
    module.pack_original(linear, scales=None, zeros=None)
    module = module.to(device=device)
    module._scaled_mm_hard_disabled = True
    return module


def _triton_v2(device, dtype):
    pytest.importorskip("triton")
    from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear

    return _gptq(TritonV2Linear, device, dtype)


# Every kernel that declares ``SUPPORTS_TRAINING`` and can be built from a
# packed checkpoint without a model. ``devices`` lists the hosts the kernel
# validates on in this environment.
KERNELS = {
    "torch": (_gptq, TorchLinear, ("cpu", "cuda")),
    "triton_v2": (_triton_v2, None, ("cuda",)),
    "torch_fused": (_gptq, TorchFusedLinear, ("cpu", "cuda")),
    "torch_aten": (_gptq, TorchAtenLinear, ("cpu", "cuda")),
    "awq_torch": (_awq, None, ("cuda",)),
    "gguf_torch": (_gguf, None, ("cpu", "cuda")),
    "fp8_torch": (_fp8, None, ("cpu", "cuda")),
}


def _cases():
    cases = []
    for name, (builder, cls, devices) in KERNELS.items():
        for device in devices:
            skip = None
            if device == "cuda" and not CUDA:
                skip = "CUDA required"
            if name == "triton_v2" and not CUDA:
                skip = "TritonV2Linear needs CUDA"
            dtype = torch.float16 if device == "cuda" else torch.float32
            cases.append(
                pytest.param(
                    name, builder, cls, device, dtype,
                    id=f"{name}-{device}",
                    marks=pytest.mark.skipif(skip is not None, reason=skip or ""),
                )
            )
    return cases


@pytest.mark.parametrize("name,builder,cls,device,dtype", _cases())
@pytest.mark.parametrize("shape", [(4, IN_FEATURES), (2, 3, IN_FEATURES)])
def test_trainable_kernel_allows_inplace_accumulation(name, builder, cls, device, dtype, shape):
    """``result += output`` (PEFT AdaLoRA) must not raise for any rank."""
    try:
        module = builder(cls, device, dtype) if cls is not None else builder(device, dtype)
    except Exception as exc:  # kernel not buildable on this host
        pytest.skip(f"{name} cannot be built on {device}: {type(exc).__name__}: {exc}")
    module.train()
    x = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)

    out = module(x)
    out += 1.0
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not CUDA, reason="TritonV2Linear needs CUDA")
@pytest.mark.parametrize("shape", [(4, IN_FEATURES), (2, 3, IN_FEATURES)])
def test_triton_v2_eval_kernel_allows_inplace_accumulation(shape):
    """Eval forwards run the custom ``QuantLinearFunction`` autograd path."""
    module = _triton_v2("cuda", torch.float16)
    module.eval()
    x = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)

    out = module(x)
    out += 1.0
    out.sum().backward()

    assert x.grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "helper,weight_shape",
    [
        (matmul_fp32_reduce, (IN_FEATURES, OUT_FEATURES)),
        (linear_fp32_reduce, (OUT_FEATURES, IN_FEATURES)),
    ],
    ids=["matmul", "linear"],
)
def test_fp32_reduction_epilogue_owns_its_storage(helper, weight_shape, dtype):
    """The epilogue may run inside a custom Function, so it must not return a view."""
    x = torch.randn(4, IN_FEATURES, dtype=dtype)
    weight = torch.randn(*weight_shape, dtype=dtype)
    bias = torch.randn(OUT_FEATURES, dtype=dtype)

    out = helper(x, weight, bias)

    assert not out._is_view(), "epilogue returned a view of an internal tensor"
    out += 1.0


@pytest.mark.skipif(not CUDA, reason="the wide-bias epilogue needs CUDA FP16")
@pytest.mark.parametrize("shape", [(4, IN_FEATURES), (2, 3, IN_FEATURES)])
def test_triton_v2_wide_bias_kernel_allows_inplace_accumulation(shape):
    """A checkpoint FP32 bias takes ``linear_wide_bias`` inside the Function."""
    module = _triton_v2("cuda", torch.float16).eval()
    # A checkpoint may store the bias wider than the activations; the epilogue
    # then keeps it out of the low-precision reduction (autocast narrows the
    # GEMM operands, so the wide-bias epilogue is the one that runs).
    module.bias.data = module.bias.data.float()
    x = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)

    with torch.autocast("cuda", dtype=torch.float16):
        out = module(x)

    assert out.shape == shape[:-1] + (OUT_FEATURES,)
    out += 1.0
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not CUDA, reason="the wide CUDA GEMM Function needs CUDA")
@pytest.mark.parametrize("shape", [(4, IN_FEATURES), (2, 3, IN_FEATURES)])
def test_cuda_wide_matmul_function_returns_owned_storage(shape):
    """``_CudaWideMatmul`` must not hand back a view created inside its forward."""
    x = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(IN_FEATURES, OUT_FEATURES, device="cuda", dtype=torch.float16) / 8
    bias = torch.randn(OUT_FEATURES, device="cuda", dtype=torch.float16)

    out = matmul_fp32_reduce(x, weight, bias)

    assert out.shape == x.shape[:-1] + (OUT_FEATURES,)
    out += 1.0
    out.sum().backward()
    assert x.grad is not None


@contextlib.contextmanager
def _peft_adalora_gptq_wrapper():
    """Yield PEFT's AdaLoRA wrapper for GPTQ modules, or skip.

    PEFT resolves an installed ``gptqmodel`` distribution on every dispatch
    probe, and a source checkout has none; expose the in-tree version for the
    duration of the test. The probe is ``lru_cache``d, so the patch has to stay
    in place from the first call until the wrapper is built.
    """
    import importlib.metadata as metadata

    original = metadata.version
    try:
        metadata.version("gptqmodel")
    except metadata.PackageNotFoundError:
        def _version(name):
            return "7.12.0" if name == "gptqmodel" else original(name)

        metadata.version = _version
    try:
        pytest.importorskip("peft")
        import peft.import_utils

        peft.import_utils.is_gptqmodel_available.cache_clear()
        from peft.tuners.adalora.config import AdaLoraConfig
        from peft.tuners.adalora.gptq import SVDQuantLinear
    except ImportError as exc:  # pragma: no cover - optional dependency probe
        pytest.skip(f"peft AdaLoRA GPTQ wrapper unavailable: {exc}")
    try:
        yield AdaLoraConfig, SVDQuantLinear
    finally:
        metadata.version = original
        peft.import_utils.is_gptqmodel_available.cache_clear()


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("name", ["torch", "triton_v2"])
def test_peft_adalora_wrapper_trains_a_gptq_kernel(name):
    """The reported failure: PEFT's wrapper does ``result += output``."""
    with _peft_adalora_gptq_wrapper() as (AdaLoraConfig, SVDQuantLinear):
        builder, cls, _devices = KERNELS[name]
        base = builder(cls, "cuda", torch.bfloat16) if cls is not None else builder("cuda", torch.bfloat16)

        config = AdaLoraConfig(r=8, target_modules=["q_proj"], task_type="CAUSAL_LM", total_step=10)
        wrapped = SVDQuantLinear(base, "default", config, r=8, lora_alpha=8).train()
        x = torch.randn(2, 3, IN_FEATURES, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        # PEFT keeps the adapter matrices in the base layer's dtype and only
        # bypasses its FP32 input cast under autocast.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = wrapped(x)
        out.float().sum().backward()

    assert x.grad is not None
    assert any(p.grad is not None for p in wrapped.parameters() if p.requires_grad)


def _awq_packed_tensors(device="cuda"):
    k = n = 128
    return (
        torch.zeros((k, n // 8), device=device, dtype=torch.int32),
        torch.ones((4, n), device=device, dtype=torch.float16),
        torch.zeros((4, n // 8), device=device, dtype=torch.int32),
        torch.zeros(n, device=device, dtype=torch.float16),
    )


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 128)])
def test_awq_gemm_function_allows_inplace_accumulation(monkeypatch, shape):
    """``AwqGemmFn`` backs the trainable ``AwqGEMMLinear`` kernel."""
    import gptqmodel.nn_modules.qlinear.gemm_awq as gemm_awq

    weight = torch.randn(128, 128, device="cuda", dtype=torch.float16) / 8
    monkeypatch.setattr(gemm_awq, "awq_dequantize_weights", lambda *args, **kwargs: weight)
    monkeypatch.setattr(gemm_awq, "_awq_fp32_output_mm", lambda x, w: x @ w)
    monkeypatch.setattr(gemm_awq, "_awq_cuda_gemm_forward", lambda x, *args, **kwargs: x @ weight)
    monkeypatch.setattr(
        gemm_awq, "_awq_cuda_gemm_forward_fp32_reduce", lambda x, *args, **kwargs: x @ weight
    )
    qweight, scales, qzeros, bias = _awq_packed_tensors()

    x = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
    out = gemm_awq.AwqGemmFn.apply(x, qweight, qzeros, scales, 4, 32, bias, 128, "cuda", False)

    assert out.shape == x.shape[:-1] + (128,)
    out += 1.0
    out.sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("shape", [(4, 128), (2, 3, 128)])
def test_awq_gemm_triton_function_allows_inplace_accumulation(monkeypatch, shape):
    """``AwqGemmTritonFn`` backs the trainable ``AwqGEMMTritonLinear`` kernel."""
    import types

    import gptqmodel.nn_modules.qlinear.gemm_awq_triton as gemm_triton
    from gptqmodel.quantization.awq.modules.triton import gemm as triton_gemm

    weight = torch.randn(128, 128, device="cuda", dtype=torch.float16) / 8
    monkeypatch.setattr(triton_gemm, "awq_dequantize_triton", lambda *args, **kwargs: weight)
    # The dequantise-then-GEMM path needs no compiled plan; the fused path is
    # covered by the kernel suites.
    monkeypatch.setattr(
        gemm_triton, "select_awq_triton_plan", lambda **kwargs: types.SimpleNamespace(fused=False)
    )
    monkeypatch.setattr(gemm_triton, "mark_awq_triton_plan_warmed", lambda *args, **kwargs: None)
    qweight, scales, qzeros, bias = _awq_packed_tensors()

    x = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
    out = gemm_triton.AwqGemmTritonFn.apply(x, qweight, qzeros, scales, 4, 32, bias, 128)

    assert out.shape == x.shape[:-1] + (128,)
    out += 1.0
    out.sum().backward()
    assert x.grad is not None
