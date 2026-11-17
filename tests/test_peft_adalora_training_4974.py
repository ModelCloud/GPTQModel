# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Trainable GPTQ kernels must not return a view from inside their Function (#4974).

``TritonV2Linear`` calls ``QuantLinearFunction.apply`` and ``TorchLinear``'s
wide-autograd path calls ``_CudaWideMatmul.apply``. Both Functions ended with
``... .to(dtype).reshape(shape)``, and ``Tensor.reshape`` returns a view even
when the shape is unchanged. Autograd forbids in-place ops on a view created
inside a custom Function, so PEFT's AdaLoRA wrapper -- which does
``result += output`` -- failed on the first forward for every GPTQ model.

The tests pin the contract (an in-place add on the training output must work),
the numerics (the same FP32 reduction, bit for bit), and the inference cost
(the fix removes a no-op reshape; it must not introduce a clone).
"""

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear._numerics import matmul_fp32_reduce
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


IN_FEATURES = 256
OUT_FEATURES = 64
GROUP_SIZE = 128

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the wide-autograd and Triton kernels need a CUDA device",
)


def _packed(kernel_cls, *, device="cpu"):
    """A small, valid GPTQ module built through the real packer."""

    dense = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False).to(torch.float16)
    with torch.no_grad():
        dense.weight.normal_(0.0, 0.02, generator=torch.Generator().manual_seed(4974))
    module = kernel_cls(
        bits=4,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    module.qzero_format(format=2)
    groups = IN_FEATURES // GROUP_SIZE
    scales = (torch.rand(OUT_FEATURES, groups) * 0.02 + 0.005).to(torch.float16)
    zeros = torch.full((OUT_FEATURES, groups), 8, dtype=torch.int32)
    g_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    module.pack_block(
        linear=dense, scales=scales, zeros=zeros, g_idx=g_idx, block_in=8192, workers=1
    )
    return module.to(device=device, dtype=torch.bfloat16)


def _triton_kernel():
    try:
        from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"TritonV2Linear unavailable: {exc}")
    return TritonV2Linear


def _dequantized_reference(module, x):
    """The kernel's own reduction over the dequantized weight, computed by hand."""

    weight = module.dequantize_weight(num_itr=module.g_idx.shape[0] // x.shape[-1])
    flat = x.reshape(-1, x.shape[-1])
    if x.device.type == "cuda":
        out = torch.mm(flat, weight, out_dtype=torch.float32).to(x.dtype)
    else:
        # CPU keeps the plain low-precision matmul.
        out = torch.matmul(flat, weight)
    return out.reshape(x.shape[:-1] + (module.out_features,))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inference_output_is_unchanged(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    module = _packed(TorchLinear, device=device).eval()
    x = torch.randn(2, 16, IN_FEATURES, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        actual = module(x)
        expected = _dequantized_reference(module, x)

    assert torch.equal(actual, expected)


@requires_cuda
def test_training_output_allows_inplace_and_backward():
    """``out += 1`` is what PEFT's AdaLoRA wrapper does to the kernel output."""

    module = _packed(TorchLinear, device="cuda").train()
    x = torch.randn(2, 16, IN_FEATURES, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    out = module(x)
    out += 1.0
    out.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()


@requires_cuda
def test_triton_training_output_allows_inplace_and_backward():
    """The same contract for ``QuantLinearFunction`` (TritonV2Linear)."""

    module = _packed(_triton_kernel(), device="cuda").eval()
    x = torch.randn(2, 16, IN_FEATURES, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    out = module(x)
    out += 1.0
    out.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()


@requires_cuda
def test_two_dimensional_reduction_is_not_a_view():
    """A Function's output must be an owned tensor, not a reshape view."""

    module = _packed(TorchLinear, device="cuda")
    x = torch.randn(8, IN_FEATURES, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = module.dequantize_weight(num_itr=1)

    out = matmul_fp32_reduce(x, weight)

    assert not out._is_view()
    out += 1.0
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reduction_matches_the_manual_epilogue(device):
    """Removing the no-op reshape must not change a single bit."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    module = _packed(TorchLinear, device=device)
    x = torch.randn(2, 16, IN_FEATURES, device=device, dtype=torch.bfloat16)
    weight = module.dequantize_weight(num_itr=1)
    if device == "cuda":
        # The wide FP32 reduction the Function performs.
        expected = torch.mm(
            x.reshape(-1, IN_FEATURES), weight, out_dtype=torch.float32
        ).to(x.dtype)
        expected = expected.reshape(x.shape[:-1] + (OUT_FEATURES,))
    else:
        expected = torch.matmul(x, weight)

    actual = matmul_fp32_reduce(x, weight)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inference_path_does_not_clone(device):
    """The fix skips a no-op reshape; it must not add an allocation."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    module = _packed(TorchLinear, device=device).eval()
    x = torch.randn(2, 16, IN_FEATURES, device=device, dtype=torch.bfloat16)

    original = torch.Tensor.clone
    clones = {"n": 0}

    def counting_clone(self, *args, **kwargs):
        clones["n"] += 1
        return original(self, *args, **kwargs)

    torch.Tensor.clone = counting_clone
    try:
        with torch.no_grad():
            module(x)
            module(x)
    finally:
        torch.Tensor.clone = original

    assert clones["n"] == 0


def _adalora_wrapper(module):
    peft = pytest.importorskip("peft", reason="AdaLoRA lives in PEFT")

    # PEFT resolves an installed ``gptqmodel`` distribution on every dispatch
    # probe (`is_gptqmodel_available`, an ``lru_cache``d function) and raises
    # ``PackageNotFoundError`` when it is missing, which is the case for a
    # source checkout. Expose the in-tree version for the construction only.
    import importlib.metadata as metadata

    original = metadata.version
    try:
        metadata.version("gptqmodel")
    except metadata.PackageNotFoundError:
        def _version(name):
            return "7.12.0" if name == "gptqmodel" else original(name)

        metadata.version = _version
    try:
        import peft.import_utils

        peft.import_utils.is_gptqmodel_available.cache_clear()
        from peft.tuners.adalora.config import AdaLoraConfig
        from peft.tuners.adalora.gptq import SVDQuantLinear

        config = AdaLoraConfig(r=8, lora_alpha=8, target_modules=["q_proj"], total_step=10)
        wrapper = SVDQuantLinear(module, "default", config, r=8, lora_alpha=8)
    except (ImportError, AttributeError) as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"PEFT AdaLoRA GPTQ wrapper unavailable: {exc}")
    finally:
        metadata.version = original
        probe = getattr(getattr(peft, "import_utils", None), "is_gptqmodel_available", None)
        if probe is not None:
            probe.cache_clear()

    # PEFT keeps the adapter matrices in FP32 and the wrapper casts the input
    # to FP32 before the low-rank product, so align the adapter dtype here.
    for name, param in wrapper.named_parameters():
        if "lora_" in name:
            param.data = param.data.float()
    return peft, wrapper


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_peft_adalora_wrapper_trains(device):
    """The real PEFT wrapper does ``result += output``; it must train."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    module = _packed(TorchLinear, device=device).train()
    _, wrapper = _adalora_wrapper(module)
    wrapper = wrapper.to(device)

    x = torch.randn(2, 16, IN_FEATURES, device=device, dtype=torch.bfloat16, requires_grad=True)
    out = wrapper(x)
    out.float().pow(2).mean().backward()

    assert torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = [p.grad for p in wrapper.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
