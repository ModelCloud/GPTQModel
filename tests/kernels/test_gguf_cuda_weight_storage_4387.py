# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4387 item 2: GGUF_CPP_CUDA keeps one device copy of the packed weight.

The first forward uploads ``qweight`` into a ggml CUDA buffer. The Torch
``qweight`` used to stay allocated next to it, so every layer held its packed
weight twice (211 + 211 MiB on Qwen2.5-0.5B q4_k_m). The ggml buffer now
becomes the storage of ``qweight`` through a zero-copy view: Torch and ggml
read the same bytes, and ``state_dict()``, ``load_state_dict()``, ``.to()``,
copies and the exact FP32 bias epilogue keep working.
"""

from __future__ import annotations

import copy
import gc
import weakref

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCudaKernel


# 896 is not a multiple of ggml-cuda's 512-column row padding, so the ggml
# buffer is larger than ``qweight`` (it cannot wrap the Torch allocation).
IN_FEATURES = 896
OUT_FEATURES = 128
BITS = "q4_k_m"


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    GGUFCudaKernel.cached_validate_once.cache_clear()
    ok, err = GGUFCudaKernel.cached_validate_once()
    if not ok:
        pytest.skip(f"llama-cpp-python CUDA unavailable: {err}")


def _packed_state(seed: int, *, bias: bool) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias, dtype=torch.float16).eval()
    torch_kernel = GGUFTorchLinear(
        bits=BITS,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        register_buffers=True,
    )
    torch_kernel.pack_original(linear, scales=None, zeros=None)
    return {name: tensor.clone() for name, tensor in torch_kernel.state_dict().items()}


def _cuda_kernel(state: dict[str, torch.Tensor], *, bias: bool) -> GGUFCudaKernel:
    kernel = GGUFCudaKernel(
        bits=BITS,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        register_buffers=True,
    )
    kernel.load_state_dict(state, strict=True)
    return kernel.eval().to("cuda")


def _input(rows: int = 5) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(11)
    return torch.randn(rows, IN_FEATURES, generator=generator, dtype=torch.float16).to("cuda")


def _forward(kernel: GGUFCudaKernel, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = kernel(x)
    torch.cuda.synchronize()
    return out


@pytest.mark.cuda
@pytest.mark.parametrize("bias", [False, True])
def test_forward_leaves_one_device_copy_of_the_packed_weight(bias: bool):
    _require_cuda()
    state = _packed_state(0, bias=bias)
    kernel = _cuda_kernel(state, bias=bias)
    x = _input()
    nbytes = kernel.qweight.numel() * kernel.qweight.element_size()

    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    out = _forward(kernel, x)
    del out
    gc.collect()
    torch.cuda.synchronize()

    # The Torch allocation of ``qweight`` was released (pre-fix: unchanged).
    assert torch.cuda.memory_allocated() <= allocated_before - nbytes
    weight = kernel._ggml_cuda_weight
    assert weight is not None
    # ``qweight`` is a view of the ggml buffer: one device copy, not two.
    assert weight.backs(kernel.qweight)
    assert kernel.qweight.data_ptr() == weight.data_ptr
    assert kernel.qweight.is_cuda and not kernel.qweight.is_inference()
    assert weight.nbytes == nbytes
    assert torch.equal(kernel.qweight.cpu(), state["qweight"])

    # The same bytes keep producing the same result (the biased layer runs the
    # exact FP32 epilogue, which dequantizes from the shared view).
    first = _forward(kernel, x)
    assert torch.equal(_forward(kernel, x), first)
    assert kernel._ggml_cuda_weight is weight


@pytest.mark.cuda
def test_state_dict_and_reload_are_byte_identical():
    _require_cuda()
    kernel = _cuda_kernel(_packed_state(1, bias=True), bias=True)
    before = {name: tensor.cpu().clone() for name, tensor in kernel.state_dict().items()}
    x = _input()
    expected = _forward(kernel, x)

    saved = {name: tensor.cpu() for name, tensor in kernel.state_dict().items()}
    assert saved.keys() == before.keys()
    for name, tensor in before.items():
        assert saved[name].dtype == tensor.dtype
        assert torch.equal(saved[name], tensor), name

    reloaded = _cuda_kernel(saved, bias=True)
    assert torch.equal(_forward(reloaded, x), expected)


@pytest.mark.cuda
def test_state_dict_view_keeps_the_ggml_buffer_alive():
    _require_cuda()
    state = _packed_state(2, bias=False)
    kernel = _cuda_kernel(state, bias=False)
    _forward(kernel, _input())

    weight_ref = weakref.ref(kernel._ggml_cuda_weight)
    exported = kernel.state_dict()["qweight"]
    del kernel
    gc.collect()

    # The exported tensor shares the ggml buffer, so the buffer must outlive
    # the module, and go away with the last tensor that uses it.
    assert weight_ref() is not None
    assert torch.equal(exported.cpu(), state["qweight"])
    del exported
    gc.collect()
    assert weight_ref() is None


@pytest.mark.cuda
def test_load_state_dict_writes_into_the_shared_buffer():
    _require_cuda()
    kernel = _cuda_kernel(_packed_state(3, bias=True), bias=True)
    x = _input()
    _forward(kernel, x)
    weight = kernel._ggml_cuda_weight

    replacement = _packed_state(4, bias=True)
    kernel.load_state_dict(replacement, strict=True)

    # The copy landed in the ggml buffer; nothing is uploaded again.
    assert kernel._ggml_cuda_weight is weight
    assert weight.backs(kernel.qweight)
    assert torch.equal(kernel.qweight.cpu(), replacement["qweight"])
    fresh = _cuda_kernel(replacement, bias=True)
    assert torch.equal(_forward(kernel, x), _forward(fresh, x))
    assert kernel._ggml_cuda_weight is weight


@pytest.mark.cuda
def test_moving_off_the_device_frees_the_upload_and_reupload_matches():
    _require_cuda()
    state = _packed_state(5, bias=False)
    kernel = _cuda_kernel(state, bias=False)
    x = _input()
    expected = _forward(kernel, x)
    weight_ref = weakref.ref(kernel._ggml_cuda_weight)

    kernel.cpu()
    gc.collect()
    assert kernel._ggml_cuda_weight is None
    assert weight_ref() is None
    assert kernel.qweight.device.type == "cpu"
    assert torch.equal(kernel.qweight, state["qweight"])

    kernel.to("cuda")
    assert torch.equal(_forward(kernel, x), expected)
    assert kernel._ggml_cuda_weight.backs(kernel.qweight)


@pytest.mark.cuda
def test_deepcopy_gets_its_own_weight_storage():
    _require_cuda()
    kernel = _cuda_kernel(_packed_state(6, bias=False), bias=False)
    x = _input()
    expected = _forward(kernel, x)

    clone = copy.deepcopy(kernel)
    # Raw ggml handles are not shared with the copy, which would free them
    # twice; the copy's ``qweight`` is its own Torch tensor.
    assert clone._ggml_cuda_weight is None
    assert not clone._ggml_cuda_plans
    assert clone.qweight.data_ptr() != kernel.qweight.data_ptr()

    del kernel
    gc.collect()
    assert torch.equal(_forward(clone, x), expected)
    assert clone._ggml_cuda_weight.backs(clone.qweight)
