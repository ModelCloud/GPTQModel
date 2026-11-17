# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GGUFCudaKernel must return an empty output for zero-row inputs (#4743).

``GGUF_CPP_CUDA`` is what AUTO picks for Q4_0/Q8_0 GGUF checkpoints on CUDA.
Before the guard it built a ggml plan for ``M == 0``, which contains
zero-sized tensors and made the arena allocation raise
``GGUFCudaKernel GGML plan has no scratch tensors``. Every other kernel,
including the CPU sibling ``GGUFCppKernel``, returns ``(..., N)`` for M=0.
"""

import threading

import pytest
import torch

import gptqmodel.nn_modules.qlinear.gguf_cpp as gguf_cpp
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCudaKernel


class _FakeWeight:
    def __init__(self, qweight):
        self.qweight = qweight

    def backs(self, qweight):
        return self.qweight is qweight

    def as_tensor(self):
        return self.qweight

    def close(self):
        pass


class _FakePlan:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def close(self):
        self.closed = True


class _FakeBridge:
    """Bridge that records calls and fails loudly if M=0 reaches the kernel."""

    def __init__(self):
        self.matmul_rows = []
        self.weight_builds = 0
        self.lock = threading.RLock()

    def cuda_backend_lock(self, device_index):
        return self.lock

    def cuda_available(self):
        return True, None

    def cuda_scratch_generation(self, device_index):
        return 0

    def build_cuda_weight(self, **kwargs):
        self.weight_builds += 1
        return _FakeWeight(kwargs["qweight"])

    def quantized_matmul_cuda(
        self, *, x, gguf_tensor_qtype, padded_in_features, out_features, plan=None, weight=None
    ):
        if x.shape[0] == 0:
            raise AssertionError("zero-row input reached the GGML CUDA kernel")
        self.matmul_rows.append(x.shape[0])
        if plan is None:
            plan = _FakePlan(x.shape[0])
        return torch.zeros((x.shape[0], out_features), dtype=x.dtype, device=x.device), plan


def _module(monkeypatch) -> GGUFCudaKernel:
    # The class caches ``validate_once``; another GGUF test in the same session
    # may already have recorded "no GGML CUDA support" for the real bridge.
    monkeypatch.setattr(
        GGUFCudaKernel,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )
    return GGUFCudaKernel(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=32,
        bias=False,
    ).to("cuda")


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(0, 64), (2, 0, 64), (2, 3, 0, 64)])
def test_zero_row_inputs_return_an_empty_output(monkeypatch, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    fake = _FakeBridge()
    monkeypatch.setattr(gguf_cpp, "_get_ggml_bridge", lambda: fake)

    module = _module(monkeypatch)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)

    out = module(x)

    assert tuple(out.shape) == shape[:-1] + (32,)
    assert out.numel() == 0
    assert out.dtype == torch.float16
    assert out.device.type == "cuda"
    # No plan, no arena growth and no weight upload happened for M=0.
    assert fake.matmul_rows == []
    assert fake.weight_builds == 0
    assert module._ggml_cuda_plans == {}


@pytest.mark.cuda
def test_non_empty_input_still_uses_the_kernel(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    fake = _FakeBridge()
    monkeypatch.setattr(gguf_cpp, "_get_ggml_bridge", lambda: fake)

    module = _module(monkeypatch)
    out = module(torch.randn(2, 3, 64, device="cuda", dtype=torch.float16))

    assert tuple(out.shape) == (2, 3, 32)
    assert fake.matmul_rows == [6]
