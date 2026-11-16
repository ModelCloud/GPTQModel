# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GGUFCudaKernel keeps one device copy of the quantized weight (#4216).

Plans are keyed by batch size, and each one used to allocate a ggml buffer that
included (and re-uploaded) the whole packed weight, so a layer kept up to
``_MAX_GGML_CUDA_PLANS`` weight copies. The weight now lives in a single
per-device buffer that every plan's graph references.
"""

import ctypes
import threading

import numpy as np
import pytest
import torch

import gptqmodel.nn_modules.qlinear.gguf_cpp as gguf_cpp
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCudaKernel, _get_ggml_bridge


def _bridge():
    try:
        return _get_ggml_bridge()
    except Exception as exc:  # pragma: no cover - optional dependency
        pytest.skip(f"llama-cpp-python GGML bridge unavailable: {exc}")


def _buffer_size(bridge, buffer) -> int:
    lib = bridge._ggml_base
    # `ggml_backend_buffer_get_size` is not part of the bridge's bound surface.
    lib.ggml_backend_buffer_get_size.argtypes = [ctypes.c_void_p]
    lib.ggml_backend_buffer_get_size.restype = ctypes.c_size_t
    return int(lib.ggml_backend_buffer_get_size(buffer))


def _packed_q8_0(bridge, out_features: int, in_features: int) -> torch.Tensor:
    weight = np.random.default_rng(0).standard_normal(
        (out_features, in_features)
    ).astype(np.float32)
    return torch.from_numpy(bridge.quantize_chunk(weight, "Q8_0"))


def _compute_plan_on_cpu(bridge, backend, plan, x: torch.Tensor) -> torch.Tensor:
    """Run a plan's graph on the CPU backend.

    The CUDA backend is unavailable in this environment; the graph shape and
    buffer bookkeeping are backend independent, so the CPU backend stands in
    for it to prove a plan can read a weight tensor owned by another context.
    """

    lib = bridge._ggml_base
    lib.ggml_backend_tensor_set(
        plan.input_tensor,
        ctypes.c_void_p(x.data_ptr()),
        0,
        x.numel() * x.element_size(),
    )
    status = lib.ggml_backend_graph_compute(backend, plan.graph)
    assert status == 0
    lib.ggml_backend_synchronize(backend)
    output = torch.empty((plan.rows, plan.out_features), dtype=plan.output_dtype)
    lib.ggml_backend_tensor_get(
        plan.output_tensor,
        ctypes.c_void_p(output.data_ptr()),
        0,
        plan.output_nbytes,
    )
    return output


def test_cuda_plans_share_one_weight_buffer_and_compute(monkeypatch):
    bridge = _bridge()
    cpu_backend = bridge._get_cpu_backend()
    monkeypatch.setattr(bridge, "_get_cuda_backend", lambda index: cpu_backend)

    in_features, out_features = 256, 64
    qweight = _packed_q8_0(bridge, out_features, in_features)
    weight = bridge.build_cuda_weight(
        device=torch.device("cpu"),
        qweight=qweight,
        gguf_tensor_qtype="Q8_0",
        padded_in_features=in_features,
        out_features=out_features,
        kernel_name="test",
    )
    plans = []
    try:
        assert weight.nbytes == qweight.numel() * qweight.element_size()

        # Build largest first: the shared arena only grows, and a grow would
        # invalidate plans bound to the previous buffer.
        for rows in (8, 4, 2, 1):
            plan = bridge.build_quantized_matmul_cuda_plan(
                backend=cpu_backend,
                weight=weight,
                padded_in_features=in_features,
                out_features=out_features,
                rows=rows,
                input_ggml_type=bridge.ggml_type_f32,
                kernel_name="test",
            )
            plans.append((rows, plan))

        scratch = bridge._cuda_scratch[0]
        assert scratch.size >= max(plan.scratch_nbytes for _, plan in plans)
        for rows, plan in plans:
            # A pre-fix plan owned a buffer that also held the packed weight.
            # Now every plan is a slice of the single device arena.
            assert plan.scratch is scratch
            assert plan.scratch_generation == scratch.generation
            assert not hasattr(plan, "buffer")
            assert plan.scratch_nbytes >= rows * (in_features + out_features) * 4
            assert plan.scratch_nbytes <= rows * (in_features + out_features) * 4 + 8192

        # The shared weight tensor is usable from a plan built in another ctx.
        # Compare against the single-shot CPU path, which runs the same ggml
        # kernel with the weight in its own context: identical numerics prove
        # the graph resolves the cross-context weight correctly.
        x = torch.from_numpy(
            np.random.default_rng(1).standard_normal((2, in_features)).astype(np.float32)
        )
        rows, plan = next(item for item in plans if item[0] == 2)
        assert rows == x.shape[0]
        actual = _compute_plan_on_cpu(bridge, cpu_backend, plan, x)
        expected = bridge.quantized_matmul_cpu(
            qweight=qweight,
            x=x,
            gguf_tensor_qtype="Q8_0",
            padded_in_features=in_features,
            out_features=out_features,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    finally:
        for _, plan in plans:
            plan.close()
        weight.close()
        bridge._cuda_scratch.pop(0, None).close()


class _FakePlan:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeWeight:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeBridge:
    def __init__(self) -> None:
        self.weight_builds = []
        self.plan_weights = []
        self.plans = []
        # The kernel serializes every ggml-cuda entry through the bridge's
        # process-wide lock (#4378), so the fake must expose it.
        self._cuda_backend_lock = threading.RLock()

    def cuda_backend_lock(self, device_index):
        return self._cuda_backend_lock

    def cuda_available(self):
        return True, None

    def cuda_scratch_generation(self, device_index):
        return 0

    def build_cuda_weight(self, **kwargs):
        weight = _FakeWeight()
        self.weight_builds.append(kwargs)
        return weight

    def quantized_matmul_cuda(
        self, *, x, gguf_tensor_qtype, padded_in_features, out_features, plan=None, weight=None
    ):
        if plan is None:
            plan = _FakePlan(x.shape[0])
            self.plans.append(plan)
            self.plan_weights.append(weight)
        return torch.zeros(
            (x.shape[0], out_features), dtype=x.dtype, device=x.device
        ), plan


@pytest.mark.cuda
def test_kernel_builds_the_weight_once_for_many_batch_sizes(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    fake = _FakeBridge()
    monkeypatch.setattr(gguf_cpp, "_get_ggml_bridge", lambda: fake)

    module = GGUFCudaKernel(
        bits="q8_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=32,
        bias=False,
    ).to("cuda")

    for rows in range(1, 11):
        module(torch.randn(rows, 64, device="cuda", dtype=torch.float16))

    # One upload for ten batch sizes, and every plan references that copy.
    assert len(fake.weight_builds) == 1
    assert fake.plan_weights and all(w is fake.plan_weights[0] for w in fake.plan_weights)
    assert len(fake.plans) == 10
    assert len(module._ggml_cuda_plans) == gguf_cpp._MAX_GGML_CUDA_PLANS
    # Evicted plans release their own activation/output buffers.
    assert sum(plan.closed for plan in fake.plans) == 10 - gguf_cpp._MAX_GGML_CUDA_PLANS

    weight = module._ggml_cuda_weight
    module.clear_weight_cache()
    assert not module._ggml_cuda_plans
    assert module._ggml_cuda_weight is None
    assert weight.closed
