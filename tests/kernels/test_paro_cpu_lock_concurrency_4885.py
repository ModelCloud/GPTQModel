# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Warm Paro CPU forwards must not serialize on the process-wide transform lock (#4885).

#4832 keeps Paro's packed CPU operands on the module so ``state_dict()`` still
emits the checkpoint layout. To make that safe it wrapped the *whole* GEMM in
``BaseQuantLinear._cpu_transform_lock``, one RLock shared by every module of
every model in the process. That serialized every warm Paro CPU matmul.

Only the cold layout transform and the operand capture need the lock. A warm
forward must capture the packed operands under it, release it, and run the
product on those locals; a concurrent ``state_dict()`` may then rebind the
module attributes without tearing the in-flight GEMM.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.utils.paro import build_identity_rotation_buffers


BITS = 4
IN_FEATURES = 512
OUT_FEATURES = 256
GROUP_SIZE = 128
CHECKPOINT_KEYS = ("qweight", "qzeros", "scales", "bias")
PATHS = [
    pytest.param("aten", id="aten"),
    # The host may lack AVX512-BF16; pure-torch stand-ins cover that layout.
    pytest.param("avx512", id="avx512"),
]


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack unpacked integer weights into the AWQ bit layout used by the kernels."""
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]

    assert unpacked.shape[1] % pack_factor == 0
    packed = torch.zeros(
        (unpacked.shape[0], unpacked.shape[1] // pack_factor),
        dtype=torch.int32,
    )
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


def _make_packed_buffers(bits: int, in_features: int, out_features: int, group_size: int):
    """Build synthetic packed AWQ tensors for Paro runtime tests."""
    groups = in_features // group_size
    int_weight = torch.randint(0, 2**bits, size=(in_features, out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, out_features), dtype=torch.int32)
    scales = (torch.rand(groups, out_features, dtype=torch.float16) * 2.0) + 0.25
    bias = torch.randn(out_features, dtype=torch.float16)

    return (
        _pack_awq_tensor(int_weight, bits),
        _pack_awq_tensor(zero_points, bits),
        scales,
        bias,
    )


def _make_paro_cpu_module(dtype: torch.dtype):
    """A real ``ParoLinear`` with deterministic packed buffers and rotation."""
    torch.manual_seed(4885)
    qweight, qzeros, scales, bias = _make_packed_buffers(
        BITS, IN_FEATURES, OUT_FEATURES, GROUP_SIZE
    )
    scales.mul_(0.01)
    module = ParoLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=True,
        krot=2,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=IN_FEATURES,
        group_size=GROUP_SIZE,
        krot=2,
        dtype=dtype,
    )
    theta.normal_()
    channel_scales.uniform_(0.75, 1.25)
    module.pairs.copy_(pairs)
    module.theta.copy_(theta)
    module.channel_scales.copy_(channel_scales)
    module.post_init()
    module.eval()
    checkpoint = {
        name: getattr(module, name).detach().clone() for name in CHECKPOINT_KEYS
    }
    return module, checkpoint


def _install_fake_avx512_kernels(monkeypatch):
    """Pure-torch stand-ins for the AVX512-BF16 packer and GEMM."""
    from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import pack_hf_int4, unpack_hf_int4
    from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
    from gptqmodel.quantization.awq.utils.packing_utils import reverse_awq_order, unpack_awq

    def fake_pack(qweight, qzeros, out_features, in_features):
        iweight, izeros = unpack_awq(qweight, qzeros, BITS)
        iweight, izeros = reverse_awq_order(iweight, izeros, BITS)
        return (
            pack_hf_int4(iweight.t().to(torch.uint8).contiguous()),
            izeros.to(torch.uint8).contiguous(),
        )

    def fake_gemm(x, qweight, qzeros, scales, group_size, bias):
        out_features, half = qweight.shape
        in_features = half * 2
        codes = unpack_hf_int4(qweight, out_features, in_features).to(torch.float32)
        groups = torch.arange(in_features) // group_size
        zeros = qzeros.float().t()[:, groups].t()
        weight_scales = scales.float().t()[:, groups].t()
        weight = (codes.t() - zeros) * weight_scales
        out = x.float() @ weight
        if bias is not None:
            out = out + bias.float()
        return out.to(x.dtype)

    monkeypatch.setattr(
        TorchCpuAvx512AwqLinear,
        "cached_validate_once",
        classmethod(lambda cls: (True, None)),
    )
    monkeypatch.setattr(TorchCpuAvx512AwqLinear, "pack_awq_int4_kernel", fake_pack)
    monkeypatch.setattr(
        TorchCpuAvx512AwqLinear,
        "gemm_int4_qzero_group_scaled_cpu_kernel",
        fake_gemm,
    )
    return TorchCpuAvx512AwqLinear, "gemm_int4_qzero_group_scaled_cpu_kernel"


def _warm_module(path: str, monkeypatch):
    """Build and warm one real Paro layer for ``path``.

    Returns ``(module, checkpoint, x, expected, gemm_owner, gemm_attr)`` where
    ``gemm_owner.gemm_attr`` is the product entry point for that layout.
    """
    from gptqmodel.nn_modules.qlinear import torch_fused

    if path == "aten":
        from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
            _has_local_int4pack_cpu_ops,
        )

        if not _has_local_int4pack_cpu_ops():
            pytest.skip("ATen CPU int4pack ops are unavailable")
        module, checkpoint = _make_paro_cpu_module(torch.float16)
        x = torch.randn(1, IN_FEATURES, dtype=torch.float16)
        owner, attr = torch_fused, "_cpu_int4pack_mm"
    else:
        owner, attr = _install_fake_avx512_kernels(monkeypatch)
        module, checkpoint = _make_paro_cpu_module(torch.bfloat16)
        x = torch.randn(1, IN_FEATURES, dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = module(x)
    # The warm forward owns the layout; the cold transform is done.
    assert module._live_cpu_layout() == path
    return module, checkpoint, x, expected, owner, attr


@pytest.mark.parametrize("path", PATHS)
def test_warm_forwards_overlap_while_the_first_gemm_is_in_flight(path, monkeypatch):
    """A warm forward must not serialize behind another forward's GEMM.

    The lock still has to cover the operand capture, so the test does not hold
    it across a whole forward (the fixed forward would wait for it). Instead it
    gates the first GEMM and proves both properties of the fix: the process-wide
    lock is free while that GEMM runs, and a second warm forward completes
    before the first GEMM is released. On the buggy build the first forward
    holds the lock across the GEMM, so both checks fail.
    """
    module, _, x, expected, owner, attr = _warm_module(path, monkeypatch)
    real_gemm = getattr(owner, attr)
    gemm_entered = threading.Event()
    release_gemm = threading.Event()
    gate_lock = threading.Lock()
    gated = False

    def gated_gemm(*args, **kwargs):
        nonlocal gated
        with gate_lock:
            block = not gated
            gated = True
        if block:
            gemm_entered.set()
            assert release_gemm.wait(30), "test gate was never released"
        return real_gemm(*args, **kwargs)

    monkeypatch.setattr(owner, attr, gated_gemm)

    errors: list[BaseException] = []
    outputs: list[torch.Tensor] = []
    second_done = threading.Event()

    def first_forward():
        try:
            with torch.inference_mode():
                outputs.append(module(x))
        except Exception as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    def second_forward():
        try:
            with torch.inference_mode():
                outputs.append(module(x))
        except Exception as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)
        finally:
            second_done.set()

    first = threading.Thread(target=first_forward)
    second = threading.Thread(target=second_forward)
    first.start()
    second_started = False
    try:
        assert gemm_entered.wait(10), "the first warm forward never reached its GEMM"

        # The GEMM must run outside the process-wide transform lock.
        lock_free = module._cpu_transform_lock.acquire(blocking=False)
        if lock_free:
            module._cpu_transform_lock.release()
        assert lock_free, (
            "a warm Paro CPU GEMM still holds the process-wide transform lock"
        )

        # A second warm forward must finish while the first GEMM is gated.
        second.start()
        second_started = True
        assert second_done.wait(10), (
            "a warm Paro CPU forward serialized behind another forward's GEMM"
        )
    finally:
        release_gemm.set()
        first.join(timeout=60)
        if second_started:
            second.join(timeout=60)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors, errors[0]
    assert len(outputs) == 2
    for output in outputs:
        assert torch.equal(output, expected)


@pytest.mark.parametrize("path", PATHS)
def test_concurrent_state_dict_during_warm_forwards_stays_consistent(path, monkeypatch):
    """Saving while warm GEMMs run must not raise or tear the checkpoint layout.

    The fix moves the product outside the lock, so ``state_dict()`` can now run
    while a GEMM is in flight. It must still emit the checkpoint tensors, and
    the concurrent products must stay bitwise identical to the single-threaded
    reference (issue #4832's invariant, issue #4885's regression).
    """
    module, checkpoint, x, expected, _, _ = _warm_module(path, monkeypatch)

    errors: list[BaseException] = []
    mismatches: list[str] = []
    stop = threading.Event()
    barrier = threading.Barrier(4)

    def forward_loop():
        barrier.wait()
        try:
            with torch.inference_mode():
                for _ in range(20):
                    out = module(x)
                    if not torch.equal(out, expected):
                        mismatches.append("concurrent forward output differs")
                        return
        except Exception as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    def save_loop():
        barrier.wait()
        try:
            while not stop.is_set():
                state = module.state_dict()
                for name, reference in checkpoint.items():
                    actual = state.get(name)
                    if actual is None:
                        mismatches.append(f"{name} missing")
                    elif actual.dtype != reference.dtype or actual.shape != reference.shape:
                        mismatches.append(
                            f"{name} {actual.dtype}{tuple(actual.shape)} != "
                            f"{reference.dtype}{tuple(reference.shape)}"
                        )
                    elif not torch.equal(actual, reference):
                        mismatches.append(f"{name} values differ")
        except Exception as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [
        threading.Thread(target=forward_loop),
        threading.Thread(target=forward_loop),
        threading.Thread(target=save_loop),
        threading.Thread(target=save_loop),
    ]
    for thread in threads:
        thread.start()
    try:
        time.sleep(0.5)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=60)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors, [f"{type(exc).__name__}: {exc}" for exc in errors[:4]]
    assert not mismatches, mismatches[:8]

    # The post-concurrency single-threaded reference must still match bitwise.
    with torch.inference_mode():
        after = module(x)
    assert torch.equal(after, expected)
