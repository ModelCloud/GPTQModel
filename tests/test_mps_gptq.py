# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import gptqmodel.quantization.gptq as gptq_module
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS")


def test_auto_import_does_not_force_mps_cpu_fallback():
    env = os.environ.copy()
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import gptqmodel.models.auto; "
                "assert 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_gptq_quantizes_on_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    device = torch.device("mps")
    layer = torch.nn.Linear(32, 24, bias=False, device=device)
    gptq = GPTQ(layer, QuantizeConfig(bits=4, group_size=8, damp_percent=0.01))
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(2, 16, 32, device=device), None)

    qweight, *_ = gptq.quantize(blocksize=16)
    torch.mps.synchronize()

    assert qweight.device.type == "mps"
    assert torch.isfinite(qweight).all()


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_unchunked_mps_hessian_queues_without_per_batch_host_sync(monkeypatch):
    """Unchunked X^T X owns no shared scratch and can remain on MPS's ordered stream."""

    torch.manual_seed(401)
    columns = 64
    layer = torch.nn.Linear(columns, 16, bias=False, device="mps", dtype=torch.float16)
    qcfg = QuantizeConfig(bits=4, group_size=32, offload_to_disk=False)
    gptq = GPTQ(layer, qcfg)
    matrix = torch.randn(19, columns, device="mps", dtype=torch.float16)
    native_sync = gptq_module.torch_sync

    def reject_sync(*_args, **_kwargs):
        raise AssertionError("the unchunked MPS Hessian path must not synchronize")

    monkeypatch.setattr(gptq_module, "torch_sync", reject_sync)
    actual = gptq.compute_hessian_xtx(matrix)
    expected = matrix.float().T @ matrix.float()
    torch.mps.synchronize()

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    sync_calls = 0

    def counted_sync(*args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        return native_sync(*args, **kwargs)

    monkeypatch.setattr(gptq_module, "torch_sync", counted_sync)
    monkeypatch.setattr(gptq_module, "_USE_GPTQ_MPS_ASYNC_HESSIAN", False)
    fallback = gptq.compute_hessian_xtx(matrix)
    assert sync_calls == 1
    torch.testing.assert_close(fallback, expected, atol=0, rtol=0)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_chunked_mps_hessian_keeps_workspace_reuse_barrier(monkeypatch):
    """Chunked accumulation must finish before its bounded workspace is returned."""

    torch.manual_seed(409)
    columns = 32
    layer = torch.nn.Linear(columns, 8, bias=False, device="mps", dtype=torch.float16)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=16,
        offload_to_disk=False,
        hessian={"chunk_size": 3, "staging_dtype": torch.float32},
    )
    gptq = GPTQ(layer, qcfg)
    matrix = torch.randn(11, columns, device="mps", dtype=torch.float16)
    sync_calls = 0
    native_sync = gptq_module.torch_sync

    def counted_sync(*args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        return native_sync(*args, **kwargs)

    monkeypatch.setattr(gptq_module, "torch_sync", counted_sync)
    actual = gptq.compute_hessian_xtx(matrix)
    expected = torch.zeros(columns, columns, device="mps", dtype=torch.float32)
    for start in range(0, matrix.shape[0], 3):
        chunk = matrix[start : start + 3].float()
        expected.addmm_(chunk.T, chunk)

    assert sync_calls == 1
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_async_mps_hessian_isolated_across_threads_and_instances():
    """Concurrent Python contexts may queue independent MPS Hessians without shared state."""

    columns = 64
    seeds = (17, 29, 43, 61, 73, 89, 101, 127, 149, 173, 197, 223)
    cases = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        matrix = torch.randn(23, columns, generator=generator, dtype=torch.float16).to(
            "mps"
        )
        layer = torch.nn.Linear(
            columns, 8, bias=False, device="mps", dtype=torch.float16
        )
        cases.append((matrix, layer))

    def compute(case):
        matrix, layer = case
        gptq = GPTQ(
            layer,
            QuantizeConfig(bits=4, group_size=32, offload_to_disk=False),
        )
        return gptq.compute_hessian_xtx(matrix), matrix.float().T @ matrix.float()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(compute, cases))
    torch.mps.synchronize()

    for actual, expected in results:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_async_mps_hessian_immediate_cpu_read_and_same_output_ordering(monkeypatch):
    """CPU reads synchronize implicitly and repeated addmm launches preserve order."""

    torch.manual_seed(419)
    columns = 64
    layer = torch.nn.Linear(columns, 8, bias=False, device="mps", dtype=torch.float16)
    gptq = GPTQ(
        layer,
        QuantizeConfig(bits=4, group_size=32, offload_to_disk=False),
    )
    matrices = [
        torch.randn(7 + seed % 5, columns, device="mps", dtype=torch.float16)
        for seed in range(12)
    ]

    monkeypatch.setattr(gptq_module, "_USE_GPTQ_MPS_ASYNC_HESSIAN", False)
    reference = torch.zeros(columns, columns, device="mps", dtype=torch.float32)
    for matrix in matrices:
        gptq.compute_hessian_xtx(matrix, out=reference)
    reference_cpu = reference.cpu()

    monkeypatch.setattr(gptq_module, "_USE_GPTQ_MPS_ASYNC_HESSIAN", True)
    actual = torch.zeros_like(reference)
    for matrix in matrices:
        gptq.compute_hessian_xtx(matrix, out=actual)

    # Tensor.cpu() is the first host observation and must wait for every queued
    # addmm without an explicit synchronization in the Hessian implementation.
    assert torch.equal(actual.cpu(), reference_cpu)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_async_mps_hessian_survives_input_deletion_and_fresh_output_gc(monkeypatch):
    """MPS command buffers retain storage, not Python tensor objects or GPTQ state."""

    torch.manual_seed(431)
    columns = 64
    layer = torch.nn.Linear(columns, 8, bias=False, device="mps", dtype=torch.float16)
    gptq = GPTQ(
        layer,
        QuantizeConfig(bits=4, group_size=32, offload_to_disk=False),
    )
    source = torch.randn(31, columns, device="mps", dtype=torch.float16)

    monkeypatch.setattr(gptq_module, "_USE_GPTQ_MPS_ASYNC_HESSIAN", False)
    reference = gptq.compute_hessian_xtx(source).cpu()
    torch.mps.empty_cache()
    baseline_bytes = torch.mps.current_allocated_memory()

    monkeypatch.setattr(gptq_module, "_USE_GPTQ_MPS_ASYNC_HESSIAN", True)
    last = None
    input_refs = []
    for _ in range(64):
        matrix = source.clone()
        input_refs.append(weakref.ref(matrix))
        last = gptq.compute_hessian_xtx(matrix)
        del matrix
        if len(input_refs) % 8 == 0:
            gc.collect()
    torch.mps.synchronize()

    assert torch.equal(last.cpu(), reference)
    del last
    gc.collect()
    torch.mps.empty_cache()
    assert all(ref() is None for ref in input_refs)
    # One allocator page of variation is acceptable; completed fresh outputs
    # and their inputs must not accumulate with the number of calls.
    assert torch.mps.current_allocated_memory() <= baseline_bytes + 1_048_576


def _quantize_mps_layer(
    use_mps_block, *, bits=4, group_size=32, sym=False, desc_act=False
):
    gptq_module._USE_GPTQ_MPS_BLOCK = use_mps_block
    torch.manual_seed(617)
    device = torch.device("mps")
    layer = torch.nn.Linear(128, 128, bias=False, device=device)
    gptq = GPTQ(
        layer,
        QuantizeConfig(
            bits=bits,
            group_size=group_size,
            sym=sym,
            damp_percent=0.01,
            desc_act=desc_act,
            act_group_aware=False,
            offload_to_disk=False,
        ),
    )
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(4, 128, device=device), None)
    output = gptq.quantize(blocksize=128)
    torch.mps.synchronize()
    return output


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
@pytest.mark.skipif(
    not hasattr(torch.mps, "compile_shader"),
    reason="Metal shader runtime is not available",
)
@pytest.mark.parametrize(
    ("bits", "group_size", "sym", "desc_act"),
    [(2, 32, False, False), (4, 64, True, True), (8, 128, False, True)],
)
def test_native_mps_block_matches_eager_gptq_end_to_end(
    monkeypatch, bits, group_size, sym, desc_act
):
    config = {
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "desc_act": desc_act,
    }
    eager_output = _quantize_mps_layer(use_mps_block=False, **config)
    launches = 0
    native_block = gptq_module.gptq_block_mps

    def counted_block(*args, **kwargs):
        nonlocal launches
        launches += 1
        return native_block(*args, **kwargs)

    monkeypatch.setattr(gptq_module, "gptq_block_mps", counted_block)
    native_output = _quantize_mps_layer(use_mps_block=True, **config)

    assert launches == 1
    for eager, native in zip(eager_output[:4], native_output[:4]):
        torch.testing.assert_close(native, eager, atol=0, rtol=0)
    assert native_output[5] == pytest.approx(eager_output[5], abs=1e-7)
    assert native_output[6:] == eager_output[6:]

    torch.manual_seed(991)
    heldout = torch.randn(16, 128, device="mps")
    eager_projection = heldout @ eager_output[0].T
    native_projection = heldout @ native_output[0].T
    torch.testing.assert_close(native_projection, eager_projection, atol=0, rtol=0)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
@pytest.mark.skipif(
    not hasattr(torch.mps, "compile_shader"),
    reason="Metal shader runtime is not available",
)
@pytest.mark.parametrize("bits", range(2, 9))
def test_async_mps_hessian_preserves_end_to_end_gptq_for_all_bits(monkeypatch, bits):
    monkeypatch.setattr(gptq_module, "_USE_GPTQ_MPS_ASYNC_HESSIAN", False)
    reference = _quantize_mps_layer(True, bits=bits, group_size=32)
    monkeypatch.setattr(gptq_module, "_USE_GPTQ_MPS_ASYNC_HESSIAN", True)
    candidate = _quantize_mps_layer(True, bits=bits, group_size=32)

    for expected, actual in zip(reference[:4], candidate[:4]):
        assert torch.equal(actual, expected)
    assert candidate[5] == pytest.approx(reference[5], abs=0)
    assert candidate[6:] == reference[6:]


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_mps_block_failure_falls_back_to_exact_eager_result(monkeypatch):
    eager_output = _quantize_mps_layer(use_mps_block=False)

    def deliberate_failure(*_args, **_kwargs):
        raise RuntimeError("deliberate Metal failure")

    monkeypatch.setattr(gptq_module, "gptq_block_mps", deliberate_failure)
    fallback_output = _quantize_mps_layer(use_mps_block=True)

    for eager, fallback in zip(eager_output[:4], fallback_output[:4]):
        torch.testing.assert_close(fallback, eager, atol=0, rtol=0)
    assert fallback_output[5:] == eager_output[5:]
