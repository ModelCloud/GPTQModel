# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Memory-shape regressions for VPTQ (issue #4391, items 2 and 3).

The quantizer used to keep three chunk-sized distance temporaries in
`_chunked_assign` and up to eight float64 `n x n` Hessian copies while
computing the same inverse Cholesky twice per module. These tests pin the
numerical result to the pre-fix algorithm bitwise and bound the number of
full-size temporaries that may be live at once.

Everything here runs on CPU with synthetic tensors; no model download or
CUDA extension is required.
"""

from __future__ import annotations

import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from gptqmodel.quantization import vptq as vptq_module
from gptqmodel.quantization.config import VPTQConfig
from gptqmodel.quantization.vptq import VPTQ, _chunked_assign


# ---------------------------------------------------------------------------
# pre-fix reference implementations
# ---------------------------------------------------------------------------
def legacy_chunked_assign(
    data, centroids, *, chunk_bytes=vptq_module._DISTANCE_CHUNK_BYTES
):
    """`_chunked_assign` exactly as it was before issue #4391 item 2."""

    num_rows = data.shape[0]
    num_centroids = centroids.shape[0]
    rows_per_chunk = max(1, chunk_bytes // max(1, num_centroids * data.element_size()))

    centroids_norm = (centroids * centroids).sum(dim=1)
    assignment = torch.empty(num_rows, dtype=torch.int64, device=data.device)
    residual = torch.empty(num_rows, dtype=data.dtype, device=data.device)

    for start in range(0, num_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, num_rows)
        block = data[start:end]
        distance = centroids_norm.unsqueeze(0) - 2.0 * (block @ centroids.t())
        best_distance, best_index = distance.min(dim=1)
        assignment[start:end] = best_index
        residual[start:end] = best_distance + (block * block).sum(dim=1)
    return assignment, residual


def legacy_inverse_hessian_cholesky(hessian, percdamp):
    """`_inverse_hessian_cholesky` exactly as it was before item 3."""

    damp = torch.diagonal(hessian).mean().to(torch.float64) * float(percdamp)
    damped = hessian.clone()
    damped.diagonal().add_(damp.to(dtype=damped.dtype))
    cholesky = torch.linalg.cholesky(damped)
    inverse = torch.cholesky_inverse(cholesky)
    return torch.linalg.cholesky(inverse, upper=True)


def _legacy_inverse_method(self, hessian):
    return legacy_inverse_hessian_cholesky(hessian, self.qcfg.percdamp)


def build_quantizer(weight, qcfg):
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    linear.weight.data.copy_(weight)
    quantizer = VPTQ(module=linear, qcfg=qcfg)
    # Synthetic Hessians stand in for calibration data, so the coverage
    # fallback must not replace them with the identity.
    quantizer.fallback = None
    return quantizer


def synthetic_case(rows=24, columns=32, seed=3):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, columns, generator=generator) * 0.05
    features = torch.randn(columns, columns, generator=generator)
    hessian = features.t() @ features
    hessian = (hessian / hessian.diagonal().mean()).double()
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, 8],
        kmeans_iters=2,
        block_size=8,
    )
    return weight, hessian, qcfg


def assert_result_bitwise_equal(actual, expected):
    assert torch.equal(actual.quantized_weight, expected.quantized_weight)
    for name in ("centroids", "indices", "res_centroids", "res_indices"):
        actual_group = getattr(actual, name)
        expected_group = getattr(expected, name)
        assert actual_group.keys() == expected_group.keys()
        for key in actual_group:
            if expected_group[key] is None:
                assert actual_group[key] is None
            else:
                assert torch.equal(actual_group[key], expected_group[key]), (
                    f"{name}[{key}] drifted from the pre-fix algorithm"
                )


# ---------------------------------------------------------------------------
# allocation accounting
# ---------------------------------------------------------------------------
def _iter_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


class FullSizeAllocationCounter(TorchDispatchMode):
    """Count live freshly-allocated tensors of exactly one `numel`/dtype.

    Views are excluded by requiring the storage to be exactly the tensor's own
    `numel`, and repeated `out=` writes to the same tensor are counted once.
    """

    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.total = 0
        self.max_live = 0
        self._live = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for tensor in _iter_tensors(output):
            if (
                tensor.dtype == self.dtype
                and tensor.numel() == self.numel
                and tensor._base is None
                and tensor.untyped_storage().nbytes()
                == self.numel * tensor.element_size()
            ):
                if id(tensor) not in self._live:
                    self.total += 1
                    self._live[id(tensor)] = weakref.ref(tensor)
        for key, ref in list(self._live.items()):
            if ref() is None:
                del self._live[key]
        self.max_live = max(self.max_live, len(self._live))
        return output


# ---------------------------------------------------------------------------
# item 2: chunked assignment
# ---------------------------------------------------------------------------
def test_chunked_assign_matches_prefix_algorithm_bitwise():
    """The `addmm` form must not change assignment or residual bits."""

    generator = torch.Generator().manual_seed(17)
    data = torch.randn(257, 4, generator=generator)
    centroids = torch.randn(11, 4, generator=generator)
    # Force several small chunks instead of one 256 MiB block.
    chunk_bytes = 11 * data.element_size() * 7

    actual_assignment, actual_residual = _chunked_assign(
        data, centroids, chunk_bytes=chunk_bytes
    )
    expected_assignment, expected_residual = legacy_chunked_assign(
        data, centroids, chunk_bytes=chunk_bytes
    )
    assert torch.equal(actual_assignment, expected_assignment)
    assert torch.equal(actual_residual, expected_residual)

    # Exact ties resolve to the lowest index in both forms.
    tie = torch.tensor([[0.5, 0.0]])
    tie_codebook = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    assert torch.equal(
        _chunked_assign(tie, tie_codebook)[0],
        legacy_chunked_assign(tie, tie_codebook, chunk_bytes=1024)[0],
    )


def test_chunked_assign_keeps_one_distance_temporary():
    """At most one chunk-sized distance buffer may be live (was three)."""

    generator = torch.Generator().manual_seed(11)
    data = torch.randn(1000, 4, generator=generator)
    centroids = torch.randn(8, 4, generator=generator)
    chunk_bytes = 8 * data.element_size() * 50
    rows_per_chunk = chunk_bytes // (8 * data.element_size())
    distance_numel = rows_per_chunk * centroids.shape[0]

    legacy_counter = FullSizeAllocationCounter(distance_numel, data.dtype)
    with legacy_counter:
        legacy_chunked_assign(data, centroids, chunk_bytes=chunk_bytes)
    # The pre-fix form allocated the matmul, the `2.0 *` product and their
    # difference; this is the regression the issue measured.
    assert legacy_counter.max_live == 3

    counter = FullSizeAllocationCounter(distance_numel, data.dtype)
    with counter:
        _chunked_assign(data, centroids, chunk_bytes=chunk_bytes)
    assert counter.total == 1000 // rows_per_chunk
    assert counter.max_live == 1


# ---------------------------------------------------------------------------
# item 3: inverse Hessian
# ---------------------------------------------------------------------------
def test_inverse_hessian_matches_prefix_algorithm_bitwise():
    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        kmeans_iters=1,
        percdamp=0.01,
    )
    quantizer = build_quantizer(torch.zeros(16, 48), qcfg)
    generator = torch.Generator().manual_seed(5)
    basis = torch.randn(48, 48, generator=generator, dtype=torch.float64)
    hessian = basis @ basis.t() + torch.eye(48, dtype=torch.float64)

    actual = quantizer._inverse_hessian_cholesky(hessian.clone())
    expected = legacy_inverse_hessian_cholesky(hessian.clone(), qcfg.percdamp)
    assert torch.equal(actual, expected)

    # And it is still a genuine inverse-Hessian factor.
    inverse = actual.t() @ actual
    damp = qcfg.percdamp * torch.diagonal(hessian).mean()
    damped = hessian + damp * torch.eye(48, dtype=torch.float64)
    reference = torch.linalg.inv(damped)
    relative = float((inverse - reference).norm() / reference.norm())
    assert relative < 1e-12


def test_inverse_hessian_factorizes_in_place():
    """The helper may hold two `n x n` float64 buffers, not four."""

    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        kmeans_iters=1,
        percdamp=0.01,
    )
    quantizer = build_quantizer(torch.zeros(16, 64), qcfg)
    generator = torch.Generator().manual_seed(6)
    basis = torch.randn(64, 64, generator=generator, dtype=torch.float64)
    hessian = basis @ basis.t() + torch.eye(64, dtype=torch.float64)
    hessian = hessian.clone()

    legacy_input = hessian.clone()
    legacy_counter = FullSizeAllocationCounter(64 * 64, torch.float64)
    with legacy_counter:
        legacy_inverse_hessian_cholesky(legacy_input, qcfg.percdamp)
    assert legacy_counter.max_live == 4

    counter = FullSizeAllocationCounter(64 * 64, torch.float64)
    with counter:
        quantizer._inverse_hessian_cholesky(hessian)
    assert counter.max_live == 2


def test_quantize_weight_computes_the_inverse_once(monkeypatch):
    """Both passes must share one factorization, even with residual enabled."""

    weight, hessian, qcfg = synthetic_case()
    quantizer = build_quantizer(weight, qcfg)

    calls = []
    original = quantizer._inverse_hessian_cholesky

    def counting(hessian_arg):
        calls.append(hessian_arg)
        return original(hessian_arg)

    passed_inverses = []
    original_pass = quantizer._vptq_pass

    def spy_pass(**kwargs):
        passed_inverses.append(kwargs["inv_hessian"])
        return original_pass(**kwargs)

    monkeypatch.setattr(quantizer, "_inverse_hessian_cholesky", counting)
    monkeypatch.setattr(quantizer, "_vptq_pass", spy_pass)
    quantizer.quantize_weight(weight, hessian.clone())

    assert len(calls) == 1, "the damped inverse Hessian was factorized twice"
    assert len(passed_inverses) == 2, "residual pass did not run"
    assert passed_inverses[0] is passed_inverses[1]


def test_quantize_weight_owns_hessian_peak_budget():
    """A handed-over Hessian keeps the peak at three `n x n` float64 buffers."""

    weight, hessian, qcfg = synthetic_case()
    quantizer = build_quantizer(weight, qcfg)
    hessian = hessian.clone()
    columns = hessian.shape[0]
    pointer = hessian.data_ptr()

    counter = FullSizeAllocationCounter(columns * columns, torch.float64)
    with counter:
        quantizer.quantize_weight(weight, hessian, owns_hessian=True)

    # One owned Hessian (reused in place by the normalization) plus the damped
    # copy and its Cholesky factor. The pre-fix path kept five.
    assert counter.max_live <= 3
    # In-place normalization proves the caller's buffer was reused, not cloned.
    assert hessian.data_ptr() == pointer


def test_quantize_weight_matches_prefix_algorithm_bitwise(monkeypatch):
    """The whole quantizer must reproduce the pre-fix packed result exactly."""

    weight, hessian, qcfg = synthetic_case()

    actual, actual_loss = build_quantizer(weight, qcfg).quantize_weight(
        weight, hessian.clone()
    )

    monkeypatch.setattr(vptq_module, "_chunked_assign", legacy_chunked_assign)
    monkeypatch.setattr(VPTQ, "_inverse_hessian_cholesky", _legacy_inverse_method)
    expected, expected_loss = build_quantizer(weight, qcfg).quantize_weight(
        weight, hessian.clone()
    )

    assert_result_bitwise_equal(actual, expected)
    assert actual_loss == expected_loss


def test_quantize_releases_the_hessian():
    """`quantize()` hands `H2` over and must not keep a second reference."""

    qcfg = VPTQConfig(
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        kmeans_iters=2,
    )
    linear = torch.nn.Linear(32, 24, bias=False)
    quantizer = VPTQ(module=linear, qcfg=qcfg)
    quantizer.add_batch(torch.randn(4, 9, 32))
    quantizer.materialize_hessian()
    assert quantizer.H2 is not None

    quantizer.quantize()
    assert quantizer.H2 is None
