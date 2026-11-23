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


class LiveFloatBytes(TorchDispatchMode):
    """Track the peak bytes of freshly-allocated floating tensors.

    Views are excluded (`_base is None`), and any tensor that collected is
    dropped, so the peak is what the assignment itself keeps alive at once. This
    counts bytes rather than tensor shapes because the exact pass tiles rows and
    candidates differently from the screen.
    """

    def __init__(self):
        self._live = {}
        self.peak_bytes = 0
        self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for tensor in _iter_tensors(output):
            if tensor._base is None and tensor.is_floating_point():
                self._live[id(tensor)] = weakref.ref(tensor), tensor.numel() * tensor.element_size()
        for key, (ref, _) in list(self._live.items()):
            if ref() is None:
                del self._live[key]
        live = sum(size for _, size in self._live.values())
        self.peak_bytes = max(self.peak_bytes, live)
        self.largest = max(self.largest, max((size for _, size in self._live.values()), default=0))
        return output


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
def _exact_oracle(data, centroids):
    """Independent float64 direct-difference assignment and distance."""

    difference = data.to(torch.float64).unsqueeze(1) - centroids.to(
        torch.float64
    ).unsqueeze(0)
    distance = difference.square().sum(dim=2)
    return distance.argmin(dim=1), distance.min(dim=1).values


def test_chunked_assign_matches_exact_oracle_not_the_expanded_legacy_form():
    """The assignment must follow the exact distance, not the old bit pattern.

    The pre-fix form evaluated `||c||^2 - 2 x.c` and added `||x||^2` back. Its
    cancellation let a one-ULP neighbour beat an exact codebook entry, so this
    pins the assignment to an independent float64 direct-difference oracle and
    keeps the legacy helper only as the documented pre-fix control.
    """

    generator = torch.Generator().manual_seed(17)
    data = torch.randn(257, 4, generator=generator)
    centroids = torch.randn(11, 4, generator=generator)
    # Force several small chunks instead of one 256 MiB block.
    chunk_bytes = 11 * data.element_size() * 7

    actual_assignment, actual_residual = _chunked_assign(
        data, centroids, chunk_bytes=chunk_bytes
    )
    expected_assignment, expected_distance = _exact_oracle(data, centroids)
    assert torch.equal(actual_assignment, expected_assignment)
    assert torch.allclose(
        actual_residual.double(), expected_distance, rtol=1e-6, atol=1e-6
    )
    # A sum of squares cannot be negative, unlike the expanded form that added
    # `||x||^2` back into a cancelled value.
    assert float(actual_residual.min()) >= 0.0

    # Exact ties resolve to the lowest index, and a one-ULP neighbour does not.
    tie = torch.tensor([[0.5, 0.0]])
    tie_codebook = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    assert int(_chunked_assign(tie, tie_codebook)[0][0]) == 0
    assert int(_exact_oracle(tie, tie_codebook)[0][0]) == 0
    assert torch.equal(
        _chunked_assign(tie, tie_codebook)[0],
        legacy_chunked_assign(tie, tie_codebook, chunk_bytes=1024)[0],
    )


def test_chunked_assign_keeps_one_distance_temporary():
    """Live distance bytes stay inside the caller's chunk budget (was three)."""

    generator = torch.Generator().manual_seed(11)
    rows, dim = 4096, 4
    data = torch.randn(rows, dim, generator=generator)
    centroids = torch.randn(512, dim, generator=generator)
    chunk_bytes = 256 * 1024

    legacy_counter = LiveFloatBytes()
    with legacy_counter:
        legacy_chunked_assign(data, centroids, chunk_bytes=chunk_bytes)
    # The pre-fix form kept the matmul result, its `2.0 *` product and their
    # difference live together: three chunk-sized score blocks.
    assert legacy_counter.largest >= chunk_bytes
    assert legacy_counter.peak_bytes > 2 * chunk_bytes

    counter = LiveFloatBytes()
    with counter:
        _chunked_assign(data, centroids, chunk_bytes=chunk_bytes)
    # One score block, plus the assignment's own row-shaped intermediates.
    assert counter.largest <= chunk_bytes
    assert counter.peak_bytes <= chunk_bytes + 4 * rows * dim * data.element_size() + 4 * rows


def test_exact_assignment_honours_the_chunk_budget_for_tall_inputs():
    """The direct pass tiles rows too, so a tall input cannot outgrow the budget."""

    generator = torch.Generator().manual_seed(13)
    rows, dim = 8192, 4
    data = torch.randn(rows, dim, generator=generator)
    centroids = torch.randn(64, dim, generator=generator)
    chunk_bytes = 64 * 1024
    # Work stays under the direct-pass threshold, so this exercises the tile.
    assert rows * centroids.shape[0] * dim <= vptq_module._EXACT_ASSIGN_WORK

    counter = LiveFloatBytes()
    with counter:
        _chunked_assign(data, centroids, chunk_bytes=chunk_bytes)

    # No single tensor is chunk sized; the difference, its square and the
    # running result are row tiles.
    assert counter.largest <= rows * dim * data.element_size()
    assert counter.peak_bytes <= chunk_bytes + 4 * rows * dim * data.element_size() + 4 * rows


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
