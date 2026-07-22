# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch

from gptqmodel.adapter.adapter import EoRAConfig, Lora
from gptqmodel.eora import eora as eora_module
from gptqmodel.looper.eora_processor import EoraProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import QuantizeConfig


def _make_eora_processor(monkeypatch, *, shared: bool):
    monkeypatch.setenv("GPTQMODEL_EORA_SHARED_COVARIANCE", "1" if shared else "0")
    qcfg = QuantizeConfig(bits=4, group_size=8, adapter=Lora(rank=2, eora_config=EoRAConfig()))
    processor = EoraProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    processor.num_batches = 3

    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
    subset = {}
    for index, name in enumerate(names):
        module = torch.nn.Linear(8, 4 + index, bias=False).eval()
        named = NamedModule(
            module,
            name=name,
            full_name=f"model.layers.0.{name}",
            layer_index=0,
        )
        processor.preprocess(named)
        subset[name] = named
    processor.prepare_subset(subset, subset_index=0, subset_total=1)
    return processor, subset, names


def _capture_covariances(processor, subset, names, batches):
    processor._mask_tls = threading.local()
    for batch_index, (inputs, keep_mask) in enumerate(batches):
        processor._set_current_batch_index(batch_index)
        processor._mask_tls.value = keep_mask
        for name in names:
            module = subset[name].module
            processor.pre_process_fwd_hook(name)(module, (inputs,), module(inputs))

    covariances = {
        name: processor._finalize_eigen_scaling_matrix(name).detach().clone()
        for name in names
    }
    stats = processor.shared_covariance_stats()
    processor.cleanup_subset(subset, subset_index=0, subset_total=1)
    return covariances, stats


def test_eora_same_input_modules_share_covariance_math_exactly(monkeypatch):
    torch.manual_seed(0)
    batches = []
    for _ in range(3):
        inputs = torch.randn(2, 5, 8)
        keep_mask = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]],
            dtype=torch.bool,
        )
        batches.append((inputs, keep_mask))

    shared_processor, shared_subset, names = _make_eora_processor(monkeypatch, shared=True)
    shared_covariances, shared_stats = _capture_covariances(
        shared_processor,
        shared_subset,
        names,
        batches,
    )

    isolated_processor, isolated_subset, _ = _make_eora_processor(monkeypatch, shared=False)
    isolated_covariances, isolated_stats = _capture_covariances(
        isolated_processor,
        isolated_subset,
        names,
        batches,
    )

    assert shared_stats == {
        "batch_requests": len(names) * len(batches),
        "batch_hits": (len(names) - 1) * len(batches),
        "batch_misses": len(batches),
    }
    assert isolated_stats == {
        "batch_requests": 0,
        "batch_hits": 0,
        "batch_misses": 0,
    }
    for name in names:
        assert torch.equal(shared_covariances[name], isolated_covariances[name])


def test_eora_covariance_reuse_requires_the_same_activation_view(monkeypatch):
    processor, subset, names = _make_eora_processor(monkeypatch, shared=True)
    processor.num_batches = 1
    processor._mask_tls = threading.local()
    processor._mask_tls.value = None
    processor._set_current_batch_index(0)

    for index, name in enumerate(names):
        inputs = torch.full((1, 3, 8), float(index + 1))
        module = subset[name].module
        processor.pre_process_fwd_hook(name)(module, (inputs,), module(inputs))

    covariances = {
        name: processor._finalize_eigen_scaling_matrix(name).detach().clone()
        for name in names
    }
    stats = processor.shared_covariance_stats()
    processor.cleanup_subset(subset, subset_index=0, subset_total=1)

    assert stats == {
        "batch_requests": len(names),
        "batch_hits": 0,
        "batch_misses": len(names),
    }
    assert not torch.equal(covariances[names[0]], covariances[names[1]])
    assert not torch.equal(covariances[names[1]], covariances[names[2]])


def test_eora_randomized_svd_is_deterministic_and_preserves_objective():
    torch.manual_seed(1)
    rows = 96
    columns = 64
    rank = 8
    signal_left = torch.randn(rows, rank)
    signal_right = torch.randn(rank, columns)
    matrix = signal_left @ signal_right + 0.02 * torch.randn(rows, columns)

    exact_u, exact_s, exact_vh = torch.linalg.svd(matrix, full_matrices=False)
    lowrank_first = eora_module._eora_randomized_svd(matrix, rank)
    lowrank_second = eora_module._eora_randomized_svd(matrix, rank)

    for first, second in zip(lowrank_first, lowrank_second):
        assert torch.equal(first, second)

    exact = exact_u[:, :rank] @ torch.diag(exact_s[:rank]) @ exact_vh[:rank]
    lowrank_u, lowrank_s, lowrank_vh = lowrank_first
    approximate = lowrank_u[:, :rank] @ torch.diag(lowrank_s[:rank]) @ lowrank_vh[:rank]
    exact_residual = torch.linalg.matrix_norm(matrix - exact)
    approximate_residual = torch.linalg.matrix_norm(matrix - approximate)
    assert approximate_residual <= exact_residual * 1.001


def test_eora_gesvda_failure_falls_back_to_exact(monkeypatch):
    matrix = torch.randn(12, 8)
    original_svd = torch.linalg.svd
    calls = []

    def fake_svd(input, *, full_matrices, driver=None):
        calls.append(driver)
        if driver == "gesvda":
            raise RuntimeError("synthetic convergence failure")
        return original_svd(input, full_matrices=full_matrices)

    monkeypatch.setattr(eora_module, "_eora_gesvda_supported", lambda _matrix: True)
    monkeypatch.setattr(torch.linalg, "svd", fake_svd)

    actual = eora_module._eora_compute_svd(matrix, rank=3, algo="auto")
    expected = original_svd(matrix, full_matrices=False)

    assert calls == ["gesvda", None]
    torch.testing.assert_close(actual[1], expected[1])


def test_eora_svd_rejects_unknown_algo():
    with pytest.raises(ValueError, match="EoRA SVD algo"):
        eora_module._eora_compute_svd(torch.eye(4), rank=2, algo="turbo")
