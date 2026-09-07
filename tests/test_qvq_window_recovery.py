# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Synthetic algebra/serialization checks; not evidence of model quality."""

import copy
import json
from dataclasses import replace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq import (
    reconstruct_qvq_inner_weight,
    repack_p32_planar_to_window,
)
from gptqmodel.quantization.qvq_rank8 import (
    P32WindowConfig,
    _metrics,
    _rank8_output_fit,
    _window_artifact_binding_digest,
    add_rank8_correction,
    apply_rank8_audit,
    export_window_package,
    fit_rank8,
    fit_rank_candidates,
    grouped_window_kernel_candidates,
    load_window_artifact,
    load_window_package,
    prepare_rank8,
    qvq_p32_window_linear,
    save_window_artifact,
    window_kernel_candidates,
    window_package_storage,
)
from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU


def fixture(hadamard=True):
    torch.manual_seed(74)
    k, n = 32, 16
    layer = QVQLinear(
        bits=2,
        in_features=k,
        out_features=n,
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=hadamard,
        output_hadamard=hadamard,
    ).eval()
    layer.trellis.random_(-2147483648, 2147483647)
    layer.SV.copy_(torch.linspace(0.2, 1.5, n))
    layer.SU.copy_(torch.linspace(0.5, 1.2, k))
    weight = reconstruct_qvq_inner_weight(
        layer.trellis,
        bits=2,
        in_features=k,
        out_features=n,
        bank_ids=layer.bank_ids,
        bank_alt_id=layer.bank_alt_id,
        v2b2_p32=True,
    )
    # Construct an exact low-rank teacher difference in transformed coordinates.
    a = torch.randn(k, 4) * 0.05
    b = torch.randn(4, n) * 0.05
    teacher = torch.nn.Linear(k, n, bias=False).eval()
    eye = torch.eye(k)
    xp = layer.transform_input(eye)
    y = xp @ (weight + a @ b)
    if hadamard:
        y = matmul_hadU(y)
    with torch.no_grad():
        teacher.weight.copy_((y * layer.SV).T)
    train, heldout = torch.randn(80, k), torch.randn(40, k)
    return layer, teacher, train, heldout


def test_window_only_constructor_accepts_payload_without_planar_trellis():
    source, _, _, _ = fixture(hadamard=False)
    window = repack_p32_planar_to_window(source.trellis, bits=source.bits)
    layer = QVQLinear(
        bits=source.bits,
        in_features=source.in_features,
        out_features=source.out_features,
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=False,
        output_hadamard=False,
        window_only=True,
        tensors={
            "window_words": window,
            "SU": source.SU,
            "SV": source.SV,
            "bank_ids": source.bank_ids,
            "bank_alt_id": source.bank_alt_id,
        },
    ).eval()
    assert layer.trellis is None
    assert layer.window_words is not None


def test_rank8_audit_gate_requires_every_document_and_records_confirmation():
    layer, teacher, train, heldout = fixture(hadamard=False)
    report = fit_rank8(
        layer,
        teacher,
        train,
        heldout,
        train_document_ids=["train"],
        heldout_document_ids=["heldout"],
        max_solver_bytes=1,
        minimum_improvement=0.0,
    )
    assert report["validated"]
    assert report["arithmetic_signature"] == "reference_fp32_v1"
    accepted = apply_rank8_audit(
        layer,
        [
            {
                "document_id": "audit-1",
                "rows": 2,
                "baseline": {"mse": 2.0, "mae": 1.0, "max": 2.0, "tail": 1.5},
                "recovered": {"mse": 1.0, "mae": 0.8, "max": 1.8, "tail": 1.4},
            },
            {
                "document_id": "audit-2",
                "rows": 2,
                "baseline": {"mse": 2.0, "mae": 1.0, "max": 2.0, "tail": 1.5},
                "recovered": {"mse": 1.5, "mae": 0.8, "max": 1.8, "tail": 1.4},
            },
        ],
    )
    assert accepted["accepted"]
    assert accepted["fit_validated"]
    assert json.loads(bytes(layer.rank8_metadata.tolist()).decode())["audit_validated"]


def test_rank8_audit_rejection_rolls_back_registered_factors():
    layer, teacher, train, heldout = fixture(hadamard=False)
    fit_rank8(
        layer,
        teacher,
        train,
        heldout,
        train_document_ids=["train"],
        heldout_document_ids=["heldout"],
        max_solver_bytes=1,
        minimum_improvement=0.0,
    )
    rejected = apply_rank8_audit(
        layer,
        [
            {
                "document_id": "audit-1",
                "rows": 2,
                "baseline": {"mse": 1.0, "mae": 1.0, "max": 2.0, "tail": 1.5},
                "recovered": {"mse": 1.1, "mae": 1.0, "max": 2.0, "tail": 1.6},
            }
        ],
    )
    assert not rejected["accepted"]
    assert layer.rank8_A is None and layer.rank8_B is None
    assert layer.rank8_metadata is None


def test_rank8_export_requires_independent_audit_confirmation():
    layer, teacher, train, heldout = fixture(hadamard=False)
    fit_rank8(
        layer, teacher, train, heldout,
        train_document_ids=("train",), heldout_document_ids=("heldout",),
    )
    with pytest.raises(ValueError, match="independent audit"):
        export_window_package(layer)


def test_rank8_large_output_uses_bounded_randomized_solver():
    torch.manual_seed(9)
    x = torch.randn(24, 32, dtype=torch.float64)
    residual = torch.randn(24, 64, dtype=torch.float64)
    weights = torch.ones((24, 1), dtype=torch.float64)
    a, b, mode = _rank8_output_fit(
        x,
        residual,
        weights,
        max_solver_bytes=1,
        rcond=1e-5,
        seed=123,
    )
    again_a, again_b, again_mode = _rank8_output_fit(
        x,
        residual,
        weights,
        max_solver_bytes=1,
        rcond=1e-5,
        seed=123,
    )
    assert mode == again_mode == "randomized_output_range"
    assert a.shape == (32, 8) and b.shape == (8, 64)
    torch.testing.assert_close(a, again_a, rtol=0, atol=0)
    torch.testing.assert_close(b, again_b, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rank8_fp32_factor_cache_is_prepared_and_versioned():
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", in_features=256, out_features=256, device="cuda").eval()
    layer.rank8_A = torch.randn(256, 8, device="cuda", dtype=torch.float16)
    layer.rank8_B = torch.randn(8, 256, device="cuda", dtype=torch.float16)
    first_a = layer._cached_rank8_factor("A")
    second_a = layer._cached_rank8_factor("A")
    assert first_a.dtype == torch.float32
    assert first_a.is_contiguous()
    assert first_a.data_ptr() == second_a.data_ptr()
    layer.rank8_A = layer.rank8_A.clone()
    refreshed_a = layer._cached_rank8_factor("A")
    assert refreshed_a.data_ptr() != first_a.data_ptr()


def test_rank8_fp32_factor_cache_accepts_inference_mode_tensors():
    """Inference tensors have no version counter but remain graph-safe constants."""
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", in_features=32, out_features=32).eval()
    with torch.inference_mode():
        layer.rank8_A = torch.randn(32, 8, dtype=torch.float16)
        layer.rank8_B = torch.randn(8, 32, dtype=torch.float16)
    prepared = layer._cached_rank8_factor("A")
    assert prepared.dtype == torch.float32
    assert prepared.is_contiguous()
    assert prepared.data_ptr() == layer._cached_rank8_factor("A").data_ptr()


def test_rank8_fp32_factor_cache_invalidates_on_in_place_mutation():
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", in_features=32, out_features=32).eval()
    layer.rank8_A = torch.randn(32, 8, dtype=torch.float16)
    first = layer._cached_rank8_factor("A")
    with torch.no_grad():
        layer.rank8_A.add_(1)
    refreshed = layer._cached_rank8_factor("A")
    assert refreshed.data_ptr() != first.data_ptr()


def test_qvq_auxiliary_cache_accepts_inference_mode_buffer():
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", in_features=32, out_features=32).eval()
    with torch.inference_mode():
        layer.SU = torch.ones(32)
    prepared = layer._cached_cast("SU", torch.float16)
    assert prepared.dtype == torch.float16
    assert prepared.data_ptr() == layer._cached_cast("SU", torch.float16).data_ptr()


def test_rank_candidate_sweep_uses_predictable_output_fit_and_reports_all_ranks():
    torch.manual_seed(1412)
    x = torch.randn((48, 32), dtype=torch.float64)
    a_true = torch.randn((32, 12), dtype=torch.float64)
    b_true = torch.randn((12, 16), dtype=torch.float64)
    residual = x @ a_true @ b_true
    result = fit_rank_candidates(
        x,
        residual,
        ranks=(2, 4, 6, 8, 12),
        max_solver_bytes=1,
    )
    assert tuple(result) == (2, 4, 6, 8, 12)
    errors = []
    for rank, candidate in result.items():
        assert candidate["A"].shape == (32, rank)
        assert candidate["B"].shape == (rank, 16)
        assert candidate["solver_mode"] == "randomized_output_range"
        errors.append(candidate["weighted_fit"]["mse"])
    assert errors[-1] < errors[0]
    with pytest.raises(ValueError, match="distinct members"):
        fit_rank_candidates(x, residual, ranks=(3,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_m": 64},
        {"block_n": 32},
        {"block_k": 16},
        {"pipeline_stages": 3},
        {"block_m": True},
        {
            "algorithm": "hopper_direct_decode_mma",
            "block_m": 64,
            "block_n": 128,
            "warp_groups": 1,
        },
        {
            "algorithm": "hopper_direct_decode_mma",
            "block_m": 64,
            "block_n": 64,
            "split_k": 2,
        },
    ],
)
def test_window_geometry_rejects_unimplemented_combinations(kwargs):
    with pytest.raises(ValueError):
        P32WindowConfig(**kwargs)


def test_unverified_rank8_arithmetic_rejects_balanced_and_quality_modes():
    layer, _, _, _ = fixture()
    _kernel_rank8(layer)
    for quality_mode in ("balanced", "quality"):
        for signature in (
            "unverified_tensor_core",
            "unverified_project_output_fused",
        ):
            with pytest.raises(ValueError, match="unverified rank8 arithmetic"):
                prepare_rank8(
                    layer,
                    P32WindowConfig(
                        recovery_mode="on",
                        quality_mode=quality_mode,
                        arithmetic_signature=signature,
                    ),
                )


def test_fully_fused_rank8_mode_requires_its_project_output_contract():
    with pytest.raises(ValueError, match="fully_fused rank8 requires"):
        P32WindowConfig(recovery_kernel="fully_fused")
    config = P32WindowConfig(
        recovery_mode="on",
        recovery_kernel="fully_fused",
        recovery_projection="project_output_fused",
        arithmetic_signature="unverified_project_output_fused",
    )
    assert P32WindowConfig.from_backend_config(config.to_backend_config()) == config


def test_ampere_window_candidates_are_explicit_and_shape_specific(monkeypatch):
    layer, _, _, _ = fixture(hadamard=False)
    prepare_rank8(layer, P32WindowConfig())
    monkeypatch.setattr(layer, "runtime_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type(
            "Props",
            (),
            {"major": 8, "minor": 0, "multi_processor_count": 108, "name": "NVIDIA A100"},
        )(),
    )
    candidates = window_kernel_candidates(layer, m=1)
    assert candidates
    assert all(candidate.algorithm == "ampere_window" for candidate in candidates)
    assert [candidate.split_k for candidate in candidates] == list(
        dict.fromkeys(candidate.split_k for candidate in candidates)
    )
    assert candidates[0].min_m == candidates[0].max_m == 1
    assert P32WindowConfig.from_backend_config(
        candidates[0].to_backend_config()
    ) == candidates[0]


def test_gfx950_window_candidates_use_real_launch_controls(monkeypatch):
    """The unified tuner exposes the existing non-unified gfx950 sweep."""

    layer, _, _, _ = fixture(hadamard=False)
    prepare_rank8(layer, P32WindowConfig())
    monkeypatch.setattr(layer, "runtime_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(torch.version, "hip", "6.3", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type(
            "Props", (), {"major": 9, "minor": 4, "name": "AMD MI355X"}
        )(),
    )
    monkeypatch.setattr(
        "gptqmodel.utils.qvq_amd.qvq_p32_amd_supported", lambda device: True
    )
    candidates = window_kernel_candidates(layer, m=128)

    assert candidates
    assert all(candidate.algorithm == "amd_gfx950" for candidate in candidates)
    assert all(candidate.block_n == 64 for candidate in candidates)
    assert all(candidate.block_k in (16, 32, 64) for candidate in candidates)
    assert all(candidate.warp_groups in (4, 8) for candidate in candidates)
    assert all(candidate.pipeline_stages in (1, 2, 3) for candidate in candidates)
    assert P32WindowConfig.from_backend_config(
        candidates[0].to_backend_config()
    ) == candidates[0]


def test_grouped_window_candidates_preserve_child_split_tuples(monkeypatch):
    """The high-level grouped API exposes the SM80 tuple tuner directly."""

    config = P32WindowConfig(quality_mode="fast")

    class Child:
        v2b2_p32 = True
        in_features = 5120
        bits = 3
        codebook_version = "V2"

        def __init__(self, out_features):
            self.out_features = out_features
            self._p32_window_config = config

        @staticmethod
        def runtime_device():
            return torch.device("cuda", 0)

    children = (Child(1024), Child(12288))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type("Properties", (), {"major": 8, "minor": 0, "multi_processor_count": 108})(),
    )
    candidates = grouped_window_kernel_candidates(children, m=1)

    assert len(candidates) >= 2
    assert all(len(candidate) == 2 for candidate in candidates)
    assert all(
        tuple(child.split_k for child in candidate) == expected
        for candidate, expected in zip(
            candidates[:2], ((56, 40), (28, 40)), strict=True
        )
    )
    assert all(
        child.algorithm == "ampere_window" for candidate in candidates for child in candidate
    )


def test_grouped_hopper_candidates_keep_child_local_split_choices(monkeypatch):
    """SM90 grouping exposes shape-valid ordered split tuples for ZML/native tuning."""

    config = P32WindowConfig(
        quality_mode="fast",
        recovery_kernel="fused_epilogue",
        arithmetic_signature="unverified_fused_epilogue",
    )

    class Child:
        v2b2_p32 = True
        in_features = 2048
        bits = 3
        vector_size = 2
        codebook_version = "V2"

        def __init__(self, out_features):
            self.out_features = out_features
            self._p32_window_config = config

        @staticmethod
        def runtime_device():
            return torch.device("cuda", 0)

    children = (Child(2048), Child(512), Child(512))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type(
            "Properties", (), {"major": 9, "minor": 0, "name": "NVIDIA H200"}
        )(),
    )
    candidates = grouped_window_kernel_candidates(children, m=128)

    # The 64 ordered split tuples remain, plus three validated unsplit BM
    # geometry choices for grouped consumers. BN128 is reserved for the
    # single-projection specialization until a generic multi-segment proof is
    # available.
    assert len(candidates) == 67
    assert all(len(candidate) == 3 for candidate in candidates)
    assert all(
        child.algorithm == "hopper_m16"
        and child.min_m == child.max_m == 128
        and child.split_k in (1, 2, 4, 8)
        and child.recovery_kernel == "fused_epilogue"
        and child.arithmetic_signature == "unverified_fused_epilogue"
        for candidate in candidates
        for child in candidate
    )
    assert any(
        tuple(child.split_k for child in candidate) == (1, 2, 4)
        for candidate in candidates
    )
    assert any(
        tuple((child.block_m, child.block_n) for child in candidate)
        == ((64, 64), (64, 64), (64, 64))
        for candidate in candidates
    )


def test_external_window_controls_roundtrip_and_cpu_candidates():
    config = P32WindowConfig(
        algorithm="hopper_direct_decode_mma",
        block_m=64,
        block_n=128,
        warp_groups=2,
        recovery_mode="auto",
        quality_mode="balanced",
    )
    assert P32WindowConfig.from_backend_config(config.to_backend_config()) == config
    with pytest.raises(TypeError):
        P32WindowConfig.from_backend_config({"unknown_tile": 64})
    layer, _, _, _ = fixture()
    prepare_rank8(layer, P32WindowConfig())
    candidates = window_kernel_candidates(layer, m=33)
    assert len(candidates) == 1
    assert candidates[0].algorithm == "production_window"
    assert candidates[0].min_m == candidates[0].max_m == 33
    assert candidates[0].recovery_mode == "off"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_m", [32, 64, 128])
@pytest.mark.parametrize("block_n", [64, 128])
@pytest.mark.parametrize("bits", [2, 2.5, 3, 3.5])
@pytest.mark.parametrize(
    "m,chunk_m",
    [(1, 0), (33, 0), (512, 0), (4097, 0), (4097, 4096), (8192, 0), (8192, 4096)],
)
def test_hopper_explicit_geometry_rank8_matrix(block_m, block_n, bits, m, chunk_m):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    torch.manual_seed(141)
    layer = (
        QVQLinear(
            bits=bits,
            in_features=256,
            out_features=256,
            bank_count=2,
            v2b2_p32=True,
        )
        .to("cuda")
        .eval()
    )
    layer.trellis.random_(-2147483648, 2147483647)
    layer.SU.fill_(1)
    layer.SV.fill_(0.1)
    layer.post_init()
    _kernel_rank8(layer)
    x = torch.randn(m, 256, device="cuda", dtype=torch.float16) * 0.1
    config = P32WindowConfig(
        algorithm="hopper_direct_decode_mma",
        block_m=block_m,
        block_n=block_n,
        chunk_m=chunk_m,
        warp_groups=block_n // 64,
        recovery_kernel="fused_epilogue",
    )
    prepare_rank8(layer, replace(config, recovery_mode="on"))
    candidates = window_kernel_candidates(layer, m=m)
    assert {(c.block_m, c.block_n) for c in candidates if c.block_m} == {
        (bm, bn) for bm in (32, 64, 128) for bn in (64, 128)
    }
    assert all(c.recovery_mode == "on" for c in candidates)
    assert {c.recovery_projection for c in candidates} == {
        "separate_reference",
        "concurrent_reference",
        "input_fused",
        "tensor_core",
        "project_output_fused",
    }
    for mode in ("off", "on"):
        prepare_rank8(
            layer, P32WindowConfig(algorithm="production_window", recovery_mode=mode)
        )
        reference = layer(x)
        prepare_rank8(layer, replace(config, recovery_mode=mode))
        actual = layer(x)
        error = (actual.float() - reference.float()).abs()
        assert torch.isfinite(actual).all()
        assert error.mean() <= 2e-3 and error.max() <= 0.046875
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(x)
        for _ in range(3):
            graph.replay()
            assert torch.equal(captured, actual)


def fit(layer, teacher, train, heldout, **kwargs):
    report = fit_rank8(
        layer,
        teacher,
        train,
        heldout,
        train_document_ids=["train"],
        heldout_document_ids=["heldout"],
        **kwargs,
    )
    # Synthetic fixtures include a third, disjoint audit fold so deployment
    # export tests exercise the same hard promotion boundary as production.
    audit = torch.randn(24, train.shape[1])
    original = getattr(layer, "_p32_window_config", P32WindowConfig())
    prepare_rank8(layer, P32WindowConfig(recovery_mode="off"))
    baseline = teacher(audit).float() - layer(audit).float()
    prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
    recovered = teacher(audit).float() - layer(audit).float()
    apply_rank8_audit(
        layer,
        [{"document_id": "audit", "rows": audit.shape[0],
          "baseline": _metrics(baseline), "recovered": _metrics(recovered)}],
        minimum_improvement=0.0,
    )
    prepare_rank8(layer, original)
    report["audit_validated"] = True
    return report


@pytest.mark.parametrize("hadamard", [False, True])
def test_recovery_teacher_error_roundtrip_and_off_identity(hadamard, tmp_path):
    layer, teacher, train, heldout = fixture(hadamard)
    original = layer(heldout)
    report = fit(
        layer,
        teacher,
        train,
        heldout,
        rank_candidates=(2, 4, 6, 8, 12),
    )
    assert report["validated"]
    assert set(report["candidates"]) == {"output_l2", "tail_weighted_output_l2"}
    assert report["rank_candidates"] == [2, 4, 6, 8, 12]
    assert set(report["rank_sweep"]) == {"output_l2", "tail_weighted_output_l2"}
    assert all(
        set(report["rank_sweep"][objective]) == {"2", "4", "6", "8", "12"}
        for objective in report["rank_sweep"]
    )
    assert torch.equal(layer(heldout), original)
    on = P32WindowConfig(recovery_mode="on")
    actual = qvq_p32_window_linear(layer, heldout, on)
    assert (actual - teacher(heldout)).square().mean() < (
        original - teacher(heldout)
    ).square().mean() * 0.01
    xp = layer.transform_input(heldout)
    inner = layer._inner_forward(xp)
    expected = (
        inner.float()
        + (xp.float() @ layer.rank8_A.float()).half().float() @ layer.rank8_B.float()
    )
    expected = layer._recover_output_compute_dtype(expected, xp.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(layer.forward_pretransformed(xp), actual, rtol=0, atol=0)
    package = export_window_package(layer)
    assert "trellis" not in package["tensors"]
    path = tmp_path / "window.pt"
    torch.save(package, path)
    loaded = load_window_package(torch.load(path, weights_only=True), config=on)
    torch.testing.assert_close(loaded(heldout), actual, rtol=0, atol=0)
    artifact_dir = tmp_path / "window-artifact"
    artifact_report = save_window_artifact(layer, artifact_dir)
    assert artifact_report["serialized_bytes"] > artifact_report["tensor_bytes"]
    artifact_loaded = load_window_artifact(artifact_dir, config=on)
    torch.testing.assert_close(artifact_loaded(heldout), actual, rtol=0, atol=0)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["payload_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="payload binding"):
        load_window_artifact(artifact_dir, config=on)
    manifest["payload_sha256"] = _window_artifact_binding_digest(
        manifest["metadata"], manifest["tensors"]
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (artifact_dir / "window_words.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_window_artifact(artifact_dir, config=on)
    storage = window_package_storage(
        [package], serialized_bytes=path.stat().st_size
    )
    assert storage["serialized_bytes"] == path.stat().st_size
    assert storage["serialized_bpw"] > storage["recovered_average_bpw"]
    assert storage["window_average_bpw"] < storage["recovered_average_bpw"]
    assert path.stat().st_size >= storage["tensor_bytes"]
    record = storage["modules"][0]
    assert record["recovered_bpw"] - record["window_bpw"] == record["rank8_delta_bpw"]
    assert record["selected"]
    assert record["selected_bpw"] == record["recovered_bpw"]
    # Ordinary module state_dict also carries recovery; reload starts off.
    shell, _, _, _ = fixture(hadamard)
    shell.load_state_dict(layer.state_dict())
    assert torch.equal(shell(heldout), original)
    prepare_rank8(shell, on)
    torch.testing.assert_close(shell(heldout), actual, rtol=0, atol=0)
    prepare_rank8(layer, P32WindowConfig())
    layer.rank8_A.fill_(float("nan"))
    assert torch.equal(layer(heldout), original)


def test_off_does_not_access_recovery():
    class Poison:
        _p32_rank8_enabled = False

        def __getattr__(self, name):
            raise AssertionError(name)

    value = torch.ones(1)
    assert add_rank8_correction(Poison(), None, value) is value


def test_forward_pretransformed_propagates_requested_store_dtype():
    layer, _, train, _ = fixture(hadamard=False)
    seen = {}
    original = layer._forward_pretransformed_compute_dtype
    original_recover = layer._recover_output_compute_dtype

    def wrapped(transformed, compute_dtype, *, output_dtype=None, rank8_hidden=None):
        seen["output_dtype"] = output_dtype
        return original(
            transformed,
            compute_dtype,
            output_dtype=output_dtype,
            rank8_hidden=rank8_hidden,
        )

    layer._forward_pretransformed_compute_dtype = wrapped

    def recover(output, compute_dtype, *, target_dtype=None):
        seen["target_dtype"] = target_dtype
        return original_recover(output, compute_dtype, target_dtype=target_dtype)

    layer._recover_output_compute_dtype = recover
    transformed = layer.transform_input(train)
    layer.forward_pretransformed(transformed, output_dtype=torch.float16)
    assert seen["output_dtype"] == torch.float16
    assert seen["target_dtype"] == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_window_package_owns_window_payload_without_planar_vram_copy(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    source = QVQLinear(
        bits=2,
        in_features=256,
        out_features=256,
        bank_count=2,
        v2b2_p32=True,
    ).to("cuda").eval()
    source.trellis.random_(-2147483648, 2147483647)
    source.SU.fill_(1)
    source.SV.fill_(0.1)
    source.post_init()
    package = export_window_package(source)
    import gptqmodel.quantization.qvq_rank8 as qvq_rank8_module

    original_repack = qvq_rank8_module.repack_p32_window_to_planar
    repack_calls = []

    def record_repack(*args, **kwargs):
        repack_calls.append(True)
        return original_repack(*args, **kwargs)

    monkeypatch.setattr(
        qvq_rank8_module, "repack_p32_window_to_planar", record_repack
    )
    loaded = load_window_package(package, device="cuda")
    assert loaded.window_only
    assert loaded.window_words.device.type == "cuda"
    assert loaded.trellis is None
    assert repack_calls == []
    legacy = load_window_package(package, device="cuda", retain_planar=True)
    assert legacy.trellis.device.type == "cpu"
    assert len(repack_calls) == 1
    retained_planar = legacy._prepare_planar_fallback()
    assert retained_planar.device.type == "cuda"
    assert len(repack_calls) == 1
    x = torch.randn(16, 256, device="cuda", dtype=torch.float16) * 0.01
    torch.testing.assert_close(loaded(x), source(x), rtol=0, atol=0)
    # Production ownership must remain window-only at the largest supported
    # prefill shape; this used to fall through to the released planar copy.
    x_large = torch.randn(8192, 256, device="cuda", dtype=torch.float16) * 0.01
    torch.testing.assert_close(loaded(x_large), source(x_large), rtol=0, atol=0)


def test_binding_and_mutation_guard():
    layer, teacher, train, heldout = fixture()
    fit(layer, teacher, train, heldout)
    package = export_window_package(layer)
    for name in ("window_words", "SU", "SV", "bank_ids", "rank8_A", "rank8_B"):
        bad = copy.deepcopy(package)
        bad["tensors"][name].flatten()[0] += 1
        with pytest.raises(ValueError, match="hash mismatch"):
            load_window_package(bad)
    prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
    layer.SV.add_(1)
    with pytest.raises(RuntimeError, match="state changed"):
        layer(heldout)
    with pytest.raises(ValueError, match="base hash mismatch"):
        prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))


@pytest.mark.parametrize("mode", ["fast", "balanced", "quality"])
def test_quality_policy(mode):
    layer, teacher, train, heldout = fixture()
    fit(layer, teacher, train, heldout)
    prepare_rank8(layer, P32WindowConfig(recovery_mode="auto", quality_mode=mode))
    assert layer._p32_rank8_enabled == (mode != "fast")


def test_fit_rank8_supports_window_owned_module():
    layer, teacher, train, heldout = fixture(hadamard=False)
    layer.window_words = repack_p32_planar_to_window(layer.trellis, bits=layer.bits)
    layer.trellis = None
    layer.window_only = True
    report = fit_rank8(
        layer,
        teacher,
        train,
        heldout,
        train_document_ids=["train"],
        heldout_document_ids=["heldout"],
        max_solver_bytes=1,
    )
    assert report["fit_device"] == "cpu"
    assert layer.rank8_metadata is not None
    assert layer.trellis is None


def test_reject_provenance_and_unsupported_contracts():
    layer, teacher, train, heldout = fixture()
    with pytest.raises(ValueError, match="calibration"):
        fit(layer, teacher, train, heldout, source_kind="ARC")
    with pytest.raises(ValueError, match="disjoint"):
        fit_rank8(
            layer,
            teacher,
            train,
            heldout,
            train_document_ids=["same"],
            heldout_document_ids=["same"],
        )
    with pytest.raises(ValueError, match="without validated"):
        prepare_rank8(layer, P32WindowConfig(recovery_mode="on"))
    for kwargs in (
        {"abi_version": 2},
        {"recovery_mode": "yes"},
        {"quality_mode": "best"},
        {"algorithm": "unknown"},
        {"recovery_kernel": "fully_fused"},
    ):
        with pytest.raises(ValueError):
            P32WindowConfig(**kwargs)
    train[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit(layer, teacher, train, heldout)


def test_empty_and_fast_package():
    layer, _, _, _ = fixture()
    package = export_window_package(layer)
    loaded = load_window_package(package)
    assert loaded(torch.empty(0, 32)).shape == (0, 16)
    assert window_package_storage([])["average_bpw"] == 0


def test_storage_reports_selected_recovery_weighted_model_average():
    base_layer, _, _, _ = fixture(False)
    base_package = export_window_package(base_layer)
    fitted_layer, teacher, train, heldout = fixture(False)
    fit(fitted_layer, teacher, train, heldout)
    fitted_package = export_window_package(fitted_layer)

    report = window_package_storage(
        [base_package, fitted_package], serialized_bytes=12345
    )
    assert report["rank8_modules"] == 1
    assert report["selected_rank8_modules"] == 1
    assert report["window_tensor_bytes"] < report["tensor_bytes"]
    assert (
        report["window_average_bpw"]
        < report["selected_average_bpw"]
        == report["recovered_average_bpw"]
    )
    assert report["serialized_bytes"] == 12345
    assert report["serialized_bpw"] == pytest.approx(
        8 * 12345 / report["weights"]
    )


def _kernel_rank8(layer):
    """Known factors for kernel algebra only; no quality claim or fitted artifact."""
    from gptqmodel.quantization.qvq_rank8 import CONTRACT, _base, _digest, _encode

    layer.rank8_A = (
        torch.randn(layer.in_features, 8, device=layer.trellis.device).half() * 0.05
    )
    layer.rank8_B = (
        torch.randn(8, layer.out_features, device=layer.trellis.device).half() * 0.05
    )
    tensors, metadata = _base(layer)
    layer.rank8_metadata = _encode(
        {
            "base_hash": _digest(tensors, metadata),
            "fit_contract": CONTRACT,
            "validated": True,
            "selected": True,
            "audit_validated": True,
            "audit_acceptance": {"accepted": True, "documents": []},
            "factors_hash": _digest({"A": layer.rank8_A, "B": layer.rank8_B}, {}),
        },
        layer.trellis.device,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "algorithm", ["auto", "hopper_m16", "hopper_direct_decode_mma"]
)
@pytest.mark.parametrize("m", [1, 16, 64, 512])
@pytest.mark.parametrize("recovery_kernel", ["separate_reference", "fused_epilogue"])
@pytest.mark.parametrize(
    "projection", ["separate_reference", "concurrent_reference", "input_fused", "tensor_core"]
)
def test_hopper_rank8_eager_graph(algorithm, m, recovery_kernel, projection):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", device="cuda").eval()
    _kernel_rank8(layer)
    x = torch.randn(m, 256, device="cuda", dtype=torch.float16) * 0.01
    off = P32WindowConfig(algorithm=algorithm)
    on = P32WindowConfig(
        algorithm=algorithm,
        recovery_mode="on",
        recovery_kernel=recovery_kernel,
        recovery_projection=projection,
    )
    base = qvq_p32_window_linear(layer, x, off)
    graphs = []
    for config in (off, on):
        prepare_rank8(layer, config)
        for _ in range(3):
            eager = layer(x)
        xp = layer.transform_input(x)
        inner = layer._inner_forward(xp)
        if config.recovery_mode == "on":
            inner = (
                inner.float()
                + (xp.float() @ layer.rank8_A.float()).half().float()
                @ layer.rank8_B.float()
            )
        reference = layer._recover_output_compute_dtype(inner, xp.dtype).half()
        drift = (eager.float() - reference.float()).abs()
        assert drift.mean() <= 2e-3 and drift.max() <= 0.046875
        if config.recovery_mode == "off":
            assert torch.equal(eager, base)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(x)
        for _ in range(10):
            graph.replay()
            assert torch.equal(output, eager)
        graphs.append((graph, output, eager.clone()))
    # Python mode now says on; the off graph must retain its captured policy.
    for graph, output, reference in graphs:
        graph.replay()
        assert torch.equal(output, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [128, 8192])
def test_hopper_rank8_concurrent_producer_is_graph_safe_on_nondefault_stream(m):
    """The shared X' producer must be warmed on the serving stream before capture.

    This exercises the stream/event path used by ``concurrent_reference``
    directly.  A graph captured on the default stream can hide accidental
    stream ownership assumptions, so the contract is checked on a dedicated
    stream and replayed after changing the static input buffer.
    """
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", device="cuda").eval()
    _kernel_rank8(layer)
    config = P32WindowConfig(
        algorithm="hopper_direct_decode_mma",
        block_m=64,
        block_n=64,
        warp_groups=1,
        recovery_mode="on",
        recovery_projection="concurrent_reference",
    )
    stream = torch.cuda.Stream()
    x = torch.randn(m, layer.in_features, device="cuda", dtype=torch.float16) * 0.01
    x_next = torch.randn_like(x)
    with torch.cuda.stream(stream):
        prepare_rank8(layer, config)
        eager = layer(x)
        stream.synchronize()
        device_index = int(x.device.index if x.device.index is not None else 0)
        stream_key = int(stream.cuda_stream)
        assert (device_index, stream_key, m, layer.in_features) in layer._qvq_rank8_concurrent_warm

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = layer(x)
        graph.replay()
        stream.synchronize()
        assert torch.equal(captured, eager)

        x.copy_(x_next)
        expected = layer(x)
        stream.synchronize()
        graph.replay()
        stream.synchronize()
        assert torch.equal(captured, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "roles", [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")]
)
@pytest.mark.parametrize("m", [1, 64])
@pytest.mark.parametrize("recovery_kernel", ["separate_reference", "fused_epilogue"])
@pytest.mark.parametrize("projection", ["separate_reference", "input_fused", "tensor_core", "mixed"])
@pytest.mark.parametrize("k", [256, 2048])
def test_hopper_grouped_rank8_independent_flags(roles, m, recovery_kernel, projection, k):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from test_qvq_grouped_runtime import _child

    from gptqmodel.nn_modules.qvq_grouped_runtime import install_qvq_hopper_groups

    children = [
        _child(role, device="cuda", seed=i + 1, in_features=k).eval() for i, role in enumerate(roles)
    ]
    for child in children:
        child.SU.copy_(children[0].SU)
    for child in children[1::2]:
        # Invalid optional payload must remain inaccessible in the shared off
        # epilogue, including when an enabled sibling uses the same launch.
        child.rank8_A = torch.full((1,), float("nan"), device="cuda")
        child.rank8_B = torch.full((1,), float("nan"), device="cuda")
        prepare_rank8(child, P32WindowConfig(recovery_kernel=recovery_kernel))
    for index, child in enumerate(children[::2]):
        _kernel_rank8(child)
        prepare_rank8(
            child,
            P32WindowConfig(
                recovery_mode="on",
                recovery_kernel=recovery_kernel,
                recovery_projection=(
                    ("input_fused", "tensor_core")[index % 2]
                    if projection == "mixed" else projection
                ),
            ),
        )
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.01
    references = [child(x) for child in children]
    parent = torch.nn.Module()
    for name, child in zip(roles, children):
        parent.add_module(name, child)
    parent.eval()
    assert sum(install_qvq_hopper_groups(parent).values()) == 1
    for _ in range(3):
        outputs = [child(x) for child in children]
    for output, reference in zip(outputs, references):
        delta = (output.float() - reference.float()).abs()
        assert delta.mean() <= 2e-3 and delta.max() <= 0.046875
    telemetry = children[0]._gptqmodel_qvq_grouped_runtime.telemetry
    if projection == "tensor_core" or (projection == "mixed" and len(roles) == 3):
        # Grouped MLP/QKV intentionally fails closed for projection modes that
        # it cannot preserve per child.  The caller then uses the exact child
        # path rather than silently replacing the selected arithmetic policy.
        assert telemetry.grouped_launches == 0
        assert telemetry.plain_fallbacks > 0
    else:
        assert telemetry.grouped_launches > 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [child(x) for child in children]
    for _ in range(10):
        graph.replay()
        for output, reference in zip(captured, outputs):
            assert torch.equal(output, reference)


def test_fit_scores_the_deployed_final_output_dtype():
    from gptqmodel.quantization.qvq_rank8 import _metrics

    layer, teacher, train, heldout = fixture(hadamard=False)
    teacher = teacher.half()
    train, heldout = train.half(), heldout.half()
    with torch.no_grad():
        expected = [_metrics(teacher(x).float() - layer(x).float()) for x in (train, heldout)]
    report = fit_rank8(
        layer, teacher, train, heldout,
        train_document_ids=("train",), heldout_document_ids=("heldout",),
    )
    assert report["baseline"] == expected
    assert report["fit_output_boundary"] == "original_activation_dtype"


def test_quantize_fit_export_is_one_module_job(tmp_path):
    from gptqmodel.quantization.qvq import quantize_qvq_linear
    from gptqmodel.quantization.qvq_rank8 import Rank8Calibration, save_window_package

    _, teacher, train, heldout = fixture()
    audit = torch.randn(24, train.shape[1])
    calibration = Rank8Calibration(
        train, heldout, ("train",), ("heldout",),
        audit_inputs=audit, audit_document_ids=("audit",), audit_row_counts=(24,),
    )
    result = quantize_qvq_linear(
        teacher.weight.detach(),
        train.T @ train / train.shape[0],
        bits=2,
        bank_count=2,
        v2b2_p32=True,
        rank8_calibration=calibration,
    )
    assert result.rank8_fit_report is not None
    assert result.rank8_fit_report["source_kind"] == "calibration"
    assert result.rank8_fit_report["audit_validated"]
    assert result.rank8_fit_report["audit_document_ids"] == ["audit"]
    layer = QVQLinear(
        bits=2,
        in_features=32,
        out_features=16,
        bank_count=2,
        v2b2_p32=True,
        tensors=result.serialized_tensors(),
    ).eval()
    storage = save_window_package(layer, tmp_path / "quantized.pt")
    assert storage["serialized_bytes"] == (tmp_path / "quantized.pt").stat().st_size
    reloaded = load_window_package(
        torch.load(tmp_path / "quantized.pt", weights_only=True)
    )
    assert torch.equal(reloaded(heldout), layer(heldout))
