# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Synthetic algebra/serialization checks; not evidence of model quality."""

import copy
from dataclasses import replace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight
from gptqmodel.quantization.qvq_rank8 import (
    P32WindowConfig,
    _rank8_output_fit,
    add_rank8_correction,
    export_window_package,
    fit_rank8,
    load_window_package,
    prepare_rank8,
    qvq_p32_window_linear,
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
        "input_fused",
        "tensor_core",
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
    return fit_rank8(
        layer,
        teacher,
        train,
        heldout,
        train_document_ids=["train"],
        heldout_document_ids=["heldout"],
        **kwargs,
    )


@pytest.mark.parametrize("hadamard", [False, True])
def test_recovery_teacher_error_roundtrip_and_off_identity(hadamard, tmp_path):
    layer, teacher, train, heldout = fixture(hadamard)
    original = layer(heldout)
    report = fit(layer, teacher, train, heldout)
    assert report["validated"]
    assert set(report["candidates"]) == {"output_l2", "tail_weighted_output_l2"}
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
@pytest.mark.parametrize("projection", ["separate_reference", "input_fused", "tensor_core"])
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
    assert children[0]._gptqmodel_qvq_grouped_runtime.telemetry.grouped_launches > 0
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
    calibration = Rank8Calibration(train, heldout, ("train",), ("heldout",))
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
