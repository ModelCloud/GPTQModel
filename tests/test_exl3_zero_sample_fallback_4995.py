# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4995: a routed EXL3 expert that captured zero tokens must not abort the run.

The native EXL3 trellis extension cannot be built in this image (the CUDA
toolkit headers and the bundled ``nvvm``/``cicc`` disagree), so the end-to-end
test stubs ``quantize_exl3`` and asserts the processor hands it the data-free
proxy Hessian with ``count=0`` -- EXL3's own uncalibrated ``fallback_quant``
path, which is the RTN-equivalent grid.
"""

from __future__ import annotations

import threading

import pytest
import torch
from torch import nn

import gptqmodel.looper.exllamav3_processor as processor_module
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
    EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT,
    finalize_capture_H,
)
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import EXL3Config, Fallback, FallbackStrategy
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.exl3_projection_checkpoint import build_projection_request
from gptqmodel.utils.exl3_remote import (
    EXL3_HESSIAN_CAPTURE_CONTRACT,
    validate_exl3_hessian_metrics,
    validate_remote_inputs,
)


WIDTH = 128
EXPECTED_TOKENS = 1_000_000
EXPERT_FULL_NAME = "model.layers.0.mlp.experts.10.down_proj"


def _named(
    full_name: str = EXPERT_FULL_NAME,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> NamedModule:
    linear = nn.Linear(WIDTH, WIDTH, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(WIDTH, WIDTH) * 0.02)
    return NamedModule(
        linear,
        full_name.rsplit(".", 1)[-1],
        full_name,
        0,
    )


def _capture(
    rows: int = 0,
    *,
    module: NamedModule | None = None,
    expected_nsamples: int | None = EXPECTED_TOKENS,
) -> tuple[GPTQ, NamedModule]:
    module = module or _named()
    capture = GPTQ(module)
    capture.expected_nsamples = expected_nsamples
    capture.quantizer.configure(perchannel=True)
    if rows:
        capture.add_batch(torch.randn(rows, WIDTH), None)
    return capture, module


def _starved_capture() -> tuple[GPTQ, NamedModule]:
    """A capture like an expert the calibration router never selected."""

    return _capture(0)


def _task_entry(
    capture: GPTQ,
    *,
    fallback: Fallback | None = None,
    qcfg: EXL3Config | None = None,
) -> dict[str, object]:
    return {
        "capture": capture,
        "qcfg": qcfg or EXL3Config(bits=4),
        "fallback": fallback,
    }


class TestResolver:
    def test_zero_samples_resolves_to_data_free_even_with_fallback_disabled(self):
        capture, _ = _starved_capture()

        assert (
            processor_module.resolve_exl3_hessian_fallback(
                capture,
                fallback=Fallback(),
                expected_nsamples=EXPECTED_TOKENS,
            )
            == "zero_samples"
        )
        # GPTQ forces RTN for a zero-sample capture even when the threshold
        # fallback is disabled: a zero Hessian has no usable curvature.
        assert (
            processor_module.resolve_exl3_hessian_fallback(
                capture,
                fallback=None,
                expected_nsamples=EXPECTED_TOKENS,
            )
            == "zero_samples"
        )

    def test_below_threshold_resolves_to_data_free(self):
        capture, _ = _capture(rows=4)

        assert (
            processor_module.resolve_exl3_hessian_fallback(
                capture,
                fallback=Fallback(threshold="0.5%"),
                expected_nsamples=EXPECTED_TOKENS,
            )
            == "below_threshold"
        )

    def test_calibrated_capture_needs_no_fallback(self):
        capture, _ = _capture(rows=8, expected_nsamples=8)

        assert (
            processor_module.resolve_exl3_hessian_fallback(
                capture,
                fallback=Fallback(threshold="0.5%"),
                expected_nsamples=8,
            )
            is None
        )

    def test_disabled_fallback_keeps_a_calibrated_capture(self):
        capture, _ = _capture(rows=8)

        assert processor_module.resolve_exl3_hessian_fallback(capture, fallback=None) is None


class TestPrepareHessian:
    def test_zero_sample_capture_gets_identity_proxy(self):
        capture, module = _starved_capture()

        hessian = processor_module.prepare_exl3_hessian(
            capture,
            target_device=torch.device("cpu"),
            module_full_name=module.full_name,
            fallback=Fallback(),
            expected_nsamples=EXPECTED_TOKENS,
        )

        assert hessian.dtype is torch.float32
        assert torch.equal(hessian, torch.eye(WIDTH))

    def test_below_threshold_capture_gets_identity_and_releases_partials(self):
        capture, module = _capture(rows=4)
        assert len(capture._device_hessian_partials) == 1

        hessian = processor_module.prepare_exl3_hessian(
            capture,
            target_device=torch.device("cpu"),
            module_full_name=module.full_name,
            fallback=Fallback(threshold="0.5%"),
            expected_nsamples=EXPECTED_TOKENS,
        )

        assert torch.equal(hessian, torch.eye(WIDTH))
        # The proxy is never read against the capture, so the fp32 partials
        # must not stay resident through the EXL3 solve.
        assert capture._device_hessian_partials == {}
        assert capture._device_sample_counts == {}
        assert capture._hessian_dirty is False

    def test_calibrated_capture_keeps_raw_xtx(self):
        rows = torch.randn(8, WIDTH)
        module = _named()
        capture = GPTQ(module)
        capture.quantizer.configure(perchannel=True)
        capture.add_batch(rows, None)

        hessian = processor_module.prepare_exl3_hessian(
            capture,
            target_device=torch.device("cpu"),
            module_full_name=module.full_name,
            fallback=Fallback(),
            expected_nsamples=8,
        )

        # EXL3 receives the raw X^T X sum, unchanged by the fallback work.
        assert torch.equal(hessian, rows.T @ rows)
        assert hessian.data_ptr() == capture.H.data_ptr()

    def test_count_zero_selects_exl3_uncalibrated_path(self):
        # ``finalize_capture_H`` is the EXL3-side switch the processor relies
        # on: count=0 -> q_fallback=True, no LDL factor (RTN-equivalent grid).
        h_data = {
            "H": torch.eye(WIDTH, dtype=torch.float32),
            "count": 0,
            "finalized": False,
        }

        q_fallback, _, L, _, diag = finalize_capture_H(
            h_data,
            {"sigma_reg": 0.025},
            False,
        )

        assert q_fallback is True
        assert L is None
        assert h_data["finalized"] is True
        assert torch.allclose(
            diag,
            torch.full((WIDTH,), 1.025),
            atol=1e-6,
        )


class TestProcessorWiring:
    def test_preprocess_keeps_the_configured_fallback(self):
        processor = processor_module.EXL3Processor.__new__(processor_module.EXL3Processor)
        processor.qcfg = EXL3Config(
            bits=4,
            fallback=Fallback(strategy=FallbackStrategy.RTN, threshold="2%"),
        )
        processor.tasks = {}
        module = _named("model.layers.0.mlp.down_proj")

        processor.preprocess(module)

        stored = processor.tasks[module.name]["fallback"]
        assert isinstance(stored, Fallback)
        assert stored.strategy is FallbackStrategy.RTN
        assert stored.threshold == "2%"

    def test_preprocess_explicit_fallback_wins(self):
        processor = processor_module.EXL3Processor.__new__(processor_module.EXL3Processor)
        processor.qcfg = EXL3Config(bits=4)
        processor.tasks = {}
        module = _named("model.layers.0.mlp.down_proj")
        override = Fallback(strategy=FallbackStrategy.RTN, threshold="9%")

        processor.preprocess(module, fallback=override)

        assert processor.tasks[module.name]["fallback"] is override

    def test_starved_module_uses_identity_proxy_and_zero_count(self, monkeypatch):
        processor = processor_module.EXL3Processor.__new__(processor_module.EXL3Processor)
        capture, module = _starved_capture()
        warnings: list[str] = []
        monkeypatch.setattr(
            processor_module.log,
            "warn",
            lambda message, *args, **kwargs: warnings.append(message % args),
        )
        entry = _task_entry(capture, fallback=Fallback())

        hessian, reason = processor._hessian_for_module(
            module,
            capture,
            entry,
            torch.device("cpu"),
        )

        assert reason == "zero_samples"
        assert torch.equal(hessian, torch.eye(WIDTH))
        assert processor_module.exl3_hessian_count(capture, reason) == 0
        assert any("no calibration samples reached this module" in line for line in warnings)

    def test_calibrated_module_keeps_its_count_and_logs_nothing(self, monkeypatch):
        processor = processor_module.EXL3Processor.__new__(processor_module.EXL3Processor)
        capture, module = _capture(rows=8, expected_nsamples=8)
        warnings: list[str] = []
        monkeypatch.setattr(
            processor_module.log,
            "warn",
            lambda message, *args, **kwargs: warnings.append(message % args),
        )
        entry = _task_entry(capture, fallback=Fallback(threshold="0.5%"))

        hessian, reason = processor._hessian_for_module(
            module,
            capture,
            entry,
            torch.device("cpu"),
        )

        assert reason is None
        assert hessian.data_ptr() == capture.H.data_ptr()
        assert processor_module.exl3_hessian_count(capture, reason) == 8
        assert warnings == []

    def test_tiny_moe_one_unrouted_expert_is_quantized_data_free(self):
        # Three routed experts of one MoE layer; the calibration router never
        # selects expert 0, which is exactly the Qwen3-30B-A3B failure (#4995).
        processor = processor_module.EXL3Processor.__new__(processor_module.EXL3Processor)
        fallback = Fallback(strategy=FallbackStrategy.RTN, threshold="0.5%")
        outcomes = {}
        for expert in range(3):
            module = _named(f"model.layers.0.mlp.experts.{expert}.down_proj")
            capture = GPTQ(module)
            capture.expected_nsamples = 1000
            capture.quantizer.configure(perchannel=True)
            if expert:
                capture.add_batch(torch.randn(16, WIDTH), None)
            hessian, reason = processor._hessian_for_module(
                module,
                capture,
                _task_entry(capture, fallback=fallback),
                torch.device("cpu"),
            )
            outcomes[expert] = (reason, processor_module.exl3_hessian_count(capture, reason), hessian)

        assert outcomes[0][0] == "zero_samples"
        assert outcomes[0][1] == 0
        assert torch.equal(outcomes[0][2], torch.eye(WIDTH))
        for expert in (1, 2):
            assert outcomes[expert][0] is None
            assert outcomes[expert][1] == 16
            assert outcomes[expert][2] is not None


def _uncalibrated_metrics() -> dict[str, object]:
    """The metric contract EXL3 emits for its uncalibrated fallback path."""

    return {
        "schema": "gptqmodel.exl3-trellis-error",
        "schema_version": 1,
        "quantizer_path": "fallback",
        "reported_metric_kind": "unweighted_reconstruction_mse",
        "reported_metric_value": 0.01,
        "hessian_domain": "regularized_exl3_search_space",
        "hessian_sample_count": 0,
        "hessian_regularization_sigma": 0.025,
        "hessian_numerical_contract": EXL3_HESSIAN_NUMERICAL_CONTRACT,
        "hessian_transform_compute_dtype": "torch.float64",
        "hessian_storage_dtype": "torch.float32",
        "hessian_regularization_placement": "before-fp64-congruence",
        "hessian_regularization_diagonal_addend": 0.025,
        "hessian_symmetry_restoration": EXL3_HESSIAN_SYMMETRY_CONTRACT,
        "hessian_symmetry_correction_max_abs": 0.0,
        "hessian_weighted_error_numerator": None,
        "hessian_weighted_reference_denominator": None,
        "hessian_weighted_relative_error": None,
        "hessian_metric_status": "not_applicable_fallback",
        "selected_global_scale": 0.5,
        "scale_search_mse": 0.01,
        "apply_out_scales": False,
        "reconstruction": {
            "domain": "regularized_exl3_search_space",
            "shape": [WIDTH, WIDTH],
            "element_count": WIDTH * WIDTH,
            "error_sum_sq": 1.0,
            "reference_sum_sq": 2.0,
            "mse": 0.01,
            "nmse": 0.5,
            "relative_frobenius": 0.7071067811865476,
            "mean_abs_error": 0.05,
            "max_abs_error": 0.1,
            "reference_finite": True,
            "error_finite": True,
            "tile_shape": [16, 16],
            "tile_count": 64,
            "tile_sse_sum": 1.0,
            "tile_sse_max": 0.1,
            "tile_sse_percentiles": {
                "p50": 0.01,
                "p90": 0.02,
                "p99": 0.03,
                "p99_9": 0.04,
            },
            "worst_tiles": [],
        },
    }


def _packed_tensors(weight: torch.Tensor, bits: int) -> dict[str, torch.Tensor]:
    in_features, stored_outputs = weight.shape
    device = weight.device
    return {
        "suh": torch.ones(in_features, dtype=torch.float16, device=device),
        "svh": torch.ones(stored_outputs, dtype=torch.float16, device=device),
        "trellis": torch.zeros(
            (in_features // 16, stored_outputs // 16, bits * 16),
            dtype=torch.int16,
            device=device,
        ),
        "mcg": torch.zeros(1, dtype=torch.int32, device=device),
    }


def _process_processor(journal_path) -> processor_module.EXL3Processor:
    processor = processor_module.EXL3Processor.__new__(processor_module.EXL3Processor)
    processor.qcfg = EXL3Config(bits=4)
    processor.tasks = {}
    processor.lm_head_name = "lm_head"
    processor.error_journal_path = str(journal_path)
    processor._stats_lock = threading.Lock()
    processor.durations = []
    processor.avg_losses = []
    processor.module_names = []
    processor.log = []
    processor.draw_progress = lambda *args, **kwargs: None
    processor.formatted_fwd_time = lambda: "0.000"
    processor.device_memory_report = lambda: "test"
    processor.log_new_row = lambda *args, **kwargs: None
    processor._remote_client_for_run = lambda _provenance: None
    return processor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="EXL3 requires CUDA")
def test_unrouted_expert_is_packed_end_to_end(tmp_path, monkeypatch):
    """The full processor path packs a zero-token expert instead of aborting."""

    device = torch.device("cuda:0")
    module = _named(EXPERT_FULL_NAME, device=device, dtype=torch.bfloat16)
    capture = GPTQ(module)
    capture.expected_nsamples = EXPECTED_TOKENS
    capture.quantizer.configure(perchannel=True)

    processor = _process_processor(tmp_path / "error-journal.jsonl")
    processor.tasks[module.name] = _task_entry(capture, fallback=Fallback())

    observed: dict[str, object] = {}

    def fake_quantize_exl3(*, weight, H_data, quant_args, return_weight_q):
        observed["count"] = H_data["count"]
        observed["hessian"] = H_data["H"].detach().clone()
        observed["bits"] = quant_args["K"]
        quant_args["error_metrics"] = _uncalibrated_metrics()
        quant_args["q_fallback"] = True
        return None, 0.01, _packed_tensors(weight, int(quant_args["K"]))

    monkeypatch.setattr(processor_module, "quantize_exl3", fake_quantize_exl3)

    stat = processor.process(module)

    # EXL3 was asked for the uncalibrated grid against the identity proxy.
    assert observed["count"] == 0
    assert torch.equal(observed["hessian"], torch.eye(WIDTH, device=device))
    # The starved expert is packed and staged, not dropped or aborted.
    assert module.module.weight.is_meta
    assert module.state["trellis"].shape == (WIDTH // 16, WIDTH // 16, observed["bits"] * 16)
    assert stat["exl3_error_ledger_record"]["quantizer_metrics"]["quantizer_path"] == "fallback"
    assert processor.log[-1]["exl3_projection_checkpoint_hit"] is False


def _remote_contract() -> dict[str, object]:
    return {
        "bits": 4,
        "codebook": "mcg",
        "hessian_capture": EXL3_HESSIAN_CAPTURE_CONTRACT,
        "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
        "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
        "apply_out_scales": None,
        "sigma_reg": 0.025,
        "seed": 787,
    }


def _remote_request(*, sample_count: int, uncalibrated: bool):
    weight = torch.randn(WIDTH, WIDTH) * 0.02
    hessian = torch.eye(WIDTH)
    request = build_projection_request(
        module_full_name=EXPERT_FULL_NAME,
        layer_index=0,
        input_weight=weight,
        hessian=hessian,
        sample_count=sample_count,
        quantizer_contract=_remote_contract(),
        family_join=None,
        route_evidence=None,
        uncalibrated=uncalibrated,
    )
    return request, {"input_weight": weight, "hessian": hessian}


class TestRemoteContract:
    def test_uncalibrated_request_may_carry_no_samples(self):
        request, tensors = _remote_request(sample_count=0, uncalibrated=True)

        assert request["uncalibrated"] is True
        validate_remote_inputs(request, tensors)

    def test_zero_samples_without_the_flag_is_rejected(self):
        request, tensors = _remote_request(sample_count=0, uncalibrated=False)

        assert "uncalibrated" not in request
        with pytest.raises(ValueError, match="contract is invalid"):
            validate_remote_inputs(request, tensors)

    def test_uncalibrated_flag_changes_the_request_identity(self):
        calibrated, _ = _remote_request(sample_count=4, uncalibrated=False)
        uncalibrated, _ = _remote_request(sample_count=4, uncalibrated=True)

        assert calibrated["request_sha256"] != uncalibrated["request_sha256"]

    def test_metrics_accept_the_uncalibrated_path_only_when_expected(self):
        metrics = _uncalibrated_metrics()

        validate_exl3_hessian_metrics(
            metrics,
            sample_count=0,
            sigma_reg=0.025,
            uncalibrated=True,
        )
        # Below-threshold captures keep a nonzero capture count but still hand
        # EXL3 count=0, so the metric contract binds the effective count.
        validate_exl3_hessian_metrics(
            metrics,
            sample_count=4,
            sigma_reg=0.025,
            uncalibrated=True,
        )
        with pytest.raises(RuntimeError, match="numerical contract"):
            validate_exl3_hessian_metrics(
                metrics,
                sample_count=0,
                sigma_reg=0.025,
                uncalibrated=False,
            )
