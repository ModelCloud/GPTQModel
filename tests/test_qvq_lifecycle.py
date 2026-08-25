# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_processor import QVQProcessor, clone_qvq_config_for_module
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import OutputAlignConfig, QVQConfig, YaqaConfig
from gptqmodel.quantization.qvq import quantize_qvq_linear


def _prepared_calibration(**kwargs):
    """Return the tiny already-tokenized calibration fixture unchanged."""

    return kwargs["calibration_dataset"]


def _processor(*, bits=2, dynamic=None, qcfg=None):
    """Build a CPU QVQ processor whose hook receives explicit keep masks."""

    calibration = [
        {
            "input_ids": torch.tensor([[1, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }
    ]
    return QVQProcessor(
        tokenizer=None,
        qcfg=qcfg
        or QVQConfig(
            bits=bits,
            rounding="block_ldlq",
            dynamic=dynamic,
            device="cpu",
            offload_to_disk=False,
        ),
        calibration=calibration,
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )


def test_qvq_yaqa_factor_chunking_preserves_decoder_layer_boundaries():
    layers = torch.nn.ModuleList(
        [
            torch.nn.ModuleDict(
                {
                    "q_proj": torch.nn.Linear(4, 4, bias=False),
                    "o_proj": torch.nn.Linear(4, 4, bias=False),
                }
            )
            for _ in range(3)
        ]
    )
    targets = {
        f"model.layers.{layer_index}.{name}": module
        for layer_index, layer in enumerate(layers)
        for name, module in layer.items()
    }
    bytes_per_layer = sum(QVQProcessor._yaqa_factor_bytes(module) for module in layers[0].values())

    chunks = QVQProcessor._yaqa_target_chunks(
        targets,
        list(layers),
        max_factor_bytes=bytes_per_layer * 2,
    )

    assert [tuple(chunk) for chunk in chunks] == [
        ("model.layers.0.q_proj", "model.layers.0.o_proj", "model.layers.1.q_proj", "model.layers.1.o_proj"),
        ("model.layers.2.q_proj", "model.layers.2.o_proj"),
    ]
    assert set().union(*(set(chunk) for chunk in chunks)) == set(targets)


def test_qvq_yaqa_packed_symmetric_chunking_uses_actual_accumulator_bytes():
    layers = torch.nn.ModuleList(
        [
            torch.nn.ModuleDict({"proj": torch.nn.Linear(8, 8, bias=False)})
            for _ in range(4)
        ]
    )
    targets = {
        f"model.layers.{layer_index}.proj": layer["proj"]
        for layer_index, layer in enumerate(layers)
    }
    full_bytes = QVQProcessor._yaqa_factor_bytes(layers[0]["proj"])
    packed_bytes = QVQProcessor._yaqa_packed_factor_bytes(layers[0]["proj"])
    assert packed_bytes == 8 * 9 * 4
    assert packed_bytes < full_bytes

    full_chunks = QVQProcessor._yaqa_target_chunks(
        targets,
        list(layers),
        max_factor_bytes=full_bytes * 2,
    )
    packed_chunks = QVQProcessor._yaqa_target_chunks(
        targets,
        list(layers),
        max_factor_bytes=full_bytes * 2,
        packed_symmetric=True,
    )

    assert [len(chunk) for chunk in full_chunks] == [2, 2]
    assert [len(chunk) for chunk in packed_chunks] == [3, 1]


def test_qvq_yaqa_default_mps_factor_budget_bounds_gram_working_set():
    with patch("torch.mps.recommended_max_memory", return_value=40 * 1024**3):
        assert QVQProcessor._yaqa_default_max_factor_bytes(torch.device("mps"), 20 * 1024**3) == 4 * 1024**3
    with patch("torch.mps.recommended_max_memory", return_value=16 * 1024**3):
        assert QVQProcessor._yaqa_default_max_factor_bytes(torch.device("mps"), 20 * 1024**3) == 4 * 1024**3
    assert QVQProcessor._yaqa_default_max_factor_bytes(torch.device("cpu"), 12345) == 12345


def test_qvq_dynamic_clone_preserves_fractional_rate_and_skip_contract():
    cfg = QVQConfig(
        bits=2,
        rounding="block_ldlq",
        dynamic={"+:model.layers.0.*": {"bits": 2.5}, "-:model.layers.1.*": {}},
        offload_to_disk=False,
    )

    overridden = clone_qvq_config_for_module(cfg, "model.layers.0.self_attn.q_proj")

    assert overridden.bits == 2.5
    assert clone_qvq_config_for_module(cfg, "model.layers.1.self_attn.q_proj") is None


def test_qvq_dynamic_clone_materializes_yaqa_rate_regularization():
    cfg = QVQConfig(
        bits=2,
        rounding="yaqa",
        yaqa=YaqaConfig(
            regularization=0.05,
            regularization_by_rate=((2.0, 0.1), (2.5, 0.2)),
        ),
        dynamic={"+:model.layers.0.*": {"bits": 2.5}},
        offload_to_disk=False,
    )

    base = clone_qvq_config_for_module(cfg, "model.layers.1.self_attn.q_proj")
    overridden = clone_qvq_config_for_module(cfg, "model.layers.0.self_attn.q_proj")

    assert base.bits == 2
    assert base.yaqa.regularization == pytest.approx(0.1)
    assert overridden.bits == 2.5
    assert overridden.yaqa.regularization == pytest.approx(0.2)


def test_qvq_dynamic_clone_can_override_yaqa_regularization_per_module():
    cfg = QVQConfig(
        bits=2,
        rounding="yaqa",
        yaqa=YaqaConfig(regularization_by_rate=((2.0, 0.05),)),
        dynamic={"+:model.layers.0.*": {"yaqa_regularization": 0.1}},
        offload_to_disk=False,
    )

    overridden = clone_qvq_config_for_module(cfg, "model.layers.0.mlp.down_proj")
    base = clone_qvq_config_for_module(cfg, "model.layers.1.mlp.down_proj")

    assert overridden.yaqa.regularization == pytest.approx(0.1)
    assert base.yaqa.regularization == pytest.approx(0.05)


def test_qvq_propagated_bank_selection_builds_a_heldout_gate_from_dense_hook():
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            format="qvq_v4",
            vector_size=4,
            bank_count=4,
            device="cpu",
            offload_to_disk=False,
        )
    )
    processor.preprocess(named)
    source = torch.randn((1, 4, 16))
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.tensor([[True, True, True, False]])
    processor.pre_process_fwd_hook("proj")(root.proj, (source,), root.proj(source))

    assert processor.tasks["proj"]["capture"].nsamples == 2
    with patch(
        "gptqmodel.looper.qvq_processor.quantize_qvq_linear",
        wraps=quantize_qvq_linear,
    ) as quantize:
        processor.process(named, device=torch.device("cpu"))
    propagated_inputs = quantize.call_args.kwargs["propagated_inputs"]
    propagated_targets = quantize.call_args.kwargs["propagated_target_output"]
    assert propagated_inputs.shape == (1, 16)
    assert propagated_targets.shape == (1, 16)
    assert callable(quantize.call_args.kwargs["propagated_acceptance"])


def test_qvq_propagated_bank_selection_false_disables_automatic_gate():
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            format="qvq_v4",
            vector_size=4,
            bank_count=4,
            propagated_bank_selection=False,
            device="cpu",
            offload_to_disk=False,
        )
    )
    processor.preprocess(named)
    assert processor.tasks["proj"]["qcfg"].propagated_bank_selection is False


def test_qvq_automatic_propagation_allows_the_reduced_hessian_count():
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            format="qvq_v4",
            vector_size=4,
            bank_count=4,
            device="cpu",
            offload_to_disk=False,
        )
    )
    processor.total_calibration_tokens = 4
    processor.preprocess(named)
    source = torch.randn((1, 4, 16))
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.ones((1, 4), dtype=torch.bool)
    processor.pre_process_fwd_hook("proj")(root.proj, (source,), root.proj(source))

    assert processor.tasks["proj"]["capture"].nsamples == 3
    processor.process(named, device=torch.device("cpu"))


def test_qvq_automatic_propagation_gate_includes_linear_bias():
    processor = _processor()
    inputs = torch.eye(2)
    bias = torch.tensor([1.0, -2.0])
    targets = inputs + bias
    processor._automatic_propagation_gate_samples["proj"] = [(inputs, targets)]
    processor._materialize_automatic_propagation_gate("proj", bias=bias)
    gate = processor._get_propagation_gate("proj", torch.device("cpu"))
    assert gate is not None
    baseline = torch.eye(2)
    proposal = 2.0 * torch.eye(2)
    assert gate[2](proposal, baseline) is False


@pytest.mark.parametrize(("format_value", "vector_size"), (("qvq", 2), ("qvq_v4", 4)))
def test_qvq_lifecycle_excludes_padding_quantizes_replays_and_installs_runtime_module(format_value, vector_size):
    torch.manual_seed(20260811)
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=True, dtype=torch.float32)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            format=format_value,
            device="cpu",
            offload_to_disk=False,
        )
    )
    processor.preprocess(named)

    source = torch.randn((1, 4, 16), dtype=torch.float32)
    # Make padding impossible to miss numerically: including either padded
    # row would dominate the valid-token second moment.
    source[:, 2:].mul_(1_000.0)
    keep_mask = torch.tensor([[True, True, False, False]])
    output = root.proj(source)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = keep_mask
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook("proj")(root.proj, (source,), output)

    capture = processor.tasks["proj"]["capture"]
    assert capture.nsamples == 2
    hessian = capture.finalize_hessian(target_device=torch.device("cpu"))
    valid_source = source[0, :2].to(torch.float32)
    # GPTQ stores 2 X^T X / N. QTIP's reference uses X^T X / N; the
    # uniform factor of two cancels from block LDL normalization and therefore
    # leaves the QTIP error-feedback recurrence exactly unchanged.
    expected_hessian = valid_source.T @ valid_source
    torch.testing.assert_close(hessian, expected_hessian, rtol=1e-6, atol=1e-6)
    processor.process(named, device=torch.device("cpu"))
    dense_replay = root.proj(source)
    # StageLayer clears transient capture tasks before its concurrent finalize
    # drain. The serialized module payload must own all runtime metadata.
    processor.clear_cache_data()

    model = SimpleNamespace(model=root)
    qmodule = processor.submodule_finalize(named, model)

    assert isinstance(qmodule, QVQLinear)
    assert root.proj is qmodule
    assert set(qmodule.state_dict()) == {"trellis", "SU", "SV", "bias"}
    assert qmodule.bits == 2
    assert qmodule.vector_size == vector_size
    # The padded rows are intentionally scaled by 1,000 to prove masking.  A
    # materialized dense replay and the factorized QVQ path use different but
    # mathematically equivalent FP32 association orders, so use a bounded
    # absolute gate in addition to a tight relative gate for this stress case.
    torch.testing.assert_close(qmodule(source), dense_replay, rtol=2e-6, atol=2e-4)


def test_qvq_lifecycle_forwards_module_scale_search_and_reports_the_guarded_decision():
    qcfg = QVQConfig(
        bits=2,
        rounding="block_ldlq",
        module_scale_search=True,
        device="cpu",
        offload_to_disk=False,
    )
    processor = _processor(qcfg=qcfg)
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float32)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    processor.preprocess(named)

    source = torch.randn((1, 4, 16), dtype=torch.float32)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.ones((1, 4), dtype=torch.bool)
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook("proj")(root.proj, (source,), root.proj(source))

    with patch("gptqmodel.looper.qvq_processor.quantize_qvq_linear", wraps=quantize_qvq_linear) as quantize:
        processor.process(named, device=torch.device("cpu"))

    assert quantize.call_args.kwargs["module_scale_search"] is True
    stat = processor.log[-1]
    assert isinstance(stat["module_scale_search_selected"], bool)
    assert stat["module_scale_multiplier"] > 0
    assert isinstance(stat["module_scale_reencoded"], bool)


def test_qvq_output_alignment_is_disabled_by_default_and_explicitly_enablable():
    processor = _processor()
    inputs = [[torch.ones(1, 1, 16)]]

    assert processor._output_alignment is None
    assert processor.uses_grouped_optimization() is False
    assert processor.needs_pristine_layer_clone() is False
    assert processor.clean_group_layer_inputs(layer_index=0, layer_inputs=inputs) is inputs
    assert processor.receive_clean_layer_inputs(layer_index=0, layer_inputs=inputs) is None

    enabled = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            output_alignment=OutputAlignConfig(),
            device="cpu",
            offload_to_disk=False,
        )
    )
    assert enabled._output_alignment is not None


def test_qvq_output_alignment_delegates_pristine_stream_ownership_to_attachment():
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            output_alignment=OutputAlignConfig(),
            device="cpu",
            offload_to_disk=False,
        )
    )
    inputs = [[torch.ones(1, 1, 16)]]
    attachment = processor._output_alignment

    with (
        patch.object(attachment, "clean_group_layer_inputs", return_value=inputs) as select,
        patch.object(attachment, "receive_clean_layer_inputs") as advance,
    ):
        assert processor.clean_group_layer_inputs(layer_index=2, layer_inputs=inputs) is inputs
        processor.receive_clean_layer_inputs(layer_index=2, layer_inputs=inputs)

    select.assert_called_once_with(layer_index=2, layer_inputs=inputs)
    advance.assert_called_once_with(layer_index=2, layer_inputs=inputs)


def test_qvq_pristine_hessian_capture_is_exact_masked_and_ignores_later_noisy_replay():
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            output_alignment=OutputAlignConfig(pristine_hessian=True),
            device="cpu",
            offload_to_disk=False,
        )
    )
    root = torch.nn.Module()
    root.proj = HookedLinear.from_linear(torch.nn.Linear(16, 16, bias=False, dtype=torch.float32))
    named = NamedModule(root.proj, name="proj", full_name="model.layers.0.proj", layer_index=0)
    processor.preprocess(named)
    original_capture = processor.tasks["proj"]["capture"]
    def prior_hook(*_args):
        return None

    root.proj.forward_hook = prior_hook
    root.proj.forward_hook_last = True

    clean = torch.randn((1, 4, 16), dtype=torch.float32)
    noisy = clean + 100.0
    keep = torch.tensor([[True, True, False, False]])
    processor._mask_tls = threading.local()
    processor._mask_tls.value = keep
    processor._set_current_batch_index(0)

    with processor.pristine_quant_input_capture(layer_index=0):
        root.proj(clean)

    assert root.proj.forward_hook is prior_hook
    assert root.proj.forward_hook_last is True
    capture = processor.tasks["proj"]["capture"]
    assert capture is not original_capture
    assert capture.nsamples == 2
    # The ordinary subset hook must not mix already-quantized/noisy inputs into
    # a Hessian committed from the pristine dense replay.
    noisy_output = F.linear(noisy, root.proj.weight)
    processor.pre_process_fwd_hook("proj")(root.proj, (noisy,), noisy_output)
    assert capture.nsamples == 2
    hessian = capture.finalize_hessian(target_device=torch.device("cpu"))
    valid = clean[0, :2].to(torch.float32)
    torch.testing.assert_close(hessian, valid.T @ valid, rtol=1e-6, atol=1e-6)


def test_qvq_pristine_hessian_capture_failure_restores_hooks_and_discards_partial_state():
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            output_alignment=OutputAlignConfig(pristine_hessian=True),
            device="cpu",
            offload_to_disk=False,
        )
    )
    root = torch.nn.Module()
    root.proj = HookedLinear.from_linear(torch.nn.Linear(16, 16, bias=False, dtype=torch.float32))
    named = NamedModule(root.proj, name="proj", full_name="model.layers.0.proj", layer_index=0)
    processor.preprocess(named)
    original_capture = processor.tasks["proj"]["capture"]
    def prior_hook(*_args):
        return None

    root.proj.forward_hook = prior_hook
    root.proj.forward_hook_last = True
    source = torch.randn((1, 4, 16), dtype=torch.float32)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.tensor([[True, True, False, False]])

    with pytest.raises(RuntimeError, match="replay failed"):
        with processor.pristine_quant_input_capture(layer_index=0):
            root.proj(source)
            raise RuntimeError("replay failed")

    assert root.proj.forward_hook is prior_hook
    assert root.proj.forward_hook_last is True
    assert processor.tasks["proj"]["capture"] is original_capture
    assert original_capture.nsamples == 0
    assert "pristine_hessian_complete" not in processor.tasks["proj"]
    assert processor._active_pristine_hessian_captures == {}


def test_qvq_pristine_hessian_capture_restores_native_hook_registration():
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            output_alignment=OutputAlignConfig(pristine_hessian=True),
            device="cpu",
            offload_to_disk=False,
        )
    )
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="model.layers.0.proj", layer_index=0)
    processor.preprocess(named)
    prior_calls = []
    prior_handle = root.proj.register_forward_hook(lambda *_args: prior_calls.append(True))
    source = torch.randn((1, 2, 16), dtype=torch.float32)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.ones((1, 2), dtype=torch.bool)

    with processor.pristine_quant_input_capture(layer_index=0):
        root.proj(source)
    root.proj(source)

    assert len(prior_calls) == 2
    assert processor.tasks["proj"]["capture"].nsamples == 2
    prior_handle.remove()


def test_qvq_pristine_hessian_capture_rejects_overlap_and_tolerates_old_capture_cleanup_failure():
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            output_alignment=OutputAlignConfig(pristine_hessian=True),
            device="cpu",
            offload_to_disk=False,
        )
    )
    root = torch.nn.Module()
    root.proj = HookedLinear.from_linear(torch.nn.Linear(16, 16, bias=False))
    named = NamedModule(root.proj, name="proj", full_name="model.layers.0.proj", layer_index=0)
    processor.preprocess(named)
    processor._active_pristine_hessian_captures = {"other": object()}
    with pytest.raises(RuntimeError, match="cannot overlap"):
        with processor.pristine_quant_input_capture(layer_index=0):
            pass
    processor._active_pristine_hessian_captures = {}
    original_capture = processor.tasks["proj"]["capture"]
    source = torch.randn((1, 2, 16), dtype=torch.float32)
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.ones((1, 2), dtype=torch.bool)

    with patch.object(original_capture, "free", side_effect=RuntimeError("cleanup failed")):
        with processor.pristine_quant_input_capture(layer_index=0):
            root.proj(source)

    assert processor.tasks["proj"]["capture"] is not original_capture
    assert processor.tasks["proj"]["pristine_hessian_complete"] is True


def test_qvq_pristine_hessian_empty_capture_failure_cleans_context():
    processor = _processor(
        qcfg=QVQConfig(
            bits=2,
            rounding="block_ldlq",
            output_alignment=OutputAlignConfig(pristine_hessian=True),
            device="cpu",
            offload_to_disk=False,
        )
    )
    root = torch.nn.Module()
    root.proj = HookedLinear.from_linear(torch.nn.Linear(16, 16, bias=False, dtype=torch.float32))
    named = NamedModule(root.proj, name="proj", full_name="model.layers.0.proj", layer_index=0)
    processor.preprocess(named)

    with pytest.raises(RuntimeError, match="empty replay failed"):
        with processor.pristine_quant_input_capture(layer_index=0):
            assert set(processor._active_pristine_hessian_captures) == {"proj"}
            raise RuntimeError("empty replay failed")

    assert processor._active_pristine_hessian_captures == {}


@pytest.mark.parametrize(
    "qcfg",
    [
        QVQConfig(
            bits=2,
            rounding="block_ldlq",
            output_alignment=OutputAlignConfig(pristine_hessian=False),
            device="cpu",
            offload_to_disk=False,
        ),
        QVQConfig(bits=2, rounding="yaqa", output_alignment=None, device="cpu", offload_to_disk=False),
    ],
)
def test_qvq_pristine_hessian_capture_is_inert_when_disabled_or_yaqa(qcfg):
    processor = _processor(qcfg=qcfg)
    if processor._output_alignment is not None:
        root = torch.nn.Module()
        root.proj = torch.nn.Linear(16, 16, bias=False)
        named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
        processor.preprocess(named)
        source = torch.randn((1, 2, 16), dtype=torch.float32)
        processor._mask_tls = threading.local()
        processor._mask_tls.value = torch.ones((1, 2), dtype=torch.bool)
        processor.pre_process_fwd_hook("proj")(root.proj, (source,), root.proj(source))
        assert processor.tasks["proj"]["capture"].nsamples == 2

    with processor.pristine_quant_input_capture(layer_index=0):
        pass

    assert processor._active_pristine_hessian_captures == {}
    assert processor._pristine_hessian_modules == {}


def test_qvq_output_alignment_uses_existing_threadx_cuda_owner_lane():
    qcfg = QVQConfig(
        bits=2,
        rounding="block_ldlq",
        output_alignment=OutputAlignConfig(),
        device="cpu",
        offload_to_disk=False,
    )
    processor = _processor(qcfg=qcfg)
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=3)
    attachment = processor._output_alignment
    processor.log.append({"layer": 3, "module": "proj"})

    with (
        patch.object(attachment, "modules_are_fully_staged", return_value=True),
        patch("gptqmodel.looper.qvq_processor.get_device", return_value=torch.device("cuda", 6)),
        patch("gptqmodel.looper.qvq_processor.DEVICE_THREAD_POOL.do") as threadx_do,
    ):
        threadx_do.return_value = {
            "accepted": 1.0,
            "seconds": 2.5,
            "train_batches": 4.0,
        }
        processor.cleanup_subset({"proj": named}, subset_index=1, subset_total=2)

    threadx_do.assert_called_once_with(torch.device("cuda", 6), attachment.align_layer, 3, finalize=True)
    assert processor._output_alignment_stats[3][0]["accepted"] == 1.0
    assert processor.log[0]["output_alignment_seconds"] == 2.5


def test_qvq_output_alignment_finally_cleanup_discards_incomplete_layer_without_masking_failure():
    qcfg = QVQConfig(
        bits=2,
        rounding="block_ldlq",
        output_alignment=OutputAlignConfig(),
        device="cpu",
        offload_to_disk=False,
    )
    processor = _processor(qcfg=qcfg)
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=3)
    attachment = processor._output_alignment

    with (
        patch.object(attachment, "modules_are_fully_staged", return_value=False),
        patch.object(attachment, "discard_layer") as discard,
        patch.object(attachment, "align_layer") as align,
    ):
        processor.cleanup_subset({"proj": named}, subset_index=1, subset_total=2)

    discard.assert_called_once_with(3)
    align.assert_not_called()


def test_qvq_output_alignment_refines_groups_and_runs_after_every_projection_subset():
    qcfg = QVQConfig(
        bits=2,
        rounding="block_ldlq",
        output_alignment=OutputAlignConfig(),
        device="cpu",
        offload_to_disk=False,
    )
    processor = _processor(qcfg=qcfg)
    assert processor.refine_subset_module_groups([["q", "k", "v"], ["o"]]) == [
        ["q"],
        ["k"],
        ["v"],
        ["o"],
    ]
    processor._output_alignment.bind_model(
        SimpleNamespace(
            get_module_tree_flags=lambda name: {
                "q": frozenset({"q"}),
                "k": frozenset({"k"}),
                "v": frozenset({"v"}),
                "o": frozenset(),
                "gate": frozenset({"gate"}),
                "up": frozenset({"up"}),
                "down": frozenset({"down"}),
            }[name]
        )
    )
    assert processor.refine_subset_module_groups(
        [["q", "k", "v"], ["o"], ["gate", "up"], ["down"]]
    ) == [["v"], ["q"], ["k"], ["o"], ["up"], ["gate"], ["down"]]

    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=3)
    attachment = processor._output_alignment
    with (
        patch.object(attachment, "modules_are_fully_staged", return_value=True),
        patch.object(attachment, "align_layer", return_value={"alignment_pass": 1.0}) as align,
    ):
        processor.cleanup_subset({"proj": named}, subset_index=0, subset_total=2)

    align.assert_called_once_with(3, finalize=False)


def test_qvq_output_alignment_rejects_moe_from_explicit_module_tree_tags_before_capture():
    qcfg = QVQConfig(
        bits=2,
        rounding="block_ldlq",
        output_alignment=OutputAlignConfig(),
        device="cpu",
        offload_to_disk=False,
    )
    processor = _processor(qcfg=qcfg)
    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    named.state["module_tree_flags"] = frozenset({"moe", "routed"})

    with pytest.raises(NotImplementedError, match="MoE decoder layers"):
        processor.preprocess(named)

    assert "proj" not in processor.tasks


def test_qvq_lifecycle_requires_yaqa_factors_before_layer_processing():
    calibration = [{"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}]
    config = QVQConfig(rounding="yaqa", offload_to_disk=False)
    processor = QVQProcessor(
        tokenizer=None,
        qcfg=config,
        calibration=calibration,
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    named = NamedModule(torch.nn.Linear(16, 16), name="proj", full_name="model.layers.0.proj", layer_index=0)

    with pytest.raises(RuntimeError, match="must be prepared"):
        processor.preprocess(named)


class _YaqaLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False)

    def forward(self, hidden_states):
        return F.silu(self.proj(hidden_states))


class _YaqaDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_YaqaLayer(), _YaqaLayer()])


class _YaqaCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.embed = torch.nn.Embedding(32, 16)
        self.model = _YaqaDecoder()
        self.lm_head = torch.nn.Linear(16, 32, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        hidden_states = self.embed(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return SimpleNamespace(logits=self.lm_head(hidden_states))


class _YaqaQModel:
    def __init__(self, qcfg):
        self.model = _YaqaCausalModel().eval()
        self.quantize_config = qcfg
        self.turtle_model = None

    @staticmethod
    def extract_layers_node():
        return "model.layers"

    @staticmethod
    def simple_layer_modules(**kwargs):
        del kwargs
        return [["proj"]]

    @staticmethod
    def should_quantize_layer(*args):
        del args
        return True

    @staticmethod
    def should_quantize_module(*args):
        del args
        return True


@pytest.mark.parametrize("bits", (1, 1.5))
def test_qvq_yaqa_lifecycle_collects_full_model_factors_and_wires_them_to_quantizer(bits):
    calibration = [
        {
            "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        }
    ]
    qcfg = QVQConfig(
        bits=bits,
        rounding="yaqa",
        yaqa={"seed": 787, "regularization": 1e-4, "minimum_sequences": 2},
        output_alignment=None,
        device="cpu",
        offload_to_disk=False,
    )
    qmodel = _YaqaQModel(qcfg)
    processor = QVQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration,
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=2,
        yaqa_calibration=[
            {
                "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]),
                "attention_mask": torch.ones((3, 2), dtype=torch.long),
            }
        ],
    )

    processor.prepare_yaqa(qmodel)

    assert processor._yaqa_stats["independent_sequences"] == 3
    assert processor._yaqa_stats["valid_output_samples"] == 6
    assert processor._yaqa_stats["activation_checkpointing"] is True
    assert processor._yaqa_stats["checkpointed_modules"] == 2
    full_name = "model.layers.0.proj"
    input_hessian = processor._yaqa_input_hessians[full_name].clone()
    output_hessian = processor._yaqa_output_hessians[full_name].clone()
    assert input_hessian.shape == (16, 16)
    assert output_hessian.shape == (16, 16)

    module = qmodel.model.model.layers[0].proj
    named = NamedModule(module, name="proj", full_name=full_name, layer_index=0)
    processor.preprocess(named)
    source = qmodel.model.embed(calibration[0]["input_ids"]).detach()
    processor._mask_tls = threading.local()
    processor._mask_tls.value = calibration[0]["attention_mask"].bool()
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook("proj")(module, (source,), module(source))

    fake_result = SimpleNamespace(
        trellis=torch.zeros((1, int(8 * bits)), dtype=torch.int32),
        SU=torch.ones(16),
        SV=torch.ones(16),
        bias=None,
        weight=module.weight.detach().clone(),
        proxy_loss=torch.tensor(1.0),
        output_scale_optimized_channels=0,
        hessian_viterbi_selected=False,
        rounding="yaqa",
        kronecker_proxy_loss=torch.tensor(0.75),
        module_scale_search_selected=False,
        module_scale_multiplier=1.0,
        module_scale_reencoded=False,
        telemetry=None,
        bank_ids=None,
        bank_alt_id=None,
        yaqa_bank_fallback_to_v2=None,
        yaqa_selector_churn=None,
        yaqa_family_changed=None,
        yaqa_block_family_id=None,
        yaqa_spectral_selected=None,
        yaqa_spectral_method=None,
        yaqa_spectral_rank=None,
        yaqa_spectral_lambda=None,
        yaqa_spectral_alpha=None,
        yaqa_spectral_svd_device=None,
        yaqa_spectral_concentration=None,
        yaqa_spectral_oracle_losses=None,
        yaqa_spectral_candidates=None,
        yaqa_spectral_absorption_efficiency=None,
        yaqa_spectral_selector_churn=None,
        yaqa_spectral_family_changed=None,
        serialized_tensors=lambda: {
            "trellis": fake_result.trellis,
            "SU": fake_result.SU,
            "SV": fake_result.SV,
        },
    )
    with patch("gptqmodel.looper.qvq_processor.quantize_qvq_linear", return_value=fake_result) as quantize:
        processor.process(named, device=torch.device("cpu"))

    torch.testing.assert_close(quantize.call_args.args[1], input_hessian)
    torch.testing.assert_close(quantize.call_args.kwargs["output_hessian"], output_hessian)
    assert quantize.call_args.kwargs["rounding"] == "yaqa"
    assert quantize.call_args.kwargs["damp_percent"] == 1e-4
    assert processor.log[-1]["yaqa_independent_sequences"] == 3
    assert processor.log[-1]["yaqa_kronecker_proxy_loss"] == 0.75
    assert full_name not in processor._yaqa_input_hessians
    assert full_name not in processor._yaqa_output_hessians


def test_qvq_yaqa_lifecycle_rejects_lazy_source_and_duplicate_prepass():
    calibration = [{"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}]
    qcfg = QVQConfig(
        bits=1,
        rounding="yaqa",
        yaqa={"minimum_sequences": 1},
        device="cpu",
        offload_to_disk=False,
    )
    qmodel = _YaqaQModel(qcfg)
    processor = QVQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration,
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    qmodel.turtle_model = object()
    with pytest.raises(RuntimeError, match="offload_to_disk=False"):
        processor.prepare_yaqa(qmodel)

    qmodel.turtle_model = None
    processor.prepare_yaqa(qmodel)
    with pytest.raises(RuntimeError, match="already prepared"):
        processor.prepare_yaqa(qmodel)


def test_qvq_yaqa_chunked_factor_passes_match_single_pass_exactly():
    yaqa_rows = [
        {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6]]),
            "attention_mask": torch.ones((3, 2), dtype=torch.long),
        }
    ]

    def collect(max_factor_bytes_per_pass):
        qcfg = QVQConfig(
            bits=2,
            rounding="yaqa",
            yaqa={
                "seed": 787,
                "minimum_sequences": 3,
                "max_factor_bytes_per_pass": max_factor_bytes_per_pass,
            },
            device="cpu",
            offload_to_disk=False,
        )
        qmodel = _YaqaQModel(qcfg)
        qmodel.model.load_state_dict(reference_state)
        processor = QVQProcessor(
            tokenizer=None,
            qcfg=qcfg,
            calibration=yaqa_rows,
            prepare_dataset_func=_prepared_calibration,
            calibration_concat_size=None,
            calibration_sort=None,
            batch_size=1,
            yaqa_calibration=yaqa_rows,
        )
        processor.prepare_yaqa(qmodel)
        return processor

    reference_model = _YaqaCausalModel().eval()
    reference_state = reference_model.state_dict()
    bytes_per_layer = 2 * 16 * 16 * 4
    single = collect(None)
    chunked = collect(bytes_per_layer)

    assert single._yaqa_stats["factor_passes"] == 1
    assert chunked._yaqa_stats["factor_passes"] == 2
    assert chunked._yaqa_stats["pass_target_counts"] == [1, 1]
    assert single._yaqa_input_hessians.keys() == chunked._yaqa_input_hessians.keys()
    for name in single._yaqa_input_hessians:
        torch.testing.assert_close(single._yaqa_input_hessians[name], chunked._yaqa_input_hessians[name], rtol=0, atol=0)
        torch.testing.assert_close(
            single._yaqa_output_hessians[name], chunked._yaqa_output_hessians[name], rtol=0, atol=0
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bits", (1, 1.5))
def test_qvq_yaqa_lifecycle_quantizes_installs_and_replays_ultralow_rates_on_cuda(bits):
    calibration = [
        {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }
    ]
    qcfg = QVQConfig(
        bits=bits,
        rounding="yaqa",
        yaqa={"seed": 787, "minimum_sequences": 1},
        dynamic={"-:.*model.layers.1.*": {}},
        device="cuda",
        offload_to_disk=False,
    )
    qmodel = _YaqaQModel(qcfg)
    processor = QVQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration,
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )
    processor.prepare_yaqa(qmodel)

    full_name = "model.layers.0.proj"
    module = qmodel.model.model.layers[0].proj
    named = NamedModule(module, name="proj", full_name=full_name, layer_index=0)
    processor.preprocess(named)
    source = qmodel.model.embed(calibration[0]["input_ids"]).detach()
    processor._mask_tls = threading.local()
    processor._mask_tls.value = calibration[0]["attention_mask"].bool()
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook("proj")(module, (source,), module(source))

    processor.process(named, device=torch.device("cuda"))
    replay = module(source.to(module.weight.device)).cpu()
    qmodule = processor.submodule_finalize(named, qmodel)

    assert isinstance(qmodule, QVQLinear)
    assert qmodule.bits == bits
    assert processor.log[-1]["rounding"] == "yaqa"
    assert processor.log[-1]["yaqa_kronecker_proxy_loss"] is not None
    torch.testing.assert_close(qmodule(source), replay, atol=2e-5, rtol=2e-5)


def test_qvq_lifecycle_rejects_non_linear_runtime_replacement_instead_of_changing_math():
    processor = _processor(bits=2)
    convolution = torch.nn.Conv1d(16, 16, kernel_size=1)
    named = NamedModule(convolution, name="conv", full_name="conv", layer_index=0)

    with pytest.raises(NotImplementedError, match="Unsupported QVQ module type: Conv1d"):
        processor._restore_module_weight(named, torch.randn(16, 16))
