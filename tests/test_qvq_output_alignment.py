# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import gptqmodel
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_output_alignment import (
    QVQOutputAlignmentAttachment,
    _FixedTrellisAlignmentLinear,
    _LayerAlignmentState,
    _ReplayBatch,
)
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.qvq import _qvq_fp16_emulated_hadamard_fallback
from gptqmodel.quantization import OutputAlignConfig
from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    reconstruct_qvq_inner_weight,
    rht_reconstruct_weight,
)
from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU, matmul_hadU_stable
from gptqmodel.utils.looper_helpers import normalize_device_like
from gptqmodel.utils.python import has_gil_disabled
from gptqmodel.utils.threadx import DeviceThreadPool


class _TinyDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 16, bias=False, dtype=torch.float32)
        self.register_buffer("saved", torch.ones(1, dtype=torch.float64))

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return (self.proj(hidden_states),)


class _TwoProjectionDecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(16, 16, bias=False, dtype=torch.float32)
        self.future = torch.nn.Linear(16, 16, bias=False, dtype=torch.float32)

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return (self.future(torch.nn.functional.silu(self.first(hidden_states))),)


def _prepare_attachment(*, target_scale=1.2, padding_outlier=False, config=None):
    torch.manual_seed(20260812)
    layer = _TinyDecoderLayer()
    trellis = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (1, qvq_words_per_tile(2)),
        dtype=torch.int32,
    )
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        in_features=16,
        out_features=16,
        codebook_version=PGC16_CODEBOOK_VERSION,
    )
    SU = torch.ones(16, dtype=torch.float32)
    SV = torch.ones(16, dtype=torch.float32)
    current_weight = rht_reconstruct_weight(inner, SU, SV)
    target_weight = rht_reconstruct_weight(inner, SU, SV * target_scale)
    layer.proj.weight.data.copy_(current_weight)

    named = NamedModule(layer.proj, name="proj", full_name="model.layers.0.proj", layer_index=0)
    named.state.update(
        {
            "trellis": trellis,
            "SU": SU.clone(),
            "SV": SV.clone(),
            "_qvq_runtime_config": (2.0, PGC16_CODEBOOK_VERSION),
            "module_tree_flags": frozenset(),
        }
    )
    attachment = QVQOutputAlignmentAttachment(
        config
        or OutputAlignConfig(
            learning_rate=0.03,
            epochs=12,
            maximum_train_batches=2,
            maximum_validation_batches=2,
            validation_fraction=0.5,
        )
    )
    attachment.bind_model(SimpleNamespace(prepare_layer_replay_kwargs=lambda **kwargs: kwargs["additional_inputs"]))
    attachment.register_module(named)
    attachment.receive_pristine_layer_module(layer_index=0, layer_module=layer)

    inputs = []
    targets = []
    masks = []
    for _ in range(4):
        source = torch.randn(1, 4, 16)
        target = source @ target_weight.t()
        mask = torch.tensor([[True, True, True, False]])
        if padding_outlier:
            target[:, -1].fill_(1e6)
        inputs.append([source])
        targets.append([target])
        masks.append(mask)
    attachment.receive_layer_forward_context(
        layer_index=0,
        layer_inputs=inputs,
        layer_input_kwargs=[{} for _ in inputs],
        layer_outputs=targets,
        position_ids=[None for _ in inputs],
        attention_masks=masks,
    )
    return attachment, layer, named, current_weight


def _prepare_sequential_attachment():
    torch.manual_seed(20260813)
    layer = _TwoProjectionDecoderLayer()
    trellis = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (1, qvq_words_per_tile(2)),
        dtype=torch.int32,
    )
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        in_features=16,
        out_features=16,
        codebook_version=PGC16_CODEBOOK_VERSION,
    )
    SU = torch.ones(16, dtype=torch.float32)
    SV = torch.ones(16, dtype=torch.float32)
    layer.first.weight.data.copy_(rht_reconstruct_weight(inner, SU, SV))
    target_first = rht_reconstruct_weight(inner, SU, SV * 1.2)
    target_future = layer.future.weight.detach().clone()
    original_future = target_future.clone()
    hooked_future = HookedLinear.from_linear(layer.future)
    layer.future = hooked_future

    named = NamedModule(layer.first, name="first", full_name="model.layers.0.first", layer_index=0)
    named.state.update(
        {
            "trellis": trellis,
            "SU": SU.clone(),
            "SV": SV.clone(),
            "_qvq_runtime_config": (2.0, PGC16_CODEBOOK_VERSION),
            "module_tree_flags": frozenset(),
        }
    )
    attachment = QVQOutputAlignmentAttachment(
        OutputAlignConfig(
            learning_rate=0.02,
            epochs=8,
            maximum_train_batches=2,
            maximum_validation_batches=2,
            validation_fraction=0.5,
        )
    )
    attachment.bind_model(SimpleNamespace(prepare_layer_replay_kwargs=lambda **kwargs: kwargs["additional_inputs"]))
    attachment.register_module(named)
    attachment.receive_pristine_layer_module(layer_index=0, layer_module=layer)

    inputs = []
    targets = []
    for _ in range(4):
        source = torch.randn(1, 4, 16)
        target = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(source, target_first)),
            target_future,
        )
        inputs.append([source])
        targets.append([target])
    attachment.receive_layer_forward_context(
        layer_index=0,
        layer_inputs=inputs,
        layer_input_kwargs=[{} for _ in inputs],
        layer_outputs=targets,
        position_ids=[None for _ in inputs],
        attention_masks=[None for _ in inputs],
    )
    return attachment, layer, named, original_future, hooked_future


def test_qvq_output_alignment_fixed_trellis_reduces_heldout_layer_error_and_updates_exact_payload():
    attachment, layer, named, _ = _prepare_attachment(padding_outlier=True)
    original_trellis = named.state["trellis"].clone()
    original_su = named.state["SU"].clone()

    result = attachment.align_layer(0)

    assert result is not None
    assert result["accepted"] == 1.0
    assert result["candidate_validation_loss"] < result["baseline_validation_loss"]
    assert result["seconds"] > 0
    assert result["train_batches"] == 2
    assert result["validation_batches"] == 2
    assert result["optimizer_steps"] == 24
    torch.testing.assert_close(named.state["trellis"], original_trellis, rtol=0, atol=0)
    torch.testing.assert_close(named.state["SU"], original_su, rtol=0, atol=0)
    assert not torch.equal(named.state["SV"], torch.ones_like(named.state["SV"]))
    expected = rht_reconstruct_weight(
        reconstruct_qvq_inner_weight(
            named.state["trellis"],
            bits=2,
            in_features=16,
            out_features=16,
            codebook_version=PGC16_CODEBOOK_VERSION,
        ),
        named.state["SU"],
        named.state["SV"],
    )
    torch.testing.assert_close(layer.proj.weight, expected, rtol=1e-5, atol=1e-5)
    assert layer.saved.dtype == torch.float64


def test_qvq_output_alignment_preserves_v2b2_p32_family_and_selector_geometry():
    torch.manual_seed(20260816)
    layer = _TinyDecoderLayer()
    trellis = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (1, qvq_words_per_tile(2)),
        dtype=torch.int32,
    )
    dense_selectors = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.uint8)
    packed_selectors = pack_qvq_binary_bank_ids(dense_selectors)
    bank_alt_id = torch.tensor([2], dtype=torch.uint8)
    SU = torch.ones(16, dtype=torch.float32)
    SV = torch.ones(16, dtype=torch.float32)
    named = NamedModule(layer.proj, name="proj", full_name="model.layers.0.proj", layer_index=0)
    named.state.update(
        {
            "trellis": trellis,
            "SU": SU,
            "SV": SV,
            "bank_ids": packed_selectors,
            "bank_alt_id": bank_alt_id,
            "_qvq_runtime_config": (2.0, PGC16_CODEBOOK_VERSION, 2, 2, 16, False, False, True),
            "module_tree_flags": frozenset(),
        }
    )
    attachment = QVQOutputAlignmentAttachment(OutputAlignConfig())

    expected_inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        in_features=16,
        out_features=16,
        codebook_version=PGC16_CODEBOOK_VERSION,
        vector_size=2,
        trellis_window=16,
        bank_ids=packed_selectors,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    canonical_inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=2,
        in_features=16,
        out_features=16,
        codebook_version=PGC16_CODEBOOK_VERSION,
    )
    temporary = attachment._build_temporary_module(named, torch.device("cpu"))
    runtime = attachment._build_runtime_module(named, torch.device("cpu"), SU=SU, SV=SV)

    assert not torch.equal(expected_inner, canonical_inner)
    torch.testing.assert_close(temporary.inner_weight, expected_inner, rtol=0, atol=0)
    torch.testing.assert_close(runtime.get_inner_weight_tensor(dtype=torch.float32), expected_inner, rtol=0, atol=0)
    assert runtime.v2b2_p32 is True
    assert runtime.bank_count == 2
    torch.testing.assert_close(runtime.bank_alt_id, bank_alt_id, rtol=0, atol=0)


def test_qvq_output_alignment_mps_tensor_device_resolves_to_single_owner_lane():
    pool = DeviceThreadPool.__new__(DeviceThreadPool)

    assert pool._key(torch.device("mps")) == "mps"
    assert pool._key(torch.device("mps:0")) == "mps"
    assert normalize_device_like(torch.device("mps:0")) == torch.device("mps")


@pytest.mark.parametrize(
    ("input_hadamard", "output_hadamard"),
    ((True, True), (False, True), (True, False), (False, False)),
)
def test_fixed_trellis_alignment_preserves_declared_transform_axes(
    input_hadamard, output_hadamard
):
    generator = torch.Generator().manual_seed(20260904)
    source = torch.randn((3, 16), generator=generator)
    inner = torch.randn((16, 16), generator=generator) * 0.1
    SU = torch.linspace(0.75, 1.25, 16)
    SV = torch.linspace(1.25, 0.75, 16)
    bias = torch.linspace(-0.1, 0.1, 16)
    module = _FixedTrellisAlignmentLinear(
        inner_weight=inner,
        SU=SU,
        SV=SV,
        bias=bias,
        output_dtype=torch.float32,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )

    expected = source * SU
    if input_hadamard:
        expected = matmul_hadU(expected)
    expected = expected @ inner
    if output_hadamard:
        expected = matmul_hadU(expected)
    expected = expected * SV + bias

    assert torch.equal(module(source), expected)
    reconstructed = QVQOutputAlignmentAttachment(OutputAlignConfig())._candidate_weights(
        {"proj": module}
    )["proj"]
    reference_weight = rht_reconstruct_weight(
        inner,
        SU,
        SV,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    assert torch.equal(reconstructed, reference_weight)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required")
def test_fixed_trellis_alignment_mps_preserves_activation_dtype():
    module = _FixedTrellisAlignmentLinear(
        inner_weight=torch.eye(16, device="mps"),
        SU=torch.ones(16, device="mps"),
        SV=torch.ones(16, device="mps"),
        bias=None,
        # Materialized module metadata can be FP32 even when the live model
        # activation contract is FP16.
        output_dtype=torch.float32,
    )

    output = module(torch.randn(2, 16, device="mps", dtype=torch.float16))

    assert output.dtype == torch.float16


def test_qvq_output_alignment_restores_a_buffer_cleared_during_candidate_construction():
    attachment, layer, _, _ = _prepare_attachment()
    original_saved = layer.saved.clone()
    candidate_weights = attachment._candidate_weights

    def clear_buffer(temporary_modules):
        layer.saved = None
        return candidate_weights(temporary_modules)

    with patch.object(attachment, "_candidate_weights", side_effect=clear_buffer):
        attachment.align_layer(0)

    assert layer.saved is not None
    assert layer.saved.dtype == torch.float64
    torch.testing.assert_close(layer.saved, original_saved, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not has_gil_disabled(), reason="requires Python free-threading with PYTHON_GIL=0")
def test_qvq_output_alignment_runs_on_the_single_threadx_cuda_owner_with_exact_fixed_trellis():
    attachment, layer, named, _ = _prepare_attachment(padding_outlier=True)
    device = torch.device("cuda:0")
    layer.to(device)
    named.module_dtype = torch.float16
    original_trellis = named.state["trellis"].clone()

    def align_on_owner():
        assert gptqmodel.DEVICE_THREAD_POOL.is_device_owner_thread(device)
        return attachment.align_layer(0)

    result = gptqmodel.DEVICE_THREAD_POOL.do(device, align_on_owner)
    torch.cuda.synchronize(device)

    assert result["accepted"] == 1.0
    assert result["gradient_scaling"] == 1.0
    assert result["runtime_candidate_validation_loss"] < result["runtime_baseline_validation_loss"]
    torch.testing.assert_close(named.state["trellis"], original_trellis, rtol=0, atol=0)
    assert layer.proj.weight.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_trellis_alignment_matches_stable_fp16_transform_with_fp32_inner_accumulation():
    device = torch.device("cuda:0")
    width = 2048
    source = torch.randn(1, width, device=device, dtype=torch.float16) * 0.01
    SU = torch.linspace(0.75, 1.25, width, device=device, dtype=torch.float32)
    SV = torch.linspace(1.25, 0.75, width, device=device, dtype=torch.float32)
    bias = torch.linspace(-0.1, 0.1, width, device=device, dtype=torch.float32)
    inner = torch.eye(width, device=device, dtype=torch.float32)
    module = _FixedTrellisAlignmentLinear(
        inner_weight=inner,
        SU=SU,
        SV=SV,
        bias=bias,
        output_dtype=torch.float16,
    )

    actual = module(source)
    transformed = matmul_hadU_stable(source.to(torch.float32) * SU)
    accumulated = transformed.to(torch.float32) @ inner
    expected = _qvq_fp16_emulated_hadamard_fallback(
        accumulated,
        post_scale=SV,
        bias=bias,
        scale_mode=3,
    ).to(torch.float16)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    module.output_dtype = None
    assert module(source.float()).dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_trellis_alignment_prevents_autocast_from_narrowing_fp32_inner_result():
    device = torch.device("cuda:0")
    width = 2048
    source = torch.ones(1, width, device=device, dtype=torch.float32)
    module = _FixedTrellisAlignmentLinear(
        inner_weight=torch.eye(width, device=device, dtype=torch.float32) * 2_000,
        SU=torch.ones(width, device=device, dtype=torch.float32),
        SV=torch.full((width,), 1e-3, device=device, dtype=torch.float32),
        bias=None,
        output_dtype=None,
    )

    # The inner result exceeds FP16 range, but the following normalized
    # Hadamard and SV scale make the completed linear output small and finite.
    # Before the explicit autocast exclusion, CUDA autocast narrowed the `@`
    # result to FP16 and irreversibly introduced infinity here.
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        actual = module(source)
    with torch.autocast(device_type="cuda", enabled=False):
        expected = module(source)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qvq_output_alignment_sequential_pass_adapts_future_dense_weights_and_retains_layer_state():
    attachment, layer, named, original_future, hooked_future = _prepare_sequential_attachment()
    original_trellis = named.state["trellis"].clone()

    result = attachment.align_layer(0, finalize=False)

    assert result["accepted"] == 1.0
    assert result["alignment_pass"] == 1.0
    assert result["staged_modules"] == 1.0
    assert result["finalize"] == 0.0
    assert 0 in attachment._layers
    assert not torch.equal(layer.future.weight, original_future)
    assert layer.future.weight.dtype == original_future.dtype
    assert layer.future is hooked_future
    torch.testing.assert_close(named.state["trellis"], original_trellis, rtol=0, atol=0)
    attachment.discard_layer(0)


def test_qvq_output_alignment_later_pass_readapts_prior_payload_and_commits_both_modules():
    attachment, layer, first, _, _ = _prepare_sequential_attachment()
    first_trellis = first.state["trellis"].clone()
    first_result = attachment.align_layer(0, finalize=False)
    first_SV_after_initial_pass = first.state["SV"].clone()

    future_trellis = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (1, qvq_words_per_tile(2)),
        dtype=torch.int32,
    )
    future_SU = torch.ones(16, dtype=torch.float32)
    future_SV = torch.ones(16, dtype=torch.float32)
    future_inner = reconstruct_qvq_inner_weight(
        future_trellis,
        bits=2,
        in_features=16,
        out_features=16,
        codebook_version=PGC16_CODEBOOK_VERSION,
    )
    layer.future.weight.data.copy_(rht_reconstruct_weight(future_inner, future_SU, future_SV))
    future = NamedModule(layer.future, name="future", full_name="model.layers.0.future", layer_index=0)
    future.state.update(
        {
            "trellis": future_trellis,
            "SU": future_SU.clone(),
            "SV": future_SV.clone(),
            "_qvq_runtime_config": (2.0, PGC16_CODEBOOK_VERSION),
            "module_tree_flags": frozenset(),
        }
    )
    attachment.register_module(future)

    second_result = attachment.align_layer(0)

    assert first_result["accepted"] == 1.0
    assert second_result["accepted"] == 1.0
    assert second_result["alignment_pass"] == 2.0
    assert second_result["staged_modules"] == 2.0
    assert not torch.equal(first.state["SV"], first_SV_after_initial_pass)
    assert not torch.equal(future.state["SV"], future_SV)
    torch.testing.assert_close(first.state["trellis"], first_trellis, rtol=0, atol=0)
    torch.testing.assert_close(future.state["trellis"], future_trellis, rtol=0, atol=0)
    for named, live in ((first, layer.first), (future, layer.future)):
        inner = reconstruct_qvq_inner_weight(
            named.state["trellis"],
            bits=2,
            in_features=16,
            out_features=16,
            codebook_version=PGC16_CODEBOOK_VERSION,
        )
        expected = rht_reconstruct_weight(inner, named.state["SU"], named.state["SV"])
        torch.testing.assert_close(live.weight, expected, rtol=1e-5, atol=1e-5)


def test_qvq_output_alignment_rejects_candidate_and_restores_weight_and_staged_auxiliaries():
    attachment, layer, named, current_weight = _prepare_attachment(
        config=OutputAlignConfig(learning_rate=0.03, epochs=1, validation_fraction=0.5)
    )
    original_SU = named.state["SU"].clone()
    original_SV = named.state["SV"].clone()

    with patch.object(attachment, "_evaluate", side_effect=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0]):
        result = attachment.align_layer(0)

    assert result["accepted"] == 0.0
    torch.testing.assert_close(layer.proj.weight, current_weight, rtol=0, atol=0)
    torch.testing.assert_close(named.state["SU"], original_SU, rtol=0, atol=0)
    torch.testing.assert_close(named.state["SV"], original_SV, rtol=0, atol=0)


def test_qvq_output_alignment_exception_after_replacement_is_fully_transactional():
    attachment, layer, named, current_weight = _prepare_attachment()
    original_module = layer.proj
    original_SU = named.state["SU"].clone()
    original_SV = named.state["SV"].clone()

    with (
        patch.object(attachment, "_evaluate", side_effect=[1.0, RuntimeError("forced validation failure")]),
        pytest.raises(RuntimeError, match="forced validation failure"),
    ):
        attachment.align_layer(0)

    assert layer.proj is original_module
    torch.testing.assert_close(layer.proj.weight, current_weight, rtol=0, atol=0)
    torch.testing.assert_close(named.state["SU"], original_SU, rtol=0, atol=0)
    torch.testing.assert_close(named.state["SV"], original_SV, rtol=0, atol=0)


def test_qvq_output_alignment_requires_disjoint_training_and_validation_batches():
    attachment, _, _, _ = _prepare_attachment()
    state = attachment._layers[0]
    state.replay_batches = state.replay_batches[:1]

    with pytest.raises(RuntimeError, match="at least two calibration batches"):
        attachment.align_layer(0)


def test_qvq_output_alignment_uses_module_tree_tags_to_reject_moe():
    attachment, layer, named, current_weight = _prepare_attachment()
    named.state["module_tree_flags"] = frozenset({"moe", "routed"})

    with pytest.raises(NotImplementedError, match="MoE decoder layers"):
        attachment.align_layer(0)

    torch.testing.assert_close(layer.proj.weight, current_weight, rtol=0, atol=0)


def test_qvq_output_alignment_nested_replay_values_are_owned_and_moved_without_aliasing():
    source = torch.tensor([1.0])
    nested = {"tuple": (source, 3), "list": [source], "scalar": "keep"}

    cloned = QVQOutputAlignmentAttachment._cpu_clone(nested)
    moved = QVQOutputAlignmentAttachment._move_value(cloned, torch.device("cpu"))

    assert moved["scalar"] == "keep"
    assert moved["tuple"][1] == 3
    assert moved["tuple"][0] is not source
    assert moved["list"][0] is not cloned["list"][0]
    torch.testing.assert_close(moved["tuple"][0], source, rtol=0, atol=0)


@pytest.mark.parametrize("value", ([], (), 1, "tensor"))
def test_qvq_output_alignment_rejects_invalid_primary_outputs(value):
    exception = ValueError if isinstance(value, (list, tuple)) else TypeError
    with pytest.raises(exception, match="empty layer output|tensor decoder-layer outputs"):
        QVQOutputAlignmentAttachment._primary(value)


def test_qvq_output_alignment_context_capture_is_bounded_idempotent_and_validates_alignment():
    attachment = QVQOutputAlignmentAttachment(
        OutputAlignConfig(maximum_train_batches=1, maximum_validation_batches=1)
    )
    tensor = torch.ones(1, 2, 16)

    with pytest.raises(RuntimeError, match="input/output batch counts"):
        attachment.receive_layer_forward_context(
            layer_index=0,
            layer_inputs=[[tensor]],
            layer_input_kwargs=[{}],
            layer_outputs=[],
            position_ids=[],
            attention_masks=[],
        )
    with pytest.raises(RuntimeError, match="input kwargs"):
        attachment.receive_layer_forward_context(
            layer_index=0,
            layer_inputs=[[tensor], [tensor]],
            layer_input_kwargs=[{}],
            layer_outputs=[[tensor], [tensor]],
            position_ids=[],
            attention_masks=[],
        )

    attachment.receive_layer_forward_context(
        layer_index=0,
        layer_inputs=[[tensor], [tensor], [tensor]],
        layer_input_kwargs=[],
        layer_outputs=[[tensor], [tensor], [tensor]],
        position_ids=[],
        attention_masks=[],
    )
    assert len(attachment._layers[0].replay_batches) == 2

    first_layer = torch.nn.Identity()
    attachment.receive_pristine_layer_module(layer_index=0, layer_module=first_layer)
    attachment.receive_pristine_layer_module(layer_index=0, layer_module=torch.nn.Linear(1, 1))
    assert attachment._layers[0].layer_module is first_layer
    assert attachment._layers[0].replay_batches[0].position_ids is None
    assert attachment._layers[0].replay_batches[0].attention_mask is None

    attachment.receive_layer_forward_context(
        layer_index=0,
        layer_inputs=[[tensor]],
        layer_input_kwargs=[{}],
        layer_outputs=[[tensor]],
        position_ids=[],
        attention_masks=[],
    )
    assert len(attachment._layers[0].replay_batches) == 2


def test_qvq_output_alignment_advances_owned_clean_inputs_and_never_trains_on_noisy_stream():
    attachment = QVQOutputAlignmentAttachment(
        OutputAlignConfig(maximum_train_batches=1, maximum_validation_batches=1)
    )
    first_input = [[torch.full((1, 2, 16), 1.0)]]
    first_output = [[torch.full((1, 2, 16), 2.0)]]
    clean_second_input = [[torch.full((1, 2, 16), 3.0)]]
    noisy_second_input = [[torch.full((1, 2, 16), 30.0)]]
    second_output = [[torch.full((1, 2, 16), 4.0)]]

    assert attachment.clean_group_layer_inputs(layer_index=0, layer_inputs=first_input) is first_input
    attachment.receive_clean_layer_inputs(layer_index=0, layer_inputs=clean_second_input)
    attachment.receive_layer_forward_context(
        layer_index=0,
        layer_inputs=first_input,
        layer_input_kwargs=[{}],
        layer_outputs=first_output,
        position_ids=[None],
        attention_masks=[None],
    )

    selected = attachment.clean_group_layer_inputs(layer_index=1, layer_inputs=noisy_second_input)
    assert selected is not clean_second_input
    torch.testing.assert_close(selected[0][0], clean_second_input[0][0], rtol=0, atol=0)
    clean_second_input[0][0].fill_(99.0)
    attachment.receive_layer_forward_context(
        layer_index=1,
        layer_inputs=noisy_second_input,
        layer_input_kwargs=[{}],
        layer_outputs=second_output,
        position_ids=[None],
        attention_masks=[None],
    )

    replay = attachment._layers[1].replay_batches[0]
    torch.testing.assert_close(replay.inputs[0], torch.full((1, 2, 16), 3.0), rtol=0, atol=0)
    assert not torch.equal(replay.inputs[0], noisy_second_input[0][0])
    assert attachment._active_clean_layer_inputs == {}
    assert attachment._clean_layer_inputs == {}

    attachment.receive_clean_layer_inputs(layer_index=1, layer_inputs=second_output)
    assert 2 in attachment._clean_layer_inputs
    attachment.clear()
    assert attachment._layers == {}
    assert attachment._clean_layer_inputs == {}
    assert attachment._active_clean_layer_inputs == {}


def test_qvq_output_alignment_forward_rebuilds_model_kwargs_and_excludes_past_cache():
    captured = {}

    class Layer(torch.nn.Module):
        def forward(self, source, **kwargs):
            captured.update(kwargs)
            return source + 1

    def prepare_layer_replay_kwargs(**kwargs):
        assert kwargs["target_device"] == torch.device("cpu")
        kwargs["additional_inputs"]["prepared"] = True
        return kwargs["additional_inputs"]

    attachment = QVQOutputAlignmentAttachment(OutputAlignConfig())
    batch = _ReplayBatch(
        inputs=[torch.zeros(1, 2, 16)],
        input_kwargs={"past_key_value": torch.ones(1), "nested": {"value": torch.ones(1)}},
        target=torch.ones(1, 2, 16),
        position_ids=torch.arange(2).unsqueeze(0),
        attention_mask=None,
    )
    with pytest.raises(RuntimeError, match="not bound"):
        attachment._forward_batch(Layer(), batch, torch.device("cpu"))

    attachment.bind_model(SimpleNamespace(prepare_layer_replay_kwargs=prepare_layer_replay_kwargs))
    output = attachment._forward_batch(Layer(), batch, torch.device("cpu"))

    assert output.shape == batch.target.shape
    assert "past_key_value" not in captured
    assert captured["attention_mask"] is None
    assert captured["use_cache"] is False
    assert captured["prepared"] is True
    assert torch.equal(captured["position_ids"], batch.position_ids)


def test_qvq_output_alignment_loss_guards_shape_padding_and_nonfinite_evaluation():
    prediction = torch.ones(1, 2, 4)
    with pytest.raises(RuntimeError, match="prediction shape"):
        QVQOutputAlignmentAttachment._masked_loss(prediction, torch.ones(1, 3, 4), None)
    with pytest.raises(RuntimeError, match="no non-padding"):
        QVQOutputAlignmentAttachment._masked_loss(
            prediction,
            prediction,
            torch.zeros(1, 2, dtype=torch.bool),
        )
    assert QVQOutputAlignmentAttachment._masked_loss(torch.ones(2, 4), torch.zeros(2, 4), None).item() == 1

    attachment = QVQOutputAlignmentAttachment(OutputAlignConfig())
    batch = _ReplayBatch([prediction], {}, prediction, None, None)
    with patch.object(attachment, "_forward_batch", return_value=torch.full_like(prediction, float("nan"))):
        with pytest.raises(RuntimeError, match="non-finite validation loss"):
            attachment._evaluate(torch.nn.Identity(), [batch], torch.device("cpu"))


def test_qvq_output_alignment_rebuilds_inference_parameters_and_buffers_for_autograd():
    with torch.inference_mode():
        layer = torch.nn.Linear(4, 4)
        layer.register_buffer("saved", torch.ones(1))
    assert layer.weight.is_inference()
    assert layer.saved.is_inference()

    QVQOutputAlignmentAttachment._rebuild_inference_tensors(layer)

    assert not layer.weight.is_inference()
    assert not layer.saved.is_inference()
    # Normal tensors are intentionally retained by identity.
    normal = torch.nn.Linear(4, 4)
    normal.register_buffer("saved", torch.ones(1))
    normal_weight = normal.weight
    normal_buffer = normal.saved
    QVQOutputAlignmentAttachment._rebuild_inference_tensors(normal)
    assert normal.weight is normal_weight
    assert normal.saved is normal_buffer


def test_qvq_output_alignment_bias_path_and_non_linear_guard():
    module = _FixedTrellisAlignmentLinear(
        inner_weight=torch.eye(16),
        SU=torch.ones(16),
        SV=torch.ones(16),
        bias=torch.full((16,), 2.0),
        output_dtype=torch.float32,
    )
    output = module(torch.zeros(1, 16))
    torch.testing.assert_close(output, torch.full((1, 16), 2.0), rtol=0, atol=0)

    root = torch.nn.Module()
    root.embedding = torch.nn.Embedding(16, 16)
    named = NamedModule(root.embedding, name="embedding", full_name="embedding", layer_index=0)
    attachment = QVQOutputAlignmentAttachment(OutputAlignConfig())
    with pytest.raises(NotImplementedError, match="nn.Linear"):
        attachment._build_temporary_module(named, torch.device("cpu"))


def test_qvq_output_alignment_state_guards_and_clear_are_exact():
    attachment = QVQOutputAlignmentAttachment(OutputAlignConfig())
    assert attachment.layer_is_fully_staged(0) is False
    assert attachment.modules_are_fully_staged(0, {"proj"}) is False
    assert attachment.align_layer(0) is None

    root = torch.nn.Module()
    root.proj = torch.nn.Linear(16, 16, bias=False)
    named = NamedModule(root.proj, name="proj", full_name="proj", layer_index=0)
    attachment.register_module(named)
    assert attachment.layer_is_fully_staged(0) is False
    assert attachment.modules_are_fully_staged(0, set()) is False
    assert attachment.modules_are_fully_staged(0, {"proj"}) is False
    named.state.update({"trellis": 1, "SU": 1, "SV": 1, "_qvq_runtime_config": 1})
    assert attachment.layer_is_fully_staged(0) is True
    assert attachment.modules_are_fully_staged(0, {"proj"}) is True
    attachment.discard_layer(0)
    assert attachment.layer_is_fully_staged(0) is False

    attachment._layers[1] = _LayerAlignmentState()
    assert attachment.align_layer(1) is None
    attachment.register_module(named)
    with pytest.raises(RuntimeError, match="did not capture decoder layer"):
        attachment.align_layer(0)
    attachment._layers[2] = _LayerAlignmentState()
    attachment.clear()
    assert attachment._layers == {}


def test_qvq_output_alignment_rejects_cross_device_decoder_layer_before_mutation():
    attachment = QVQOutputAlignmentAttachment(OutputAlignConfig())
    attachment._layers[0] = _LayerAlignmentState(
        modules={
            "cpu": SimpleNamespace(
                module=SimpleNamespace(weight=torch.empty(1)),
                state={"trellis": 1, "SU": 1, "SV": 1, "_qvq_runtime_config": 1},
            ),
            "meta": SimpleNamespace(
                module=SimpleNamespace(weight=torch.empty(1, device="meta")),
                state={"trellis": 1, "SU": 1, "SV": 1, "_qvq_runtime_config": 1},
            ),
        },
        layer_module=torch.nn.Identity(),
        replay_batches=[object(), object()],
    )

    with pytest.raises(RuntimeError, match="share a device"):
        attachment.align_layer(0)


def test_qvq_output_alignment_identity_and_nonfinite_training_failures_restore_exact_state():
    attachment, layer, named, current_weight = _prepare_attachment()
    replacement = torch.nn.Linear(16, 16, bias=False)
    layer.proj = replacement
    with pytest.raises(RuntimeError, match="lost live module identity"):
        attachment.align_layer(0)
    assert layer.proj is replacement

    attachment, layer, named, current_weight = _prepare_attachment()
    original_module = layer.proj
    original_SU = named.state["SU"].clone()
    original_SV = named.state["SV"].clone()
    with (
        patch.object(attachment, "_evaluate", side_effect=[1.0, 1.0, 1.0]),
        patch.object(attachment, "_masked_loss", return_value=torch.tensor(float("nan"), requires_grad=True)),
        pytest.raises(RuntimeError, match="non-finite training loss"),
    ):
        attachment.align_layer(0)
    assert layer.proj is original_module
    torch.testing.assert_close(layer.proj.weight, current_weight, rtol=0, atol=0)
    torch.testing.assert_close(named.state["SU"], original_SU, rtol=0, atol=0)
    torch.testing.assert_close(named.state["SV"], original_SV, rtol=0, atol=0)

    attachment, layer, named, current_weight = _prepare_attachment(
        config=OutputAlignConfig(learning_rate=0.03, epochs=1, validation_fraction=0.5)
    )
    original_module = layer.proj
    with (
        patch.object(
            attachment,
            "_evaluate",
            side_effect=[1.0, 1.0, 1.0, 1.0, 1.0, RuntimeError("dense candidate failed")],
        ),
        pytest.raises(RuntimeError, match="dense candidate failed"),
    ):
        attachment.align_layer(0)
    assert layer.proj is original_module
    torch.testing.assert_close(layer.proj.weight, current_weight, rtol=0, atol=0)
