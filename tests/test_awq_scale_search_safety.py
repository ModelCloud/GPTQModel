# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""CPU oracles for AWQ candidate coordinates, replay coverage and rollback."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState, _AWQScaleGroupError
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.config import AWQConfig


class _RoutedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=True)
        self.router = nn.Linear(2, 2, bias=True)
        self.shared_gate = nn.Linear(2, 1, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[0.8, -0.2], [0.1, 0.6]]))
            self.router.weight.copy_(torch.eye(2))

    def forward(self, x):
        logits = self.router(x)
        selected = F.one_hot(logits.argmax(-1), num_classes=2).to(x.dtype)
        return self.proj(x) * selected * self.shared_gate(x).sigmoid()


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(2)
        self.block = _RoutedBlock()

    def forward(self, x):
        return self.block(self.norm(x))


def _processor(layer, chunked=True, fallback=None):
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu", group_size=2, sym=True,
                       scale_search_chunked_activations=chunked, fallback=fallback),
        calibration=CalibrationConfig(),
        gptq_model=SimpleNamespace(rotary_embedding=None, model=layer),
        model=layer,
        calculate_w_wq_diff=False,
    )
    processor._quant_batch_size = 0
    return processor


def _snapshot(layer):
    return {name: value.clone() for name, value in layer.state_dict().items()}


def _assert_pristine(layer, before):
    for name, value in layer.state_dict().items():
        assert torch.equal(value, before[name]), name


def _search(processor, layer, x):
    return processor._search_best_scale(
        layer, layer.norm, [layer.block.proj], x, module2inspect=layer.block,
        scale_consumers=["block.router", "block.shared_gate"],
    )


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_identity_quantization_preserves_every_candidate_router(chunked, dtype, monkeypatch):
    torch.manual_seed(0)
    layer = _Layer().to(dtype)
    processor = _processor(layer, chunked)
    x = torch.tensor([[2., 1.], [1., 3.]], dtype=dtype)
    reference = layer.block.router(x).clone()
    before = _snapshot(layer)
    logits = []
    handle = layer.block.router.register_forward_hook(lambda _m, _args, out: logits.append(out.clone()))
    monkeypatch.setattr(processor, "_pseudo_quantize_tensor_into", lambda src, dst, **_kw: dst.copy_(src))
    try:
        _search(processor, layer, x)
    finally:
        handle.remove()
    assert len(logits) == 21
    for candidate in logits:
        assert torch.equal(candidate, reference)
        assert torch.equal(candidate.argmax(-1), reference.argmax(-1))
    _assert_pristine(layer, before)


def _quantize_oracle(weight):
    scale = weight.abs().amax(-1, keepdim=True).clamp_min(1e-5) / 7
    return (weight / scale).round().clamp(-8, 7) * scale


@pytest.mark.parametrize("chunked", [False, True])
@torch.inference_mode()
def test_search_matches_independent_folded_model_oracle(chunked):
    torch.manual_seed(5)
    layer = _Layer()
    processor = _processor(layer, chunked)
    x = torch.tensor([[2., 1.], [0.3, -1.2], [-0.2, 0.8]])
    baseline = layer.block(x)
    before = _snapshot(layer)
    weight = layer.block.proj.weight.clone()
    w_mean = (weight.abs() / (weight.abs().amax(-1, keepdim=True) + 1e-6)).mean(0)
    x_mean = x.abs().mean(0)
    losses, candidates = [], []
    for index in range(20):
        ratio = index / 20
        scale = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp_min(1e-4)
        scale /= (scale.max() * scale.min()).sqrt()
        folded = copy.deepcopy(layer.block)
        folded.proj.weight.copy_(_quantize_oracle(weight * scale))
        folded.router.weight.mul_(scale)
        folded.shared_gate.weight.mul_(scale)
        output = folded(x / scale)
        losses.append((baseline - output).float().square().mean().item())
        candidates.append(scale)
    best = min(range(20), key=losses.__getitem__)
    _prev, _targets, actual, loss = _search(processor, layer, x)
    torch.testing.assert_close(actual, candidates[best], rtol=1e-6, atol=1e-6)
    assert loss == pytest.approx(losses[best], rel=1e-6, abs=1e-9)
    _assert_pristine(layer, before)

    # Final application must agree with a separately transformed complete layer.
    raw_x = torch.tensor([[1., 4.], [2., -3.]])
    oracle = copy.deepcopy(layer)
    oracle.norm.weight.div_(actual)
    oracle.norm.bias.div_(actual)
    oracle.block.proj.weight.copy_(_quantize_oracle(weight * actual))
    oracle.block.router.weight.mul_(actual)
    oracle.block.shared_gate.weight.mul_(actual)
    apply_scale(layer, [("norm", ["block.proj"], actual, loss)],
                consumer_map={"norm": ["block.router", "block.shared_gate"]})
    layer.block.proj.weight.copy_(_quantize_oracle(layer.block.proj.weight))
    torch.testing.assert_close(layer(raw_x), oracle(raw_x), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failure", ["quantize", "reference", "replay"])
@torch.inference_mode()
def test_search_restores_weights_after_failure(chunked, failure, monkeypatch):
    torch.manual_seed(0)
    layer = _Layer()
    processor = _processor(layer, chunked)
    x = torch.tensor([[2., 1.]])
    before = _snapshot(layer)
    handles = []
    if failure == "quantize":
        def fail_quantize(src, dst, **_kw):
            dst.zero_()
            raise RuntimeError("injected quantizer failure")
        monkeypatch.setattr(processor, "_pseudo_quantize_tensor_into", fail_quantize)
    else:
        calls = []

        def fail_replay(_module, _args):
            calls.append(1)
            if len(calls) == (1 if failure == "reference" else 3):
                raise RuntimeError("injected replay failure")
        handles.append(layer.block.register_forward_pre_hook(fail_replay))
    try:
        with pytest.raises(RuntimeError, match="injected"):
            _search(processor, layer, x)
    finally:
        for handle in handles:
            handle.remove()
    _assert_pristine(layer, before)
    assert not layer.block.proj._forward_pre_hooks
    assert not layer.block.router._forward_pre_hooks
    assert not layer.block.shared_gate._forward_pre_hooks


class _ParallelLayer(nn.Module):
    def __init__(self, omit_right=False):
        super().__init__()
        self.norm = nn.LayerNorm(2)
        self.left = nn.Linear(2, 2)
        self.right = nn.Linear(2, 2)
        self.omit_right = omit_right

    def forward(self, x):
        return self.left(x) if self.omit_right else self.left(x) + self.right(x)


@torch.inference_mode()
def test_incomplete_candidate_chunk_stream_raises_and_restores_weights(monkeypatch):
    layer = _Layer()
    processor = _processor(layer)
    original_replay = processor._iter_prepared_forward_outputs
    before = _snapshot(layer)
    calls = []

    def missing_candidate_chunks(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            yield from original_replay(*args, **kwargs)

    monkeypatch.setattr(processor, "_iter_prepared_forward_outputs", missing_candidate_chunks)
    with pytest.raises(ValueError, match="zip"):
        _search(processor, layer, torch.tensor([[2., 1.]]))
    _assert_pristine(layer, before)


def _parallel_group(layer, boundary):
    return {
        "prev_op": layer.norm, "layers": [layer.left, layer.right],
        "inp": torch.tensor([[2., 1.]]), "module2inspect": boundary,
    }


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("incomplete", ["boundary", "execution"])
def test_incomplete_objective_rejected_before_candidate_mutation(chunked, incomplete, monkeypatch):
    layer = _ParallelLayer(omit_right=incomplete == "execution")
    processor = _processor(layer, chunked)
    boundary = layer.left if incomplete == "boundary" else layer
    before = _snapshot(layer)

    def unexpected_quantize(*_args, **_kwargs):
        pytest.fail("invalid replay must be rejected before quantizing a candidate")
    monkeypatch.setattr(processor, "_pseudo_quantize_tensor_into", unexpected_quantize)
    with pytest.raises(_AWQScaleGroupError, match="right"):
        processor._search_best_scale(layer, **_parallel_group(layer, boundary))
    _assert_pristine(layer, before)
    assert not layer.left._forward_pre_hooks
    assert not layer.right._forward_pre_hooks


@pytest.mark.parametrize("chunked", [False, True])
def test_complete_parallel_objective_is_supported(chunked):
    torch.manual_seed(7)
    layer = _ParallelLayer()
    processor = _processor(layer, chunked)
    group = _parallel_group(layer, layer)
    x = group["inp"]
    weights = [target.weight.detach().clone() for target in group["layers"]]
    normalized = torch.cat([weight.abs() / (weight.abs().amax(-1, keepdim=True) + 1e-6) for weight in weights])
    w_mean = normalized.mean(0)
    x_mean = x.abs().mean(0)
    with torch.no_grad():
        reference = layer(x)
        losses, scales = [], []
        for index in range(20):
            ratio = index / 20
            scale = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp_min(1e-4)
            scale /= (scale.max() * scale.min()).sqrt()
            output = sum(F.linear(x, _quantize_oracle(weight * scale) / scale, target.bias)
                         for weight, target in zip(weights, group["layers"]))
            losses.append((reference - output).square().mean().item())
            scales.append(scale)
    result = processor._search_best_scale(layer, **group)
    best = min(range(20), key=losses.__getitem__)
    assert result[1] == ("left", "right")
    torch.testing.assert_close(result[2], scales[best], rtol=1e-6, atol=1e-6)
    assert result[3] == pytest.approx(losses[best], abs=1e-9, rel=1e-6)


@pytest.mark.parametrize("invalid", ["source", "target", "width", "consumer", "tied"])
def test_invalid_group_bindings_are_rejected(invalid):
    layer = _ParallelLayer()
    group = _parallel_group(layer, layer)
    if invalid == "source":
        group["prev_op"] = nn.LayerNorm(2)
    elif invalid == "target":
        group["layers"] = [nn.Linear(2, 2)]
    elif invalid == "width":
        layer.right = nn.Linear(3, 2)
        group["layers"] = [layer.left, layer.right]
    elif invalid == "consumer":
        group["scale_consumers"] = ["nonexistent.router"]
    else:
        layer.right.weight = layer.left.weight
    with pytest.raises(_AWQScaleGroupError):
        _processor(layer)._search_best_scale(layer, **group)


@pytest.mark.parametrize("conflict", ["source", "target", "tied", "consumer"])
def test_conflicting_groups_rejected_before_search(conflict):
    layer = _ParallelLayer()
    layer.other_norm = nn.LayerNorm(2)
    groups = [{"prev_op": layer.norm, "layers": [layer.left]},
              {"prev_op": layer.other_norm, "layers": [layer.right]}]
    if conflict == "source":
        groups[1]["prev_op"] = layer.norm
    elif conflict == "target":
        groups[1]["layers"] = [layer.left]
    elif conflict == "tied":
        layer.right.weight = layer.left.weight
    else:
        groups[0]["_scale_consumers"] = ["right"]
    with pytest.raises(_AWQScaleGroupError):
        AWQProcessor._validate_scale_group_conflicts(layer, groups)


@pytest.mark.parametrize("allow_fallback", [False, True])
def test_layer_uses_configured_fallback_for_incomplete_group(allow_fallback, monkeypatch, caplog):
    layer = _ParallelLayer()
    processor = _processor(layer, fallback={} if allow_fallback else None)
    state = _AWQLayerState(layer_module=layer, modules={
        name: NamedModule(getattr(layer, name), name=name, full_name=f"layer.{name}", layer_index=0)
        for name in ("left", "right")
    })
    features = {name: torch.ones(1, 2) for name in state.modules}
    monkeypatch.setattr(processor, "_layer_input_features", lambda _state: features)
    processor.gptq_model.awq_get_modules_for_scaling = lambda *_args: [_parallel_group(layer, layer.left)]
    quantized = []
    monkeypatch.setattr(processor, "apply_quant", lambda modules, scales_list: quantized.append((set(modules), scales_list)))
    if allow_fallback:
        processor._quantize_layer(0, state)
        assert (set(features), []) in quantized
        assert state.quantized
        assert "configured plain-quantization fallback" in caplog.text
    else:
        with pytest.raises(_AWQScaleGroupError, match="right"):
            processor._quantize_layer(0, state)
        assert not state.quantized
        assert not quantized


@pytest.mark.parametrize("filtered", ["low_sample", "missing_features"])
@pytest.mark.parametrize("conflict", ["source", "tied_target"])
def test_filtering_cannot_hide_a_conflicting_fold(filtered, conflict, monkeypatch):
    layer = _ParallelLayer()
    layer.other_norm = nn.LayerNorm(2)
    before = _snapshot(layer)
    processor = _processor(layer, fallback={})
    state = _AWQLayerState(layer_module=layer, modules={
        name: NamedModule(getattr(layer, name), name=name, full_name=f"layer.{name}", layer_index=0)
        for name in ("left", "right")
    })
    features = {name: torch.ones(1, 2) for name in state.modules}
    groups = [
        {"prev_op": layer.norm, "layers": [layer.left], "inp": features["left"], "module2inspect": layer.left},
        {"prev_op": layer.norm if conflict == "source" else layer.other_norm,
         "layers": [layer.right], "inp": features["right"], "module2inspect": layer.right},
    ]
    if conflict == "tied_target":
        layer.right.weight = layer.left.weight
        before = _snapshot(layer)
    if filtered == "missing_features":
        del features["right"]
    monkeypatch.setattr(processor, "_layer_input_features", lambda _state: features)
    monkeypatch.setattr(processor, "_should_fallback_group",
                        lambda names, _features: filtered == "low_sample" and names == ["right"])
    processor.gptq_model.awq_get_modules_for_scaling = lambda *_args: groups

    def must_not_search(*_args, **_kwargs):
        pytest.fail("A filtered conflict reached candidate search")

    monkeypatch.setattr(processor, "_search_best_scale", must_not_search)
    with pytest.raises(_AWQScaleGroupError):
        processor._quantize_layer(0, state)
    _assert_pristine(layer, before)
    assert not state.quantized
