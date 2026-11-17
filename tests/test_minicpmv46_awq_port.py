# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native MiniCPM-V 4.6 DeltaNet has four consumers of its input RMS norm."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import MiniCPMV4_6Config, MiniCPMV4_6ForConditionalGeneration, Qwen3_5TextConfig

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQScaleGroupError
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.models.definitions.minicpmv_4_6 import MiniCPMV4_6QModel
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.awq.utils.module import get_op_name
from gptqmodel.quantization.config import AWQConfig


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _config(heads=2, head_dim=8, hidden=32):
    text = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=hidden, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        head_dim=8, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=1, linear_num_value_heads=heads,
        linear_key_head_dim=8, linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=2, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    text._attn_implementation = "eager"
    return MiniCPMV4_6Config(
        text_config=text,
        vision_config={"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "image_size": 8, "patch_size": 2},
        insert_layer_id=0, image_size=8, image_token_id=62, video_token_id=63,
    )


@torch.inference_mode()
def _case(heads=2, head_dim=8, dtype=torch.float32):
    torch.manual_seed(4175)
    model = MiniCPMV4_6ForConditionalGeneration(_config(heads, head_dim)).to(dtype).eval()
    wrapper = MiniCPMV4_6QModel.__new__(MiniCPMV4_6QModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    layer = model.model.language_model.layers[0]
    # Avoid zero RMS gains in Qwen's 1+weight convention and amplify all branches.
    layer.input_layernorm.weight.copy_(torch.linspace(-.35, .45, 32, dtype=dtype))
    for name in ("in_proj_a", "in_proj_b", "in_proj_z"):
        getattr(layer.linear_attn, name).weight.mul_(4)
    ids = torch.tensor([[0, 0, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 9, 10], [4, 5, 6, 7, 8, 0, 0]])
    mask = ids.ne(0).long()
    features, contexts, outputs, handles = {}, {}, {}, []

    def record(name):
        def hook(_module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            features[name] = x.clone()
            contexts[name] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        return hook

    try:
        for name, module in layer.named_modules():
            handles.append(module.register_forward_pre_hook(record(name), with_kwargs=True))
        handles.append(layer.register_forward_hook(lambda _m, _a, out: outputs.update(layer=out.clone())))
        reference = model(ids, attention_mask=mask, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    return SimpleNamespace(model=model, wrapper=wrapper, layer=layer, ids=ids, mask=mask,
                           features=features, contexts=contexts, reference=reference, output=outputs["layer"])


def _groups(case):
    return case.wrapper.awq_get_modules_for_scaling(
        case.layer, case.features,
        {"_awq_feature_kwargs": {"linear_attn.in_proj_qkv": case.contexts["linear_attn"]}},
    )


def _input_group(case):
    groups = [g for g in _groups(case) if g["prev_op"] is case.layer.input_layernorm]
    assert len(groups) == 1, "native linear-attention input norm must fold exactly once"
    return groups[0]


def _processor(case, chunked=True, fallback=None):
    proc = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu", group_size=8, sym=True, fallback=fallback,
                       scale_search_chunked_activations=chunked),
        calibration=CalibrationConfig(), gptq_model=case.wrapper, model=case.model,
        calculate_w_wq_diff=False,
    )
    proc._quant_batch_size = 0
    return proc


def _search(proc, case, group):
    return proc._search_best_scale(
        case.layer, group["prev_op"], group["layers"], group["inp"],
        module2inspect=group["module2inspect"], kwargs=group["kwargs"],
        scale_consumers=group.get("_scale_consumers"),
    )


def _snapshot(module):
    return {name: value.clone() for name, value in module.state_dict().items()}


def _pristine(module, before):
    assert module.state_dict().keys() == before.keys()
    for name, value in module.state_dict().items():
        assert torch.equal(value, before[name]), name


@pytest.mark.parametrize("heads,head_dim", [(1, 8), (2, 8), (1, 32), (2, 32)])
@torch.inference_mode()
def test_exact_source_four_consumers_and_neutral_full_layer(heads, head_dim):
    case = _case(heads, head_dim)
    group = _input_group(case)
    attn = case.layer.linear_attn
    assert group["layers"] == [attn.in_proj_qkv, attn.in_proj_z]
    assert group["_scale_consumers"] == ["linear_attn.in_proj_b", "linear_attn.in_proj_a"]
    assert group["module2inspect"] is attn
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
        torch.testing.assert_close(case.features[f"linear_attn.{name}"], group["inp"], rtol=0, atol=0)
    # Gated RMS norm is per VALUE HEAD and has a nonlinear z input. Even the
    # head_dim==hidden collision must never make it the input-projection source.
    assert all(g["prev_op"] is not attn.norm for g in _groups(case))
    before = _snapshot(case.layer)
    exercised = []

    def continuation():
        prefix = case.model(case.ids, attention_mask=case.mask, use_cache=True)
        return case.model(torch.tensor([[11], [12], [13]]), past_key_values=prefix.past_key_values,
                          attention_mask=torch.cat((case.mask, torch.ones(3, 1, dtype=torch.long)), -1)).logits

    continued = continuation()
    for current in _groups(case):
        source = get_op_name(case.layer, current["prev_op"])
        targets = tuple(get_op_name(case.layer, m) for m in current["layers"])
        scales = torch.linspace(.3, 2.3, current["inp"].shape[-1])
        apply_scale(case.layer, [(source, targets, scales, 0)],
                    consumer_map={source: current.get("_scale_consumers", [])})
        actual = case.model(case.ids, attention_mask=case.mask, use_cache=False).logits
        torch.testing.assert_close(actual, case.reference, rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(continuation(), continued, rtol=2e-5, atol=1e-6)
        case.layer.load_state_dict(before)
        exercised.append(source)
    assert exercised.count("input_layernorm") == 1
    assert "post_attention_layernorm" in exercised
    assert "mlp.up_proj" in exercised
    _pristine(case.layer, before)


@pytest.mark.parametrize("omitted", ["linear_attn.in_proj_b", "linear_attn.in_proj_a"])
@torch.inference_mode()
def test_omitting_either_unquantized_consumer_corrupts_full_native_output(omitted):
    case = _case(2, 32)
    group = _input_group(case)
    consumers = [name for name in group["_scale_consumers"] if name != omitted]
    targets = [get_op_name(case.layer, m) for m in group["layers"]]
    apply_scale(case.layer, [("input_layernorm", targets, torch.linspace(.3, 2.3, 32), 0)],
                consumer_map={"input_layernorm": consumers})
    actual = case.model(case.ids, attention_mask=case.mask, use_cache=False).logits
    assert (actual - case.reference).abs().max() > .01


def test_real_module_plan_keeps_unquantized_consumers_and_live_owners():
    from gptqmodel.models.loader import _convert_model_with_defuser
    from gptqmodel.models.module_plan import ModulePlan

    case = _case()
    _convert_model_with_defuser(MiniCPMV4_6QModel, case.model, cleanup_original=False)
    plan = ModulePlan.from_tree(MiniCPMV4_6QModel.module_tree)
    group = _input_group(case)
    for module in group["layers"]:
        name = get_op_name(case.layer, module)
        assert plan.find(name).quantize
        assert case.layer.state_dict()[name + ".weight"].data_ptr() == module.weight.data_ptr()
    for name in group["_scale_consumers"]:
        spec = plan.find(name)
        assert not spec.quantize and "!" in spec.flags
        assert case.layer.state_dict()[name + ".weight"].data_ptr() == case.layer.get_submodule(name).weight.data_ptr()
    assert plan.find("linear_attn.in_proj_qkv_decoy") is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [1, 2])
@torch.inference_mode()
def test_captured_native_padding_causality_and_partial_chunks(dtype, heads):
    case = _case(heads, 32, dtype)
    group = _input_group(case)
    proc = _processor(case)
    mixer, x, kwargs = group["module2inspect"], group["inp"], group["kwargs"]
    native = mixer(x, **kwargs)
    assert kwargs["attention_mask"].shape == (3, 7)
    assert torch.count_nonzero(kwargs["attention_mask"] == 0) == 4
    for chunk in (0, 1, 2):
        proc._quant_batch_size = chunk
        replay = proc._module_forward(x, mixer, kwargs)
        torch.testing.assert_close(replay, native, rtol=2e-3, atol=2e-3)
    changed = x.clone()
    changed[:, -2:] += torch.linspace(-2, 2, x.shape[-1], dtype=dtype)
    changed_out = mixer(changed, **kwargs)
    torch.testing.assert_close(changed_out[:, :-2], native[:, :-2], rtol=0, atol=0)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_identity_quantization_scores_every_candidate_and_keeps_b_a(chunked, dtype, monkeypatch):
    case = _case(2, 32, dtype)
    group = _input_group(case)
    proc = _processor(case, chunked)
    before = _snapshot(case.layer)
    references = {name: getattr(case.layer.linear_attn, name)(group["inp"])
                  for name in ("in_proj_b", "in_proj_a")}
    observations = {name: [] for name in references}
    outputs, handles = [], []
    for name in references:
        handles.append(getattr(case.layer.linear_attn, name).register_forward_hook(
            lambda _m, _a, out, name=name: observations[name].append(out.clone())))
    handles.append(case.layer.linear_attn.register_forward_hook(lambda _m, _a, out: outputs.append(out.clone())))
    monkeypatch.setattr(proc, "_pseudo_quantize_tensor_into", lambda src, dst, **_kw: dst.copy_(src))
    try:
        _search(proc, case, group)
    finally:
        for handle in handles:
            handle.remove()
    assert len(outputs) == 21  # native reference plus ALL twenty candidates
    tolerance = 1e-6 if dtype == torch.float32 else 2e-3
    for out in outputs[1:]:
        torch.testing.assert_close(out, outputs[0], rtol=tolerance, atol=tolerance)
    for name, observed in observations.items():
        assert len(observed) == 21
        for output in observed:
            torch.testing.assert_close(output, references[name], rtol=0, atol=0)
    _pristine(case.layer, before)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("heads", [1, 2])
@torch.inference_mode()
def test_search_full_and_partial_batch_native_context_agree(chunked, heads):
    case = _case(heads, 32)
    group = _input_group(case)
    proc = _processor(case, chunked)
    before = _snapshot(case.layer)
    whole = _search(proc, case, group)
    for chunk in (1, 2):
        proc._quant_batch_size = chunk
        proc.max_chunk_memory = group["inp"][0].numel() * 4 * 3 * chunk
        result = _search(proc, case, group)
        torch.testing.assert_close(result[2], whole[2], rtol=1e-6, atol=1e-6)
        assert result[3] == pytest.approx(whole[3], rel=1e-5, abs=1e-10)
    _pristine(case.layer, before)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("failure", ["reference", "quantizer", "candidate"])
@torch.inference_mode()
def test_failure_rolls_back_weights_and_observation_hooks(chunked, failure, monkeypatch):
    case = _case()
    group = _input_group(case)
    proc = _processor(case, chunked)
    before = _snapshot(case.layer)
    baseline_hooks = {id(m): (len(m._forward_hooks), len(m._forward_pre_hooks)) for m in case.layer.modules()}
    handles, calls = [], []
    if failure == "quantizer":
        def fail(src, dst, **_kwargs):
            dst.fill_(7)
            raise RuntimeError("injected quantizer failure")
        monkeypatch.setattr(proc, "_pseudo_quantize_tensor_into", fail)
    else:
        def fail(_module, _args):
            calls.append(1)
            if len(calls) == (1 if failure == "reference" else 3):
                raise RuntimeError("injected replay failure")
        handles.append(case.layer.linear_attn.register_forward_pre_hook(fail))
    try:
        with pytest.raises(RuntimeError, match="injected"):
            _search(proc, case, group)
    finally:
        for handle in handles:
            handle.remove()
    _pristine(case.layer, before)
    for module in case.layer.modules():
        assert baseline_hooks[id(module)] == (len(module._forward_hooks), len(module._forward_pre_hooks))
    monkeypatch.undo()
    retry = _search(proc, case, group)
    assert torch.isfinite(retry[2]).all()
    _pristine(case.layer, before)


@torch.inference_mode()
def test_incomplete_branch_observation_fails_closed(monkeypatch):
    case = _case()
    group = _input_group(case)
    # Preserve native execution, but hide one branch's module call from the
    # observation hooks. A complete source group must reject the replay.
    b = case.layer.linear_attn.in_proj_b
    original = b.forward
    monkeypatch.setattr(b, "_call_impl", original)
    with pytest.raises(_AWQScaleGroupError, match="in_proj_b"):
        _search(_processor(case), case, group)


def _quantize_oracle(weight):
    groups = weight.reshape(-1, 8)
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-5) / 7
    return ((groups / scale).round().clamp(-8, 7) * scale).reshape_as(weight)


@pytest.mark.parametrize("chunked", [False, True])
@torch.inference_mode()
def test_search_matches_independently_folded_four_branch_quantized_objective(chunked):
    case = _case(2, 32)
    group = _input_group(case)
    proc = _processor(case, chunked)
    before = _snapshot(case.layer)
    x = group["inp"]
    mixer = case.layer.linear_attn
    baseline = mixer(x, **group["kwargs"])
    weights = [m.weight.clone() for m in group["layers"]]
    w_mean = torch.cat([
        (w.reshape(-1, 8).abs() / (w.reshape(-1, 8).abs().amax(-1, keepdim=True) + 1e-6)).reshape_as(w)
        for w in weights
    ]).mean(0)
    x_mean = x.abs().reshape(-1, x.shape[-1]).mean(0)
    candidates, losses = [], []
    for index in range(20):
        ratio = index / 20
        scale = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp_min(1e-4)
        scale /= (scale.max() * scale.min()).sqrt()
        folded = copy.deepcopy(mixer)
        folded.in_proj_qkv.weight.copy_(_quantize_oracle(weights[0] * scale))
        folded.in_proj_z.weight.copy_(_quantize_oracle(weights[1] * scale))
        folded.in_proj_b.weight.mul_(scale)
        folded.in_proj_a.weight.mul_(scale)
        output = folded(x / scale, **group["kwargs"])
        losses.append((baseline - output).float().square().mean().item())
        candidates.append(scale)
    assert max(losses) > min(losses) * 1.05, "oracle must distinguish candidate scales"
    best = min(range(20), key=losses.__getitem__)
    result = _search(proc, case, group)
    torch.testing.assert_close(result[2], candidates[best], rtol=1e-6, atol=1e-6)
    assert result[3] == pytest.approx(losses[best], rel=1e-5, abs=1e-10)
    _pristine(case.layer, before)

    # Independent complete decoder transformation, including Qwen's 1+gain.
    oracle = copy.deepcopy(case.layer)
    scale = result[2]
    oracle.input_layernorm.weight.copy_((1 + oracle.input_layernorm.weight) / scale - 1)
    oracle.linear_attn.in_proj_qkv.weight.copy_(_quantize_oracle(weights[0] * scale))
    oracle.linear_attn.in_proj_z.weight.copy_(_quantize_oracle(weights[1] * scale))
    oracle.linear_attn.in_proj_b.weight.mul_(scale)
    oracle.linear_attn.in_proj_a.weight.mul_(scale)
    expected = oracle(case.features[""], **case.contexts[""])
    apply_scale(case.layer, [result], consumer_map={"input_layernorm": group["_scale_consumers"]})
    for target in group["layers"]:
        target.weight.copy_(_quantize_oracle(target.weight))
    actual = case.layer(case.features[""], **case.contexts[""])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("invalid", ["flat", "stale-mask", "causal-mask", "cache"])
@torch.inference_mode()
def test_unsupported_replay_rejects_stale_masks_and_mutable_cache(invalid):
    case = _case()
    group = _input_group(case)
    x, kwargs = group["inp"], dict(group["kwargs"])
    before = _snapshot(case.layer)
    if invalid == "flat":
        x = x.reshape(-1, x.shape[-1])
    elif invalid == "stale-mask":
        kwargs["attention_mask"] = kwargs["attention_mask"][:, :-1]
    elif invalid == "causal-mask":
        kwargs["attention_mask"] = torch.ones(3, 1, 7, 7)
    else:
        prefix = case.model(case.ids, attention_mask=case.mask, use_cache=True)
        cache = prefix.past_key_values
        conv = cache.get_conv_state(0).clone()
        recurrent = cache.get_recurrent_state(0).clone()
        kwargs["cache_params"] = cache
    with pytest.raises(_AWQScaleGroupError, match="unsupported replay"):
        _processor(case)._module_forward(x, group["module2inspect"], kwargs)
    if invalid == "cache":
        torch.testing.assert_close(cache.get_conv_state(0), conv, rtol=0, atol=0)
        torch.testing.assert_close(cache.get_recurrent_state(0), recurrent, rtol=0, atol=0)
    _pristine(case.layer, before)


@pytest.mark.parametrize("fallback", [None, {}])
def test_bad_context_obeys_configured_plain_fallback(monkeypatch, fallback):
    from gptqmodel.looper.named_module import NamedModule

    case = _case()
    group = _input_group(case)
    group["kwargs"]["attention_mask"] = torch.ones(3, 6)
    proc = _processor(case, fallback=fallback)
    proc.apply_clip = False
    state = proc._get_layer_state(0)
    state.layer_module = case.layer
    state.modules = {name: NamedModule(module, name, name, 0)
                     for name, module in case.layer.named_modules()
                     if isinstance(module, nn.Linear) and not name.endswith(("in_proj_a", "in_proj_b"))}
    proc._feature_task_names = set()
    proc._awq_feature_kwargs = {}
    monkeypatch.setattr(proc, "_layer_input_features", lambda _: case.features)
    monkeypatch.setattr(case.wrapper, "awq_get_modules_for_scaling", lambda *_: [group])
    calls = []
    monkeypatch.setattr(proc, "apply_quant", lambda named, scales_list: calls.append((set(named), scales_list)))
    expected = set(state.modules)
    before = _snapshot(case.layer)
    if fallback is None:
        with pytest.raises(_AWQScaleGroupError, match="captured batch/sequence mask"):
            proc._quantize_layer(0, state)
        assert not state.quantized and not calls
    else:
        proc._quantize_layer(0, state)
        assert state.quantized
        assert set().union(*(names for names, _ in calls)) == expected
        assert all(scales == [] for _, scales in calls)
    _pristine(case.layer, before)


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_public_tiny_awq_quantizes_packs_and_roundtrips_all_linears(tmp_path, heads, dtype):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        MiniCPMV4_6ImageProcessor,
        MiniCPMV4_6Processor,
        MiniCPMV4_6VideoProcessor,
        PreTrainedTokenizerFast,
    )

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    torch.manual_seed(19)
    model = MiniCPMV4_6ForConditionalGeneration(_config(heads, 32, hidden=64)).eval()
    model.save_pretrained(tmp_path)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3,
                                   "hello": 4, "world": 5}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
                                       eos_token="[EOS]", unk_token="[UNK]", extra_special_tokens={
                                           name: f"<{name}>" for name in (
                                               "image_token", "video_token", "image_start_token", "image_end_token",
                                               "slice_start_token", "slice_end_token", "image_id_start_token", "image_id_end_token",
                                           )
                                       })
    MiniCPMV4_6Processor(image_processor=MiniCPMV4_6ImageProcessor(),
                        video_processor=MiniCPMV4_6VideoProcessor(), tokenizer=tokenizer).save_pretrained(tmp_path)
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None, offload_to_disk=False)
    loaded = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=dtype)
    plan = loaded.module_plan()
    for name in ("linear_attn.in_proj_b", "linear_attn.in_proj_a"):
        assert not plan.find(name).quantize
    source = _snapshot(model.to(dtype))
    expected = {name for name, m in loaded.model.named_modules()
                if isinstance(m, nn.Linear) and ".language_model.layers." in name
                and not name.endswith(("in_proj_a", "in_proj_b"))}
    rows = [{"input_ids": torch.tensor([[4, 5, 4, 5, 4, 5, 4, 5]]),
             "attention_mask": torch.ones(1, 8, dtype=torch.long)} for _ in range(2)]
    loaded.quantize(rows, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    packed = {name for name, m in loaded.model.named_modules() if isinstance(m, AwqTorchLinear)}
    assert packed == expected
    assert len(packed) == 13
    norm = "model.language_model.layers.0.input_layernorm.weight"
    assert not torch.equal(loaded.model.state_dict()[norm], source[norm]), "real norm scale search/fold must execute"
    ids = torch.tensor([[4, 5, 4, 5]])
    with torch.inference_mode():
        reference = loaded.model(ids, use_cache=False).logits
    assert torch.isfinite(reference).all()
    state = _snapshot(loaded.model)
    for cycle in range(2):
        path = tmp_path / f"quantized-{cycle}"
        loaded.save(path)
        assert list(path.glob("*.safetensors")) and not list(path.glob("*.bin"))
        loaded = GPTQModel.load(str(path), backend=BACKEND.TORCH, device="cpu", dtype=dtype)
        _pristine(loaded.model, state)
        with torch.inference_mode():
            actual = loaded.model(ids, use_cache=False).logits
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
