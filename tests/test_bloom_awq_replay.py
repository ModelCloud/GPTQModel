# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Bloom replay must preserve native causal, padding and residual semantics."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, build_alibi_tensor
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor, _AWQScaleGroupError
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.bloom import BloomQModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.config import AWQConfig


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _context(x, heads):
    batch, seq = x.shape[:2]
    return {
        "attention_mask": torch.full((batch, 1, seq, seq), torch.finfo(x.dtype).min, dtype=x.dtype).triu(1),
        "alibi": build_alibi_tensor(torch.ones(batch, seq, dtype=torch.long), heads, x.dtype),
    }


def _case(*, batch=3, heads=4, post_layernorm=False, dtype=torch.float32):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = (
            BloomForCausalLM(
                BloomConfig(
                    hidden_size=32,
                    n_layer=1,
                    n_head=heads,
                    vocab_size=64,
                    apply_residual_connection_post_layernorm=post_layernorm,
                    attention_dropout=0,
                    hidden_dropout=0,
                )
            )
            .to(dtype)
            .eval()
        )
        ids = torch.randint(3, 64, (batch, 8))
    wrapper = BloomQModel.__new__(BloomQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    layer = model.transformer.h[0]
    features, calls, handles = {}, {}, []

    def record(name):
        def hook(_module, args, kwargs):
            features[name] = args[0].detach().clone()
            calls[name] = (args, dict(kwargs))

        return hook

    mask = torch.ones_like(ids)
    if batch > 1:
        mask[0, :2] = 0
    if batch > 2:
        mask[2, -2:] = 0
    try:
        for name, module in layer.named_modules():
            handles.append(module.register_forward_pre_hook(record(name), with_kwargs=True))
        with torch.inference_mode():
            reference = model(ids, attention_mask=mask, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()
    args, kwargs = calls["self_attention"]
    kwargs["residual"] = args[1]
    return SimpleNamespace(
        model=model,
        wrapper=wrapper,
        layer=layer,
        ids=ids,
        mask=mask,
        features=features,
        x=args[0],
        kwargs=kwargs,
        reference=reference,
    )


def _processor(case, *, chunked=True, fallback=None):
    return AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu", group_size=8, fallback=fallback, scale_search_chunked_activations=chunked),
        calibration=CalibrationConfig(batch_size=1),
        gptq_model=case.wrapper,
        model=case.model,
    )


def test_bloom_awq_replay_supplies_residual_for_mlp():
    """AWQ block replay must pass the residual argument BloomMLP.forward requires."""

    class _MLP(nn.Module):
        def forward(self, hidden_states, residual):
            raise AssertionError("replay kwargs are what this test exercises")

    instance = BloomQModel.__new__(BloomQModel)
    x = torch.randn(4, 16)
    kwargs = {}

    new_x, new_kwargs = BloomQModel.awq_module_replay_inputs(instance, _MLP(), x, kwargs)

    assert new_x.dim() == 3
    assert new_x.shape == (1, 4, 16)
    assert "residual" in new_kwargs
    assert torch.equal(new_kwargs["residual"], new_x)


def test_bloom_awq_replay_supplies_attention_scaffolding():
    """AWQ block replay must pass residual, attention_mask, and alibi for attention."""

    class _Attention(nn.Module):
        num_heads = 4

        def forward(self, hidden_states, residual, alibi, attention_mask):
            raise AssertionError("replay kwargs are what this test exercises")

    instance = BloomQModel.__new__(BloomQModel)
    x = torch.randn(2, 8, 32)
    kwargs = _context(x, _Attention.num_heads)

    new_x, new_kwargs = BloomQModel.awq_module_replay_inputs(instance, _Attention(), x, kwargs)

    assert new_x.shape == (2, 8, 32)
    assert torch.equal(new_kwargs["residual"], new_x)
    assert "attention_mask" in new_kwargs
    assert new_kwargs["attention_mask"].shape == (2, 1, 8, 8)
    assert "alibi" in new_kwargs
    # alibi shape: (batch_size * num_heads, 1, seq_len)
    assert new_kwargs["alibi"].shape == (2 * 4, 1, 8)
    assert new_kwargs["attention_mask"] is kwargs["attention_mask"]
    assert new_kwargs["alibi"] is kwargs["alibi"]


def test_bloom_real_submodules_execute_cleanly_with_replay_inputs():
    """Real BloomMLP and BloomAttention must execute without TypeError under replay inputs."""

    config = BloomConfig(
        hidden_size=64,
        n_layer=2,
        n_head=4,
        vocab_size=128,
    )
    model = BloomForCausalLM(config).eval()
    instance = BloomQModel.__new__(BloomQModel)
    instance.__dict__["model"] = model

    block = model.transformer.h[0]
    mlp = block.mlp
    attn = block.self_attention

    x = torch.randn(1, 16, 64)

    # 1. Real BloomMLP forward
    new_x_mlp, kwargs_mlp = BloomQModel.awq_module_replay_inputs(instance, mlp, x, {})
    out_mlp = mlp(new_x_mlp, **kwargs_mlp)
    assert out_mlp.shape == (1, 16, 64)

    # 2. Real BloomAttention forward
    new_x_attn, kwargs_attn = BloomQModel.awq_module_replay_inputs(instance, attn, x, _context(x, attn.num_heads))
    out_attn, _ = attn(new_x_attn, **kwargs_attn)
    assert out_attn.shape == (1, 16, 64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_bloom_replay_preserves_native_causality(dtype):
    case = _case(batch=1, dtype=dtype)
    attn = case.layer.self_attention
    x, kwargs = case.wrapper.awq_module_replay_inputs(attn, case.x, case.kwargs)
    expected, _ = attn(case.x, **case.kwargs)
    actual, probs = attn(x, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(probs.triu(1)) == 0
    changed = x.clone()
    changed[:, -1] = torch.linspace(-8, 8, x.shape[-1], dtype=dtype)
    changed_out, _ = attn(changed, **kwargs)
    torch.testing.assert_close(changed_out[:, 0], actual[:, 0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "key,stale",
    [
        ("attention_mask", False),
        ("alibi", False),
        ("attention_mask", True),
        ("alibi", True),
    ],
)
def test_bloom_rejects_missing_or_stale_attention_context(key, stale):
    case = _case()
    kwargs = dict(case.kwargs)
    kwargs[key] = kwargs[key][..., :-1] if stale else None
    with pytest.raises(_AWQScaleGroupError, match=key):
        _processor(case)._module_forward(case.x, case.layer.self_attention, kwargs)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("chunk_size", [1, 2])
@torch.inference_mode()
def test_bloom_microbatch_replay_matches_native_padding_context(heads, chunk_size):
    case = _case(heads=heads)
    expected = case.layer.self_attention(case.x, **case.kwargs)[0]
    processor = _processor(case)
    processor._quant_batch_size = chunk_size
    actual = processor._module_forward(case.x, case.layer.self_attention, case.kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("chunked", [False, True])
@torch.inference_mode()
def test_bloom_scale_search_agrees_with_unchunked_native_context(chunked):
    case = _case()
    processor = _processor(case, chunked=chunked)
    before = {name: value.clone() for name, value in case.layer.state_dict().items()}

    def search():
        return processor._search_best_scale(
            case.layer,
            case.layer.input_layernorm,
            [case.layer.self_attention.query_key_value],
            case.x,
            module2inspect=case.layer.self_attention,
            kwargs=case.kwargs,
        )

    processor._quant_batch_size = 0
    whole = search()
    processor._quant_batch_size = 1
    processor.max_chunk_memory = case.x[0].numel() * case.x.element_size() * 3
    chunked_result = search()
    torch.testing.assert_close(chunked_result[2], whole[2], rtol=1e-6, atol=1e-6)
    assert chunked_result[3] == pytest.approx(whole[3], rel=1e-5, abs=1e-9)
    for name, value in case.layer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("post_layernorm", [False, True])
@torch.inference_mode()
def test_bloom_declared_folds_preserve_full_native_outputs(post_layernorm):
    case = _case(batch=1, post_layernorm=post_layernorm)
    groups = case.wrapper.awq_get_modules_for_scaling(case.layer, case.features, {})
    if post_layernorm:
        assert groups == []
    else:
        assert groups
    folded = 0
    for group in groups:
        source = group["prev_op"]
        if source.weight.dim() == 2 and source.weight.shape[0] != group["inp"].shape[-1]:
            continue
        source_name = next(name for name, module in case.layer.named_modules() if module is source)
        target_names = tuple(
            next(name for name, module in case.layer.named_modules() if module is target) for target in group["layers"]
        )
        scales = torch.linspace(0.5, 2, group["inp"].shape[-1])
        apply_scale(case.layer, [(source_name, target_names, scales, 0)])
        folded += 1
    assert folded == (0 if post_layernorm else 2)
    actual = case.model(case.ids, attention_mask=case.mask, use_cache=False).logits
    torch.testing.assert_close(actual, case.reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("fallback", [None, {}])
def test_bloom_post_layernorm_mode_quantizes_without_folding(monkeypatch, fallback):
    case = _case(batch=1, post_layernorm=True)
    processor = _processor(case, fallback=fallback)
    state = processor._get_layer_state(0)
    state.layer_module = case.layer
    state.modules = {
        name: NamedModule(module, name, name, 0)
        for name, module in case.layer.named_modules()
        if isinstance(module, nn.Linear)
    }
    expected_names = set(state.modules)
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    monkeypatch.setattr(processor, "_layer_input_features", lambda _: case.features)
    calls = []
    monkeypatch.setattr(processor, "apply_quant", lambda named, scales_list: calls.append((set(named), scales_list)))
    processor._quantize_layer(0, state)
    assert state.quantized
    assert calls == [(expected_names, [])]
    assert len(calls[0][0]) == 4


@pytest.mark.parametrize("allow_fallback", [False, True])
def test_bloom_missing_context_uses_configured_fallback(monkeypatch, allow_fallback, caplog):
    case = _case(batch=1)
    processor = _processor(case, fallback={} if allow_fallback else None)
    processor.apply_clip = False
    state = processor._get_layer_state(0)
    state.layer_module = case.layer
    name = "self_attention.query_key_value"
    state.modules = {name: NamedModule(case.layer.self_attention.query_key_value, name, name, 0)}
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    monkeypatch.setattr(processor, "_layer_input_features", lambda _: {name: case.x})
    monkeypatch.setattr(
        case.wrapper,
        "awq_get_modules_for_scaling",
        lambda *_: [
            {
                "prev_op": case.layer.input_layernorm,
                "layers": [case.layer.self_attention.query_key_value],
                "inp": case.x,
                "module2inspect": case.layer.self_attention,
                "kwargs": {},
            }
        ],
    )
    calls = []
    monkeypatch.setattr(processor, "apply_quant", lambda named, scales_list: calls.append((set(named), scales_list)))
    before = {key: value.clone() for key, value in case.layer.state_dict().items()}
    if allow_fallback:
        processor._quantize_layer(0, state)
        assert state.quantized
        assert ({name}, []) in calls
        assert all(scales == [] for _, scales in calls)
        assert "configured plain-quantization fallback" in caplog.text
    else:
        with pytest.raises(_AWQScaleGroupError, match="attention_mask"):
            processor._quantize_layer(0, state)
        assert not state.quantized
        assert not calls
    for key, value in case.layer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("post_layernorm", [False, True])
def test_bloom_awq_cpu_quantization_completes(tmp_path, post_layernorm):
    config = BloomConfig(
        hidden_size=64,
        n_layer=1,
        n_head=4,
        vocab_size=16,
        apply_residual_connection_post_layernorm=post_layernorm,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        BloomForCausalLM(config).save_pretrained(tmp_path)
    tokenizer = Tokenizer(
        WordLevel({"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="[UNK]")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]"
    ).save_pretrained(tmp_path)
    assert list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.bin"))
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    model = GPTQModel.load(str(tmp_path), quantize_config=qcfg, backend=BACKEND.TORCH, dtype=torch.float16)
    calibration = [
        {"input_ids": torch.tensor([[4, 5, 6, 7, 4, 5, 6, 7]]), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        for _ in range(2)
    ]
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)
    assert sum(isinstance(module, AwqTorchLinear) for module in model.model.modules()) == 4
    with torch.inference_mode():
        logits = model.model(**calibration[0], use_cache=False).logits
    assert torch.isfinite(logits).all()


def test_bloom_awq_replay_rebuilds_alibi_for_packed_positions():
    """Packed replay must rebuild ALiBi from the retained rows' positions (#4176 item 3).

    ``BloomQModel.awq_module_replay_inputs`` accepted a captured alibi whose
    shape happened to equal the packed replay's ``(batch*heads, 1, N)``. With
    variable-length calibration the packed rows come from different documents
    and offsets, so a shape-compatible alibi is *stale*: it encodes the captured
    sample's ``0..N-1`` positions instead of the retained rows' real ones.
    """

    from transformers.models.bloom.modeling_bloom import _get_alibi_slopes

    heads = 4

    class _Attention(nn.Module):
        num_heads = heads

        def forward(self, hidden_states, residual, alibi, attention_mask):
            return hidden_states, None

    instance = BloomQModel.__new__(BloomQModel)
    # Five packed rows sampled at positions 0, 2, 4, 6, 8 of one document.
    positions = torch.tensor([[0, 2, 4, 6, 8]])
    seq = int(positions.shape[-1])
    x = torch.randn(1, seq, 32)
    mask = torch.full((1, 1, seq, seq), torch.finfo(x.dtype).min, dtype=x.dtype).triu(1)
    stale = build_alibi_tensor(torch.ones(1, seq, dtype=torch.long), heads, x.dtype)
    # The stale alibi has the accepted shape, so the old check let it through.
    assert stale.shape == (heads, 1, seq)

    _, kwargs = BloomQModel.awq_module_replay_inputs(
        instance,
        _Attention(),
        x,
        {
            "attention_mask": mask,
            "alibi": stale,
            "_awq_rotary_position_ids": positions,
        },
    )

    slopes = _get_alibi_slopes(heads, positions.device).to(torch.float32)
    expected = (
        slopes.view(1, heads, 1, 1) * positions.to(torch.float32).view(1, 1, 1, -1)
    ).reshape(heads, 1, seq).to(x.dtype)
    torch.testing.assert_close(kwargs["alibi"], expected, rtol=0, atol=0)
    assert not torch.equal(kwargs["alibi"], stale), "stale ALiBi was accepted unchanged"


def test_bloom_awq_replay_rebuilds_alibi_matching_the_native_builder():
    """A contiguous packed replay must reproduce `build_alibi_tensor` exactly.

    The captured alibi comes from a padded sample, so it is stale even though
    its shape matches the packed replay. The rebuild must replace it with the
    tensor the native builder produces for the retained positions.
    """

    heads = 4

    class _Attention(nn.Module):
        num_heads = heads

        def forward(self, hidden_states, residual, alibi, attention_mask):
            return hidden_states, None

    instance = BloomQModel.__new__(BloomQModel)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        seq = 6
        x = torch.randn(1, seq, 32, dtype=dtype)
        mask = torch.full((1, 1, seq, seq), torch.finfo(dtype).min, dtype=dtype).triu(1)
        positions = torch.arange(seq).unsqueeze(0)
        native = build_alibi_tensor(torch.ones(1, seq, dtype=torch.long), heads, dtype)
        # Leading padding shifts every captured position by one.
        padded_mask = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.long)
        stale = build_alibi_tensor(padded_mask, heads, dtype)
        assert stale.shape == native.shape and not torch.equal(stale, native)

        _, kwargs = BloomQModel.awq_module_replay_inputs(
            instance,
            _Attention(),
            x,
            {
                "attention_mask": mask,
                "alibi": stale,
                "_awq_rotary_position_ids": positions,
            },
        )

        torch.testing.assert_close(kwargs["alibi"], native, rtol=0, atol=0)


def test_bloom_awq_replay_refuses_a_mismatched_alibi_with_shape_context():
    """A mismatched ALiBi must be refused with the expected and actual shapes."""

    heads = 4

    class _Attention(nn.Module):
        num_heads = heads

        def forward(self, hidden_states, residual, alibi, attention_mask):
            return hidden_states, None

    instance = BloomQModel.__new__(BloomQModel)
    seq = 8
    x = torch.randn(1, seq, 32)
    kwargs = _context(x, heads)
    # A stale alibi from a shorter sample: no position ids to rebuild from.
    kwargs["alibi"] = kwargs["alibi"][..., :5]

    with pytest.raises(NotImplementedError) as excinfo:
        BloomQModel.awq_module_replay_inputs(instance, _Attention(), x, kwargs)

    message = str(excinfo.value)
    assert "alibi" in message
    assert "expected" in message and "got" in message
