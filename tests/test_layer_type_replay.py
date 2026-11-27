# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Replay each decoder layer with the mask and rotary embeddings of its own layer type.

Issue #4004: input capture recorded only the *first* decoder layer's
`attention_mask` / `position_embeddings` and layer replay handed that pair to every
layer. `transformers` 5 builds one mask per layer type
(`causal_mask_mapping[config.layer_types[i]]`), and Gemma3 additionally builds one
rotary tuple per type, so full-attention layers of a mixed-`layer_types` model were
calibrated with the sliding-window mask and the local RoPE.

These tests pin the invariant directly: for every decoder layer, the kwargs that
`BaseQModel.prepare_layer_replay_kwargs` produces must equal the kwargs the model's
own forward passes to that layer, and replaying the stack must reproduce the native
last hidden state.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from gptqmodel.models.base import _LAYER_TYPE_REPLAY_KWARGS_KEY
from gptqmodel.models.definitions.gemma3 import Gemma3QModel
from gptqmodel.models.definitions.gemma3n import (
    Gemma3nForConditionalGenerationGPTQ,
    Gemma3nTextQModel,
)
from gptqmodel.models.definitions.gemma4 import (
    Gemma4ForConditionalGenerationGPTQ,
    Gemma4TextQModel,
)
from gptqmodel.models.definitions.gemma4_unified import (
    Gemma4UnifiedForConditionalGenerationGPTQ,
    Gemma4UnifiedTextQModel,
)
from gptqmodel.models.definitions.gpt_oss import GPTOSSGPTQ
from gptqmodel.models.definitions.llama4 import Llama4QModel
from gptqmodel.models.definitions.olmo3 import Olmo3QModel
from gptqmodel.models.definitions.raon1_mini import Raon1MiniQModel


SEQ = 6
SLIDING_WINDOW = 2


class _StopCapture(Exception):
    """Abort the calibration forward from the first-layer hook, as `StopForward` does."""


def _gemma3_text():
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM

    torch.manual_seed(0)
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=SLIDING_WINDOW,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    return Gemma3ForCausalLM(config).eval(), Gemma3QModel


def _gpt_oss():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    torch.manual_seed(0)
    config = GptOssConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_local_experts=2,
        num_experts_per_tok=1,
        sliding_window=SLIDING_WINDOW,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    return GptOssForCausalLM(config).eval(), GPTOSSGPTQ


def _llama4_text_chunked():
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4ForCausalLM

    torch.manual_seed(0)
    config = Llama4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_chunk_size=2,
        # Llama4 declares the chunked type first, so it is the base type whose
        # mask `support_batch_quantize=False` replay used to discard.
        layer_types=["chunked_attention", "full_attention"],
        num_local_experts=2,
        num_experts_per_tok=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    return Llama4ForCausalLM(config).eval(), Llama4QModel


def _olmo3_text():
    from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
    from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM

    torch.manual_seed(0)
    config = Olmo3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        sliding_window=SLIDING_WINDOW,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    return Olmo3ForCausalLM(config).eval(), Olmo3QModel


def _layer_type(layer: nn.Module, config) -> str | None:
    attention = getattr(layer, "self_attn", None)
    declared = getattr(attention, "layer_type", None)
    if not isinstance(declared, str):
        declared = getattr(layer, "layer_type", None)
    if isinstance(declared, str):
        return declared

    index = getattr(attention, "layer_idx", None)
    layer_types = getattr(config, "layer_types", ())
    if isinstance(index, int) and 0 <= index < len(layer_types):
        return layer_types[index]
    return None


def _native_forward(model, input_ids, attention_mask):
    """Record what the model's own forward hands to every decoder layer."""

    recorded = {}
    outputs = []
    handles = []

    def make_hook(index):
        def hook(module, args, kwargs):
            recorded[index] = {
                "attention_mask": kwargs.get("attention_mask"),
                "position_embeddings": kwargs.get("position_embeddings"),
                "hidden_states": args[0].detach().clone(),
            }

        return hook

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_pre_hook(make_hook(index), with_kwargs=True))
        handles.append(layer.register_forward_hook(lambda mod, args, out: outputs.append(out)))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return recorded, outputs


def _capture_and_replay(
    model,
    qmodel_cls,
    input_ids,
    attention_mask,
    *,
    drop_captured_mask: bool = False,
    capture_example=None,
):
    """Run the capture hook once, then replay every layer through the default hooks.

    ``drop_captured_mask`` reproduces `support_batch_quantize=False` replay,
    where the forward executor hands every layer ``attention_mask=None``. The
    per-layer-type capture must still restore the base type's window/chunk mask.
    """

    qmodel = object.__new__(qmodel_cls)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    layers = model.model.layers
    captured = {}

    def capture_first(module, args, kwargs):
        replay_kwargs = {
            key: value
            for (key, value) in kwargs.items()
            if key not in ("hidden_states", "attention_mask", "position_ids")
        }
        replay_kwargs = qmodel.capture_first_layer_input_kwargs(
            args=args,
            kwargs=kwargs,
            batch_device=torch.device("cpu"),
            layer_input_kwargs=replay_kwargs,
        )
        # Mirrors `StageInputsCapture.cache_inputs`: the per-layer-type capture is
        # an optional hook, absent on a definition that does not provide it.
        capture_layer_type_kwargs = getattr(
            qmodel, "capture_layer_type_replay_kwargs", None
        )
        if callable(capture_layer_type_kwargs):
            replay_kwargs = capture_layer_type_kwargs(
                args=args,
                kwargs=kwargs,
                batch_device=torch.device("cpu"),
                layer_input_kwargs=replay_kwargs,
            )
        captured["hidden_states"] = args[0].detach().clone()
        captured["attention_mask"] = kwargs.get("attention_mask")
        captured["kwargs"] = replay_kwargs
        raise _StopCapture

    handle = layers[0].register_forward_pre_hook(capture_first, with_kwargs=True)
    qmodel.begin_input_capture_example(
        capture_example
        if capture_example is not None
        else {"input_ids": input_ids, "attention_mask": attention_mask},
        torch.device("cpu"),
    )
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except _StopCapture:
        # Expected control-flow exception from `capture_first` to stop early after capture.
        ...
    finally:
        handle.remove()
        qmodel.end_input_capture_example()

    replayed = []
    hidden_states = captured["hidden_states"]
    for layer in layers:
        additional_inputs = {
            key: value
            for (key, value) in captured["kwargs"].items()
            if key not in ("past_key_values", "past_key_value")
        }
        additional_inputs["attention_mask"] = (
            None if drop_captured_mask else captured["attention_mask"]
        )
        additional_inputs["position_ids"] = torch.arange(
            hidden_states.shape[1], dtype=torch.long
        ).unsqueeze(0)
        additional_inputs["use_cache"] = False
        additional_inputs = qmodel.prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=[hidden_states],
            additional_inputs=additional_inputs,
            target_device=torch.device("cpu"),
        )
        # The capture bookkeeping must be consumed by the replay hook, never
        # forwarded to the decoder layer.
        assert not [key for key in additional_inputs if key.startswith("_gptqmodel")], (
            f"private replay key leaked into {type(layer).__name__} kwargs"
        )
        replayed.append(dict(additional_inputs))
        with torch.inference_mode():
            output = layer(hidden_states, **additional_inputs)
        hidden_states = output[0] if isinstance(output, tuple) else output
    return replayed, hidden_states


def _assert_masks_equal(layer_index, layer_type, expected, actual):
    if expected is None or actual is None:
        assert expected is None and actual is None, (
            f"layer {layer_index} ({layer_type}): native mask is "
            f"{'None' if expected is None else 'materialized'} but replay mask is "
            f"{'None' if actual is None else 'materialized'}"
        )
        return
    assert torch.equal(expected, actual), (
        f"layer {layer_index} ({layer_type}): replay attention_mask differs from the "
        "mask the model builds for that layer type"
    )


@pytest.mark.parametrize("padded", [False, True], ids=["dense", "padded"])
@pytest.mark.parametrize(
    "builder,qmodel_cls",
    [
        (_gemma3_text, Gemma3QModel),
        (_gpt_oss, GPTOSSGPTQ),
        (_llama4_text_chunked, Llama4QModel),
        (_olmo3_text, Olmo3QModel),
    ],
    ids=["gemma3_text", "gpt_oss", "llama4_text_chunked", "olmo3_text"],
)
def test_replay_kwargs_match_native_layer_type(builder, qmodel_cls, padded):
    model, _ = builder()
    input_ids = torch.arange(3, 3 + SEQ).unsqueeze(0)
    attention_mask = torch.ones(1, SEQ, dtype=torch.long)
    if padded:
        attention_mask[0, :2] = 0

    native, native_outputs = _native_forward(model, input_ids, attention_mask)
    replayed, replayed_hidden = _capture_and_replay(
        model, qmodel_cls, input_ids, attention_mask
    )

    layer_types = [getattr(model.config, "layer_types", None)]
    assert layer_types[0] is not None, "test model must declare per-layer attention types"
    assert len(set(layer_types[0])) == 2, "test model must mix two attention types"

    for index, layer in enumerate(model.model.layers):
        layer_type = _layer_type(layer, model.config)
        _assert_masks_equal(
            index,
            layer_type,
            native[index]["attention_mask"],
            replayed[index].get("attention_mask"),
        )
        native_pe = native[index]["position_embeddings"]
        replay_pe = replayed[index].get("position_embeddings")
        assert native_pe is not None and replay_pe is not None, (
            f"layer {index} ({layer_type}): replay dropped position_embeddings"
        )
        for native_component, replay_component in zip(native_pe, replay_pe):
            assert torch.equal(native_component, replay_component), (
                f"layer {index} ({layer_type}): replay position_embeddings differs from "
                "the rotary embeddings the model builds for that layer type"
            )

    last_native = native_outputs[-1]
    last_native = last_native[0] if isinstance(last_native, tuple) else last_native
    torch.testing.assert_close(replayed_hidden, last_native, rtol=0, atol=0)


@pytest.mark.parametrize("padded", [False, True], ids=["dense", "padded"])
def test_batch_feature_capture_source_rebuilds_every_layer_type_mask(padded):
    """Processor batches are ``BatchFeature`` (a ``UserDict``), not plain dicts.

    Issue #6138: the per-layer-type replay rebuild returned early for every
    ``BatchFeature`` source, so a mixed ``layer_types`` model replayed layer 0's
    mask into the other type. Qwen3.5-MoE crashes in its first
    ``full_attention`` layer; Gemma 3 VL silently loses its sliding window.
    """

    from transformers import BatchFeature

    model, qmodel_cls = _gemma3_text()
    input_ids = torch.arange(3, 3 + SEQ).unsqueeze(0)
    attention_mask = torch.ones(1, SEQ, dtype=torch.long)
    if padded:
        attention_mask[0, :2] = 0
    example = BatchFeature(
        {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    assert not isinstance(example, dict)

    native, native_outputs = _native_forward(model, input_ids, attention_mask)
    replayed, replayed_hidden = _capture_and_replay(
        model,
        qmodel_cls,
        input_ids,
        attention_mask,
        capture_example=example,
    )

    layer_types = list(model.config.layer_types)
    assert len(set(layer_types)) == 2, "test model must mix two attention types"
    for index, layer in enumerate(model.model.layers):
        _assert_masks_equal(
            index,
            _layer_type(layer, model.config),
            native[index]["attention_mask"],
            replayed[index].get("attention_mask"),
        )

    last_native = native_outputs[-1]
    last_native = last_native[0] if isinstance(last_native, tuple) else last_native
    torch.testing.assert_close(replayed_hidden, last_native, rtol=0, atol=0)


def test_qwen3_5_moe_quantize_batch_size_two_rebuilds_full_attention_mask(
    tmp_path, monkeypatch
):
    """End-to-end #6138: a padded processor batch must not crash SDPA.

    The tiny Qwen3.5-MoE checkpoint mixes ``linear_attention`` and
    ``full_attention``. Without the per-layer-type rebuild, layer 0's raw 2-D
    ``int64`` padding mask is replayed into the first full-attention layer and
    ``scaled_dot_product_attention`` rejects it. The processor is a stub so the
    test owns the padded ``BatchFeature`` and needs no Hub download.
    """

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BatchFeature, PreTrainedTokenizerFast

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.models.local import LocalModelLoader
    from gptqmodel.models.ports.native.auto import NativeForImageTextToText
    from gptqmodel.models.ports.native.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
    )
    from gptqmodel.quantization.config import CalibrationConfig, QuantizeConfig

    config = Qwen3_5MoeConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 128,
            "partial_rotary_factor": 1.0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_interleaved": True,
                "mrope_section": [1, 1, 2],
            },
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "pad_token_id": 0,
            "eos_token_id": None,
            "bos_token_id": 1,
            "tie_word_embeddings": False,
            "attn_output_gate": True,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 64,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    NativeForImageTextToText._from_config(config).eval().save_pretrained(tmp_path)

    # The loader requires tokenizer files even though this test's processor is
    # a stub; a WordLevel tokenizer keeps the checkpoint self-contained.
    backend = Tokenizer(WordLevel({f"token_{i}": i for i in range(128)}, unk_token="token_0"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="token_0", pad_token="token_0"
    ).save_pretrained(tmp_path)

    class _PaddingProcessor:
        def apply_chat_template(self, batch, tokenize=False, add_generation_prompt=True):
            return [f"sample {index}" for index, _ in enumerate(batch)]

        def __call__(self, *, text, images, videos, padding, return_tensors):
            assert padding is True and return_tensors == "pt"
            lengths = [4 + 3 * index for index in range(len(text))]
            width = max(lengths)
            input_ids = torch.zeros((len(text), width), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row, length in enumerate(lengths):
                input_ids[row, :length] = torch.arange(1, length + 1)
                attention_mask[row, :length] = 1
            return BatchFeature(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )

    monkeypatch.setattr(
        LocalModelLoader, "load_processor", lambda *args, **kwargs: _PaddingProcessor()
    )
    model = GPTQModel.load(
        str(tmp_path),
        QuantizeConfig(device="cpu", bits=4, group_size=32, fallback=None),
        backend=BACKEND.TORCH,
        dtype=torch.float32,
    )
    conversations = [
        [{"role": "user", "content": "GPTQModel quantizes large language models."}],
        [{"role": "user", "content": "Short one."}],
        [{"role": "user", "content": "The quick brown fox jumps over the lazy dog today."}],
        [{"role": "user", "content": "Tiny text here."}],
    ]

    prepared = model.prepare_dataset(conversations, batch_size=2)
    assert all(isinstance(batch, BatchFeature) for batch in prepared)
    assert [batch["input_ids"].shape[0] for batch in prepared] == [2, 2]
    assert model.model.config.text_config.layer_types == [
        "linear_attention",
        "full_attention",
    ]

    model.quantize(
        CalibrationConfig(dataset=conversations, batch_size=2, min_length=1),
        backend=BACKEND.TORCH,
    )


@pytest.mark.parametrize("padded", [False, True], ids=["dense", "padded"])
@pytest.mark.parametrize(
    "builder,qmodel_cls",
    [
        (_gemma3_text, Gemma3QModel),
        (_gpt_oss, GPTOSSGPTQ),
        (_llama4_text_chunked, Llama4QModel),
        (_olmo3_text, Olmo3QModel),
    ],
    ids=["gemma3_text", "gpt_oss", "llama4_text_chunked", "olmo3_text"],
)
def test_base_layer_type_mask_survives_support_batch_quantize_false(
    builder, qmodel_cls, padded
):
    """`support_batch_quantize=False` replay must rebuild the base-type mask.

    Issue #4038: the forward executor drops the captured mask for those models,
    so every layer of the first declared type ran with full causal attention
    instead of the sliding/chunked mask HF passes.
    """

    model, _ = builder()
    input_ids = torch.arange(3, 3 + SEQ).unsqueeze(0)
    attention_mask = torch.ones(1, SEQ, dtype=torch.long)
    if padded:
        attention_mask[0, :2] = 0

    native, _ = _native_forward(model, input_ids, attention_mask)
    replayed, _ = _capture_and_replay(
        model,
        qmodel_cls,
        input_ids,
        attention_mask,
        drop_captured_mask=True,
    )

    base_type = model.config.layer_types[0]
    materialized = 0
    for index, layer in enumerate(model.model.layers):
        layer_type = _layer_type(layer, model.config)
        if layer_type != base_type:
            continue
        if native[index]["attention_mask"] is not None:
            materialized += 1
        _assert_masks_equal(
            index,
            layer_type,
            native[index]["attention_mask"],
            replayed[index].get("attention_mask"),
        )
    if padded:
        # Left padding forces every base-type layer to receive a materialized
        # mask, so the equality check above cannot pass vacuously.
        assert materialized > 0


@pytest.mark.parametrize(
    "qmodel_cls",
    [
        Olmo3QModel,
        Gemma4TextQModel,
        Gemma4ForConditionalGenerationGPTQ,
        Gemma4UnifiedTextQModel,
        Gemma4UnifiedForConditionalGenerationGPTQ,
        Gemma3nTextQModel,
        Gemma3nForConditionalGenerationGPTQ,
        Raon1MiniQModel,
    ],
    ids=[
        "olmo3",
        "gemma4_text",
        "gemma4",
        "gemma4_unified_text",
        "gemma4_unified",
        "gemma3n_text",
        "gemma3n",
        "raon1_mini",
    ],
)
def test_model_override_consumes_private_layer_type_capture_key(qmodel_cls):
    """Model-specific replay overrides must call the base hook, not shadow it.

    Issue #4038: OLMo3/Gemma3n/Gemma4/Raon overrode
    `prepare_layer_replay_kwargs` without `super()`, so
    `_gptqmodel_layer_type_replay_kwargs` leaked into the decoder layer call and
    full-attention layers kept layer 0's sliding mask.
    """

    qmodel = object.__new__(qmodel_cls)
    nn.Module.__init__(qmodel)
    qmodel.model = nn.Module()
    layer_type_mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
    layer = types.SimpleNamespace(layer_type="full_attention")

    prepared = qmodel.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[],
        additional_inputs={
            _LAYER_TYPE_REPLAY_KWARGS_KEY: {
                "base_type": "sliding_attention",
                "layer_types": ("sliding_attention", "full_attention"),
                "by_type": {"full_attention": {"attention_mask": layer_type_mask}},
            },
            "use_cache": False,
        },
        target_device=torch.device("cpu"),
    )

    assert not [key for key in prepared if key.startswith("_gptqmodel")], (
        f"{qmodel_cls.__name__} leaked private replay state into layer kwargs"
    )
    assert prepared["attention_mask"] is not None


def _gemma3_layer_type_kwargs(model, qmodel):
    """Capture the per-layer-type replay kwargs exactly as the input stage does."""

    input_ids = torch.arange(3, 3 + SEQ).unsqueeze(0)
    attention_mask = torch.ones(1, SEQ, dtype=torch.long)
    hidden = model.model.embed_tokens(input_ids)
    position_ids = torch.arange(SEQ, dtype=torch.long).unsqueeze(0)
    qmodel.begin_input_capture_example(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        torch.device("cpu"),
    )
    try:
        layer_type_kwargs = qmodel._build_layer_type_replay_kwargs(
            args=(hidden,),
            kwargs={
                "hidden_states": hidden,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_device=torch.device("cpu"),
        )
    finally:
        qmodel.end_input_capture_example()
    assert layer_type_kwargs is not None
    return layer_type_kwargs


def test_awq_token_row_replay_uses_the_layer_attention_type():
    """AWQ token-row replay must use the layer's own mask, not layer 0's (#4176 item 2).

    ``_token_row_replay_kwargs`` read only ``inputs_cache.attention_masks``,
    which holds the first decoder layer's mask. On a mixed sliding/full model
    (gemma-3, OLMo 3) a full-attention layer was therefore replayed with the
    sliding window: zero allowed pairs at distance >= ``sliding_window``.
    """

    from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
    from gptqmodel.quantization.config import FORMAT, METHOD, CalibrationConfig, QuantizeConfig

    model, qmodel_cls = _gemma3_text()
    qmodel = object.__new__(qmodel_cls)
    nn.Module.__init__(qmodel)
    qmodel.model = model

    layers = model.model.layers
    layer_types = list(model.config.layer_types)
    assert layer_types == ["sliding_attention", "full_attention"]

    # The capture stage stores layer 0's (sliding) mask in `attention_masks`
    # and every declared type in the private replay key.
    native, _ = _native_forward(
        model,
        torch.arange(3, 3 + SEQ).unsqueeze(0),
        torch.ones(1, SEQ, dtype=torch.long),
    )
    sliding_mask = native[0]["attention_mask"]
    assert sliding_mask is not None

    processor = AWQProcessor(
        tokenizer=None,
        qcfg=QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=8),
        calibration=CalibrationConfig(),
        gptq_model=qmodel,
        model=model,
    )
    module_name = "self_attn.q_proj"
    full_layer = layers[1]
    state = _AWQLayerState(modules={module_name: object()}, layer_module=full_layer)
    processor.gptq_model.awq_input_feature_aggregation = lambda name: (
        {"mode": "token_rows"} if name == module_name else None
    )
    processor.inputs_cache = types.SimpleNamespace(
        attention_masks=[sliding_mask],
        position_ids=[torch.arange(SEQ, dtype=torch.long).unsqueeze(0)],
        layer_input_kwargs=[
            {_LAYER_TYPE_REPLAY_KWARGS_KEY: _gemma3_layer_type_kwargs(model, qmodel)}
        ],
    )
    processor.tasks[module_name] = {
        "inputs": [torch.zeros(1, SEQ, 4)],
        "batch_indices": [0],
    }

    processor._layer_input_features(state)

    replay = processor._awq_feature_kwargs[module_name]
    mask = replay["attention_mask"]
    allowed = mask[0, 0]
    if allowed.dtype != torch.bool:
        allowed = allowed >= 0
    # A full-attention layer must see across the whole sequence. The sliding
    # mask (window=2) allows nothing at distance >= 2.
    assert bool(allowed[SEQ - 1, 0]), (
        "full-attention layer replayed with the sliding mask: "
        f"allowed pairs at distance {SEQ - 1} = {bool(allowed[SEQ - 1, 0])}"
    )
    assert bool(allowed[SEQ - 1, :].all())
    # And it must not become one causal block across documents: with one
    # document here the mask is the native causal grid.
    assert not bool(allowed[0, SEQ - 1])
