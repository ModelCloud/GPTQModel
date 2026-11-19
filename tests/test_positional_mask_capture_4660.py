# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Capture a decoder mask that a model passes positionally.

Issue #4660: `transformers` 5.22 `GPT2Model.forward` calls every block as

```python
hidden_states = block(
    hidden_states,
    past_key_values,
    causal_mask,            # attention_mask as args[2]
    encoder_hidden_states,
    encoder_attention_mask=...,
    use_cache=...,
    position_ids=...,
    **kwargs,
)
```

The first-layer capture hook read only `kwargs["attention_mask"]`, so GPT-2
captured no mask at all. Layer replay then ran every layer with
`attention_mask=None`: under eager attention there is no causal mask (layers 1..N
were calibrated on bidirectional activations) and under SDPA padding was
attended to. These tests pin the invariant at the real consumer:
`StageInputsCapture.cache_inputs` must record the mask the block received, and
replaying the cached layers must reproduce the native layer output.
"""

from __future__ import annotations

import threading

import pytest
import torch
from torch import nn

from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.models.definitions.gpt2 import GPT2QModel
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.attn_mask import normalize_seq_mask


SEQ = 9
BATCH = 3
PAD = 3


def _block_argument(module, args, kwargs, name):
    """Bind a block call and read one parameter, independent of the implementation.

    `transformers` 5.22 `GPT2Model.forward` passes the causal mask as `args[2]`,
    so a correct capture cannot look at `kwargs` alone. This helper is a local
    oracle: it binds the block's own signature instead of importing the helper
    under test.
    """

    import inspect

    signature = inspect.signature(module.forward)
    bound = signature.bind_partial(*args, **kwargs)
    if name in bound.arguments:
        return bound.arguments[name]
    extra = bound.arguments.get("kwargs")
    if isinstance(extra, dict):
        return extra.get(name)
    return None


def _calibration():
    ids = torch.randint(4, 64, (BATCH, SEQ))
    mask = torch.ones_like(ids)
    ids[:, -PAD:] = 0
    mask[:, -PAD:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def _gpt2(attn_implementation: str):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=3,
        n_head=2,
        n_inner=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation=attn_implementation,
    )
    return GPT2LMHeadModel(config).eval()


def _qmodel(definition, model):
    qmodel = object.__new__(definition)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.quantize_config = QuantizeConfig(device="cpu")
    qmodel.turtle_model = None
    qmodel.quant_region_timer = None
    # `cache_inputs` materializes the first layer through this lock.
    qmodel._turtle_lock = threading.RLock()
    return qmodel


class _Looper:
    def __init__(self, qmodel):
        self.gptq_model = qmodel

    def _batch_row_count(self, layer_inputs):
        return int(layer_inputs[0].shape[0])


def _capture(qmodel, layer, example):
    return StageInputsCapture(_Looper(qmodel)).cache_inputs(
        layers=[layer],
        calibration_data=[example],
        use_cache=False,
    )


def _native_layer_inputs(model, example, position_ids=None):
    """The hidden state, mask, and positions the model hands to every block.

    ``position_ids`` lets the oracle run with the same derived positions the
    capture uses.  Since #4930 capture derives them from the mask, so the
    default ``arange`` would only disagree on the pad slots.
    """

    recorded = {}
    outputs = []
    handles = []

    def make_hook(index):
        def hook(module, args, kwargs):
            recorded[index] = {
                "hidden": args[0].detach().clone(),
                "mask": _block_argument(module, args, kwargs, "attention_mask"),
                "position_ids": _block_argument(module, args, kwargs, "position_ids"),
            }

        return hook

    handles.extend(
        block.register_forward_pre_hook(make_hook(index), with_kwargs=True)
        for index, block in enumerate(model.transformer.h)
    )
    handles.extend(
        block.register_forward_hook(lambda module, args, output: outputs.append(output))
        for block in model.transformer.h
    )
    try:
        with torch.inference_mode():
            if position_ids is None:
                model(**example, use_cache=False)
            else:
                model(**example, position_ids=position_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return recorded, outputs


def _derived_position_ids(example):
    """The ids ``generate()``/capture derive from a padded batch mask (#4861)."""

    mask = example["attention_mask"]
    position_ids = mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(mask == 0, 1)
    return position_ids


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_capture_records_positional_mask(attn_implementation):
    """The mask exists only in `args[2]`; capture must not report `None`."""

    example = _calibration()
    model = _gpt2(attn_implementation)
    qmodel = _qmodel(GPT2QModel, model)
    cache = _capture(qmodel, model.transformer.h[0], example)

    native, _ = _native_layer_inputs(model, example)
    assert native[0]["mask"] is not None, "GPT-2 must pass a materialized mask"
    captured = cache.attention_masks[0]
    assert captured is not None, (
        "capture dropped the positional attention mask; replay would run GPT-2 "
        "without causality"
    )
    torch.testing.assert_close(captured, native[0]["mask"])


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_replayed_layers_reproduce_native_layer_output(attn_implementation):
    """Every replayed layer must match the model's own layer output."""

    example = _calibration()
    model = _gpt2(attn_implementation)
    qmodel = _qmodel(GPT2QModel, model)
    cache = _capture(qmodel, model.transformer.h[0], example)
    native, outputs = _native_layer_inputs(
        model, example, position_ids=_derived_position_ids(example)
    )

    hidden = cache.layer_inputs[0][0]
    mask = cache.attention_masks[0]
    position_ids = cache.position_ids[0]
    assert mask is not None, "replay has no mask to apply"
    torch.testing.assert_close(hidden, native[0]["hidden"])

    for index, block in enumerate(model.transformer.h):
        extra = {
            key: value
            for key, value in cache.layer_input_kwargs[0].items()
            if key not in ("past_key_values", "past_key_value")
        }
        extra["attention_mask"] = mask
        extra["position_ids"] = position_ids
        extra["use_cache"] = False
        extra = qmodel.prepare_layer_replay_kwargs(
            layer=block,
            layer_input=[hidden],
            additional_inputs=extra,
            target_device=torch.device("cpu"),
        )
        with torch.inference_mode():
            hidden = block(hidden, **extra)
        torch.testing.assert_close(
            hidden,
            outputs[index],
            rtol=0,
            atol=0,
            msg=f"layer {index} replay diverged from the native forward",
        )


def test_keep_mask_excludes_padding_for_a_padded_gpt2_batch():
    """Pad rows must not enter a GPTQ Hessian from a replayed GPT-2 batch."""

    example = _calibration()
    model = _gpt2("sdpa")
    qmodel = _qmodel(GPT2QModel, model)
    cache = _capture(qmodel, model.transformer.h[0], example)

    keep = normalize_seq_mask(
        cache.attention_masks[0], seq_len=cache.layer_inputs[0][0].shape[1]
    )

    assert keep is not None
    assert keep.tolist() == [[True] * (SEQ - PAD) + [False] * PAD] * BATCH


def test_keyword_mask_family_keeps_its_captured_mask():
    """A block that takes `attention_mask` by keyword is unchanged."""

    from transformers import OPTConfig, OPTForCausalLM

    from gptqmodel.models.definitions.opt import OptQModel

    torch.manual_seed(0)
    config = OPTConfig(
        vocab_size=64,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
    )
    model = OPTForCausalLM(config).eval()
    qmodel = _qmodel(OptQModel, model)
    cache = _capture(qmodel, model.model.decoder.layers[0], _calibration())

    captured = cache.attention_masks[0]
    assert captured is not None
    # OPT hands its block an additive causal mask through the keyword slot.
    assert tuple(captured.shape) == (BATCH, 1, SEQ, SEQ)


def test_binding_does_not_retain_the_decoder_block():
    """A cached signature must not pin the module or its device tensors."""

    import gc
    import weakref

    from gptqmodel.utils.inspect import bound_forward_argument

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

        def forward(self, hidden_states, attention_mask=None):
            return hidden_states

    block = _Block()
    bound_forward_argument(block, (torch.zeros(1, 2),), {}, "attention_mask")
    reference = weakref.ref(block)
    del block
    gc.collect()

    assert reference() is None
