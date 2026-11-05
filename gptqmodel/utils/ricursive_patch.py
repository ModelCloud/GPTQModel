# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Monkeypatch for the Ricursive Ripple ``generate`` bug.

``ricursiveAI/Ripple-65B-Preview`` ships ``prepare_inputs_for_generation`` in its
own ``modeling_ricursive.py`` that does::

    if past_key_values is not None:
        if input_ids.shape[1] != 1:
            input_ids = input_ids[:, cache_position]

Under transformers >= 5 ``generate`` pre-creates an *empty* ``DynamicCache`` and
passes ``cache_position=None`` for the prefill call, so that line evaluates
``input_ids[:, None]`` and injects a spurious axis: ``(batch, seq)`` becomes
``(batch, 1, seq)``. The embedding lookup then yields 4-D hidden states and the
first decoder layer dies on ``bsz, q_len, _ = hidden_states.shape``.

The fix reconstructs ``cache_position`` when it is missing and only slices
``input_ids`` when the cache actually holds tokens, which is the behaviour the
upstream code was trying to express.
"""

import torch


_PATCH_FLAG = "_gptqmodel_ricursive_generate_patched"

# original class -> class that actually carries the patch (may be a GenerationMixin subclass)
_PATCHED_CLASSES: dict = {}


def _cache_seq_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    getter = getattr(past_key_values, "get_seq_length", None)
    if callable(getter):
        return int(getter())
    try:
        return int(past_key_values[0][0].shape[-2])
    except Exception:
        return 0


def _prepare_inputs_for_generation(
    self,
    input_ids: torch.LongTensor,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=None,
    **kwargs,
):
    # transformers 5.x no longer moves remote-code inputs onto the model device for this
    # model, so a plain CPU `input_ids` reaches the embedding lookup and raises. Its own
    # generation helper does this move, so mirror it here.
    device = getattr(self, "device", None)
    if device is not None and getattr(device, "type", None) != "meta":
        if input_ids is not None and input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask is not None and attention_mask.device != device:
            attention_mask = attention_mask.to(device)
        if inputs_embeds is not None and inputs_embeds.device != device:
            inputs_embeds = inputs_embeds.to(device)

    past_length = _cache_seq_length(past_key_values)

    if cache_position is None:
        # With a non-empty cache and the full sequence supplied, only the trailing
        # (uncached) tokens are new. Otherwise every token in `input_ids` is new.
        if past_length > 0 and input_ids.shape[1] > 1:
            new_len = input_ids.shape[1] - past_length
            cache_position = torch.arange(past_length, past_length + new_len, device=input_ids.device)
        else:
            cache_position = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)

    # Only slice when the cache is warm and `input_ids` still carries cached tokens.
    # The upstream code sliced unconditionally, which turned a ``None`` cache position
    # into ``input_ids[:, None]`` and added a bogus axis.
    if past_length > 0 and input_ids.shape[1] > cache_position.numel():
        input_ids = input_ids[:, cache_position]

    if inputs_embeds is not None and cache_position.numel() and int(cache_position[0]) == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": cache_position.unsqueeze(0),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        }
    )
    return model_inputs


def _ensure_generation_config(model) -> None:
    """Give a now-generative instance the ``generation_config`` that ``can_generate()`` promises.

    Transformers only populates it for classes that advertised generation support at init
    time. Once we install ``GenerationMixin`` the class reports ``can_generate() is True``,
    and callers (e.g. GPTQModel's ``autofix_hf_model_config``) immediately read it.
    """
    if model is None or not getattr(model, "can_generate", lambda: False)():
        return
    if hasattr(model, "generation_config"):
        return
    from transformers import GenerationConfig

    try:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    except Exception:  # pragma: no cover - exotic configs fall back to defaults
        model.generation_config = GenerationConfig()


def apply_ricursive_generate_patch(model=None, model_cls=None) -> bool:
    """Install the ``prepare_inputs_for_generation`` fix. Returns True if it was applied.

    Safe to call repeatedly: the patch is recorded on the class itself.
    """
    cls = model_cls if model_cls is not None else type(model)
    if cls is None:
        return False

    if getattr(cls, _PATCH_FLAG, False):
        # Class already patched; a freshly loaded instance may still be missing the
        # `generation_config` that the patch now makes reachable.
        _ensure_generation_config(model)
        return False
    if not hasattr(cls, "prepare_inputs_for_generation") and not hasattr(cls, "generate"):
        return False

    cached = _PATCHED_CLASSES.get(cls)
    if cached is not None:
        if model is not None and type(model) is not cached:
            model.__class__ = cached
        _ensure_generation_config(model)
        return False

    # transformers 5 only injects `GenerationMixin` for remote-code classes it recognizes as
    # generative. A checkpoint rebuilt with `save_pretrained` can miss that injection, leaving
    # the class without `generate` at all -- do what `add_generation_mixin_to_remote_model`
    # would have done.
    target = cls
    if not hasattr(cls, "generate"):
        from transformers.generation.utils import GenerationMixin

        target = type(cls.__name__, (cls, GenerationMixin), {**cls.__dict__})

    target.prepare_inputs_for_generation = _prepare_inputs_for_generation
    setattr(target, _PATCH_FLAG, True)
    _PATCHED_CLASSES[cls] = target

    if model is not None and type(model) is not target:
        model.__class__ = target
    _ensure_generation_config(model)
    return True
