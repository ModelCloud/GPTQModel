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

# set on the inner `RicursiveModel` class once `_update_causal_mask` is wrapped
_MASK_PATCH_FLAG = "_gptqmodel_ricursive_mask_patched"

# set on the attention class once `_apply_sliding_window_mask` is wrapped
_WINDOW_PATCH_FLAG = "_gptqmodel_ricursive_window_patched"

# original class -> class that actually carries the patch (may be a GenerationMixin subclass)
_PATCHED_CLASSES: dict = {}


def _max_cache_shape(_cache):
    """Stand-in for the removed transformers 4.x ``get_max_cache_shape()``.

    It must report **unknown** (``None``), not the cache capacity. The caller sizes
    the causal mask from this value and the attention layer sizes it from the live
    key tensor, so returning the capacity (transformers 5's ``get_max_length()``
    reports the configured maximum, e.g. ``sliding_window``) makes the two disagree
    whenever the cache is not full.

    ``None`` selects the caller's own fallback, ``cache_position[-1] + 1``, which is
    exactly the number of keys the attention layer holds.
    """
    return None


def _ensure_cache_compat(past_key_values):
    """Give a transformers-5 cache the ``get_max_cache_shape`` the model calls."""
    if past_key_values is None or hasattr(past_key_values, "get_max_cache_shape"):
        return past_key_values
    try:
        past_key_values.get_max_cache_shape = lambda: _max_cache_shape(past_key_values)
    except Exception:  # pragma: no cover - caches with __slots__ simply stay unpatched
        pass
    return past_key_values


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
    position_ids=None,
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
        # transformers derives `position_ids` from the *caller's* tensors before this
        # hook runs, so a CPU prompt yields CPU positions even after the move above.
        # RoPE then multiplies them against CUDA frequencies and fails.
        if position_ids is not None and position_ids.device != device:
            position_ids = position_ids.to(device)

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
    sliced = past_length > 0 and input_ids.shape[1] > cache_position.numel()
    if sliced:
        input_ids = input_ids[:, cache_position]

    if inputs_embeds is not None and cache_position.numel() and int(cache_position[0]) == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    # Derive per-row positions from the padding mask. Using `cache_position` for every
    # row is only correct when nothing is padded: with left padding the real tokens of
    # a short row sit at higher indices, so a shared position vector shifts its RoPE
    # phases and the row decodes to garbage.
    if position_ids is not None:
        # A caller-supplied vector still covers the pre-slice sequence.
        if sliced:
            position_ids = position_ids[:, cache_position]
    elif attention_mask is not None and attention_mask.dim() == 2:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        position_ids = position_ids[:, -input_ids.shape[1]:]
    else:
        position_ids = cache_position.unsqueeze(0)

    model_inputs.update(
        {
            "position_ids": position_ids,
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


def _update_causal_mask(
    self,
    attention_mask,
    input_tensor,
    cache_position,
    past_key_values,
):
    """Corrected ``RicursiveModel._update_causal_mask``.

    The published version has two defects that only show up when an explicit mask is
    built (i.e. for padded batches):

    * it seeds the mask with ``finfo.min`` and only writes the allowed diagonal via
      ``triu`` when ``sequence_length != 1``. Every decode step therefore comes out
      fully masked, so a padded batch decodes to a constant token;
    * it uses ``1.0`` (not ``0.0``) for allowed positions, so SDPA adds a constant
      bias to every score.

    It also calls ``past_key_values.get_max_cache_shape()``, which transformers 5
    removed. This implementation derives the key span from ``cache_position`` -- the
    same value the attention layer sizes its key tensor from -- and returns a mask
    that is ``0.0`` where a key is visible and ``finfo.min`` elsewhere.
    """
    if attention_mask is None:
        return None
    if torch.all(attention_mask == 1):
        return None

    dtype, device = input_tensor.dtype, input_tensor.device
    sequence_length = input_tensor.shape[1]

    if cache_position is None:
        query_positions = torch.arange(sequence_length, device=device)
        target_length = sequence_length
    else:
        query_positions = cache_position.to(device=device).reshape(-1)
        target_length = int(query_positions[-1].item()) + 1

    # Padding can outrun the position span when the caller keeps a longer mask.
    target_length = max(target_length, attention_mask.shape[-1])

    key_positions = torch.arange(target_length, device=device)
    allowed = key_positions[None, :] <= query_positions[:, None]
    allowed = allowed[None, None, :, :] & attention_mask[:, None, None, :].to(torch.bool)

    causal_mask = torch.zeros(
        (attention_mask.shape[0], 1, sequence_length, target_length),
        dtype=dtype,
        device=device,
    )
    return causal_mask.masked_fill(~allowed, torch.finfo(dtype).min)


def _patch_causal_mask(model) -> bool:
    """Install the corrected mask builder on the inner ``RicursiveModel`` class."""
    inner = getattr(model, "model", None)
    if inner is None:
        return False

    cls = type(inner)
    if getattr(cls, _MASK_PATCH_FLAG, False):
        return False

    if not callable(getattr(cls, "_update_causal_mask", None)):
        return False

    cls._update_causal_mask = _update_causal_mask
    setattr(cls, _MASK_PATCH_FLAG, True)
    return True


def _patch_sliding_window_mask(model) -> bool:
    """Skip the per-step sliding-window mask when it cannot exclude any key.

    ``_apply_sliding_window_mask`` materialises an ``arange``, a broadcast comparison
    and a ``[1, 1, q, kv]`` boolean tensor on **every** decode step for each of the
    sliding layers (64 of 80 in Ripple-65B). When ``kv_len <= sliding_window`` the
    window admits every key, so the result is either an all-True mask or an unchanged
    padding mask -- and for an unpadded batch SDPA's ``is_causal`` path already
    applies exactly the causal mask this function would build.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return False

    cls = type(layers[0].self_attn)
    if getattr(cls, _WINDOW_PATCH_FLAG, False):
        return False

    original = getattr(cls, "_apply_sliding_window_mask", None)
    if not callable(original):
        return False

    def _apply_sliding_window_mask(self, attention_mask, q_len, kv_len, dtype, device, cache_position):
        if attention_mask is not None and attention_mask.shape[-1] > kv_len:
            # transformers 5 builds a hybrid cache: sliding layers get a
            # DynamicSlidingWindowLayer that evicts from the left, so they hold only
            # the most recent `sliding_window` keys while full layers keep growing.
            # The causal mask is built once per forward from the full span, so trim it
            # to the keys this layer actually holds.
            attention_mask = attention_mask[..., -kv_len:]

        window = getattr(self.config, "sliding_window", None)
        if window is not None and kv_len <= window:
            # The window cannot bind, so the tensor would come out all-True.
            if attention_mask is None:
                # q_len > 1: SDPA's is_causal path builds the same causal mask.
                # q_len == 1: every key is visible by definition.
                return None
            if q_len == 1:
                return attention_mask
        return original(self, attention_mask, q_len, kv_len, dtype, device, cache_position)

    cls._apply_sliding_window_mask = _apply_sliding_window_mask
    # Keep the upstream implementation reachable so tests can compare the two paths.
    _apply_sliding_window_mask._gptqmodel_original = original
    setattr(cls, _WINDOW_PATCH_FLAG, True)
    return True


def apply_ricursive_generate_patch(model=None, model_cls=None) -> bool:
    """Install the ``prepare_inputs_for_generation`` fix. Returns True if it was applied.

    Safe to call repeatedly: the patch is recorded on the class itself.
    """
    cls = model_cls if model_cls is not None else type(model)
    if cls is None:
        return False

    # Always attempted: a class can be generate-patched while a fresh instance still
    # needs the mask shim, and vice versa.
    _patch_causal_mask(model)
    _patch_sliding_window_mask(model)

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
