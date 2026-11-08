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

Two classes of patch -- do not mix them when contributing upstream
-----------------------------------------------------------------
Everything in this module is a runtime monkeypatch applied by GPTQModel. It is
**not** the same thing as the native ``modeling_ricursive.py`` that ships in a
checkpoint, and the two upstream destinations have different bars.

**1. Correctness and memory fixes -- upstream-eligible.**
These change behaviour only where the shipped code is wrong or unbounded:
the ``cache_position``/``input_ids`` slicing bug, the all-masked decode row,
the sliding-window key anchoring, mask alignment after eviction, forwarding
``cache_position`` into ``update()``, honouring ``logits_to_keep``, and chunking
the MLP so prefill does not materialise ``tokens x 40960`` three times over.
They are architecture-agnostic, add no dependency, and are safe to propose to
``ricursiveAI/Ripple-65B-Preview``. See ``UPSTREAM_ELIGIBLE_PATCHES``.

**2. Fast paths -- local only, never upstream.**
These are performance rewrites of code that is already correct, and they are
bound to *this* deployment: Ripple's exact 80-query/8-KV head split, this Torch
build's ``enable_gqa`` and fused ``rms_norm`` support, and the assumption that a
wholesale replacement of the attention forward is acceptable. A general upstream
model file must run everywhere and must not depend on ModelCloud's runtime, so
these are never sent upstream. See ``LOCAL_ONLY_FAST_PATHS``.

The upstream model repository (``ricursiveAI/Ripple-65B-Preview``) therefore
receives accuracy and memory-usage changes **only**. The ModelCloud checkpoint
build may bake the fast paths into its own ``modeling_ricursive.py``; the
upstream file must not.
"""

import torch
import torch.nn.functional as F


_PATCH_FLAG = "_gptqmodel_ricursive_generate_patched"

# set on the inner `RicursiveModel` class once `_update_causal_mask` is wrapped
_MASK_PATCH_FLAG = "_gptqmodel_ricursive_mask_patched"

# set on the attention class once `_apply_sliding_window_mask` is wrapped
_WINDOW_PATCH_FLAG = "_gptqmodel_ricursive_window_patched"
_NORM_PATCH_FLAG = "_gptqmodel_ricursive_norm_patched"
_GQA_PATCH_FLAG = "_gptqmodel_ricursive_gqa_patched"

# original class -> class that actually carries the patch (may be a GenerationMixin subclass)
_PATCHED_CLASSES: dict = {}


# --- upstream contribution policy -------------------------------------------
# Fixes for wrong or unbounded behaviour. Architecture-agnostic, no extra
# dependency, safe to propose to ricursiveAI/Ripple-65B-Preview. This is the
# complete set that may be upstreamed.
UPSTREAM_ELIGIBLE_PATCHES = (
    "prepare_inputs_for_generation",  # spurious axis on prefill
    "_update_causal_mask",            # all-masked decode row
    "_apply_sliding_window_mask",     # key anchoring after eviction
    "_patch_causal_mask",
    "_patch_sliding_window_mask",
    "_patch_logits_to_keep",          # memory: no full [B, T, vocab] projection
    "_patch_chunked_mlp",             # memory: bounded prefill activation peak
)

# Performance-only rewrites bound to this runtime and architecture. NEVER
# upstream: they assume Ripple's exact 80-query/8-KV head split, this Torch
# build's `enable_gqa` / fused `rms_norm` support, and that replacing the whole
# attention forward is acceptable. A general upstream file must run everywhere.
LOCAL_ONLY_FAST_PATHS = (
    "_patch_fused_rms_norm",
    "_patch_gqa_attention",
)


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


def _cache_layer_seq_length(past_key_values, layer_idx: int):
    """Actual number of keys written for ``layer_idx``, or ``None`` if unknown.

    ``Cache.get_seq_length(layer_idx)`` counts *written* tokens. For a
    preallocated static layer that is smaller than the tensor ``update()``
    returns until the cache fills; for an evicting sliding-window layer it is
    larger, because the layer only returns a trailing window.
    """
    if past_key_values is None:
        return None
    getter = getattr(past_key_values, "get_seq_length", None)
    if not callable(getter):
        return None
    try:
        return int(getter(layer_idx))
    except TypeError:  # transformers 4.x cache without a layer argument
        pass
    except Exception:  # pragma: no cover - unknown cache implementations
        return None
    try:
        return int(getter())
    except Exception:  # pragma: no cover - unknown cache implementations
        return None


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
        # A full prompt can include cached tokens, while verification may pass
        # only a new token block. A block no longer than the cache must be new;
        # for a longer block, a mask with exactly the cached prefix plus new
        # tokens identifies new-only input when its final slot is populated.
        # A longer mask, or one with unused capacity at the end, may be padded
        # to a static cache's capacity and describes a full prompt instead.
        input_length = input_ids.shape[1]
        candidate_full_prompt = past_length > 0 and input_length > past_length
        new_block_mask = (
            candidate_full_prompt
            and attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.shape[-1] == past_length + input_length
            and bool(attention_mask[..., -1].any())
        )
        full_prompt = candidate_full_prompt and not new_block_mask
        new_len = input_length - past_length if full_prompt else input_length
        cache_position = torch.arange(past_length, past_length + new_len, device=input_ids.device)

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
    if use_cache is not False:
        # Greedy/sampling generation only ever reads the last position. Keeping
        # just that row avoids projecting the whole prompt through the 100k-way
        # LM head, which is several GiB of transient memory at eval batch sizes.
        model_inputs["logits_to_keep"] = 1
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
    """Install the corrected mask builder on the inner ``RicursiveModel`` class.

    ``[upstream-eligible: correctness]`` The shipped builder returns an
    all-masked row at decode, which is a plain bug.
    """
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

    ``[upstream-eligible: correctness]`` Key anchoring after eviction is wrong
    upstream; the skip is a safe consequence of fixing it.

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

    def _apply_sliding_window_mask(
        self, attention_mask, q_len, kv_len, dtype, device, cache_position, total_kv=None
    ):
        if attention_mask is not None and attention_mask.shape[-1] > kv_len:
            # transformers 5 builds a hybrid cache: sliding layers get a
            # DynamicSlidingWindowLayer that evicts from the left, so they hold only
            # the most recent `sliding_window` keys while full layers keep growing.
            # The causal mask is built once per forward from the full span, so trim it
            # to the keys this layer actually holds.
            #
            # The retained keys sit at the absolute positions
            # `[total_kv - kv_len, total_kv)`. Slicing `[-kv_len:]` is only correct
            # when the mask length equals `total_kv`; with a mask sized to the
            # allocated cache capacity it selects the unwritten tail instead.
            mask_len = attention_mask.shape[-1]
            if total_kv is not None:
                start = max(0, min(total_kv - kv_len, mask_len - kv_len))
                attention_mask = attention_mask[..., start : start + kv_len]
            else:
                attention_mask = attention_mask[..., -kv_len:]

        window = getattr(self.config, "sliding_window", None)
        if window is not None and kv_len <= window:
            # The window cannot bind, so the tensor would come out all-True.
            if attention_mask is None:
                # SDPA's implicit causal mask is aligned top-left, which only
                # matches this query block when it is the whole sequence
                # (prefill) or a single position (decode). A multi-token block
                # verified against a warm cache sits at the *end* of the key
                # span, so the implicit mask would hide valid context; fall
                # through to the explicit offset-aware mask.
                if q_len == 1 or q_len == kv_len:
                    return None
            if q_len == 1:
                return attention_mask
        return original(self, attention_mask, q_len, kv_len, dtype, device, cache_position)

    cls._apply_sliding_window_mask = _apply_sliding_window_mask
    # Keep the upstream implementation reachable so tests can compare the two paths.
    _apply_sliding_window_mask._gptqmodel_original = original
    setattr(cls, _WINDOW_PATCH_FLAG, True)
    return True


def _sdpa_supports_gqa() -> bool:
    """True when ``scaled_dot_product_attention`` accepts ``enable_gqa``."""
    try:
        q = torch.zeros(1, 4, 1, 8)
        k = torch.zeros(1, 2, 1, 8)
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


def _patch_fused_rms_norm(model) -> bool:
    """Route ``RicursiveRMSNorm`` through Torch's fused ``rms_norm``.

    ``[LOCAL-ONLY FAST PATH -- never upstream]``
    This is a speed optimisation, not a fix. It is safe here because it is
    applied at load time by GPTQModel against a known Torch build, but it is
    not something to push into ``ricursiveAI/Ripple-65B-Preview``: it depends
    on ``F.rms_norm`` existing and on this model's specific norm placement, and
    a fast path in shipped model code would silently change numerics for every
    consumer of the checkpoint.

    Ripple-65B applies six RMS norms per layer (input, post-attention,
    pre-feedforward, post-feedforward, plus the per-head ``q_norm``/``k_norm``),
    so a decode step runs 480 of them. The reference implementation expands to
    ``float() -> pow -> mean -> rsqrt -> mul -> to(dtype)`` -- roughly six kernel
    launches and two dtype conversions per norm on a tensor that is only
    ``(batch, 1, hidden)``. That is pure launch overhead at decode time.

    ``F.rms_norm`` accumulates in FP32 and emits the input dtype in a single
    kernel, which is the same arithmetic the reference performs.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return False

    # Norms live on the layer (4 of them) and on the attention module (q_norm/k_norm).
    probe = getattr(layers[0], "input_layernorm", None)
    if probe is None or not hasattr(probe, "weight") or not hasattr(probe, "eps"):
        return False
    norm_cls = type(probe)
    if getattr(norm_cls, _NORM_PATCH_FLAG, False):
        return False

    original = norm_cls.forward

    def forward(self, hidden_states):
        weight = self.weight
        if weight.dtype != hidden_states.dtype:
            weight = weight.to(hidden_states.dtype)
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), weight, self.eps)

    # Keep the upstream implementation reachable so tests can compare the two paths.
    forward._gptqmodel_original = original
    norm_cls.forward = forward
    setattr(norm_cls, _NORM_PATCH_FLAG, True)
    return True


_MLP_CHUNK_PATCH_FLAG = "_gptqmodel_ricursive_mlp_chunk_patch"


def _patch_chunked_mlp(model) -> bool:
    """Bound the prefill activation peak of ``RicursiveMLP`` by chunking rows.

    ``[upstream-eligible: memory usage]``
    Pure row-wise split of an exact computation -- the arithmetic and the
    reduction order are unchanged, only the peak transient is smaller. Safe to
    propose upstream, and worth it: this is the difference between OOM and
    not-OOM on a 48 GiB card at long prompts.

    Ripple's MLP expands 5120 -> 40960, so a prefill of ``batch x prompt`` tokens
    materialises three intermediate tensors of ``tokens x 40960`` at once
    (``gate``, ``up`` and the ``silu(gate) * up`` product). At the batch sizes a
    long-prompt benchmark uses that is several GiB of transient memory, and it is
    what pushes a 33 GiB 4-bit model over a 48 GiB card.

    Every operation in the MLP is row-wise, so splitting the token dimension is
    exact: each row is still projected by the same kernels with the same
    reduction order, and the pieces are concatenated back. Only the peak is
    reduced. ``RIPPLE_MLP_CHUNK`` overrides the row budget; set it to
    ``0`` to disable.
    """
    import os

    try:
        chunk = int(os.environ.get("RIPPLE_MLP_CHUNK", "2048"))
    except ValueError:
        chunk = 2048
    if chunk <= 0:
        return False

    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return False
    probe = getattr(layers[0], "mlp", None)
    if probe is None or not callable(getattr(probe, "forward", None)):
        return False
    mlp_cls = type(probe)
    if getattr(mlp_cls, _MLP_CHUNK_PATCH_FLAG, False):
        return False

    original = mlp_cls.forward

    def forward(self, x, *args, **kwargs):
        if not torch.is_tensor(x) or x.dim() not in (2, 3):
            return original(self, x, *args, **kwargs)
        shape = x.shape
        hidden = shape[-1]
        rows = 1
        for size in shape[:-1]:
            rows *= size
        if rows <= chunk:
            return original(self, x, *args, **kwargs)
        flat = x.reshape(rows, hidden)
        pieces = [
            original(self, flat[start:start + chunk], *args, **kwargs)
            for start in range(0, rows, chunk)
        ]
        out = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        return out.reshape(*shape[:-1], out.shape[-1])

    forward._gptqmodel_original = original
    mlp_cls.forward = forward
    setattr(mlp_cls, _MLP_CHUNK_PATCH_FLAG, True)
    return True


def _patch_gqa_attention(model) -> bool:
    """Let SDPA broadcast the grouped KV heads instead of materialising them.

    ``[LOCAL-ONLY FAST PATH -- never upstream]``
    Two reasons this must not ship upstream. It requires
    ``F.scaled_dot_product_attention(..., enable_gqa=True)``, which older Torch
    builds reject, and it wholesale replaces the attention forward -- so it is
    bound to the exact 80-query-head / 8-KV-head layout and the exact SDPA call
    shape of this checkpoint revision. A shipped model that swapped its own
    attention for this would break on any upstream refactor or Torch downgrade.
    GPTQModel probes ``_sdpa_supports_gqa()`` and degrades to the reference
    forward; the reference code cannot probe for itself safely.

    The reference attention calls ``repeat_kv`` before SDPA. Ripple-65B has 80
    query heads and 8 KV heads, so that expands the cache 10x with a strided
    ``copy_`` on every layer of every decode step, and it also makes Flash
    Attention read 10x more K/V. Measured on a 16-row decode step at 760 cached
    keys: the ``repeat_kv`` copy cost 52.7 ms and Flash Attention 45.6 ms, i.e.
    63% of the step, purely to undo the grouping.

    ``enable_gqa=True`` performs the same arithmetic without the expansion and is
    bit-identical to the expanded path (verified: max abs diff 0.0).
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return False
    if not _sdpa_supports_gqa():
        return False

    cls = type(layers[0].self_attn)
    if getattr(cls, _GQA_PATCH_FLAG, False):
        return False

    # Only replace the forward when the checkpoint's attention still has the shape
    # this implementation targets. A future upstream refactor must degrade to the
    # shipped code rather than to a broken hybrid.
    required = (
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "layer_type",
        "q_norm",
        "k_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    probe = layers[0].self_attn
    if any(not hasattr(probe, attr) for attr in required):
        return False
    if getattr(probe.config, "use_gated_attention", False) and not hasattr(probe, "g_proj"):
        return False

    import importlib

    module = importlib.import_module(cls.__module__)
    apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb", None)
    if apply_rotary_pos_emb is None:
        return False

    original = cls.forward

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        value_residual=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        value_states = hidden_states
        if value_residual is not None:
            if not self.config.use_valres:
                raise RuntimeError("value_residual was supplied but config.use_valres is false")
            blend = torch.sigmoid(self.value_res_lambda[0]).to(dtype=hidden_states.dtype)
            value_states = torch.lerp(
                value_residual.to(dtype=hidden_states.dtype), hidden_states, blend
            )
        v = self.v_proj(value_states)

        q = self.q_norm(q.view(bsz, q_len, self.num_heads, self.head_dim))
        k = self.k_norm(k.view(bsz, q_len, self.num_kv_heads, self.head_dim))
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.layer_type == "sliding_attention":
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        total_kv = None
        if past_key_value is not None:
            # Forward the absolute positions being written: a transformers 4.x
            # StaticCache falls back to `arange(q_len)` without them and the
            # first decode step overwrites the prompt instead of appending.
            cache_kwargs = (
                {"cache_position": cache_position} if cache_position is not None else None
            )
            if cache_kwargs is not None:
                try:
                    k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
                except TypeError:
                    k, v = past_key_value.update(k, v, self.layer_idx)
            else:
                k, v = past_key_value.update(k, v, self.layer_idx)
            # Number of keys this layer has actually written. A preallocated
            # (StaticCache) layer returns its full capacity from `update()`,
            # including slots that have not been filled yet; an evicting
            # (sliding-window) layer reports more than the trailing window it
            # returns. The mask must use the real positions, not the tensor size.
            total_kv = _cache_layer_seq_length(past_key_value, self.layer_idx)

        # Drop the unwritten tail of a preallocated cache. Without this a
        # six-token prefill into a sixteen-slot StaticCache looks like a
        # ten-token prefix, and the end-anchored verify mask below would let
        # each query attend slots belonging to a different absolute position.
        if total_kv is not None and 0 < total_kv < k.shape[2]:
            k = k[:, :, :total_kv]
            v = v[:, :, :total_kv]

        # Align an incoming mask that is longer than the keys this layer holds.
        # The shared mask is built over absolute key positions
        # `0..target_length-1`; keep the slice that matches the keys present.
        #
        # The keys returned by `update()` occupy the absolute positions
        # `[total_kv - kv_len, total_kv)`. Anchoring on the tensor tail is only
        # correct when the mask length equals `total_kv`; with a mask sized to
        # the *allocated cache capacity* it selects the unwritten tail instead
        # and silently produces wrong attention.
        if attention_mask is not None and attention_mask.shape[-1] > k.shape[2]:
            kv_len_now = k.shape[2]
            mask_len = attention_mask.shape[-1]
            if total_kv is not None:
                start = max(0, min(total_kv - kv_len_now, mask_len - kv_len_now))
                attention_mask = attention_mask[..., start : start + kv_len_now]
            else:
                # Unknown cache: the keys are the trailing window.
                attention_mask = attention_mask[..., -kv_len_now:]

        # No repeat_kv: SDPA broadcasts the grouped heads itself.
        mask_kwargs = {
            "attention_mask": attention_mask,
            "q_len": q.shape[2],
            "kv_len": k.shape[2],
            "dtype": q.dtype,
            "device": q.device,
            "cache_position": cache_position,
        }
        try:
            attention_mask = self._apply_sliding_window_mask(total_kv=total_kv, **mask_kwargs)
        except TypeError:
            # Class without the window patch: the mask is already aligned above.
            attention_mask = self._apply_sliding_window_mask(**mask_kwargs)

        if attention_mask is None and q.shape[2] > 1 and k.shape[2] > q.shape[2]:
            # Multi-token block verified against a warm cache. SDPA's implicit
            # causal mask is aligned to the top-left corner, which would hide
            # every key before the block; build the mask from the actual query
            # positions and the absolute positions of the returned keys.
            if cache_position is not None and cache_position.numel() == q.shape[2]:
                query_positions = cache_position.to(
                    device=q.device, dtype=torch.long
                ).reshape(-1)
            else:
                query_positions = torch.arange(
                    k.shape[2] - q.shape[2], k.shape[2], device=q.device
                )
            if total_kv is not None and total_kv > k.shape[2]:
                # Evicting cache: the returned keys are the trailing window that
                # ends at the newest query, not positions 0..kv_len-1.
                last_query = int(query_positions[-1].item())
                key_positions = torch.arange(
                    last_query - k.shape[2] + 1, last_query + 1, device=q.device
                )
            else:
                key_positions = torch.arange(k.shape[2], device=q.device)
            attention_mask = (
                key_positions[None, :] <= query_positions[:, None]
            )[None, None]

        is_causal = bool(attention_mask is None and q.shape[2] > 1)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            enable_gqa=self.num_kv_groups > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        if self.config.use_gated_attention:
            gate = self.g_proj(hidden_states)
            if self.config.gate_full_precision:
                gate = gate.float()
            gate = torch.sigmoid(gate).to(attn_output.dtype)
            attn_output = attn_output.view(bsz, q_len, -1) * gate
        else:
            attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    forward._gptqmodel_original = original
    cls.forward = forward
    setattr(cls, _GQA_PATCH_FLAG, True)
    return True


_LOGITS_TO_KEEP_PATCH_FLAG = "_gptqmodel_logits_to_keep_patch"


def _patch_logits_to_keep(model) -> bool:
    """Honour the ``logits_to_keep`` forward argument that speculative decoders pass.

    ``[upstream-eligible: contract + memory]``
    The shipped forward accepts ``logits_to_keep`` via ``**kwargs`` and ignores
    it, so it silently violates its own documented signature. Honouring it is a
    pure fix: the returned rows are unchanged for callers that do not pass it,
    and callers that do pass it stop getting a wrong-shaped result plus an
    unnecessary ``batch x seq x vocab`` allocation.

    DFlash-style drafters call the target with ``logits_to_keep=1`` and expect a
    single position back.  The reference Ripple implementation accepts the
    argument through ``**kwargs`` and silently ignores it, so the caller receives
    one row per prompt token and the speculation loop breaks.  Slicing after the
    LM head keeps the output identical while restoring the documented contract.
    """

    if model is None:
        return False
    cls = type(model)
    if getattr(cls, _LOGITS_TO_KEEP_PATCH_FLAG, False):
        return False
    if not callable(getattr(cls, "forward", None)):
        return False

    original = cls.forward

    def forward(self, *args, **kwargs):
        keep = kwargs.pop("logits_to_keep", None)
        head = getattr(self, "lm_head", None)
        # A labeled call needs the full sequence: the causal-LM loss shifts logits
        # and labels by one position, so trimming the LM head input to a short
        # window leaves the shift empty (and the loss NaN). `labels` is positional
        # argument 5 in the reference forward, or a keyword argument.
        labels = kwargs.get("labels")
        if labels is None and len(args) > 5:
            labels = args[5]
        slice_input = bool(keep) and head is not None and labels is None
        if slice_input:
            # Slice the LM head's *input* so the full [batch, seq, vocab]
            # projection never materialises: prefill on a long prompt otherwise
            # costs several GiB that a single position needs.
            head._gptqmodel_keep_last = int(keep)
        try:
            outputs = original(self, *args, **kwargs)
        finally:
            if head is not None:
                head._gptqmodel_keep_last = None
        if keep and getattr(outputs, "logits", None) is not None:
            # The loss (if any) was computed on the full logits above; trimming
            # the returned tensor afterwards keeps the documented output shape.
            outputs.logits = outputs.logits[..., -int(keep):, :]
        return outputs

    forward._gptqmodel_original = original
    cls.forward = forward
    setattr(cls, _LOGITS_TO_KEEP_PATCH_FLAG, True)

    head = getattr(model, "lm_head", None)
    if head is not None and not getattr(head, "_gptqmodel_keep_wrapped", False):
        base = head.forward

        def head_forward(hidden_states, *args, **kwargs):
            keep = getattr(head, "_gptqmodel_keep_last", None)
            if keep and hidden_states.dim() == 3 and hidden_states.shape[1] > keep:
                hidden_states = hidden_states[:, -keep:, :]
            return base(hidden_states, *args, **kwargs)

        head.forward = head_forward
        head._gptqmodel_keep_wrapped = True
    return True


def apply_ricursive_generate_patch(model=None, model_cls=None) -> bool:
    """Install the ``prepare_inputs_for_generation`` fix. Returns True if it was applied.

    Safe to call repeatedly: the patch is recorded on the class itself.
    """
    cls = model_cls if model_cls is not None else type(model)
    if cls is None:
        return False

    # Two disjoint patch classes -- see the module docstring.
    #
    #   LOCAL_ONLY_FAST_PATHS   speed only; installed here, never upstreamed.
    #   UPSTREAM_ELIGIBLE_PATCHES   correctness + memory; candidates for
    #                               ricursiveAI/Ripple-65B-Preview.
    #
    # The fast paths below are numerically identical to the reference
    # implementation but are bound to this model's head layout and this Torch
    # build, so they are installed even when the checkpoint already ships the
    # transformers-5 correctness fixes inline -- the two concerns are
    # independent, and a fixed checkpoint still benefits from them locally.
    _patch_fused_rms_norm(model)
    _patch_gqa_attention(model)
    _patch_logits_to_keep(model)
    _patch_chunked_mlp(model)

    # A checkpoint that already ships the native transformers-5 fixes needs nothing
    # from us. `modeling_ricursive.py` sets `_ricursive_tf5_native_fix` on the class.
    if getattr(cls, "_ricursive_tf5_native_fix", None):
        _ensure_generation_config(model)
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
