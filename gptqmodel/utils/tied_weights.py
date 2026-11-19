# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Resolve the embedding pairs a checkpoint really ties.

Transformers ties ``lm_head.weight`` to the input embedding from the *model
config*. A composite wrapper can leave its top-level ``tie_word_embeddings`` at
the ``PretrainedConfig`` default ``True`` while the decoder's own config says
``False`` and the checkpoint stores a distinct head (InternVL3-hf, #4799).
GPT-QModel's lazy loader materializes a meta shell in pieces and used to call
``model.tie_weights()`` after every step; that recomputes the mapping from the
top-level flag and silently replaces the real head with the embedding.

This module resolves the mapping once, from evidence instead of one flag:

* the wrapper's ``tie_word_embeddings`` flag and its declared
  ``_tied_weights_keys`` mapping (``all_tied_weights_keys``), which is exactly
  what Transformers' ``tie_weights()`` consults. A nested decoder's
  ``text_config.tie_word_embeddings`` does not gate the wrapper's head: for a
  composite such as InternVL the wrapper ties ``lm_head`` while its text config
  defaults to ``False`` (#4973). A wrapper that says ``False`` still ties
  nothing (#4369);
* the checkpoint: a pair whose source tensor is absent cannot be tied, and a
  pair whose target tensor is present *and differs* from the source is a real
  separate head, not a tie (#4799). A declared pair whose target the checkpoint
  omits is the normal tied-save layout, so it stays tied;
* the model's own ``all_tied_weights_keys`` for the candidate pairs.

The result is stamped on the model as a snapshot and re-applied after every
lazy materialization instead of re-running ``tie_weights()``. Callers that
re-tie a model after offload/restore use the same snapshot, and the writer uses
it to decide whether the output head may be dropped as a duplicate.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


# Snapshot attribute stamped on the loaded model: ``{target: source}`` parameter
# names for the pairs the checkpoint really ties.
TIED_WEIGHTS_SNAPSHOT_ATTR = "_gptqmodel_tied_weights_keys"


def text_config_tie_word_embeddings(config) -> Optional[bool]:
    """Return the decoder text config's flag, or ``None`` without one.

    Composite wrappers expose their decoder through ``get_text_config()``. The
    same config object (plain decoder models) carries no separate answer, so it
    returns ``None`` and the caller keeps the root flag.
    """

    getter = getattr(config, "get_text_config", None)
    if not callable(getter):
        return None
    try:
        text_config = getter()
    except Exception:  # pragma: no cover - defensive: exotic remote-code configs
        return None
    if text_config is None or text_config is config:
        return None
    flag = getattr(text_config, "tie_word_embeddings", None)
    return flag if isinstance(flag, bool) else None


def word_embeddings_tied(config) -> bool:
    """Return whether both the wrapper and its decoder declare tied embeddings.

    ``lm_head`` is only tied when *both* levels agree. A composite wrapper that
    defaults to ``True`` must not override a decoder that says ``False``
    (InternVL3-hf, #4799), and a decoder that defaults to ``True`` must not
    override a wrapper that says ``False`` (Qwen3-VL, #4369).

    This is the config-only fallback for callers that have no resolved
    snapshot. ``resolve_tied_weights_keys`` deliberately does *not* use it: the
    checkpoint is the authority on whether a declared pair is really tied, and
    Transformers ties the wrapper's head from the wrapper flag alone (#4973).
    """

    if config is None:
        return False
    if not bool(getattr(config, "tie_word_embeddings", False)):
        return False
    text_tied = text_config_tie_word_embeddings(config)
    return True if text_tied is None else text_tied


def _declared_tied_weights_keys(model) -> Dict[str, str]:
    declared = getattr(model, "all_tied_weights_keys", None)
    if isinstance(declared, dict):
        return dict(declared)
    legacy = getattr(model, "_tied_weights_keys", None)
    if isinstance(legacy, dict):
        return dict(legacy)
    return {}


def resolve_tied_weights_keys(
    model,
    *,
    checkpoint_has_tensor: Optional[Callable[[str], Optional[bool]]] = None,
    checkpoint_tensors_equal: Optional[Callable[[str, str], Optional[bool]]] = None,
) -> Dict[str, str]:
    """Return the ``{target: source}`` pairs this checkpoint really ties.

    ``checkpoint_has_tensor`` and ``checkpoint_tensors_equal`` take runtime
    parameter names (``lm_head.weight``) and may return ``None`` when the
    checkpoint cannot answer; an unknown answer never drops a declared tie.

    The wrapper's flag and declared mapping decide *which* pairs may be tied;
    the checkpoint only refutes a declared pair. In particular a decoder
    ``text_config.tie_word_embeddings=False`` does not untie the wrapper's head
    (#4973): Transformers' own ``tie_weights()`` ignores it for the wrapper.
    """

    declared = _declared_tied_weights_keys(model)
    if not declared:
        return {}
    config = getattr(model, "config", None)
    # Transformers ties a composite's head from the *wrapper* flag; a nested
    # decoder's flag only gates that decoder's own declared pairs. Keeping the
    # wrapper flag as the gate also preserves #4369 for legacy remote-code
    # models whose synthesized `all_tied_weights_keys` includes nested entries.
    if config is None or not bool(getattr(config, "tie_word_embeddings", False)):
        return {}

    resolved: Dict[str, str] = {}
    for target, source in declared.items():
        if not isinstance(target, str) or not isinstance(source, str):
            continue
        if checkpoint_has_tensor is not None:
            if checkpoint_has_tensor(source) is False:
                # A tied checkpoint may store the shared tensor under either
                # member. Keep the declared pair when the checkpoint stores the
                # other side, so the missing member is materialized from the
                # stored one instead of staying on meta (#5399); drop it only
                # when neither member is present.
                if checkpoint_has_tensor(target) is not True:
                    continue
                resolved[target] = source
                continue
            if (
                checkpoint_tensors_equal is not None
                and checkpoint_has_tensor(target) is True
                and checkpoint_tensors_equal(target, source) is False
            ):
                # The checkpoint stores both sides with different values, so the
                # head is a real separate tensor, not a tie.
                continue
        resolved[target] = source
    return resolved


def snapshot_tied_weights_keys(
    model,
    *,
    checkpoint_has_tensor: Optional[Callable[[str], Optional[bool]]] = None,
    checkpoint_tensors_equal: Optional[Callable[[str, str], Optional[bool]]] = None,
) -> Dict[str, str]:
    """Resolve and stamp the snapshot on ``model``; return the mapping."""

    resolved = resolve_tied_weights_keys(
        model,
        checkpoint_has_tensor=checkpoint_has_tensor,
        checkpoint_tensors_equal=checkpoint_tensors_equal,
    )
    setattr(model, TIED_WEIGHTS_SNAPSHOT_ATTR, resolved)
    # Keep transformers' own bookkeeping consistent: `tie_weights(
    # recompute_mapping=False)` and the save-time duplicate filter read this.
    try:
        model.all_tied_weights_keys = dict(resolved)
    except Exception:  # pragma: no cover - defensive: read-only proxies
        pass
    return resolved


def tied_weights_keys(model) -> Optional[Dict[str, str]]:
    """Return the stamped snapshot, or ``None`` when the model has none."""

    snapshot = getattr(model, TIED_WEIGHTS_SNAPSHOT_ATTR, None)
    if isinstance(snapshot, dict):
        return snapshot
    return None


def lm_head_is_tied(model, lm_head_name: str = "lm_head") -> Optional[bool]:
    """Return the snapshot's verdict for the output head.

    ``None`` means no snapshot exists and the caller must keep its own
    config-based decision.
    """

    snapshot = tied_weights_keys(model)
    if snapshot is None:
        return None
    if not snapshot:
        return False
    target = f"{lm_head_name}.weight"
    if target in snapshot:
        return True
    return any(name.endswith(".weight") and name.startswith(f"{lm_head_name}.") for name in snapshot)


def _get_parameter_or_buffer(model, name: str):
    getter = getattr(model, "get_parameter_or_buffer", None)
    if callable(getter):
        try:
            return getter(name)
        except Exception:  # pragma: no cover - defensive
            return None
    if "." in name:
        parent_name, attribute = name.rsplit(".", 1)
        try:
            parent = model.get_submodule(parent_name)
        except Exception:  # pragma: no cover - defensive
            return None
    else:
        parent, attribute = model, name
    return getattr(parent, attribute, None)


def retie_tied_weights(model) -> bool:
    """Re-apply the snapshot's pairs after a materialization.

    Falls back to the model's own ``tie_weights()`` when no snapshot was
    stamped, so models loaded outside the lazy path keep their behavior.
    """

    snapshot = tied_weights_keys(model)
    if snapshot is None:
        tie = getattr(model, "tie_weights", None)
        if not callable(tie):
            return False
        config = getattr(model, "config", None)
        if config is not None and not word_embeddings_tied(config):
            return False
        tie()
        return True

    for target, source in snapshot.items():
        source_param = _get_parameter_or_buffer(model, source)
        if source_param is None:
            continue
        if "." in target:
            parent_name, attribute = target.rsplit(".", 1)
            try:
                parent = model.get_submodule(parent_name)
            except Exception:  # pragma: no cover - defensive
                continue
        else:
            parent, attribute = model, target
        setattr(parent, attribute, source_param)
        adjust_bias = getattr(model, "_adjust_bias", None)
        if callable(adjust_bias):
            try:
                adjust_bias(parent, source_param)
            except Exception:  # pragma: no cover - defensive
                pass
    return bool(snapshot)


def tensors_equal_in_chunks(slice_a, slice_b, *, chunk_rows: int = 8) -> bool:
    """Compare two safetensors ``get_slice`` handles without loading them whole."""

    shape_a = tuple(slice_a.get_shape())
    if shape_a != tuple(slice_b.get_shape()):
        return False
    if not shape_a:
        return bool(torch.equal(slice_a[:], slice_b[:]))
    rows = shape_a[0]
    step = max(1, min(chunk_rows, rows))
    for start in range(0, rows, step):
        end = min(start + step, rows)
        if not torch.equal(slice_a[start:end], slice_b[start:end]):
            return False
    return True


def lm_head_quantization_is_tied(gptq_model, lm_head_name: Optional[str] = None) -> bool:
    """Return whether quantizing ``lm_head`` would target a tied output head.

    Shared by ``ModuleLooper`` (#4034) and ``WeightOnlyLooper`` (#5027), which
    used ``find_tied_parameters`` alone and therefore missed a lazily loaded
    shell.

    Real storage identity is authoritative. Composite/VLM configs commonly
    leave a nested ``text_config.tie_word_embeddings=True`` while the loaded
    model keeps independent input and output embeddings (for example
    Qwen3-VL), so nested config flags must not be treated as proof of tying
    (issue #4369).

    ``find_tied_parameters`` only compares materialized parameters. A lazily
    loaded shell keeps both sides as independent META parameters until the
    loader ties them, so identity never matches and quantization continues
    until the source reload fails with a misleading "missing checkpoint
    tensors" error. Detect that shape from the checkpoint instead: a tied HF
    checkpoint omits ``lm_head.weight``, so the head has no independent source
    tensor to quantize. The declared tied key at the root config is the final
    fallback for checkpoints that do store ``lm_head.weight`` (Qwen3) while the
    loader still ties it.
    """

    # Deferred: `utils/model.py` imports this module, so a module-level import
    # here would be circular.
    from accelerate.utils import find_tied_parameters

    from .model import get_module

    model = getattr(gptq_model, "model", None)
    if model is None:
        return False
    lm_head_name = lm_head_name or getattr(gptq_model, "lm_head", "lm_head")

    # 1. Actual storage identity: the input embedding and output head are the
    #    same parameter (or share materialized storage).
    try:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
    except (AttributeError, TypeError):
        input_embeddings = output_embeddings = None
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    if torch.is_tensor(input_weight) and torch.is_tensor(output_weight):
        if output_weight is input_weight:
            return True
        # Meta tensors all report data_ptr() == 0, so only compare storage for
        # materialized parameters.
        if not (
            getattr(input_weight, "is_meta", False)
            or getattr(output_weight, "is_meta", False)
        ):
            try:
                same_storage = (
                    input_weight.untyped_storage().data_ptr()
                    == output_weight.untyped_storage().data_ptr()
                    and input_weight.data_ptr() == output_weight.data_ptr()
                )
            except Exception:  # pragma: no cover - backend without untyped_storage
                same_storage = False
            if same_storage:
                return True

    # 2. Accelerate's tied-parameter report (materialized parameters only).
    if any(
        f"{lm_head_name}.weight" in names
        for names in find_tied_parameters(model)
    ):
        return True

    lm_head_module = get_module(model, key=lm_head_name)
    if lm_head_module is None:
        return False
    weight = getattr(lm_head_module, "weight", None)
    if not torch.is_tensor(weight) or not getattr(weight, "is_meta", False):
        # A materialized head either has its own storage or shares it; the
        # identity check above already covered the shared case.
        return False
    checker = getattr(getattr(gptq_model, "turtle_model", None), "has_checkpoint_tensor", None)
    if not callable(checker):
        return False
    if not checker(target_model=model, target_submodule=lm_head_module):
        return True

    # A tied checkpoint may still contain `lm_head.weight` (Qwen3 stores it
    # even though the head is tied). The load-time snapshot resolved this from
    # the decoder config and the checkpoint, so prefer it over the wrapper's
    # config flag (#4799, #4369).
    snapshot = lm_head_is_tied(model, lm_head_name)
    if snapshot is not None:
        return snapshot

    config = getattr(model, "config", None)
    if not bool(getattr(config, "tie_word_embeddings", False)):
        return False
    tied_keys = getattr(model, "_tied_weights_keys", None)
    tied_head = f"{lm_head_name}.weight"
    if isinstance(tied_keys, dict):
        return tied_head in tied_keys
    if tied_keys:
        return tied_head in set(tied_keys)
    return False
