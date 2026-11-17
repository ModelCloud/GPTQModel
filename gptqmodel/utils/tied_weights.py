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

* the decoder text config (``config.get_text_config().tie_word_embeddings``)
  when the wrapper exposes one, so a composite default cannot tie a decoder
  that declares itself untied, and a decoder default cannot tie a wrapper that
  declares itself untied (#4369);
* the checkpoint: a pair whose source tensor is absent cannot be tied, and a
  pair whose target tensor is present *and differs* from the source is a real
  separate head, not a tie;
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
    """

    declared = _declared_tied_weights_keys(model)
    if not declared:
        return {}
    if not word_embeddings_tied(getattr(model, "config", None)):
        return {}

    resolved: Dict[str, str] = {}
    for target, source in declared.items():
        if not isinstance(target, str) or not isinstance(source, str):
            continue
        if checkpoint_has_tensor is not None:
            if checkpoint_has_tensor(source) is False:
                # Nothing to tie to: the source tensor is not in the checkpoint.
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
