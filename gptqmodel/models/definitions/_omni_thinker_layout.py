# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint <-> runtime key layout shared by the Qwen Omni families.

Issue #4881: transformers 5.22 builds the Omni thinker with the towers under
``thinker.model.visual``/``thinker.model.audio_tower`` and the decoder under
``thinker.model.language_model``. A Qwen2.5-Omni and a Qwen3-Omni checkpoint
instead store ``thinker.visual.*``, ``thinker.audio_tower.*`` and
``thinker.model.*`` (including ``thinker.model.embed_tokens``). The
transformers conversion registry carries those renames only for the nested
``qwen2_5_omni_thinker``/``qwen3_omni_moe_thinker`` model types, so resolving
the top-level wrapper finds no map and the lazy loader leaves the embeddings,
the rotary embeddings and both towers on META. Input capture then fails (the
reported ``Tensor.item() cannot be called on meta tensors`` in
``create_causal_mask``) and a save would write runtime keys that a reload
cannot map back (#4758).

The two families share this checkpoint layout, so the map lives here rather
than being duplicated per definition. It is the inverse of the transformers
``Qwen2_5OmniThinkerForConditionalGeneration`` conversion map, which the
Qwen3-Omni thinker inherits through ``_MODEL_TO_CONVERSION_PATTERN``.
"""

from __future__ import annotations

from types import SimpleNamespace

from ...utils.structure import LazyTurtle


class OmniThinkerCheckpointLayoutMixin:
    """Resolve the Omni thinker's runtime names to source HF checkpoint keys."""

    HF_CONVERSION_MAP_REVERSED = (
        SimpleNamespace(
            source_patterns=[r"^thinker\.model\.visual\.(.+)$"],
            target_patterns=[r"^thinker.visual.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"^thinker\.model\.audio_tower\.(.+)$"],
            target_patterns=[r"^thinker.audio_tower.\1"],
            operations=[],
        ),
        SimpleNamespace(
            source_patterns=[r"^thinker\.model\.language_model\.(.+)$"],
            target_patterns=[r"^thinker.model.\1"],
            operations=[],
        ),
    )

    @classmethod
    def checkpoint_key_for_save(cls, name: str) -> str:
        """Write checkpoint keys in the source HF layout (issues #4758/#4881).

        Transformers 5.22 keeps the thinker towers under ``thinker.model.*``
        and the decoder under ``thinker.model.language_model.*`` in memory, but
        an Omni checkpoint stores ``thinker.audio_tower.*``,
        ``thinker.visual.*`` and ``thinker.model.embed_tokens.*``. Saving the
        in-memory names writes a checkpoint the loader cannot map back, so the
        towers and embeddings reload as ``torch.empty`` garbage.
        """
        return LazyTurtle.convert_hf_checkpoint_key(name, cls.HF_CONVERSION_MAP_REVERSED)

    @classmethod
    def checkpoint_key_for_load(cls, name: str) -> str:
        """Map a source-layout checkpoint key back onto the runtime module."""
        return LazyTurtle.convert_hf_checkpoint_key(
            name,
            cls.HF_CONVERSION_MAP_REVERSED,
            to_runtime=True,
        )
