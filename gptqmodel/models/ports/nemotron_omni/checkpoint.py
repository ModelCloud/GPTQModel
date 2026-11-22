# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Flatten native scoped renames for LazyTurtle's per-leaf materialization."""

from types import SimpleNamespace


def reversed_mapping():
    from transformers.core_model_loading import Chunk

    pairs = [
        (r"^language_model\.model\.", "language_model.backbone."),
        (r"^multi_modal_projector\.layer_norm\.", "mlp1.0."),
        (r"^multi_modal_projector\.linear_1\.", "mlp1.1."),
        (r"^multi_modal_projector\.linear_2\.", "mlp1.3."),
        (r"^audio_tower\.", "sound_encoder.encoder."),
        (r"^embed_audio\.layer_norm\.", "sound_projection.norm."),
        (r"^embed_audio\.linear_1\.", "sound_projection.linear1."),
        (r"^embed_audio\.linear_2\.", "sound_projection.linear2."),
        (r"^vision_model\.embeddings\.video_patch_projection", "vision_model.radio_model.model.patch_generator.video_embedder"),
        (r"^vision_model\.embeddings\.patch_projection", "vision_model.radio_model.model.patch_generator.embedder"),
        (r"^vision_model\.embeddings\.position_embedding", "vision_model.radio_model.model.patch_generator.pos_embed"),
        (r"^vision_model\.embeddings\.cls_register_token", "vision_model.radio_model.model.patch_generator.cls_token.token"),
        (r"^vision_model\.encoder\.layer\.(\d+)\.attention\.o_proj\.",
         r"vision_model.radio_model.model.blocks.\1.attn.proj."),
        (r"^vision_model\.encoder\.layer\.(\d+)\.", r"vision_model.radio_model.model.blocks.\1."),
    ]
    entries = [SimpleNamespace(source_patterns=[a], target_patterns=[b], operations=[]) for a, b in pairs]
    # Put the packed-QKV rules first: a general block-prefix rename would lose
    # the slice owner before the Chunk converter sees it.
    for suffix in ("weight", "bias"):
        entries.insert(0, SimpleNamespace(
            source_patterns=[rf"^vision_model\.encoder\.layer\.(\d+)\.attention\.{name}\.{suffix}$"
                             for name in ("q_proj", "k_proj", "v_proj")],
            target_patterns=[rf"vision_model.radio_model.model.blocks.\1.attn.qkv.{suffix}"],
            operations=[Chunk(dim=0)],
        ))
    return entries
