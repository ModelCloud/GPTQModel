# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Scoped legacy aliases for lazy loading, without reapplying root HF renames."""

from types import SimpleNamespace

from transformers.core_model_loading import Chunk


def checkpoint_mapping():
    pairs = [
        (r"model\.language_model\.embed_tokens\.", "model.embed_tokens."),
        (r"model\.language_model\.layers\.", "model.layers."),
        (r"model\.language_model\.norm\.", "model.norm."),
        (r"model\.view_separator$", "model.view_seperator"),
        (r"model\.multi_modal_projector\.", "model.projector.layers."),
    ]
    sam = "model.vision_tower.sam_encoder."
    for native, source in [
        (r"layers\.(\d+)\.layer_norm1\.", r"blocks.\1.norm1."),
        (r"layers\.(\d+)\.layer_norm2\.", r"blocks.\1.norm2."),
        (r"layers\.(\d+)\.", r"blocks.\1."),
        (r"patch_embed\.projection\.", "patch_embed.proj."),
        (r"neck\.conv1\.", "neck.0."), (r"neck\.layer_norm1\.", "neck.1."),
        (r"neck\.conv2\.", "neck.2."), (r"neck\.layer_norm2\.", "neck.3."),
        (r"proj\.conv1\.", "net_2."), (r"proj\.conv2\.", "net_3."),
        ("", ""),
    ]:
        pairs.append((sam.replace(".", r"\.") + native, "model.sam_model." + source))
    vision = r"model\.vision_tower\.vision_encoder\."
    for native, source in [(r"encoder\.", "transformer."), ("", "")]:
        pairs.append((vision + native, "model.vision_model." + source))
    mappings = [SimpleNamespace(source_patterns=["^" + native], target_patterns=["^" + source], operations=[])
                for native, source in pairs]
    # QKV is checkpoint-fused only in the second vision encoder.
    for suffix in ("weight", "bias"):
        mappings.insert(0, SimpleNamespace(
            source_patterns=[vision + rf"encoder\.layers\.(\d+)\.self_attn\.{projection}_proj\.{suffix}"
                             for projection in ("q", "k", "v")],
            target_patterns=[rf"model.vision_model.transformer.layers.\1.self_attn.qkv_proj.{suffix}"],
            operations=[Chunk(dim=0)],
        ))
    return mappings
