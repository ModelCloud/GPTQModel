# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Checkpoint-key conversion for Inkling 2 Mini.

The released checkpoint is a pre-quantized export: the language model lives
under ``model.llm`` and keeps the exporter's projection names (``wq_du``,
``wo_ud``, ``k_sconv``, ...), while the routed experts are packed block-scaled
tensors (NVFP4 on layers 0-19, MXFP8 on layers 20-23). Every rule below is a
pure rename, so the packed expert payload lands in
:class:`~gptqmodel.models.ports.inkling2_loop.experts.Inkling2LoopPackedExperts`
untouched and no weight is re-quantized or decoded to bf16.
"""

from __future__ import annotations

import re
from typing import Optional

from transformers.core_model_loading import WeightRenaming


_INDEX_GROUP = r"(\d+)"
_INDEX_TEMPLATE = "\\1"


def _key_template(pattern: str) -> str:
    r"""Turn a rename pattern into the key template it describes.

    ``layers\.(\d+)\.ln_1\.`` becomes ``layers.\1.ln_1.``: anchors and regex
    escapes are dropped, and the index capture becomes the ``\\1`` template.
    """

    template = pattern.removeprefix("^").removesuffix("$")
    template = template.replace(_INDEX_GROUP, _INDEX_TEMPLATE)
    return template.replace("\\.", ".")


def _key_pattern(template: str) -> str:
    """Regex matching ``template`` literally, with ``\\1`` as a capture group."""

    return re.escape(template).replace(re.escape(_INDEX_TEMPLATE), _INDEX_GROUP)


class CheckpointRename(WeightRenaming):
    """A pure rename that also reverses into a usable save mapping.

    Transformers reverses a rename by swapping the two pattern strings, but a
    source pattern is a regex (anchors, escaped dots) while a target pattern is a
    key template. Swapping them verbatim made ``save_pretrained`` write keys such
    as ``...embed.weight$``; the reverse direction is built from the templates
    instead.
    """

    def reverse_transform(self) -> WeightRenaming:
        source = self.source_patterns[0]
        target = _key_pattern(_key_template(self.target_patterns[0]))
        template = _key_template(source)
        # Mirror the anchor: a prefix rule must stay a prefix rule, or the save
        # side would only rewrite keys that end at the prefix.
        if source.startswith("^") and not source.endswith("$"):
            return CheckpointRename(source_patterns="^" + target, target_patterns=template)
        if source.endswith("$"):
            return CheckpointRename(source_patterns=target + "$", target_patterns=template)
        return CheckpointRename(source_patterns=target, target_patterns=template)


class IndexShiftRename(WeightRenaming):
    """Rename with a captured index shifted by ``offset``.

    The checkpoint lists the per-layer attention-residual queries from layer 1
    (layer 0 has no such input), so ``attn_queries.i`` belongs to ``layers.i+1``.
    ``reverse_transform`` (used by ``save_pretrained`` to write checkpoint names
    back) needs its own patterns: the reverse of a ``\\1`` template is not a
    valid regex, so the reverse direction is spelled out explicitly.
    """

    def __init__(
        self,
        source_patterns: str,
        target_patterns: str,
        offset: int = 1,
        reverse_source_patterns: Optional[str] = None,
        reverse_target_patterns: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(source_patterns, target_patterns, **kwargs)
        self.offset = offset
        self.reverse_source_patterns = reverse_source_patterns
        self.reverse_target_patterns = reverse_target_patterns

    @staticmethod
    def _template(pattern: str, index: int) -> str:
        return _key_template(pattern).replace("\\1", str(index), 1)

    def rename_source_key(self, source_key: str):
        match = re.search(self.source_patterns[0], source_key)
        if match is None:
            return source_key, None
        self._was_used = True
        target = self._template(self.target_patterns[0], int(match.group(1)) + self.offset)
        return source_key.replace(match.group(0), target, 1), self.source_patterns[0]

    def reverse_transform(self) -> WeightRenaming:
        if self.reverse_source_patterns is None:
            return super().reverse_transform()
        return IndexShiftRename(
            source_patterns=self.reverse_source_patterns,
            target_patterns=self.reverse_target_patterns,
            offset=-self.offset,
            reverse_source_patterns=self.source_patterns[0],
            reverse_target_patterns=self.target_patterns[0],
        )


#: Vision tower names.
_VISION_RENAMES = [
    (r"^model\.visual\.backbone\.embeddings\.", "model.vision_tower.embeddings."),
    (r"^model\.visual\.backbone\.ln_pre\.", "model.vision_tower.pre_layernorm."),
    (r"^model\.visual\.backbone\.layers\.", "model.vision_tower.layers."),
    (r"^model\.visual\.proj_in\.", "model.vision_tower.merger.linear_1."),
    (r"^model\.visual\.proj_out\.", "model.vision_tower.merger.linear_2."),
    (r"\.embeddings\.position_embedding$", ".embeddings.position_embedding.weight"),
    (r"(vision_tower\.layers\.\d+)\.ln_1\.", r"\1.layernorm_before."),
    (r"(vision_tower\.layers\.\d+)\.ln_2\.", r"\1.layernorm_after."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_fc\.", r"\1.mlp.fc1."),
    (r"(vision_tower\.layers\.\d+)\.mlp\.c_proj\.", r"\1.mlp.fc2."),
    (r"(vision_tower\.layers\.\d+)\.ls_1\.gamma$", r"\1.lambda_1"),
    (r"(vision_tower\.layers\.\d+)\.ls_2\.gamma$", r"\1.lambda_2"),
]

#: Attention block names.
_ATTENTION_RENAMES = [
    (r"\.attn\.wq_du\.", ".self_attn.q_proj.linear."),
    (r"\.attn\.wk_dv\.", ".self_attn.k_proj.linear."),
    (r"\.attn\.wv_dv\.", ".self_attn.v_proj.linear."),
    (r"\.attn\.wo_ud\.", ".self_attn.o_proj.linear."),
    (r"\.attn\.gate_attn\.", ".self_attn.o_proj.gate_proj."),
    (r"\.attn\.wr_du\.", ".self_attn.relative_position_bias.linear."),
    (r"\.attn\.rel_logits_proj\.", ".self_attn.relative_position_bias."),
    (r"\.attn\.q_l2_clip\.", ".self_attn.q_proj.norm."),
    (r"\.attn\.rel_l2_clip\.", ".self_attn.relative_position_bias.norm."),
    (r"\.attn\.attn_sink_proj\.", ".self_attn.sink_proj."),
    (r"\.attn\.k_sconv\.", ".self_attn.k_proj.sconv."),
    (r"\.attn\.v_sconv\.", ".self_attn.v_proj.sconv."),
    (r"\.attn_sconv\.", ".post_attention_convolution."),
    (r"\.mlp_sconv\.", ".post_mlp_convolution."),
    (r"\.attn_norm\.", ".input_layernorm."),
    (r"\.mlp_norm\.", ".post_attention_layernorm."),
    (r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm."),
    # short-convolution leaves, after the wrappers above have been renamed
    (r"(sconv|convolution)\.weight$", r"\1.conv1d.weight"),
    (r"(sconv|convolution)\.static_w$", r"\1.static_weight"),
    (r"(sconv|convolution)\.weight_proj$", r"\1.weight_proj.weight"),
]

#: Message gates, router bias and the packed-expert containers.
_GATE_AND_EXPERT_RENAMES = [
    (r"\.controllers\.attn\.", ".q_message_gate."),
    (r"\.controllers\.k\.", ".k_message_gate."),
    (r"\.controllers\.v\.", ".v_message_gate."),
    (r"\.controllers\.mlp\.", ".mlp_message_gate."),
    # Only the message gates use the `down_weight` spelling. The packed
    # experts also end in `down_weight` after their own rename, so these rules
    # must not fire on them. They are spelled out per gate instead of using one
    # alternation because the runtime->checkpoint walker derives its reverse
    # target regex from this pattern and drops non-capturing groups.
    (r"\.q_message_gate\.down_weight$", ".q_message_gate.down_proj.weight"),
    (r"\.k_message_gate\.down_weight$", ".k_message_gate.down_proj.weight"),
    (r"\.v_message_gate\.down_weight$", ".v_message_gate.down_proj.weight"),
    (r"\.mlp_message_gate\.down_weight$", ".mlp_message_gate.down_proj.weight"),
    (r"\.gain_weight$", ".message_gate_proj.weight"),
    (r"\.forget_weight$", ".forget_gate_proj.weight"),
    (r"\.mlp\.gate\.bias$", ".mlp.gate.e_score_correction_bias"),
    (r"\.w13_weight$", ".gate_up_weight"),
    (r"\.w13_weight_scale$", ".gate_up_scales"),
    (r"\.w13_weight_amax$", ".gate_up_amax"),
    (r"\.w13_weight_outer_scale_denominator$", ".gate_up_outer_scale"),
    (r"\.w2_weight$", ".down_weight"),
    (r"\.w2_weight_scale$", ".down_scales"),
    (r"\.w2_weight_amax$", ".down_amax"),
    (r"\.w2_weight_outer_scale_denominator$", ".down_outer_scale"),
]


def build_checkpoint_mapping():
    mapping = [
        CheckpointRename(source_patterns=r"^model\.llm\.unembed\.weight$", target_patterns=r"lm_head.weight"),
        CheckpointRename(source_patterns=r"^model\.llm\.", target_patterns=r"model.language_model."),
        # Keep both rules anchored: the loader applies every rule repeatedly
        # until none matches, and an unanchored `.embed_norm.` rule would keep
        # re-wrapping its own output.
        CheckpointRename(
            source_patterns=r"^model\.language_model\.embed_norm\.",
            target_patterns=r"model.language_model.embed_tokens.embed_norm.",
        ),
        CheckpointRename(
            source_patterns=r"^embed_norm\.",
            target_patterns=r"embed_tokens.embed_norm.",
        ),
        CheckpointRename(source_patterns=r"\.embed\.weight$", target_patterns=r".embed_tokens.weight"),
    ]
    mapping += [CheckpointRename(source_patterns=src, target_patterns=dst) for src, dst in _VISION_RENAMES]
    mapping += [CheckpointRename(source_patterns=src, target_patterns=dst) for src, dst in _ATTENTION_RENAMES]
    mapping += [CheckpointRename(source_patterns=src, target_patterns=dst) for src, dst in _GATE_AND_EXPERT_RENAMES]
    mapping += [
        IndexShiftRename(
            source_patterns=r"attn_res\.attn_queries\.(\d+)$",
            target_patterns=r"layers.\1.attention_input.query",
            offset=1,
            reverse_source_patterns=r"layers\.(\d+)\.attention_input\.query$",
            reverse_target_patterns=r"attn_res.attn_queries.\1",
        ),
        IndexShiftRename(
            source_patterns=r"attn_res\.mlp_queries\.(\d+)$",
            target_patterns=r"layers.\1.mlp_input.query",
            offset=0,
            reverse_source_patterns=r"layers\.(\d+)\.mlp_input\.query$",
            reverse_target_patterns=r"attn_res.mlp_queries.\1",
        ),
        CheckpointRename(source_patterns=r"attn_res\.output_query$", target_patterns=r"final_input.query"),
        CheckpointRename(
            source_patterns=r"\.middle_cycle_cross_pass_queries$",
            target_patterns=r".message_input.query",
        ),
        CheckpointRename(
            source_patterns=r"\.middle_cycle_cross_pass_kv_queries$",
            target_patterns=r".message_input.kv_query",
        ),
    ]
    return mapping


__all__ = [
    "build_checkpoint_mapping",
    "CheckpointRename",
    "IndexShiftRename",
]
