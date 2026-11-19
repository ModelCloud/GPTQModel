# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""GPT-QModel support for Decision Transformer offline-RL checkpoints (#4755)."""

from transformers import AutoModel

from ..base import BaseQModel


class DecisionTransformerQModel(BaseQModel):
    """Quantization definition for ``DecisionTransformerModel`` checkpoints.

    Decision Transformer stacks a plain GPT-2 decoder under ``encoder`` and
    surrounds it with tiny state/action/return heads. Only the decoder blocks are
    quantized: the heads are 1-8 features wide, so a 4-bit group of 32 would be
    degenerate. ``embed_*``/``predict_*`` therefore stay in their native dtype,
    matching how OpenVLA keeps its non-decoder stack native.

    These checkpoints never ship a tokenizer and their forward consumes named
    tensors (``states``/``actions``/``rewards``/``returns_to_go``/``timesteps``)
    instead of ``input_ids``, so tokenizer loading is skipped and calibration
    rows are forwarded as keyword arguments.
    """

    # `DecisionTransformerModel` is not a causal LM; `AutoModel` resolves it.
    loader = AutoModel

    # Decision Transformer checkpoints have no tokenizer to load, and calibration
    # samples are named tensors rather than token ids.
    require_load_tokenizer = False
    calibration_inputs = "tensor_kwargs"

    module_tree = [
        "encoder",
        "h",
        "#",
        {
            "ln_1": ("ln_1:!:norm",),
            "attn": ("c_attn:0", "c_proj:1"),
            "ln_2": ("ln_2:!:norm",),
            "mlp": ("c_fc:0", "c_proj:1"),
        }
    ]

    # Root-level modules `DecisionTransformerModel.forward` consumes outside the
    # decoder: `embed_*` runs before the first block, and `predict_*` is used by
    # any post-quantize forward. They must be materialized from a lazy (meta)
    # shell even though the module tree only declares the decoder.
    _ROOT_MODULES = (
        "embed_timestep",
        "embed_return",
        "embed_state",
        "embed_action",
        "embed_ln",
        "predict_state",
        "predict_action",
        "predict_return",
    )

    @classmethod
    def get_base_modules(cls, model):
        base_modules = super().get_base_modules(model)
        seen = set(base_modules)
        for name in cls._ROOT_MODULES:
            if getattr(model, name, None) is not None and name not in seen:
                base_modules.append(name)
                seen.add(name)
        return base_modules


__all__ = ["DecisionTransformerQModel"]
