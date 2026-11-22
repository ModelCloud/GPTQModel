# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


from collections.abc import Mapping

from torch import nn

from ...quantization.awq.utils.module import get_op_name
from ..base import BaseQModel
from ..local import LocalModelLoader


class Phi4MMGPTQ(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.phi4.configuration_phi4.Phi4MMConfig",
        model_class="gptqmodel.models.ports.phi4.modeling_phi4.Phi4MMForCausalLM",
        processor_class="gptqmodel.models.ports.phi4.processing_phi4.Phi4MMProcessor",
        tokenizer_class="gptqmodel.models.ports.phi4.tokenization_phi4.Phi4MMTokenizer",
    )
    require_pkgs = []
    require_load_processor = True
    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": {
                "qkv_proj": {
                    "base_layer": ("base_layer",),
                    "lora_A": ("vision:!", "speech:!"),
                    "lora_B": ("vision:!", "speech:!"),
                },
                "o_proj": {
                    "base_layer": ("base_layer",),
                    "lora_A": ("vision:!", "speech:!"),
                    "lora_B": ("vision:!", "speech:!"),
                },
            },
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": {
                "gate_up_proj": {
                    "base_layer": ("base_layer",),
                    "lora_A": ("vision:!", "speech:!"),
                    "lora_B": ("vision:!", "speech:!"),
                },
                "down_proj": {
                    "base_layer": ("base_layer",),
                    "lora_A": ("vision:!", "speech:!"),
                    "lora_B": ("vision:!", "speech:!"),
                },
            },
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Every projection is a `ModalityLinear`: `base_layer(x)` plus, in vision
        # or speech mode, `lora_B(lora_A(x))`. A norm fold into `base_layer` is
        # exact only when both `lora_A` adapters, which read the same `x`, are
        # compensated. The generic run-based consumer scan cannot see them
        # (they are declared after the base layer) and instead picks the
        # previous projection's `lora_A`, so replace the consumers. A fold into
        # a projection output would also have to scale its `lora_B` rows;
        # keep only norm-sourced groups.
        nodes = []
        for node in super().awq_get_modules_for_scaling(module, input_feat, module_kwargs):
            if isinstance(node.get("prev_op"), nn.Linear):
                continue
            consumers = []
            for target in node["layers"]:
                name = get_op_name(module, target)
                owner_name = name.removesuffix(".base_layer")
                adapters = getattr(module.get_submodule(owner_name), "lora_A", None) if owner_name != name else None
                if adapters is None:
                    break
                consumers.extend(f"{owner_name}.lora_A.{adapter}" for adapter in adapters)
            else:
                node["_scale_consumers"] = consumers
                # The adapters only run in vision/speech mode, and the AWQ
                # replay of `self_attn`/`mlp` runs outside the decoder layer's
                # mode context, so they never execute there. Score the base
                # projection (its quantization error does not depend on the
                # additive adapter delta) and still fold into the adapters.
                node["_awq_allow_partial_replay"] = True
                nodes.append(node)
        return nodes

    def prepare_dataset(self, calibration_dataset, batch_size=1, **kwargs):
        from ._vlm_calibration import finalize_conversation_batches, pop_conversation_calibration_options

        rows = [calibration_dataset] if isinstance(calibration_dataset, Mapping) else list(calibration_dataset)
        if not any(
            isinstance(row, Mapping)
            and any(key in row for key in ("input_image_embeds", "input_audio_embeds", "input_mode"))
            for row in rows
        ):
            return super().prepare_dataset(rows, batch_size=batch_size, **kwargs)
        if batch_size != 1:
            raise ValueError("phi4mm prepared media rows are already batches; use batch_size=1")
        if not all(isinstance(row, Mapping) and "input_ids" in row for row in rows):
            raise ValueError("phi4mm media calibration requires processor-produced input batches")
        sort, min_length = pop_conversation_calibration_options(kwargs, label="phi4mm")
        return finalize_conversation_batches(
            [dict(row) for row in rows], sort=sort, min_length=min_length, label="phi4mm"
        )


__all__ = ["Phi4MMGPTQ"]
