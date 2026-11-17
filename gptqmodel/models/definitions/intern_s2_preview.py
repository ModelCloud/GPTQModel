# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict

from transformers import ProcessorMixin

from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..local import LocalModelLoader
from ..moe_lifecycle import GateUpDownMoELifecycleHooks
from ._vlm_calibration import prepare_vlm_calibration_dataset


class InternS2PreviewQModel(BaseQModel):
    require_pkgs = ["sentencepiece>=0.2.0"]

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.intern_s2_preview.configuration_interns2_preview.InternS2PreviewConfig",
        model_class="gptqmodel.models.ports.intern_s2_preview.modeling_interns2_preview.InternS2PreviewForConditionalGeneration",
        processor_class="gptqmodel.models.ports.intern_s2_preview.processing_interns2_preview.InternS2PreviewProcessor",
        tokenizer_class="gptqmodel.models.ports.intern_s2_preview.tokenization_interns1.InternS1Tokenizer",
    )

    require_load_processor = True
    layer_modules_strict = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    dynamic_expert_index = "num_experts"
    defuser_module_paths = ("model.language_model",)
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"
    out_of_model_tensors = {"prefixes": ["mtp"]}

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # The DeltaNet tree is Qwen3.5's: Q/K/V and the linear-attention
            # input projections all read the same `input_layernorm` output.
            # Missing `:in=x` tags made the second branch fold into
            # `in_proj_qkv` while the norm scale was never compensated.
            "self_attn": (
                "q_norm:!",
                "q_proj:0:in=x",
                "k_norm:!",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "linear_attn:!": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0:in=x",
                "in_proj_z:1:in=x",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert:0": (
                    "gate_proj:0:in=x",
                    "up_proj:0:in=x",
                    "down_proj:1",
                ),
                "experts:0": {
                    "#": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
                },
            },
        },
    ]

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Hybrid recurrence, gated full attention and packed MoE are not
        # qualified for complete AWQ search/replay. Quantize without folding.
        return []

    def _materialize_core_module(self, parent, attr_name: str):
        module = getattr(parent, attr_name)
        setattr(
            parent,
            attr_name,
            self.shell_module_materialize(module, self.quantize_config.device),
        )

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        self._materialize_core_module(core_model.language_model, "embed_tokens")
        self._materialize_core_module(core_model.language_model, "rotary_emb")
        self._materialize_core_module(core_model, "visual")

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        core_modules = (
            (core_model.language_model, "embed_tokens"),
            (core_model.language_model, "rotary_emb"),
            (core_model, "visual"),
        )

        if self.quantize_config.offload_to_disk:
            for parent, attr_name in core_modules:
                offload_to_disk(
                    model=parent,
                    module=getattr(parent, attr_name),
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        for parent, attr_name in core_modules:
            setattr(parent, attr_name, move_to(getattr(parent, attr_name), device=CPU))

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        return processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
            return_dict=True,
            return_tensors="pt",
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        return prepare_vlm_calibration_dataset(
            self,
            calibration_dataset,
            batch_size=batch_size,
            options=kwargs,
            preprocess_func=self.preprocess_dataset,
        )


__all__ = ["InternS2PreviewQModel"]
