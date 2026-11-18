# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import ProcessorMixin

from ...utils.looper_helpers import normalize_device_like
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig
from ..local import LocalModelLoader
from ._vlm_calibration import prepare_vlm_calibration_dataset


class ApertusQModel(BaseQModel):
    fusable = FusableConfig("apertus").merge(LlamaFusableConfig(), profiles=("qkv",))

    # q/k/v share the post-norm hidden state. Apertus' MLP has no gate projection,
    # so there is no gate/up pair to tag. Verified against a real (tiny, CPU)
    # forward in tests/module_tree/test_shared_input_cpu_forward.py.
    shared_input_verified_model_types = frozenset({"apertus"})

    pre_lm_head_norm_module = "model.norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "attention_layernorm": ("attention_layernorm:!",),
            "self_attn": (
                "q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x",
                "q_norm:!", "k_norm:!", "o_proj:1",
            ),
            "feedforward_layernorm": ("feedforward_layernorm:!",),
            # The MLP activation (xIELU by default) is not positively
            # homogeneous, so ``up_proj`` cannot be an AWQ scale source for
            # ``down_proj``.
            "mlp": ("up_proj:0:nosrc", "act_fn:!", "down_proj:1"),
        }
    ]


class Apertus1p5TextQModel(ApertusQModel):
    """Apertus 1.5 text backbone, including extended/pruned vocabulary layouts."""
    fusable = None  # No qualified executor for this definition.
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.apertus.configuration_apertus1p5.Apertus1p5TextConfig",
        model_class="gptqmodel.models.ports.apertus.modeling_apertus1p5.Apertus1p5TextForCausalLM",
    )


class Apertus1p5QModel(BaseQModel):
    """Apertus 1.5 multimodal wrapper with an Apertus language backbone."""

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.apertus.configuration_apertus1p5.Apertus1p5Config",
        model_class="gptqmodel.models.ports.apertus.modeling_apertus1p5.Apertus1p5ForConditionalGeneration",
        processor_class="gptqmodel.models.ports.apertus.processing_apertus1p5.Apertus1p5Processor",
    )
    require_load_processor = True
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "attention_layernorm": ("attention_layernorm:!",),
            "self_attn": (
                "q_proj:0:in=x",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "q_norm:!",
                "k_norm:!",
                "o_proj:1",
            ),
            "feedforward_layernorm": ("feedforward_layernorm:!",),
            "mlp": ("up_proj:0:nosrc", "act_fn:!", "down_proj:1"),
        },
    ]

    def _materialize_module(self, parent, name: str, module_path: str):
        target_device = normalize_device_like(self.quantize_config.device) or CPU
        setattr(
            parent,
            name,
            self.shell_module_materialize(
                getattr(parent, name),
                target_device,
                module_path=module_path,
            ),
        )

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model, "embed_tokens", "model.language_model.embed_tokens"),
            (language_model, "norm", "model.language_model.norm"),
            (language_model, "rotary_emb", "model.language_model.rotary_emb"),
            (core_model, "vision_tokenizer", "model.vision_tokenizer"),
            (core_model, "audio_tokenizer", "model.audio_tokenizer"),
        )
        for parent, name, module_path in modules:
            self._materialize_module(parent, name, module_path)

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model, "embed_tokens"),
            (language_model, "norm"),
            (language_model, "rotary_emb"),
            (core_model, "vision_tokenizer"),
            (core_model, "audio_tokenizer"),
        )

        if self.quantize_config.offload_to_disk:
            for parent, name in modules:
                offload_to_disk(
                    model=parent,
                    module=getattr(parent, name),
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        for parent, name in modules:
            setattr(parent, name, move_to(getattr(parent, name), device=CPU))

    def load_processor(self):
        return type(self).loader.load_processor(
            self.model_local_path,
            config=self.model.config,
            tokenizer=getattr(self.tokenizer, "tokenizer", self.tokenizer),
        )

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        return processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        return prepare_vlm_calibration_dataset(
            self, calibration_dataset, batch_size=batch_size, options=kwargs
        )


__all__ = ["Apertus1p5QModel", "Apertus1p5TextQModel", "ApertusQModel"]
