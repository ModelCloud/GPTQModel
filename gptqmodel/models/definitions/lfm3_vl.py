# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict

from transformers import ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..local import LocalModelLoader
from .lfm3 import LFM3QModel


class LFM3VLQModel(LFM3QModel):
    """LiquidAI LFM3-VL: LFM3 text decoder + LFM3 vision tower.

    Only the text decoder's linear projections are quantized.  The vision tower
    (patch embedding, gated depthwise convolutions, attention/MLP projections) and
    the multimodal projector stay in their checkpoint dtype, matching the way
    other VL definitions keep their vision encoders unquantized.

    Native Transformers converts the checkpoint's attention gate and vision
    patch/norm names. Definition-owned aliases select those exact native
    classes, independently of checkpoint auto_map and caller trust flags.
    """

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.lfm3_vl.native.Lfm3VlConfig",
        model_class="gptqmodel.models.ports.lfm3_vl.native.Lfm3VlForConditionalGeneration",
        processor_class="gptqmodel.models.ports.lfm3_vl.native.Lfm3VlNativeProcessor",
    )

    pre_lm_head_norm_module = "model.language_model.embedding_norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "operator_norm": ("operator_norm:!",),
            "post_operator_norm": ("post_operator_norm:!:postnorm",),
            "conv": {
                "in_proj": ("in_proj:0",),
                "out_proj": ("out_proj:1",),
                "conv:!": (),
            },
            "self_attn": (
                "q_proj:0",
                "q_norm:0:!",
                "k_proj:0",
                "k_norm:0:!",
                "v_proj:0",
                "g_proj:!",
                "o_proj:1",
            ),
            "ffn_norm": ("ffn_norm:!",),
            "post_ffn_norm": ("post_ffn_norm:!:postnorm",),
            "feed_forward": ("w1:0", "w3:0", "w2:1"),
        },
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True
    require_trust_remote_code = False
    layer_modules_strict = False

    def _language_model(self):
        return self.model.model.language_model

    def _text_config(self):
        return self.model.model.language_model.config

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        language_model = core_model.language_model
        self.shell_module_materialize(
            language_model.embed_tokens,
            self.quantize_config.device,
            module_path="model.language_model.embed_tokens",
        )
        self.shell_module_materialize(
            language_model.rotary_emb,
            self.quantize_config.device,
            module_path="model.language_model.rotary_emb",
        )
        self.shell_module_materialize(
            language_model.embedding_norm,
            self.quantize_config.device,
            module_path="model.language_model.embedding_norm",
        )
        self.shell_module_materialize(
            core_model.vision_tower,
            self.quantize_config.device,
            module_path="model.vision_tower",
        )
        self.shell_module_materialize(
            core_model.multi_modal_projector,
            self.quantize_config.device,
            module_path="model.multi_modal_projector",
        )
        self._start_outer_input_capture()

    def pre_quantize_generate_hook_end(self):
        self._stop_outer_input_capture()
        core_model = self.model.model
        language_model = core_model.language_model
        modules = (
            (language_model, "embed_tokens"),
            (language_model, "rotary_emb"),
            (language_model, "embedding_norm"),
            (core_model, "vision_tower"),
            (core_model, "multi_modal_projector"),
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

    def load_processor(self) -> ProcessorMixin:
        return self.loader.load_processor(
            self.model_local_path,
            config=self.model.config,
            tokenizer=getattr(self.tokenizer, "tokenizer", self.tokenizer),
        )

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        # Native hybrid conv/attention has sigmoid gates, Q/K head norms and
        # sandwich norms. Its complete source/consumer folds are unqualified;
        # keep every declared linear on the existing plain AWQ path.
        return []

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        return processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.load_processor()
        calib_data = []
        for batch in batched_conversations(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            # Native LFM3-VL's patch embedding consumes the model dtype;
            # media processors emit FP32 pixels even for an FP16 checkpoint.
            calib_data.append(self.prepare_inputs_for_conversations(processor, batch).to(dtype=self.model.dtype))
        del processor
        return calib_data


__all__ = ["LFM3VLQModel"]
