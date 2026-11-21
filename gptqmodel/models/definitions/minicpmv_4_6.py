# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Dict

from transformers import ProcessorMixin

from ...utils.calibration import batched_conversations
from ...utils.model import MODALITY, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..fusable import FusableConfig
from ..local import LocalModelLoader
from ._vlm_calibration import (
    finalize_conversation_batches,
    pop_conversation_calibration_options,
    split_calibration_dataset,
)
from .qwen3_5 import Qwen3_5QModel


class MiniCPMV4_6QModel(BaseQModel):
    fusable = FusableConfig("minicpmv4_6").merge(Qwen3_5QModel.fusable)

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForImageTextToText",
        processor_class="gptqmodel.models.ports.native.auto.NativeProcessor",
        tokenizer_class="gptqmodel.models.ports.native.auto.NativeTokenizer",
    )

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Parallel Q/K/V read the same input norm. The DeltaNet
            # `in_proj_qkv`/`in_proj_z` branches read that same norm output
            # through the layer input, so all four must fold into one source
            # (Qwen3.5's tree). Without the tags the second branch folds into
            # `in_proj_qkv` and the norm scale is never compensated.
            "self_attn": (
                "q_norm:!",
                "q_proj:0:in=x",
                "k_norm:!",
                "k_proj:0:in=x",
                "v_proj:0:in=x",
                "o_proj:1",
            ),
            "linear_attn": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0:in=x",
                "in_proj_z:1:in=x",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0:in=x", "up_proj:0:in=x", "down_proj:1"),
        }
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True
    require_trust_remote_code = False
    layer_modules_strict = False

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        if getattr(module, "block_type", None) != "linear_attention":
            return nodes

        # Transformers 5.23 Qwen3_5DecoderLayer feeds input_layernorm to all
        # four DeltaNet input projections. Missing self_attn alternatives and
        # the *later*, per-head gated norm are not execution boundaries here.
        # Keep the generic absent-block barrier; bind this native edge directly.
        attn = module.linear_attn
        internal = {id(child) for child in attn.modules()}
        nodes = [node for node in nodes
                 if node["prev_op"] is not module.input_layernorm
                 and id(node["prev_op"]) not in internal
                 and not any(id(target) in internal for target in node["layers"])]
        feature = "linear_attn.in_proj_qkv"
        if feature not in input_feat:
            return nodes
        kwargs = dict(module_kwargs or {})
        per_feature = kwargs.pop("_awq_feature_kwargs", {})
        kwargs.update(per_feature.get(feature, {}))
        # The decoder calls the same cache `past_key_values`; DeltaNet calls
        # it `cache_params`. Never silently lose a captured recurrent state.
        if "past_key_values" in kwargs:
            kwargs.setdefault("cache_params", kwargs.pop("past_key_values"))
        node = {
            "prev_op": module.input_layernorm,
            "layers": [attn.in_proj_qkv, attn.in_proj_z],
            "inp": input_feat[feature],
            "module2inspect": attn,
            "kwargs": kwargs,
            "_input_feature_name": feature,
            "_scale_consumers": ["linear_attn.in_proj_b", "linear_attn.in_proj_a"],
        }
        # No internal DeltaNet fold: convolution, recurrent mixing and gated
        # per-head RMS normalization separate the inputs from out_proj.
        return self._filter_awq_scaling_sources(module, [node, *nodes])

    def awq_module_replay_inputs(self, module, x, module_kwargs):
        x, kwargs = super().awq_module_replay_inputs(module, x, module_kwargs)
        # Calibration is an uncached prefill. A mutable recurrent/KV cache
        # would advance on the reference and again on each candidate replay.
        if any(kwargs.get(key) is not None for key in ("cache_params", "past_key_values")):
            raise NotImplementedError("MiniCPM-V 4.6 AWQ requires uncached calibration context")
        if getattr(module, "layer_type", None) == "linear_attention":
            if x.ndim != 3:
                raise NotImplementedError("MiniCPM-V 4.6 DeltaNet replay requires batch/sequence/hidden inputs")
            mask = kwargs.get("attention_mask")
            if mask is not None and (mask.ndim != 2 or tuple(mask.shape) != tuple(x.shape[:2])):
                raise NotImplementedError("MiniCPM-V 4.6 DeltaNet replay requires its captured batch/sequence mask")
        return x, kwargs

    def pre_quantize_generate_hook_start(self):
        language_model = self.model.model.language_model
        self.shell_module_materialize(language_model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(language_model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.model.vision_tower, self.quantize_config.device)
        self.shell_module_materialize(self.model.model.merger, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        language_model = self.model.model.language_model
        if self.quantize_config.offload_to_disk:
            offload_to_disk(
                model=language_model,
                module=language_model.embed_tokens,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=language_model,
                module=language_model.rotary_emb,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model,
                module=self.model.model.vision_tower,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            offload_to_disk(
                model=self.model,
                module=self.model.model.merger,
                disk_path=self.quantize_config.offload_to_disk_path,
            )
            return

        language_model.embed_tokens = move_to(language_model.embed_tokens, device=CPU)
        language_model.rotary_emb = move_to(language_model.rotary_emb, device=CPU)
        self.model.model.vision_tower = move_to(self.model.model.vision_tower, device=CPU)
        self.model.model.merger = move_to(self.model.model.merger, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        processor = getattr(self, "processor", None)
        if processor is not None:
            return processor
        return self.loader.load_processor(
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
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]

        downsample_mode = "16x"  # Using `downsample_mode="4x"` for Finer Detail

        inputs = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
            downsample_mode=downsample_mode,
            max_slice_nums=36,
        )
        return inputs

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        label = type(self).__name__
        pretokenized, conversations = split_calibration_dataset(calibration_dataset, label=label)
        if pretokenized is not None:
            return super().prepare_dataset(pretokenized, batch_size=batch_size, **kwargs)

        sort, min_length = pop_conversation_calibration_options(kwargs, label=label)
        processor = self.load_processor()
        calib_data = []
        for batch in batched_conversations(conversations, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(
                self.prepare_inputs_for_conversations(
                    processor,
                    batch,
                )
            )
        del processor
        return finalize_conversation_batches(
            calib_data, sort=sort, min_length=min_length, label=label
        )

    def move_input_capture_example(self, example, data_device):
        for key, value in example.items():
            example[key] = nested_move_to(value, device=data_device)

        return self.finalize_input_capture_example(example)

    def run_input_capture(self, example, use_cache: bool, data_device):
        return self.model.generate(
            **example,
            use_cache=use_cache,
        )
