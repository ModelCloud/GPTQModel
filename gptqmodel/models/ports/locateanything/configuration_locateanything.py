# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Adapted by ModelCloud for Transformers 5. The pinned source repository's
# accompanying NVIDIA LICENSE applies in addition to this preserved header.
"""Composition config without checkpoint class discovery."""

from transformers.configuration_utils import PretrainedConfig

from ..native.qwen2.configuration_qwen2 import Qwen2Config
from ..native.qwen3.configuration_qwen3 import Qwen3Config


class MoonViTConfig(PretrainedConfig):
    model_type = "moonvit"

    def __init__(self, patch_size=14, init_pos_emb_height=64, init_pos_emb_width=64,
                 num_attention_heads=16, num_hidden_layers=27, hidden_size=1152,
                 intermediate_size=4304, merge_kernel_size=(2, 2), **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.merge_kernel_size = list(merge_kernel_size)


class LocateAnythingConfig(PretrainedConfig):
    model_type = "locateanything"
    sub_configs = {"vision_config": MoonViTConfig, "text_config": Qwen2Config}

    def __init__(self, vision_config=None, text_config=None, use_backbone_lora=0,
                 use_llm_lora=0, downsample_ratio=0.5, template=None, loss_version="v1",
                 mlp_checkpoint=False, image_token_index=151667, box_start_token_id=151668,
                 box_end_token_id=151669, coord_start_token_id=151677, coord_end_token_id=152677,
                 ref_start_token_id=151672, ref_end_token_id=151673, none_token_id=4064, **kwargs):
        # The checkpoint advertises an optional Magi kernel. Dense SDPA implements
        # its exact inference visibility, without importing an optional CUDA package.
        implementation = kwargs.pop("attn_implementation", kwargs.pop("_attn_implementation", None))
        if implementation == "magi":
            implementation = "sdpa"
        if implementation not in (None, "eager", "sdpa"):
            raise ValueError("LocateAnything's block-diffusion wrapper supports eager or sdpa attention")
        super().__init__(**kwargs)
        self._attn_implementation = implementation or "sdpa"
        vision = vision_config.to_dict() if isinstance(vision_config, PretrainedConfig) else dict(vision_config or {})
        if vision.get("model_type", "moonvit") != "moonvit":
            raise ValueError("LocateAnything requires MoonViT vision_config")
        self.vision_config = MoonViTConfig(**vision)
        text = text_config.to_dict() if isinstance(text_config, PretrainedConfig) else dict(text_config or {})
        architecture = (text.get("architectures") or ["Qwen2ForCausalLM"])[0]
        classes = {"Qwen2ForCausalLM": Qwen2Config, "Qwen3ForCausalLM": Qwen3Config}
        if architecture not in classes:
            raise ValueError(f"Unsupported LocateAnything decoder: {architecture}")
        text["architectures"] = [architecture]
        text.pop("_attn_implementation", None)
        self.text_config = classes[architecture](**text)
        self.vision_config._attn_implementation = self._attn_implementation
        self.text_config._attn_implementation = self._attn_implementation
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.loss_version = loss_version
        self.mlp_checkpoint = mlp_checkpoint
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.image_token_index = image_token_index
        self.box_start_token_id = box_start_token_id
        self.box_end_token_id = box_end_token_id
        self.coord_start_token_id = coord_start_token_id
        self.coord_end_token_id = coord_end_token_id
        self.ref_start_token_id = ref_start_token_id
        self.ref_end_token_id = ref_end_token_id
        self.none_token_id = none_token_id
