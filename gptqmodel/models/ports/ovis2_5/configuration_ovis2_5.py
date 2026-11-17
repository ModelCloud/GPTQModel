# Copyright (C) 2025 AIDC-AI
# SPDX-License-Identifier: Apache-2.0
# GPT-QModel adaptation; pinned source and changes: README.md, NOTICE.
"""Ovis2.5's Qwen3 decoder and custom SigLIP2 NaViT configuration."""

from transformers import PretrainedConfig, Qwen3Config


class Siglip2NavitConfig(PretrainedConfig):
    model_type = "siglip2_navit"

    def __init__(self, hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                 num_attention_heads=16, num_channels=3, num_patches=-1, image_size=512,
                 patch_size=16, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
                 attention_dropout=0., hidden_stride=2, window_size=112,
                 fullatt_block_indexes=None, temporal_patch_size=1,
                 preserve_original_pe=True, use_rope=True, **kwargs):
        super().__init__(**kwargs)
        for name, value in locals().copy().items():
            if name not in {"self", "kwargs", "__class__"}:
                setattr(self, name, value)
        if temporal_patch_size != 1:
            raise ValueError("Ovis2.5 source patch projection supports temporal_patch_size=1 only")
        if hidden_size % num_attention_heads or hidden_size // num_attention_heads % 4:
            raise ValueError("Ovis2.5 vision head dimension must be divisible by four")
        if hidden_stride < 1 or window_size < hidden_stride * patch_size:
            raise ValueError("Ovis2.5 requires a positive merge stride and at least one window cell")
        if num_patches > 0 and int(num_patches ** .5) ** 2 != num_patches:
            raise ValueError("Ovis2.5 positional embedding table must be square")


class Ovis2_5_Config(PretrainedConfig):
    model_type = "ovis2_5"
    sub_configs = {"llm_config": Qwen3Config, "vit_config": Siglip2NavitConfig}

    def __init__(self, llm_config=None, vit_config=None, visual_vocab_size=65536,
                 hidden_size=None, min_pixels=448 * 448, max_pixels=None, **kwargs):
        llm_config = {} if llm_config is None else llm_config
        vit_config = {} if vit_config is None else vit_config
        if isinstance(llm_config, dict):
            if llm_config.get("model_type", "qwen3") != "qwen3":
                raise ValueError("Ovis2.5 requires the released Qwen3 decoder layout")
            llm_config = Qwen3Config(**llm_config)
        if isinstance(vit_config, dict):
            if vit_config.get("model_type", "siglip2_navit") != "siglip2_navit":
                raise ValueError("Ovis2.5 requires the custom Siglip2Navit vision layout")
            vit_config = Siglip2NavitConfig(**vit_config)
        if type(llm_config) is not Qwen3Config or type(vit_config) is not Siglip2NavitConfig:
            raise ValueError("Ovis2.5 nested configurations must be package-selected classes")
        self.llm_config = llm_config
        self.vit_config = vit_config
        kwargs.setdefault("tie_word_embeddings", llm_config.tie_word_embeddings)
        super().__init__(**kwargs)
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.visual_vocab_size = visual_vocab_size
        self.hidden_size = hidden_size if hidden_size is not None else llm_config.hidden_size
        if self.hidden_size != llm_config.hidden_size or visual_vocab_size <= 4:
            raise ValueError("Ovis2.5 requires matching decoder width and a nonempty visual vocabulary")
        # Both published releases share this layout; 9B's source has a larger default image budget.
        self.max_pixels = max_pixels if max_pixels is not None else (
            1792 * 1792 if llm_config.hidden_size == 4096 else 1344 * 1792)
        self.min_pixels = min_pixels
        self.vocab_size = llm_config.vocab_size

    def get_text_config(self, decoder=False, **kwargs):
        return self.llm_config
