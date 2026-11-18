# SPDX-FileCopyrightText: 2024 AIDC-AI
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned Ovis2.6 configuration; decoder families are package-selected."""

from transformers import PretrainedConfig

from ..native.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from ..native.qwen3_next.configuration_qwen3_next import Qwen3NextConfig


class Siglip2NavitConfig(PretrainedConfig):
    model_type = "siglip2_navit"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_channels=3,
        num_patches=-1,
        image_size=512,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        hidden_stride=2,
        window_size=112,
        fullatt_block_indexes=None,
        temporal_patch_size=1,
        preserve_original_pe=True,
        use_rope=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        for name, value in {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_channels": num_channels,
            "num_patches": num_patches,
            "image_size": image_size,
            "patch_size": patch_size,
            "hidden_act": hidden_act,
            "layer_norm_eps": layer_norm_eps,
            "attention_dropout": attention_dropout,
            "hidden_stride": hidden_stride,
            "window_size": window_size,
            "fullatt_block_indexes": fullatt_block_indexes,
            "temporal_patch_size": temporal_patch_size,
            "preserve_original_pe": preserve_original_pe,
            "use_rope": use_rope,
        }.items():
            setattr(self, name, value)
        if temporal_patch_size != 1:
            raise ValueError("Released Ovis2.6 vision uses temporal_patch_size=1")
        if hidden_size % num_attention_heads or hidden_size // num_attention_heads % 4:
            raise ValueError("Vision head width must be divisible by four for 2-D RoPE")
        if min(patch_size, hidden_stride, num_hidden_layers) <= 0 or window_size < patch_size * hidden_stride:
            raise ValueError("Invalid Ovis2.6 vision geometry")


class Ovis2_6_Moe_Config(PretrainedConfig):
    model_type = "ovis2_6_moe"
    sub_configs = {"llm_config": Qwen3MoeConfig, "vit_config": Siglip2NavitConfig}
    decoder_config_class = Qwen3MoeConfig

    def __init__(self, llm_config=None, vit_config=None, visual_vocab_size=65536, hidden_size=None, **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        if isinstance(llm_config, dict):
            declared = llm_config.get("model_type", self.decoder_config_class.model_type)
            if declared != self.decoder_config_class.model_type:
                raise ValueError(f"{self.model_type} requires {self.decoder_config_class.model_type}, not {declared}")
            llm_config = self.decoder_config_class(**llm_config)
        self.llm_config = llm_config if llm_config is not None else self.decoder_config_class()
        if type(self.llm_config) is not self.decoder_config_class:
            raise ValueError("Unqualified Ovis2.6 decoder configuration")
        if isinstance(vit_config, dict):
            if vit_config.get("model_type", "siglip2_navit") != "siglip2_navit":
                raise ValueError("Ovis2.6 requires its NaViT vision layout")
            vit_config = Siglip2NavitConfig(**vit_config)
        self.vit_config = vit_config if vit_config is not None else Siglip2NavitConfig()
        if type(self.vit_config) is not Siglip2NavitConfig:
            raise ValueError("Unqualified Ovis2.6 vision configuration")
        self.hidden_size = self.llm_config.hidden_size if hidden_size is None else hidden_size
        if self.hidden_size != self.llm_config.hidden_size or visual_vocab_size <= 4:
            raise ValueError("Invalid Ovis2.6 visual/text embedding widths")
        self.visual_vocab_size = visual_vocab_size
        # Legacy quantization consumers read the decoder geometry on the wrapper.
        for name in (
            "vocab_size",
            "num_hidden_layers",
            "num_experts",
            "num_experts_per_tok",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        ):
            setattr(self, name, getattr(self.llm_config, name))
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(self, name, None) is None:
                setattr(self, name, getattr(self.llm_config, name, None))

    def get_text_config(self, decoder=False, encoder=False):
        return getattr(self, "llm_config", self)


class Ovis2_6_Next_Config(Ovis2_6_Moe_Config):
    model_type = "ovis2_6_next"
    sub_configs = {"llm_config": Qwen3NextConfig, "vit_config": Siglip2NavitConfig}
    decoder_config_class = Qwen3NextConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
