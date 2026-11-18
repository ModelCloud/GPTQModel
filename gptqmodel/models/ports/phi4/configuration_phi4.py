# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
"""Translate the pinned Microsoft phi4mm metadata, not checkpoint Python."""

from copy import deepcopy

from transformers import Phi4MultimodalConfig


class Phi4MMConfig(Phi4MultimodalConfig):
    model_type = "phi4mm"
    base_model_tp_plan = {}
    base_model_pp_plan = {}

    def __init__(
        self,
        vision_lora=None,
        speech_lora=None,
        audio_processor=None,
        embd_layer=None,
        vision_config=None,
        audio_config=None,
        **kwargs,
    ):
        # Released phi4mm config.json files spell RoPE as ``rope_scaling``; convert it once at parse time.
        legacy_rope = kwargs.pop("rope_scaling", None)
        rope = kwargs.get("rope_parameters") or legacy_rope or {}
        self.phi4mm_rope_type = kwargs.pop("phi4mm_rope_type", rope.get("rope_type", rope.get("type", "default")))
        if self.phi4mm_rope_type not in ("default", "longrope", "su", "yarn"):
            raise ValueError("Unsupported phi4mm rotary policy")
        if rope:
            rope = deepcopy(rope)
            rope.pop("type", None)
            rope["rope_type"] = "longrope" if self.phi4mm_rope_type in ("su", "yarn") else self.phi4mm_rope_type
            kwargs["rope_parameters"] = rope
        self.vision_lora = deepcopy(vision_lora or {"r": 256, "lora_alpha": 512, "dp": 0.0})
        self.speech_lora = deepcopy(speech_lora or {"r": 320, "lora_alpha": 640, "dp": 0.01})
        self.audio_processor = deepcopy(audio_processor)
        self.embd_layer = deepcopy(embd_layer)
        for name in ("vision", "speech"):
            spec = getattr(self, name + "_lora")
            if spec["r"] <= 0 or not 0 <= spec.get("dp", 0) < 1:
                raise ValueError(f"Invalid {name} LoRA dimensions/dropout")
        if audio_config is None and audio_processor is not None:
            if audio_processor.get("name") != "cascades":
                raise ValueError("phi4mm requires its cascades audio encoder")
            src = audio_processor["config"]
            names = {
                "attention_dim": "hidden_size",
                "linear_units": "intermediate_size",
                "attention_heads": "num_attention_heads",
                "depthwise_seperable_out_channel": "depthwise_separable_out_channel",
            }
            direct = (
                "num_blocks",
                "activation",
                "chunk_size",
                "left_chunk",
                "dropout_rate",
                "ext_pw_out_channel",
                "depthwise_multiplier",
                "kernel_size",
                "conv_activation",
                "input_size",
                "conv_glu_type",
                "time_reduction",
            )
            audio_config = {names.get(k, k): v for k, v in src.items() if k in names or k in direct}
            audio_config["bias_max_distance"] = src.get("relative_attention_bias_args", {}).get(
                "t5_bias_max_distance", 500
            )
            audio_config["nemo_conv_channels"] = src.get("nemo_conv_settings", {}).get("conv_channels", 1024)
        if embd_layer is not None:
            image = embd_layer.get("image_embd_layer", {})
            if image.get("hd_transform_order", "sub_glb") != "sub_glb":
                raise ValueError("Unqualified phi4mm HD transform order")
        kwargs.setdefault("num_attention_heads", 24)
        kwargs.setdefault("partial_rotary_factor", 0.75)
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("eos_token_id", 199999)
        super().__init__(vision_config=vision_config, audio_config=audio_config, **kwargs)
