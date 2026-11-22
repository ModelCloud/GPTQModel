# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Translate released Nano-Omni metadata without checkpoint class discovery."""

from copy import deepcopy

from ..native.nemotron_h.configuration_nemotron_h import NemotronHConfig
from ..native.nemotron_h_omni.configuration_nemotron_h_omni import (
    NemotronH_Omni_Reasoning_V3_Config as NativeOmniConfig,
)
from ..native.parakeet.configuration_parakeet import ParakeetEncoderConfig
from ..native.radio.configuration_radio import RadioConfig


def _native_config(value, cls, allowed_types):
    if isinstance(value, cls):
        return value
    payload = deepcopy(value or {})
    model_type = payload.pop("model_type", None)
    if model_type is not None and model_type not in allowed_types:
        raise ValueError(f"Nemotron Omni requires {cls.__name__}, not {model_type!r}")
    payload.pop("auto_map", None)
    return cls(**payload)


def _radio_config(value, temporal_size):
    if isinstance(value, RadioConfig):
        return value
    payload = deepcopy(value or {})
    if "args" in payload:
        payload.pop("model_type", None)
        args = payload.pop("args") or {}
        if args.get("model", "vit_huge_patch16_224") != "vit_huge_patch16_224":
            raise ValueError("Unqualified legacy RADIO architecture in Nemotron Omni")
        if any(payload.get(key) for key in ("adaptor_names", "adaptor_configs", "feature_normalizer_config",
                                            "inter_feature_normalizer_config", "vitdet_window_size")):
            raise ValueError("Legacy RADIO adaptors/normalizers/window attention require explicit qualification")
        payload.update(
            hidden_size=1280, num_hidden_layers=32, num_attention_heads=16, mlp_ratio=4.0,
            max_img_size=payload.get("max_resolution", 2048),
            num_cls_tokens=sum(bool(t.get("use_summary", True)) for t in args.get("teachers", [1, 2, 3])
                               if isinstance(t, dict)) if args.get("teachers") else 3,
            num_registers=args.get("register_multiple", 10) - 3,
        )
    payload.setdefault("video_temporal_patch_size", temporal_size)
    return _native_config(payload, RadioConfig, {"radio"})


class NemotronOmniConfig(NativeOmniConfig):
    model_type = "NemotronH_Nano_Omni_Reasoning_V3"

    def __init__(self, text_config=None, llm_config=None, vision_config=None, audio_config=None,
                 sound_config=None, ps_version="v2", legacy_radio_layout=None, **kwargs):
        for main, alias, cls, allowed in ((text_config, llm_config, NemotronHConfig, {"nemotron_h"}),
                                          (audio_config, sound_config, ParakeetEncoderConfig,
                                           {"parakeet", "parakeet_encoder"})):
            if main is not None and alias is not None:
                values = [_native_config(v, cls, allowed).to_dict() for v in (main, alias)]
                for value in values:
                    for key in ("dtype", "_name_or_path", "transformers_version"):
                        value.pop(key, None)
                if values[0] != values[1]:
                    raise ValueError("Conflicting native and legacy Nemotron Omni subconfig aliases")
        temporal_size = kwargs.get("video_temporal_patch_size", 2)
        text_config = _native_config(text_config if text_config is not None else llm_config,
                                     NemotronHConfig, {"nemotron_h"})
        self.legacy_radio_layout = (isinstance(vision_config, dict) and "args" in vision_config
                                    if legacy_radio_layout is None else legacy_radio_layout)
        vision_config = _radio_config(vision_config, temporal_size)
        audio = audio_config if audio_config is not None else sound_config
        if audio is not None:
            if not isinstance(audio, ParakeetEncoderConfig):
                audio = {"attention_bias": False, "scale_input": False, **deepcopy(audio)}
            audio = _native_config(audio, ParakeetEncoderConfig, {"parakeet", "parakeet_encoder"})
        if ps_version not in {"v1", "v2"}:
            raise ValueError("Nemotron Omni ps_version must be v1 or v2")
        self.ps_version = ps_version
        for legacy, native in (("img_context_token_id", "image_token_id"),
                               ("video_context_token_id", "video_token_id"),
                               ("sound_context_token_id", "audio_token_id")):
            if legacy in kwargs:
                value = kwargs.pop(legacy)
                if native in kwargs and kwargs[native] != value:
                    raise ValueError(f"Conflicting {legacy} and {native}")
                kwargs[native] = value
        kwargs.pop("auto_map", None)
        kwargs.pop("model_type", None)
        super().__init__(text_config=text_config, vision_config=vision_config, audio_config=audio, **kwargs)

    @property
    def llm_config(self):
        return self.text_config
