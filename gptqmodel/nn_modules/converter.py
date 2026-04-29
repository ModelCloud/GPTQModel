# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium


def _resolve_text_decoder_config(config):
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return text_config

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        resolved = get_text_config()
        if resolved is not None:
            return resolved

    return config


MODULE_CONVERTER_MAP = {
    # llama4/gpt_oss are handled by Defuser>=0.0.10 during model load.
    # qwen2_moe/qwen3_moe/qwen3_next/qwen3_omni_moe are handled by Defuser>=0.0.10 during model load.
}
