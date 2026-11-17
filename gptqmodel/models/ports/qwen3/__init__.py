# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definition-scoped access to the exact native Transformers Qwen3 classes."""

from transformers import AutoModelForCausalLM

from ...local import LocalModelLoader


class Qwen3Loader:
    """Do not lend the Qwen3 binding to derived definitions such as Qwen3.5."""

    def __get__(self, instance, owner):
        if owner.__dict__.get("loader") is not self:
            return AutoModelForCausalLM
        return LocalModelLoader(
            config_class="gptqmodel.models.ports.qwen3.native.NativeQwen3Config",
            model_class="gptqmodel.models.ports.qwen3.native.Qwen3ForCausalLM",
            tokenizer_class="gptqmodel.models.ports.qwen3.native.Qwen3Tokenizer",
        )


__all__ = ["Qwen3Loader"]
