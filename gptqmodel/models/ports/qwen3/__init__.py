# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Definition-scoped access to the vendored Transformers 5.23 Qwen3 classes."""

from ...local import LocalModelLoader


class Qwen3Loader:
    """Do not lend the Qwen3 binding to derived definitions such as Qwen3.5."""

    def __get__(self, instance, owner):
        if owner.__dict__.get("loader") is not self:
            # Derived definitions keep the loader they would inherit without this binding.
            binder = next(cls for cls in owner.__mro__ if cls.__dict__.get("loader") is self)
            return super(binder, owner).loader
        return LocalModelLoader(
            config_class="gptqmodel.models.ports.qwen3.native.NativeQwen3Config",
            model_class="gptqmodel.models.ports.qwen3.native.Qwen3ForCausalLM",
            tokenizer_class="gptqmodel.models.ports.qwen3.native.Qwen3Tokenizer",
        )


__all__ = ["Qwen3Loader"]
