# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Straightforward monkey patch helpers for nogil runtimes."""

from .safe import ThreadSafe

_PATCHED_ATTR = "_gptqmodel_locked_save_file"


def patch_safetensors_save_file() -> None:
    from safetensors import torch as safetensors_torch

    if getattr(safetensors_torch.save_file, _PATCHED_ATTR, False):
        return

    wrapper = ThreadSafe(safetensors_torch).save_file
    setattr(wrapper, _PATCHED_ATTR, True)
    safetensors_torch.save_file = wrapper


__all__ = ["patch_safetensors_save_file"]
