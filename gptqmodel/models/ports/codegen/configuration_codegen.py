# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CodeGen layout metadata; remote import hints never select an implementation."""

from transformers.models.codegen.configuration_codegen import CodeGenConfig as NativeCodeGenConfig


_MISSING = object()


class CodeGenConfig(NativeCodeGenConfig):
    """Keep the Q,V,K row partitioning explicit across checkpoint round trips.

    Official CodeGen2 1B/3.7B configs carry ``head_dim`` and use TPU-v3's
    eight partitions. CodeGen1 and CodeGen2 7B/16B omit it and use four.
    ``qkv_layout`` overrides this legacy metadata for explicitly migrated
    checkpoints; no model name, auto_map or trust flag participates.
    """

    def __init__(self, qkv_layout=None, head_dim=_MISSING, **kwargs):
        full_rotary = "rotary_dim" in kwargs and kwargs["rotary_dim"] is None
        if full_rotary:
            kwargs.pop("rotary_dim")
        super().__init__(**kwargs)
        if qkv_layout is None:
            if head_dim is None:
                raise ValueError("CodeGen legacy head_dim is null; specify explicit qkv_layout migration metadata")
            qkv_layout = "qvk_8" if head_dim is not _MISSING else "qvk_4"
        self.qkv_layout = qkv_layout
        if head_dim is not _MISSING and head_dim is not None:
            self.head_dim = head_dim
        self.validate_layout(full_rotary=full_rotary)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # HF normally applies API overrides *after* constructing the config.
        # Merge structural overrides first so layout selection/validation sees
        # the same geometry that the model will consume, without mutating data.
        config_dict = dict(config_dict)
        for key in ("qkv_layout", "head_dim", "n_embd", "n_head", "hidden_size", "num_attention_heads", "rotary_dim"):
            if key in kwargs:
                config_dict[key] = kwargs.pop(key)
        return super().from_dict(config_dict, **kwargs)

    def validate_layout(self, *, full_rotary=False):
        qkv_layout = self.qkv_layout
        if qkv_layout not in ("qvk_4", "qvk_8"):
            raise ValueError("CodeGen qkv_layout must be 'qvk_4' or 'qvk_8'; migrate ambiguous checkpoints explicitly")
        partitions = self.qkv_partitions
        if (type(self.n_head) is not int or type(self.n_embd) is not int or self.n_embd <= 0
                or self.n_head <= 0 or self.n_embd % self.n_head or self.n_head % partitions):
            raise ValueError(f"CodeGen head geometry must divide hidden size and {partitions} QKV partitions")
        expected_head_dim = self.n_embd // self.n_head
        head_dim = getattr(self, "head_dim", None)
        if head_dim is not None:
            if type(head_dim) is not int or head_dim != expected_head_dim:
                raise ValueError("CodeGen legacy head_dim conflicts with n_embd / n_head; specify valid migration metadata")
        # Full-head RoPE has no separate checkpoint tensor. Canonicalizing None
        # also avoids the native model constructor's min(None, ...) failure.
        if full_rotary:
            self.rotary_dim = expected_head_dim
        if (type(self.rotary_dim) is not int or self.rotary_dim <= 0
                or self.rotary_dim % 2 or self.rotary_dim > expected_head_dim):
            raise ValueError("CodeGen rotary_dim must be positive, even and no larger than the attention head")

    @property
    def qkv_partitions(self):
        return 8 if self.qkv_layout == "qvk_8" else 4
