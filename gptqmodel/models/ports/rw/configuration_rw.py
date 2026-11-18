# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Historical RW config semantics: TII, BigScience and Hugging Face (Apache-2.0).
"""Translate the two historical RW layouts to native Falcon, without auto_map."""

from ..native.falcon.configuration_falcon import FalconConfig


class RWConfig(FalconConfig):
    """Canonical saves use ``falcon`` and need no legacy dispatch on reload.

    RefinedWebModel is the RW-1B/7B MHA-or-MQA decoder. RefinedWeb is the
    40B grouped-QKV decoder with separate attention and MLP layer norms.
    Their original defaults differ from FalconConfig's defaults.
    """

    def __init__(self, **kwargs):
        model_type = str(kwargs.pop("model_type", "RefinedWebModel")).lower()
        if model_type not in {"refinedweb", "refinedwebmodel", "falcon"}:
            raise ValueError(f"Unqualified RW model_type: {model_type!r}")
        for old, new in (
            ("n_layer", "num_hidden_layers"),
            ("n_head", "num_attention_heads"),
            ("n_head_kv", "num_kv_heads"),
            ("n_embed", "hidden_size"),
        ):
            if old in kwargs:
                value = kwargs.pop(old)
                if new in kwargs and kwargs[new] != value:
                    raise ValueError(f"Conflicting RW config fields: {old} and {new}")
                kwargs[new] = value

        if model_type != "falcon":
            grouped = model_type == "refinedweb"
            for name, expected in (
                ("new_decoder_architecture", grouped),
                ("num_ln_in_parallel_attn", 2 if grouped else 1),
            ):
                if kwargs.get(name) is not None and kwargs[name] != expected:
                    raise ValueError(f"{model_type} requires {name}={expected!r}")
                kwargs[name] = expected
            kwargs.setdefault("vocab_size", 250880)
            kwargs.setdefault("hidden_size", 64)
            kwargs.setdefault("num_hidden_layers", 2)
            kwargs.setdefault("num_attention_heads", 8)
            if kwargs.get("ffn_hidden_size") not in (None, 4 * kwargs["hidden_size"]):
                raise ValueError("Historical RW requires ffn_hidden_size=4*hidden_size")
            kwargs.setdefault("multi_query", False)
            # The RefinedWeb (40B) source always runs both norms in parallel,
            # regardless of the legacy parallel_attn metadata.
            if grouped:
                kwargs["parallel_attn"] = True
            else:
                kwargs.setdefault("parallel_attn", False)
            kwargs.setdefault("bos_token_id", 1)
            kwargs.setdefault("eos_token_id", 2)

        heads = kwargs.get("num_attention_heads", 71)
        kv_heads = kwargs.get("num_kv_heads")
        if not isinstance(heads, int) or heads <= 0:
            raise ValueError("num_attention_heads must be a positive integer")
        if kv_heads is not None and (
            not isinstance(kv_heads, int) or kv_heads <= 0 or heads % kv_heads
        ):
            raise ValueError("num_kv_heads must be a positive divisor of num_attention_heads")
        legacy_kv_heads = {heads, 1} if kwargs.get("multi_query") else {heads}
        if model_type == "refinedwebmodel" and kv_heads is not None and kv_heads not in legacy_kv_heads:
            raise ValueError("RefinedWebModel has MHA/MQA, not grouped n_head_kv")
        kwargs["architectures"] = ["FalconForCausalLM"]
        # These are data, but dropping them also makes canonical saves independent
        # of checkpoint Python for consumers outside GPT-QModel.
        kwargs.pop("auto_map", None)
        super().__init__(**kwargs)


__all__ = ["RWConfig"]
