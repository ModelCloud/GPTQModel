# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from os import fspath

from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from ...utils.backend import BACKEND, PROFILE, normalize_backend
from ...utils.hf import get_config_selection_kwargs, load_hf_tokenizer, normalize_torch_dtype_kwarg
from ...utils.model import MODALITY
from ..base import BaseQModel
from ..deepseek_v41.configuration import DeepSeekV41FlashConfig
from ..fusable import require_fusable_config


# GPT-QModel can dispatch the official config on Transformers versions that do
# not recognize the architecture. The weight loader always uses our own class.
if "deepseek_v41" not in CONFIG_MAPPING:
    AutoConfig.register("deepseek_v41", DeepSeekV41FlashConfig)


_INFERENCE_ONLY = "DeepSeek V4.1 Flash support is inference-only. Load the published FP8 / FP4 checkpoint."


class DeepSeekV41FlashQModel(BaseQModel):
    require_trust_remote_code = False
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    layer_modules_strict = False  # Compressor and indexer owners vary across layers.
    lm_head = "head"
    pre_lm_head_norm_module = "norm"
    dynamic_expert_index = "n_routed_experts"

    module_tree = ["layers", "#", {
        "attn_hc:!": (),
        "attn_norm:!": (),
        "attn:!": {
            "": ("wq_a:!", "q_norm:!", "wq_b:!", "wkv:!", "kv_norm:!", "wo_a:!", "wo_b:!"),
            "compressor": ("wkv:!", "wgate:!", "norm:!"),
            "indexer": ("wq_b:!", "weights_proj:!", "wk:!", "k_norm:!"),
        },
        "ffn_hc:!": (),
        "ffn_norm:!": (),
        "ffn:moe": {
            "gate:!": (),
            "experts": {"#": ("w1:!", "w3:!", "w2:!")},
            "shared_experts": ("w1:!", "w3:!", "w2:!"),
        },
    }]

    @classmethod
    def from_pretrained(cls, pretrained_model_id_or_path, quantize_config=None, profile=PROFILE.AUTO, **kwargs):
        if quantize_config is not None:
            raise NotImplementedError(_INFERENCE_ONLY)
        if profile != PROFILE.AUTO:
            raise ValueError("DeepSeek V4.1 Flash supports the default load profile")
        return cls.from_quantized(pretrained_model_id_or_path, **kwargs)

    @classmethod
    def from_quantized(
        cls,
        model_id_or_path,
        device_map=None,
        device=None,
        backend=BACKEND.AUTO,
        adapter=None,
        trust_remote_code=False,
        fuse=False,
        module_tree=None,
        kernel=None,
        p2p=None,
        runtime=None,
        max_memory=None,
        **kwargs,
    ):
        if adapter is not None or any(kwargs.pop(name, None) for name in (
            "allow_remote_adapter", "adapter_revision", "adapter_sha256",
        )):
            raise ValueError("DeepSeek V4.1 Flash does not support adapters")
        if fuse:
            require_fusable_config(cls, "deepseek_v41")
        if module_tree is not None or normalize_backend(backend) not in (BACKEND.AUTO, BACKEND.TORCH):
            raise ValueError("DeepSeek V4.1 Flash uses its built-in architecture and Torch backend")

        from ..deepseek_v41.checkpoint import load_checkpoint

        model_id_or_path = fspath(model_id_or_path)
        tokenizer_trust = kwargs.pop("tokenizer_trust_remote_code", trust_remote_code)
        normalize_torch_dtype_kwarg(kwargs, api_name="DeepSeekV41FlashQModel.from_quantized")
        dtype = kwargs.pop("dtype", "auto")
        selection = get_config_selection_kwargs(kwargs)
        unsupported = set(kwargs) - selection.keys()
        if unsupported:
            raise TypeError(f"Unsupported DeepSeek V4.1 Flash load arguments: {', '.join(sorted(unsupported))}")
        model, local_path, selection = load_checkpoint(
            model_id_or_path, selection=selection, dtype=dtype, device=device, device_map=device_map, max_memory=max_memory,
        )
        tokenizer = load_hf_tokenizer(
            model_id_or_path, model_config=model.config, trust_remote_code=tokenizer_trust, **selection,
        )
        return cls(
            model, quantized=False, quantize_config=None, tokenizer=tokenizer, model_local_path=local_path,
        ).eval()

    def quantize(self, *args, **kwargs):
        raise NotImplementedError(_INFERENCE_ONLY)
