# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from os import fspath

from ...quantization.config import METHOD, QuantizeConfig
from ...utils.backend import BACKEND, PROFILE, normalize_backend
from ...utils.hf import get_config_selection_kwargs, load_hf_tokenizer, normalize_torch_dtype_kwarg
from ...utils.model import MODALITY
from .._const import DEFAULT_MAX_SHARD_SIZE
from ..base import BaseQModel
from ..fusable import require_fusable_config
from ..local import LocalModelLoader


class DeepSeekV41FlashQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.deepseek_v41_flash.configuration.DeepSeekV41FlashConfig",
        model_class="gptqmodel.models.ports.deepseek_v41_flash.modeling.DeepSeekV41FlashForCausalLM",
    )
    require_trust_remote_code = False
    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    layer_modules_strict = False  # Compressor and indexer owners vary across layers.
    lm_head = "head"
    pre_lm_head_norm_module = "norm"
    dynamic_expert_index = "n_routed_experts"

    module_tree = ["layers", "#", {
        "engram:!": {"": ("embed:!", "wkv:!")},
        "attn_hc:!": (),
        "attn_norm:!": (),
        "attn:!": {
            "": ("wq_a", "q_norm:!", "wq_b", "wkv", "kv_norm:!", "wo_a", "wo_b"),
            "compressor": ("wkv", "wgate", "norm:!"),
            "indexer": ("wq_b", "weights_proj", "wk", "k_norm:!"),
        },
        "ffn_hc:!": (),
        "ffn_norm:!": (),
        "ffn:moe": {
            "gate:!": (),
            "experts": {"#": ("w1", "w3", "w2")},
            "shared_experts": ("w1", "w3", "w2"),
        },
    }]

    @classmethod
    def from_pretrained(cls, pretrained_model_id_or_path, quantize_config=None, profile=PROFILE.AUTO, **kwargs):
        if quantize_config is not None and quantize_config.method != METHOD.GPTQ:
            raise ValueError("DeepSeek V4.1 projection quantization uses GPTQ")
        if profile != PROFILE.AUTO:
            raise ValueError("DeepSeek V4.1 Flash supports the default load profile")
        loaded = cls.from_quantized(pretrained_model_id_or_path, **kwargs)
        if quantize_config is not None:
            if loaded.quantized:
                raise ValueError("DeepSeek V4.1 GPTQ requires a published source checkpoint")
            loaded.quantize_config = quantize_config
        return loaded

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
        backend = normalize_backend(backend)
        if module_tree is not None or backend not in (BACKEND.AUTO, BACKEND.MARLIN, BACKEND.TORCH):
            raise ValueError("DeepSeek V4.1 Flash uses its built-in module tree and the auto, marlin, or torch backend")

        from ..ports.deepseek_v41_flash.checkpoint import load_checkpoint

        model_id_or_path = fspath(model_id_or_path)
        tokenizer_trust = kwargs.pop("tokenizer_trust_remote_code", trust_remote_code)
        kwargs.pop("code_revision", None)
        normalize_torch_dtype_kwarg(kwargs, api_name="DeepSeekV41FlashQModel.from_quantized")
        dtype = kwargs.pop("dtype", "auto")
        selection = get_config_selection_kwargs(kwargs)
        unsupported = set(kwargs) - selection.keys()
        if unsupported:
            raise TypeError(f"Unsupported DeepSeek V4.1 Flash load arguments: {', '.join(sorted(unsupported))}")
        model, local_path, selection = load_checkpoint(
            model_id_or_path, selection=selection, dtype=dtype, device=device, device_map=device_map, max_memory=max_memory,
            backend=backend, loader=cls.loader,
        )
        tokenizer = load_hf_tokenizer(
            model_id_or_path, model_config=model.config, trust_remote_code=tokenizer_trust, **selection,
        )
        model.initialize_engram(tokenizer)
        quantized = bool(model.config.gptqmodel_v41_projections)
        quantize_config = QuantizeConfig.from_quant_config(model.config.quantization_config) if quantized else None
        return cls(
            model, quantized=quantized, quantize_config=quantize_config, tokenizer=tokenizer, model_local_path=local_path,
        ).eval()

    def quantize(self, calibration=None, *, projections=None):
        from ..ports.deepseek_v41_flash.gptq import quantize_projections

        with self.model.calibration():
            return quantize_projections(self, calibration, projections)

    def save(
        self, save_dir, safetensors_metadata=None, max_shard_size=DEFAULT_MAX_SHARD_SIZE,
        meta_quantizer=None, eora_path=None, split_by=None,
    ):
        from ..ports.deepseek_v41_flash.writer import save_checkpoint

        if eora_path is not None or split_by is not None:
            raise ValueError("DeepSeek V4.1 checkpoint save uses a single indexed model with all its components")
        return save_checkpoint(
            self, save_dir, safetensors_metadata=safetensors_metadata,
            max_shard_size=max_shard_size, meta_quantizer=meta_quantizer,
        )

    def save_quantized(self, save_dir, **kwargs):
        if not self.quantized:
            raise ValueError("Call quantize() before saving a DeepSeek V4.1 GPTQ checkpoint")
        return self.save(save_dir, **kwargs)

    save_pretrained = save
