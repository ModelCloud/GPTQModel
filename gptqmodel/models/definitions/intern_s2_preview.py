# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from functools import wraps
from importlib import import_module
from inspect import signature
from typing import Dict

from transformers import AutoModelForImageTextToText, AutoProcessor, ProcessorMixin

from ...utils.calibration import batched
from ...utils.model import MODALITY, move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel
from ..moe_lifecycle import GateUpDownMoELifecycleHooks


class InternS2PreviewQModel(BaseQModel):
    require_pkgs = ["sentencepiece>=0.2.0"]

    loader = AutoModelForImageTextToText

    require_load_processor = True
    require_trust_remote_code = True
    require_monkeypatch = True
    layer_modules_strict = False

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]

    dynamic_expert_index = "num_experts"
    defuser_module_paths = ("model.language_model",)
    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    pre_lm_head_norm_module = "model.language_model.norm"
    rotary_embedding = "model.language_model.rotary_emb"
    out_of_model_tensors = {"prefixes": ["mtp"]}

    awq_scale_optimize_shape_dependent_modules = ["self_attn.o_proj"]

    module_tree = [
        "model",
        "language_model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_norm:!",
                "q_proj:0",
                "k_norm:!",
                "k_proj:0",
                "v_proj:0",
                "o_proj:1",
            ),
            "linear_attn": (
                "norm:!",
                "conv1d:!",
                "in_proj_qkv:0",
                "in_proj_z:1",
                "in_proj_b:!:1",
                "in_proj_a:!:1",
                "out_proj:2",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "shared_expert_gate": ("shared_expert_gate:!",),
                "shared_expert:0": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts:0": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(
            self.model_local_path,
            trust_remote_code=True,
        )

    def monkey_patch(self):
        modeling_module = import_module(
            type(self.model.model.language_model).__module__
        )

        # Intern-S2's remote code owns a legacy stateful cache with top-level
        # conv_states/recurrent_states. Newer Transformers versions otherwise
        # pre-create their generic DynamicCache, whose state lives per layer.
        # Let the remote model create InternS2PreviewDynamicCache in forward.
        model_cls = type(self.model)
        legacy_cache_cls = getattr(
            modeling_module,
            "InternS2PreviewDynamicCache",
            None,
        )
        if legacy_cache_cls is not None and getattr(model_cls, "_is_stateful", False):
            model_cls._supports_default_dynamic_cache = classmethod(
                lambda cls: False
            )

            if not hasattr(legacy_cache_cls, "get_query_offset"):
                def get_query_offset(cache_self, layer_idx=0):
                    return cache_self.get_seq_length(layer_idx)

                legacy_cache_cls.get_query_offset = get_query_offset

            get_mask_sizes = legacy_cache_cls.get_mask_sizes
            if (
                "cache_position" in signature(get_mask_sizes).parameters
                and not getattr(get_mask_sizes, "_gptqmodel_accepts_query_length", False)
            ):
                @wraps(get_mask_sizes)
                def get_mask_sizes_compat(cache_self, query_length, layer_idx):
                    if hasattr(query_length, "shape"):
                        return get_mask_sizes(cache_self, query_length, layer_idx)

                    past_seen_tokens = cache_self.get_seq_length(layer_idx)
                    return query_length + past_seen_tokens, 0

                get_mask_sizes_compat._gptqmodel_accepts_query_length = True
                legacy_cache_cls.get_mask_sizes = get_mask_sizes_compat

        create_causal_mask = modeling_module.create_causal_mask

        if "cache_position" in signature(create_causal_mask).parameters or getattr(
            create_causal_mask, "_gptqmodel_accepts_cache_position", False
        ):
            return

        @wraps(create_causal_mask)
        def create_causal_mask_compat(*args, cache_position=None, **kwargs):
            del cache_position
            return create_causal_mask(*args, **kwargs)

        create_causal_mask_compat._gptqmodel_accepts_cache_position = True
        modeling_module.create_causal_mask = create_causal_mask_compat

    def _materialize_core_module(self, parent, attr_name: str):
        module = getattr(parent, attr_name)
        setattr(
            parent,
            attr_name,
            self.shell_module_materialize(module, self.quantize_config.device),
        )

    def pre_quantize_generate_hook_start(self):
        core_model = self.model.model
        self._materialize_core_module(core_model.language_model, "embed_tokens")
        self._materialize_core_module(core_model.language_model, "rotary_emb")
        self._materialize_core_module(core_model, "visual")

    def pre_quantize_generate_hook_end(self):
        core_model = self.model.model
        core_modules = (
            (core_model.language_model, "embed_tokens"),
            (core_model.language_model, "rotary_emb"),
            (core_model, "visual"),
        )

        if self.quantize_config.offload_to_disk:
            for parent, attr_name in core_modules:
                offload_to_disk(
                    model=parent,
                    module=getattr(parent, attr_name),
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        for parent, attr_name in core_modules:
            setattr(parent, attr_name, move_to(getattr(parent, attr_name), device=CPU))

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        del kwargs
        processor = self.load_processor()
        calibration_data = []
        for batch in batched(
            calibration_dataset,
            batch_size,
            process_func=self.preprocess_dataset,
        ):
            calibration_data.append(
                processor.apply_chat_template(
                    batch,
                    tokenize=True,
                    add_generation_prompt=True,
                    processor_kwargs={"padding": True},
                    return_dict=True,
                    return_tensors="pt",
                )
            )
        del processor
        return calibration_data


__all__ = ["InternS2PreviewQModel"]
