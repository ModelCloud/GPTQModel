# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from copy import deepcopy
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModel, AutoProcessor, ProcessorMixin
from transformers.generation.utils import GenerationMixin

from ...utils.audio import process_audio_info
from ...utils.calibration import batched
from ...utils.image import fetch_image
from ...utils.model import MODALITY, move_to, nested_move_to
from ...utils.offload import offload_to_disk
from .._const import CPU
from ..base import BaseQModel


class Cache:
    is_compileable = False

    def __init__(self):
        super().__init__()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_cache_shape(self) -> Optional[int]:
        raise NotImplementedError("Make sure to implement `get_max_cache_shape` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].numel():
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx].numel():
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    @property
    def seen_tokens(self):
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class DynamicCache(Cache):
    def __init__(self, config=None, _distributed_cache_data: Iterable = None, offloading: bool = False, **kwargs) -> None:
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.offloading = offloading

        if _distributed_cache_data is not None:
            for key_states, value_states in _distributed_cache_data:
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                not self.key_cache[layer_idx].numel()
            ):
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        is_empty_layer = (
            len(self.key_cache) == 0
            or len(self.key_cache) <= layer_idx
            or not self.key_cache[layer_idx].numel()
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        return self.get_seq_length(layer_idx) + query_length, 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    @property
    def is_sliding(self) -> List[bool]:
        return [False] * len(self.key_cache)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int):
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            if self.key_cache[idx].numel():
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["DynamicCache"]:
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["DynamicCache"]) -> "DynamicCache":
        cache = cls()
        for idx in range(len(splits[0])):
            key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx].numel()]
            value_cache = [current.value_cache[idx] for current in splits if current.value_cache[idx].numel()]
            if key_cache != []:
                layer_keys = torch.cat(key_cache, dim=0)
                layer_values = torch.cat(value_cache, dim=0)
                cache.update(layer_keys, layer_values, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]


def _patch_minicpmo_remote_prepare_inputs_for_generation(remote_module) -> None:
    remote_module.prepare_inputs_for_generation = GenerationMixin.prepare_inputs_for_generation

class MiniCPMOQModel(BaseQModel):
    loader = AutoModel

    pre_lm_head_norm_module = "llm.model.norm"

    require_trust_remote_code = True

    module_tree = [
        "llm",
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:1", "v_proj:2", "o_proj:3"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj", "up_proj", "down_proj"),
        }
    ]

    modality = [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]
    require_load_processor = True

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        from transformers import cache_utils
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from transformers.generation import utils
        cache_utils.Cache = Cache
        cache_utils.DynamicCache = DynamicCache
        utils.DynamicCache = DynamicCache

        tts_config_cls = get_class_from_dynamic_module(
            "configuration_minicpmo.MiniCPMTTSConfig",
            model_local_path,
        )

        print("tts_config_cls", tts_config_cls)

        if not hasattr(tts_config_cls, "top_p"):
            tts_config_cls.top_p = 1.0

        if not hasattr(tts_config_cls, "top_k"):
            tts_config_cls.top_k = 50

        if not hasattr(tts_config_cls, "repetition_penalty"):
            tts_config_cls.repetition_penalty = 1.0

        # MiniCPM-o remote code binds `DynamicCache` into module globals at import
        # time (`from transformers.cache_utils import DynamicCache`). Rebind those
        # globals as well so the compat cache is used even if the dynamic modules
        # were imported before this hook ran.
        remote_model_cls = get_class_from_dynamic_module(
            "modeling_minicpmo.MiniCPMO",
            model_local_path,
        )
        remote_module = import_module(remote_model_cls.__module__)
        remote_module.Cache = Cache
        remote_module.DynamicCache = DynamicCache
        _patch_minicpmo_remote_prepare_inputs_for_generation(remote_module)

    def pre_quantize_generate_hook_start(self):
        self.shell_module_materialize(self.model.llm.model.embed_tokens, self.quantize_config.device)
        self.shell_module_materialize(self.model.llm.model.rotary_emb, self.quantize_config.device)
        self.shell_module_materialize(self.model.vpm, self.quantize_config.device)
        self.shell_module_materialize(self.model.resampler, self.quantize_config.device)
        self.shell_module_materialize(self.model.apm, self.quantize_config.device)
        self.shell_module_materialize(self.model.audio_avg_pooler, self.quantize_config.device)
        self.shell_module_materialize(self.model.audio_projection_layer, self.quantize_config.device)

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            for module in (
                self.model.llm.model.embed_tokens,
                self.model.llm.model.rotary_emb,
            ):
                offload_to_disk(
                    model=self.model.llm.model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )

            for module in (
                self.model.vpm,
                self.model.resampler,
                self.model.apm,
                self.model.audio_avg_pooler,
                self.model.audio_projection_layer,
            ):
                offload_to_disk(
                    model=self.model,
                    module=module,
                    disk_path=self.quantize_config.offload_to_disk_path,
                )
            return

        self.model.llm.model.embed_tokens = move_to(self.model.llm.model.embed_tokens, device=CPU)
        self.model.llm.model.rotary_emb = move_to(self.model.llm.model.rotary_emb, device=CPU)
        self.model.vpm = move_to(self.model.vpm, device=CPU)
        self.model.resampler = move_to(self.model.resampler, device=CPU)
        self.model.apm = move_to(self.model.apm, device=CPU)
        self.model.audio_avg_pooler = move_to(self.model.audio_avg_pooler, device=CPU)
        self.model.audio_projection_layer = move_to(self.model.audio_projection_layer, device=CPU)
        if hasattr(self.model, "tts"):
            self.model.tts = move_to(self.model.tts, device=CPU)

    def preprocess_dataset(self, sample: Dict) -> Dict:
        return sample

    def load_processor(self) -> ProcessorMixin:
        return AutoProcessor.from_pretrained(self.model_local_path, trust_remote_code=True)

    @staticmethod
    def _normalize_conversation(
        conversation: list[dict],
    ) -> tuple[list[dict], list, list[int]]:
        normalized = []
        images = []
        audio_parts = []

        for index, message in enumerate(deepcopy(conversation)):
            content = message.get("content", "")
            if isinstance(content, str):
                normalized.append(message)
                continue

            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue

                item_type = item.get("type")
                if item_type == "image":
                    images.append(fetch_image(item))
                    text_parts.append("<image>./</image>")
                elif item_type == "audio":
                    audio_parts.append(index)
                    text_parts.append("<audio>./</audio>")
                elif item_type == "text":
                    text_parts.append(item.get("text", ""))
                else:
                    raise ValueError(f"Unsupported MiniCPM-o content type: {item_type}")

            message["content"] = "\n".join(part for part in text_parts if part)
            normalized.append(message)

        return normalized, images, audio_parts

    @classmethod
    def prepare_inputs_for_conversations(
        cls,
        processor: ProcessorMixin,
        conversations: list[dict] | list[list[dict]],
    ):
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]

        prompts = []
        images = []
        audios = []
        audio_parts = []

        for conversation in conversations:
            normalized, image_inputs, audio_part_inputs = cls._normalize_conversation(conversation)
            audio_inputs = process_audio_info(conversation, use_audio_in_video=False) or []

            prompts.append(
                processor.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                )
            )
            images.append(image_inputs)
            audios.append(audio_inputs)
            audio_parts.append(audio_part_inputs)

        inputs = processor(
            prompts,
            images,
            audios,
            audio_parts,
            return_tensors="pt",
        )
        inputs.pop("image_sizes")
        return inputs

    def prepare_dataset(self, calibration_dataset, batch_size: int = 1, **kwargs):
        processor = self.processor or self.load_processor()
        calib_data = []
        for batch in batched(calibration_dataset, batch_size, process_func=self.preprocess_dataset):
            calib_data.append(
                self.prepare_inputs_for_conversations(
                    processor,
                    batch,
                )
            )
        return calib_data

    def move_input_capture_example(self, example, data_device):
        for key, value in example.items():
            example[key] = nested_move_to(value, device=data_device)

        return self.finalize_input_capture_example(example)

    def run_input_capture(self, example, use_cache: bool, data_device):
        generation_config = self.model.prepare_generation_config(do_sample=True)
        generation_config["use_cache"] = use_cache

        return self.model.generate(
            **example,
            tokenizer=self.model.tokenizer,
            **generation_config,
        )
