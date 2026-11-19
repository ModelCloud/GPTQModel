# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Clean-room implementation from cleanroom/locateanything/SPEC.md; see NOTICE.
"""LocateAnything vision-language model and its localization decoding loop (SPEC sections 1, 4, 5, 8)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import can_return_tuple

from .configuration_locateanything import LocateAnythingConfig, normalize_attn_implementation
from .generate_utils import build_token_ids, handle_pattern, select_multi_position
from .modeling_decoder import LocateAnythingQwen2ForCausalLM, LocateAnythingQwen3ForCausalLM
from .modeling_vit import MoonViTPositionTable, MoonViTPreTrainedModel


_GENERATION_MODES = ("fast", "slow", "hybrid")


def _decoder_architecture(config: LocateAnythingConfig) -> str:
    architectures = getattr(config.text_config, "architectures", None) or ["Qwen2ForCausalLM"]
    return architectures[0]


def _wrap_lora(module: nn.Module, rank: int, target_modules: list[str], task_type: str | None = None):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=task_type,
    )
    return get_peft_model(module, lora_config)


class LocateAnythingPreTrainedModel(PreTrainedModel):
    config_class = LocateAnythingConfig
    config: LocateAnythingConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = False
    _skip_keys_device_placement = "past_key_values"

    def _check_and_adjust_attn_implementation(self, attn_implementation, is_init_check=False, allow_all_kernels=False):
        attn_implementation = normalize_attn_implementation(attn_implementation)
        return super()._check_and_adjust_attn_implementation(
            attn_implementation, is_init_check=is_init_check, allow_all_kernels=allow_all_kernels
        )

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", None)
        if std is None:
            std = self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None and not getattr(module.weight, "_is_hf_initialized", False):
                init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, MoonViTPositionTable):
            init.normal_(module.weight, mean=0.0, std=std)
        else:
            super()._init_weights(module)


class LocateAnythingForConditionalGeneration(LocateAnythingPreTrainedModel):
    _tied_weights_keys = {"language_model.lm_head.weight": "language_model.model.embed_tokens.weight"}

    def __init__(self, config: LocateAnythingConfig):
        config._attn_implementation = normalize_attn_implementation(config._attn_implementation)
        super().__init__(config)
        architecture = _decoder_architecture(config)
        self._no_split_modules = ["Qwen3DecoderLayer"] if "Qwen3" in architecture else ["Qwen2DecoderLayer"]
        self.template = config.template
        self.mlp_checkpoint = config.mlp_checkpoint
        self.image_token_index = config.image_token_index

        vision_width = config.vision_config.hidden_size
        text_width = config.text_config.hidden_size
        self.vision_model = MoonViTPreTrainedModel(config.vision_config)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(4 * vision_width, eps=1e-5),
            nn.Linear(4 * vision_width, text_width),
            nn.GELU(),
            nn.Linear(text_width, text_width),
        )
        if "Qwen3" in architecture:
            self.language_model = LocateAnythingQwen3ForCausalLM(config.text_config)
        else:
            self.language_model = LocateAnythingQwen2ForCausalLM(config.text_config)

        if config.use_backbone_lora:
            self.vision_model = _wrap_lora(
                self.vision_model, config.use_backbone_lora, ["wqkv", "wo", "fc0", "fc1"]
            )
        if config.use_llm_lora:
            self.language_model = _wrap_lora(
                self.language_model,
                config.use_llm_lora,
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )

        self.token_ids = build_token_ids(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def extract_feature(self, pixel_values: torch.Tensor, image_grid_hws) -> list[torch.Tensor]:
        return self.vision_model(pixel_values, torch.as_tensor(image_grid_hws))

    @staticmethod
    def _filter_by_flags(features: list[torch.Tensor], image_flags) -> list[torch.Tensor]:
        if image_flags is None:
            return features
        flags = [int(flag) for flag in torch.as_tensor(image_flags).flatten().tolist()]
        if sum(flags) <= 0:
            return features
        kept, cursor = [], 0
        for flag in flags:
            if flag > 0:
                kept.extend(features[cursor : cursor + flag])
                cursor += flag
            else:
                cursor += 1
        return kept

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        image_flags: torch.Tensor | None = None,
        past_key_values=None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        visual_features = None
        if pixel_values is not None:
            features = self._filter_by_flags(self.extract_feature(pixel_values, image_grid_hws), image_flags)
            if features and input_ids is not None and bool((input_ids == self.image_token_index).any()):
                visual_features = self.mlp1(torch.cat(features, dim=0))

        outputs = self.language_model(
            input_ids=input_ids,
            visual_features=visual_features,
            image_token_index=self.image_token_index,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            vocab_size = self.config.text_config.vocab_size
            shift_logits = logits[:, :-1].reshape(-1, vocab_size)
            shift_labels = labels[:, 1:].reshape(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        visual_features=None,
        image_grid_hws=None,
        tokenizer=None,
        n_future_tokens=6,
        **kwargs,
    ):
        """Localization decoding: speculative block (MTP) rounds with an autoregressive fallback."""
        del attention_mask
        if input_ids is None:
            input_ids = kwargs.pop("inputs", None)
        if input_ids is None:
            raise ValueError("input_ids is required")
        block_size = getattr(self.config.text_config, "block_size", 6)
        if n_future_tokens != block_size:
            raise ValueError(f"n_future_tokens ({n_future_tokens}) must equal the block size ({block_size})")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=self.vision_model.device, dtype=self.vision_model.dtype)
        if tokenizer is None:
            raise ValueError("A tokenizer is required for LocateAnything localization generate")
        if isinstance(image_grid_hws, np.ndarray):
            image_grid_hws = torch.from_numpy(image_grid_hws).to(torch.int32)
        assert input_ids.shape[0] == 1, "LocateAnything localization generate supports batch size 1"
        assert kwargs.get("use_cache"), "LocateAnything localization generate requires use_cache=True"
        mode = kwargs.get("generation_mode", "hybrid")
        assert mode in _GENERATION_MODES, f"generation_mode must be one of {_GENERATION_MODES}"
        verbose = bool(kwargs.get("verbose", False))

        token_ids = self.token_ids
        mask_token_id = token_ids["default_mask_token_id"]
        prompt_len = input_ids.shape[1]
        max_len = min(tokenizer.model_max_length, prompt_len + kwargs.get("max_new_tokens", 2048))

        if visual_features is None and pixel_values is not None and image_grid_hws is not None:
            features = self.extract_feature(pixel_values, image_grid_hws)
            visual_features = self.mlp1(torch.cat(features, dim=0)) if features else None

        language_model = self.language_model
        device = input_ids.device
        generated = input_ids
        cache = None
        use_mtp = mode != "slow"
        first_call = True
        history = []
        switches = 0
        mtp_rounds = 0
        ar_rounds = 0

        while generated.shape[1] < max_len:
            length = generated.shape[1]
            if use_mtp:
                window = torch.full((1, block_size - 1), mask_token_id, dtype=generated.dtype, device=device)
                sequence = torch.cat([generated, generated[:, -1:], window], dim=1)
                positions = torch.arange(sequence.shape[1], device=device)
                positions[-block_size:] -= 1
            else:
                sequence = generated
                positions = torch.arange(length, device=device)
            model_inputs = language_model.prepare_inputs_for_generation(
                sequence,
                past_key_values=cache,
                attention_mask=None,
                position_ids=positions.unsqueeze(0),
                use_cache=True,
            )
            if first_call:
                model_inputs["visual_features"] = visual_features
                model_inputs["image_token_index"] = self.image_token_index
                first_call = False
            outputs = language_model(**model_inputs)
            cache = outputs.past_key_values
            excess = cache.get_seq_length() - length
            if excess > 0:
                cache.crop(-excess)

            if use_mtp:
                mtp_rounds += 1
                logits = outputs.logits[:, -block_size:]
                _probs, _confidence, chosen, box = select_multi_position(
                    logits, generated, token_ids, **kwargs
                )
                box_row = box[0]
                candidate = chosen[0] if bool((box_row == 0).all()) else box_row
                result = handle_pattern(candidate.tolist(), token_ids, generation_mode=mode)
                kind = result["type"]
                tokens = list(result["tokens"])
            else:
                ar_rounds += 1
                logits = outputs.logits[:, -1:]
                _probs, _confidence, chosen, _box = select_multi_position(
                    logits, generated, token_ids, **kwargs
                )
                token = int(chosen.reshape(-1)[-1])
                if mode == "hybrid":
                    if token == token_ids["box_end_token_id"]:
                        kind = "box_end_ar"
                    elif (
                        token_ids["coord_start_token_id"] <= token <= token_ids["coord_end_token_id"]
                        or token == token_ids["none_token_id"]
                    ):
                        kind = "coord_ar"
                    else:
                        kind = "im_end"
                else:
                    kind = "im_end" if token == token_ids["im_end_token_id"] else "continue_ar"
                tokens = [token]

            appended = torch.tensor([tokens], dtype=generated.dtype, device=device)
            generated = torch.cat([generated, appended], dim=1)
            if verbose:
                history.append(("ar" if "ar" in kind else "mtp", tokenizer.decode(tokens)))
            if kind == "im_end":
                break
            if mode == "hybrid":
                if kind == "error_box":
                    use_mtp = False
                    switches += 1
                elif kind == "box_end_ar":
                    use_mtp = True

        response = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=False)[0]
        if verbose:
            new_tokens = generated.shape[1] - prompt_len
            info = (
                f"mode={mode} new_tokens={new_tokens} mtp_rounds={mtp_rounds} "
                f"ar_rounds={ar_rounds} switches={switches}"
            )
            print(info)
            return response, history, info
        return response


__all__ = ["LocateAnythingForConditionalGeneration", "LocateAnythingPreTrainedModel"]
