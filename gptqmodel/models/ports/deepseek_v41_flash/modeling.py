# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.generic import can_return_tuple

from .attention import DeepSeekV41FlashCache, V41Attention, V41AttentionState, V41DraftAttention
from .configuration import DeepSeekV41FlashConfig, attention_schedule
from .engram import V41Engram, V41EngramHash, engram_layers
from .layers import V41HyperConnection, V41MoE, V41PackedLinear, V41RMSNorm, collapse_streams, expand_streams
from .vision import V41Aligner, V41VisionTower


_CALIBRATION_MODELS = ContextVar("deepseek_v41_calibration_models", default=())


@dataclass
class V41CausalLMOutput(CausalLMOutputWithPast):
    mtp_features: torch.Tensor | None = None


class V41DecoderLayer(nn.Module):
    attention_class = V41Attention

    def __init__(self, config, dtype, layer_id, schedule, *, vision, engram_rows=None):
        super().__init__()
        self.attn_norm = V41RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.ffn_norm = V41RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.attn_hc = V41HyperConnection(config)
        self.ffn_hc = V41HyperConnection(config)
        self.attn = self.attention_class(config, dtype, layer_id, schedule)
        self.ffn = V41MoE(config, dtype, vision=vision)
        self.engram = None if engram_rows is None else V41Engram(config, engram_rows, dtype)

    def forward(self, hidden, pre_mix, positions, history, shared, image_mask):
        next_pre, post, mix = self.attn_hc(hidden)
        attn_input = self.attn_norm(collapse_streams(hidden, pre_mix))
        hidden = expand_streams(self.attn(attn_input, positions, history, shared), hidden, post, mix)

        pre_mix, post, mix = self.ffn_hc(hidden)
        ffn_input = self.ffn_norm(collapse_streams(hidden, next_pre))
        return expand_streams(self.ffn(ffn_input, image_mask), hidden, post, mix), pre_mix


class V41DraftLayer(V41DecoderLayer):
    attention_class = V41DraftAttention

    def __init__(self, config, dtype, stage, *, vision):
        super().__init__(config, dtype, config.num_hidden_layers + stage, (0, None, None), vision=vision)
        width = config.hidden_size
        if stage == 0:
            self.main_proj = V41PackedLinear(width * len(config.dspark_target_layer_ids), width)
            self.main_norm = V41RMSNorm(width, config.rms_norm_eps, dtype)
        if stage == config.num_nextn_predict_layers - 1:
            self.norm = V41RMSNorm(width, config.rms_norm_eps, dtype)
            rank, vocab_size = config.dspark_markov_rank, config.vocab_size
            self.markov_head = nn.ModuleDict({
                "embed": nn.Embedding(vocab_size, rank, dtype=dtype),
                "head": nn.Linear(rank, vocab_size, bias=False, dtype=torch.float32),
            })
            self.confidence_head = nn.ModuleDict({
                "proj": nn.Linear(width + rank, 1, bias=False, dtype=torch.float32),
            })


def _draft_config(text):
    targets = text.dspark_target_layer_ids
    depth = text.num_nextn_predict_layers
    if (
        depth < 0 or not targets or len(set(targets)) != len(targets)
        or any(not 0 <= target < text.num_hidden_layers for target in targets)
        or text.dspark_block_size < 1 or not 0 <= text.dspark_noise_token_id < text.vocab_size
        or text.dspark_markov_rank < 1 or any(text.compress_ratios[text.num_hidden_layers:text.num_hidden_layers + depth])
    ):
        raise ValueError("Invalid DeepSeek V4.1 dSpark configuration")
    config = copy(text)
    config.n_routed_experts = text.dspark_n_routed_experts or text.n_routed_experts
    config.num_experts_per_tok = text.dspark_num_experts_per_tok or text.num_experts_per_tok
    if not 0 < config.num_experts_per_tok <= config.n_routed_experts:
        raise ValueError("Invalid DeepSeek V4.1 dSpark expert counts")
    return config


class DeepSeekV41FlashForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DeepSeekV41FlashConfig
    _no_split_modules = ["V41DecoderLayer", "V41DraftLayer", "V41VisionTower", "V41Aligner"]
    _is_stateful = True

    def __init__(self, config, *, dtype=None):
        super().__init__(config)
        text = config.text_config
        if text is None:
            raise ValueError("DeepSeek V4.1 Flash requires text_config")
        dtype = dtype or config.dtype or torch.bfloat16
        if (
            text.num_key_value_heads != 1
            or text.sliding_window <= 0
            or not 0 < text.qk_rope_head_dim <= min(text.head_dim, text.index_head_dim)
            or text.qk_rope_head_dim % 2
        ):
            raise ValueError("Invalid DeepSeek V4.1 attention geometry")
        schedule = attention_schedule(text)
        tables = engram_layers(text)
        vision = config.vision_config is not None and config.vision_config.num_hidden_layers > 0
        self.embed = nn.Embedding(text.vocab_size, text.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            V41DecoderLayer(text, dtype, index, entry, vision=vision, engram_rows=tables.get(index))
            for index, entry in enumerate(schedule)
        )
        self.engram_hash = None
        self.mtp = nn.ModuleList()
        if text.num_nextn_predict_layers:
            draft = _draft_config(text)
            self.mtp.extend(
                V41DraftLayer(draft, dtype, stage, vision=vision) for stage in range(text.num_nextn_predict_layers)
            )
        self.norm = V41RMSNorm(text.hidden_size, text.rms_norm_eps, dtype)
        self.head = nn.Linear(text.hidden_size, text.vocab_size, bias=False, dtype=torch.float32)
        self.vision = V41VisionTower(config.vision_config, dtype) if vision else None
        if vision:
            self.aligner = V41Aligner(config.vision_config, text.hidden_size, dtype)
            self.image_start = nn.Parameter(torch.empty(text.hidden_size, dtype=dtype))
            self.image_end = nn.Parameter(torch.empty(text.hidden_size, dtype=dtype))
            self.image_newline = nn.Parameter(torch.empty(text.hidden_size, dtype=dtype))

    def initialize_engram(self, tokenizer):
        if self.config.text_config.engram_layer_ids:
            self.engram_hash = V41EngramHash(self.config.text_config, tokenizer).to(self.device)

    @contextmanager
    def calibration(self):
        token = _CALIBRATION_MODELS.set((*_CALIBRATION_MODELS.get(), self))
        try:
            yield
        finally:
            _CALIBRATION_MODELS.reset(token)

    @property
    def device(self):
        return self.embed.weight.device

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    @classmethod
    def _supports_default_dynamic_cache(cls):
        # Hugging Face generation lets forward create and return the architecture's cache.
        return False

    def get_image_features(self, pixel_values, image_grid_thw):
        if self.vision is None or image_grid_thw is None:
            raise ValueError("DeepSeek V4.1 images require a vision checkpoint and image_grid_thw")
        grids = image_grid_thw.tolist()
        if not grids or any(frames != 1 or height <= 0 or width <= 0 for frames, height, width in grids):
            raise ValueError("DeepSeek V4.1 image grids must have shape (images, 3) with one frame per image")
        result = []
        sizes = [height * width for _, height, width in grids]
        for patches, (_, height, width) in zip(pixel_values.split(sizes), grids):
            features = self.aligner(self.vision(patches, height, width), height, width)
            endings = self.image_newline.to(features).expand(features.size(0), 1, -1)
            rows = torch.cat((features, endings), dim=1).flatten(0, 1)
            result.append(torch.cat((self.image_start.to(rows)[None], rows, self.image_end.to(rows)[None])))
        return result

    @can_return_tuple
    @torch.no_grad()
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        image_grid_thw=None,
        logits_to_keep=0,
        engram_input_ids=None,
        output_mtp_features=False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids and inputs_embeds")
        if attention_mask is not None and (attention_mask.ndim != 2 or not attention_mask.bool().all()):
            raise ValueError("DeepSeek V4.1 Flash currently requires unpadded sequences")
        use_cache = self.config.text_config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and (not use_cache or not isinstance(past_key_values, DeepSeekV41FlashCache)):
            raise ValueError("DeepSeek V4.1 requires its own cache and use_cache=True")
        cache = past_key_values or DeepSeekV41FlashCache(len(self.layers))
        if len(cache.layers) != len(self.layers):
            raise ValueError("DeepSeek V4.1 cache depth differs from the decoder")
        start = cache.length

        hidden = self.embed(input_ids.to(self.device)) if input_ids is not None else inputs_embeds.to(self.embed.weight)
        batch, length, _ = hidden.shape
        if cache.batch_size is not None and cache.batch_size != batch:
            raise ValueError("DeepSeek V4.1 cache batch size differs from the input")
        if length == 0 or start + length > self.config.text_config.max_position_embeddings:
            raise ValueError("Invalid DeepSeek V4.1 input length")
        image_mask = None
        if pixel_values is not None:
            if input_ids is None or start:
                raise ValueError("DeepSeek V4.1 images must be supplied with input_ids on prefill")
            image_mask = input_ids.to(self.device) == self.config.image_token_id
            images = torch.cat(self.get_image_features(pixel_values, image_grid_thw), dim=0).to(hidden)
            if images.size(0) != image_mask.sum().item():
                raise ValueError("DeepSeek V4.1 image features and placeholder count differ")
            hidden = hidden.clone()
            hidden[image_mask] = images

        streams = self.config.text_config.hc_mult
        engram_context = engram_tail = None
        if self.config.text_config.engram_layer_ids:
            if self.engram_hash is None:
                raise RuntimeError("DeepSeek V4.1 Engram must be initialized with the checkpoint tokenizer")
            ids = input_ids if input_ids is not None else engram_input_ids
            if ids is None or ids.shape != (batch, length) or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("DeepSeek V4.1 Engram requires token IDs for every input embedding")
            engram_context, engram_tail = self.engram_hash.prepare(ids.to(self.device), cache.engram_tail, image_mask)

        targets = self.config.text_config.dspark_target_layer_ids if output_mtp_features and self not in _CALIBRATION_MODELS.get() else ()
        taps = {}
        state = V41AttentionState()
        pre = hidden.new_zeros((batch, length, streams), dtype=torch.float32)
        pre[..., 0] = 1
        hidden = hidden.unsqueeze(-2).expand(-1, -1, streams, -1)
        positions = torch.arange(start, start + length, device=hidden.device)
        histories = cache.stage() if past_key_values is not None else cache.layers
        for index, (layer, history) in enumerate(zip(self.layers, histories)):
            device = layer.attn_hc.fn.device
            positions = positions.to(device)
            image_mask = None if image_mask is None else image_mask.to(device)
            hidden = hidden.to(device)
            if layer.engram is not None:
                ids = self.engram_hash(engram_context, index)
                hidden = layer.engram(hidden, ids, image_mask)
            if index in targets:
                taps[index] = hidden.mean(dim=-2).to(self.device)
            hidden, pre = layer(hidden, pre.to(device), positions, history, state, image_mask)

        hidden = collapse_streams(hidden, pre)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        hidden = self.norm(hidden.to(self.norm.weight.device))
        logits = F.linear(hidden.to(self.head.weight), self.head.weight).to(self.device)
        features = torch.cat([taps[index] for index in targets], dim=-1) if targets else None
        cache.layers = histories
        cache.engram_tail = engram_tail
        cache.length = start + length
        cache.batch_size = batch
        return V41CausalLMOutput(logits=logits, past_key_values=cache if use_cache else None, mtp_features=features)

    @torch.no_grad()
    def forward_spec(self, input_ids, main_hidden, start_pos=0, *, cache, temperature=0.0):
        """Seed the draft cache or propose tokens from accepted backbone features."""
        if self in _CALIBRATION_MODELS.get():
            raise RuntimeError("DeepSeek V4.1 draft execution is disabled during quantization")
        if not self.mtp:
            raise ValueError("DeepSeek V4.1 checkpoint has no draft blocks")
        text = self.config.text_config
        batch, length, width = main_hidden.shape
        if (
            not isinstance(cache, DeepSeekV41FlashCache)
            or start_pos != cache.draft_length or start_pos + length > cache.length
            or batch != cache.batch_size or length < 1 or width != text.hidden_size * len(text.dspark_target_layer_ids)
            or input_ids.shape != (batch,) or temperature < 0
        ):
            raise ValueError("Invalid DeepSeek V4.1 draft inputs or cache position")

        first = self.mtp[0]
        kept = min(length, first.attn.window)
        main = first.main_norm(first.main_proj(main_hidden[:, -kept:].to(first.main_norm.weight)))
        windows = []
        for index, layer in enumerate(self.mtp):
            device = layer.attn_hc.fn.device
            positions = torch.arange(start_pos + length - kept, start_pos + length, device=device)
            previous = cache.draft_windows[index] if start_pos else None
            windows.append(layer.attn.accepted_kv(main.to(device), positions, previous))
        if not start_pos:
            cache.draft_windows = tuple(windows)
            cache.draft_length = length
            return None

        input_ids = input_ids.to(self.device)
        tokens = input_ids.new_full((batch, text.dspark_block_size), text.dspark_noise_token_id)
        tokens[:, 0] = input_ids
        hidden = self.embed(tokens).unsqueeze(-2).expand(-1, -1, text.hc_mult, -1)
        pre = hidden.new_zeros((*tokens.shape, text.hc_mult), dtype=torch.float32)
        pre[..., 0] = 1
        for layer, window in zip(self.mtp, windows):
            device = window.device
            positions = torch.arange(start_pos + length, start_pos + length + text.dspark_block_size, device=device)
            hidden, pre = layer(hidden.to(device), pre.to(device), positions, window, None, None)

        last = self.mtp[-1]
        hidden = collapse_streams(hidden, pre)
        logits = self.head(last.norm(hidden).to(self.head.weight)).to(hidden.device)
        start_tokens = input_ids.to(hidden.device)
        tokens = start_tokens.new_empty(batch, text.dspark_block_size + 1)
        tokens[:, 0] = start_tokens
        embeddings = []
        for position in range(text.dspark_block_size):
            markov = last.markov_head["embed"](tokens[:, position])
            embeddings.append(markov)
            scores = logits[:, position]
            scores.add_(last.markov_head["head"](markov.float()))
            tokens[:, position + 1] = (
                torch.multinomial((scores / temperature).softmax(-1), 1).squeeze(-1)
                if temperature > 0 else scores.argmax(dim=-1)
            )
        confidence_input = torch.cat((hidden, torch.stack(embeddings, dim=1)), dim=-1)
        confidence = last.confidence_head["proj"](confidence_input.float()).squeeze(-1)
        draft = tuple(value.to(self.device) for value in (tokens, logits, confidence))
        cache.draft_windows = tuple(windows)
        cache.draft_length = start_pos + length
        return draft
