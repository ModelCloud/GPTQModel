# Copyright (c) Alibaba Cloud.
# Modified by ModelCloud.ai for Transformers 5; see NOTICE and LICENSE.
"""Original Qwen fused-QKV decoder with cache-owned NTK state.

No checkpoint Python, optional CUDA imports, mutable module rotary tables, or
automatic dtype conversion. The caller/HF dtype owns parameter precision.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers import initialization as init
from transformers.cache_utils import DynamicCache
from transformers.generation import LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_qwen import QWenConfig


class QWenCache(DynamicCache):
    """NTK bases belong to one decoding sequence, never to the model."""

    def __init__(self):
        super().__init__()
        self.ntk_alpha = None

    def reorder_cache(self, beam_idx):
        super().reorder_cache(beam_idx)
        if self.ntk_alpha is not None:
            self.ntk_alpha = self.ntk_alpha.index_select(0, beam_idx.to(self.ntk_alpha.device))

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        if self.ntk_alpha is not None:
            self.ntk_alpha = self.ntk_alpha.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        if self.ntk_alpha is not None:
            self.ntk_alpha = self.ntk_alpha[indices.to(self.ntk_alpha.device)]

    def reset(self):
        super().reset()
        self.ntk_alpha = None


class QWenLegacyCache(tuple):
    """Source [batch, sequence, head, dim] tuple layout plus its NTK metadata."""

    def __new__(cls, layers, ntk_alpha=None):
        obj = super().__new__(cls, layers)
        obj.ntk_alpha = ntk_alpha
        return obj


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states):
        wide = hidden_states.float()
        return (wide * torch.rsqrt(wide.square().mean(-1, keepdim=True) + self.eps)).to(
            hidden_states.dtype
        ) * self.weight


def apply_rotary_pos_emb(tensor, positions, alpha, config):
    dim = int(config.kv_channels * config.rotary_pct)
    # Source rotates in FP32, including low-precision checkpoints. Recompute
    # frequencies locally: no stale meta buffers or cross-request NTK table.
    base = config.rotary_emb_base * alpha.float().to(tensor.device) ** (dim / (dim - 2))
    exponent = torch.arange(0, dim, 2, device=tensor.device, dtype=torch.float32) / dim
    angles = positions.float()[None, :, None] / base[:, None, None] ** exponent[None, None, :]
    angles = torch.cat((angles, angles), dim=-1)[:, None]
    rot = tensor[..., :dim].float()
    first, second = rot.chunk(2, dim=-1)
    rot = rot * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()
    return torch.cat((rot, tensor[..., dim:].float()), dim=-1).to(tensor.dtype)


class QWenAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)  # bias always present
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=not config.no_bias)
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

    def forward(
        self,
        hidden_states,
        positions,
        ntk_alpha,
        attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        head_mask=None,
    ):
        batch, length, _ = hidden_states.shape
        c = self.config
        q, k, v = [
            x.view(batch, length, c.num_attention_heads, c.kv_channels).transpose(1, 2)
            for x in self.c_attn(hidden_states).chunk(3, dim=-1)
        ]
        q = apply_rotary_pos_emb(q, positions, ntk_alpha, c)
        k = apply_rotary_pos_emb(k, positions, ntk_alpha, c)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        key_length = k.shape[-2]
        if c.use_logn_attn and not self.training and key_length > c.seq_length:
            gain = torch.log(positions.float() + 1) / math.log(c.seq_length)
            q = q * gain.clamp_min(1).to(q.dtype)[None, None, :, None]
        allowed = torch.arange(key_length, device=q.device)[None, :] <= positions[:, None]
        allowed = allowed[None, None].expand(batch, 1, length, key_length)
        additive_bias = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if attention_mask.shape != (batch, key_length):
                    raise ValueError("Qwen attention_mask must cover the cached prefix and new tokens")
                allowed = allowed & attention_mask[:, None, None, :].bool()
            elif attention_mask.ndim == 4:
                if attention_mask.dtype == torch.bool:
                    allowed = allowed & attention_mask
                else:
                    additive_bias = attention_mask.to(device=q.device, dtype=q.dtype)
            else:
                raise ValueError("Qwen attention_mask must have rank 2 or 4")
        weights = None
        sdpa_mask = allowed if additive_bias is None else additive_bias.masked_fill(~allowed, float("-inf"))
        if c._attn_implementation == "sdpa" and not output_attentions and head_mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=sdpa_mask, dropout_p=c.attn_dropout_prob if self.training else 0.0
            )
        else:
            # The upstream flag scale_attn_weights is stored, but QWenAttention
            # always scales, even when that config field is False.
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(c.kv_channels)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            if additive_bias is not None:
                scores = scores + additive_bias
            weights = F.softmax(scores.float() if c.softmax_in_fp32 else scores, dim=-1).to(q.dtype)
            weights = self.attn_dropout(weights)
            if head_mask is not None:
                weights = weights * head_mask
            out = weights @ v
        out = out.transpose(1, 2).reshape(batch, length, c.hidden_size)
        return self.c_proj(out), weights


class QWenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.intermediate_size // 2
        self.w1 = nn.Linear(config.hidden_size, width, bias=not config.no_bias)
        self.w2 = nn.Linear(config.hidden_size, width, bias=not config.no_bias)
        self.c_proj = nn.Linear(width, config.hidden_size, bias=not config.no_bias)

    def forward(self, hidden_states):
        return self.c_proj(self.w1(hidden_states) * F.silu(self.w2(hidden_states)))


class QWenBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.attn = QWenAttention(config, layer_idx)
        self.ln_2 = RMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = QWenMLP(config)

    def forward(self, hidden_states, use_cache=False, **kwargs):
        attn, weights = self.attn(self.ln_1(hidden_states), **kwargs)
        hidden_states = hidden_states + attn
        return hidden_states + self.mlp(self.ln_2(hidden_states)), weights


class QWenPreTrainedModel(PreTrainedModel):
    config_class = QWenConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["QWenBlock"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        # TF5 `initialization` helpers skip tensors already loaded from a checkpoint.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)


class QWenModel(QWenPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.use_cache_quantization:
            raise ValueError("Legacy uint8 Qwen cache is unqualified; use_cache_quantization=False")
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.emb_dropout_prob)
        self.h = nn.ModuleList([QWenBlock(config, idx) for idx in range(config.num_hidden_layers)])
        self.ln_f = RMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        position_ids=None,
        token_type_ids=None,
        head_mask=None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        c = self.config
        use_cache = c.use_cache if use_cache is None else use_cache
        return_dict = c.return_dict if return_dict is None else return_dict
        output_attentions = c.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = c.output_hidden_states if output_hidden_states is None else output_hidden_states
        legacy = isinstance(past_key_values, tuple)
        if legacy:
            old = past_key_values
            past_key_values = QWenCache()
            past_key_values.ntk_alpha = getattr(old, "ntk_alpha", None)
            for idx, (key, value) in enumerate(old):
                past_key_values.update(key.transpose(1, 2), value.transpose(1, 2), idx)
            if c.use_dynamic_ntk and past_key_values.ntk_alpha is None:
                raise ValueError("Dynamic NTK continuation requires Qwen cache-owned ntk_alpha metadata")
        if past_key_values is not None and not isinstance(past_key_values, QWenCache):
            raise ValueError("Original Qwen requires QWenCache, not a foreign/static/offloaded cache")
        cache = past_key_values if past_key_values is not None else (QWenCache() if use_cache else None)
        past_length = cache.get_seq_length() if cache is not None else 0
        x = self.wte(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = x.shape
        total = past_length + length
        if attention_mask is not None:
            if attention_mask.ndim == 2 and attention_mask.shape != (batch, total):
                raise ValueError("Qwen attention_mask must cover the cached prefix and new tokens")
            if attention_mask.ndim not in (2, 4):
                raise ValueError("Qwen attention_mask must have rank 2 or 4")
            if attention_mask.ndim == 4 and attention_mask.shape[-2:] != (length, total):
                raise ValueError("Qwen 4D attention_mask has incompatible query/key lengths")
        positions = torch.arange(past_length, total, device=x.device)
        # Upstream ignores position_ids/token_type_ids for rotary lookup; use
        # physical prefix offsets, including left padding, just as the source.
        if past_length:
            alpha = cache.ntk_alpha
            if alpha is None or alpha.shape[0] != batch:
                raise ValueError("Qwen cache NTK metadata does not match this batch")
        else:
            alpha = torch.ones(batch, device=x.device, dtype=torch.float32)
            if c.use_dynamic_ntk and not self.training:
                sizes = torch.full((batch,), total, device=x.device, dtype=torch.float32)
                if attention_mask is not None and attention_mask.ndim == 2 and total > c.seq_length:
                    sizes = attention_mask.sum(-1).float().to(x.device)
                alpha = (2 ** torch.ceil(torch.log2(sizes.clamp_min(1) / c.seq_length) + 1) - 1).clamp_min(1)
            if cache is not None:
                cache.ntk_alpha = alpha
        hidden, attentions = (() if output_hidden_states else None), (() if output_attentions else None)
        x = self.drop(x)
        for idx, block in enumerate(self.h):
            if output_hidden_states:
                hidden += (x,)
            mask = head_mask[idx] if head_mask is not None and head_mask.ndim > 1 else head_mask
            if mask is not None and mask.ndim == 1:
                mask = mask[None, :, None, None]
            x, weights = block(
                x,
                positions=positions,
                ntk_alpha=alpha,
                attention_mask=attention_mask,
                past_key_values=cache,
                output_attentions=output_attentions,
                head_mask=mask,
            )
            if output_attentions:
                attentions += (weights,)
        x = self.ln_f(x)
        if output_hidden_states:
            hidden += (x,)
        result_cache = cache if use_cache else None
        if legacy and result_cache is not None:
            result_cache = QWenLegacyCache(
                [(layer.keys.transpose(1, 2), layer.values.transpose(1, 2)) for layer in cache.layers], alpha
            )
        output = BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=result_cache, hidden_states=hidden, attentions=attentions
        )
        return output if return_dict else output.to_tuple()


class StopWordsLogitsProcessor(LogitsProcessor):
    """Source behavior: request EOS on the step *after* a stop suffix."""

    def __init__(self, stop_words_ids, eos_token_id):
        if not stop_words_ids or any(
            not ids or any(not isinstance(i, int) or i < 0 for i in ids) for ids in stop_words_ids
        ):
            raise ValueError("stop_words_ids must be nonempty lists of nonnegative IDs")
        self.stop_words_ids = [ids for ids in stop_words_ids if ids != [eos_token_id]]
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        for idx, row in enumerate(input_ids):
            if any(len(ids) <= len(row) and row[-len(ids) :].tolist() == ids for ids in self.stop_words_ids):
                scores[idx, self.eos_token_id] = float(2**15)
        return scores


class QWenLMHeadModel(QWenPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.transformer = QWenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False  # The forward creates QWenCache with NTK ownership.

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        logits_to_keep=0,
        return_dict=None,
        **kwargs,
    ):
        result = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
            **kwargs,
        )
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(result.last_hidden_state[:, selection, :])
        loss = None
        if labels is not None:
            full = self.lm_head(result.last_hidden_state) if logits_to_keep else logits
            loss = F.cross_entropy(
                full[:, :-1].float().reshape(-1, self.config.vocab_size), labels[:, 1:].to(full.device).reshape(-1)
            )
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=result.past_key_values,
            hidden_states=result.hidden_states,
            attentions=result.attentions,
        )
        return output if (self.config.return_dict if return_dict is None else return_dict) else output.to_tuple()

    def generate(self, *args, generation_config=None, logits_processor=None, stop_words_ids=None, **kwargs):
        gc = generation_config if generation_config is not None else self.generation_config
        stop_words_ids = stop_words_ids if stop_words_ids is not None else getattr(gc, "stop_words_ids", None)
        if stop_words_ids:
            eos = kwargs.get("eos_token_id", gc.eos_token_id)
            eos = eos[0] if isinstance(eos, list) else eos
            logits_processor = LogitsProcessorList(list(logits_processor or []))
            logits_processor.append(StopWordsLogitsProcessor(stop_words_ids, eos))
        return super().generate(
            *args, generation_config=generation_config, logits_processor=logits_processor, **kwargs
        )

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        query,
        history=None,
        system="You are a helpful assistant.",
        append_history=True,
        generation_config=None,
        **kwargs,
    ):
        from .tokenization_qwen import make_context

        gc = generation_config if generation_config is not None else self.generation_config
        if getattr(gc, "chat_format", None) != "chatml":
            raise ValueError("Original Qwen chat requires a ChatML generation_config")
        history = list(history or [])
        _, context = make_context(
            tokenizer,
            query,
            history,
            system,
            max_window_size=kwargs.pop("max_window_size", getattr(gc, "max_window_size", 6144)),
        )
        stop = list(kwargs.pop("stop_words_ids", None) or []) + [[tokenizer.im_end_id], [tokenizer.im_start_id]]
        ids = torch.tensor([context], device=self.device)
        output = self.generate(ids, generation_config=gc, stop_words_ids=stop, **kwargs)[0, len(context) :].tolist()
        response = tokenizer.decode(output, skip_special_tokens=False)
        response = response.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
        if append_history:
            history.append((query, response))
        return response, history
