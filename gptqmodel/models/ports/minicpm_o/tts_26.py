# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The DVAE codec, repetition penalty, gen_logits, ConditionalChatTTS.generate and
# ConditionalChatTTS.decode_to_mel_specs are ModelCloud.ai clean-room implementations
# (Apache-2.0) written from a behavioral specification; see NOTICE.

import logging
import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from transformers import LogitsProcessor, PreTrainedModel, TopKLogitsWarper, TopPLogitsWarper
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

from ..native.llama.configuration_llama import LlamaConfig
from .cache import cache_from_pairs, cache_pairs
from .native import LlamaModel


logger = logging.getLogger(__name__)

from transformers import PretrainedConfig


class ConditionalChatTTSConfig(PretrainedConfig):
    model_type = "conditional_chattts"

    def __init__(
        self,
        llm_dim: int = 2560,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 20,
        max_position_embeddings: int = 4096,
        num_audio_tokens: int = 626,
        num_text_tokens: int = 21178,
        num_mel_bins: int = 100,
        num_vq: int = 4,
        use_speaker_embedding: bool = True,
        use_llm_hidden_state: bool = False,
        spk_emb_token_id: int = 21143,
        num_spk_embs: int = 1,
        audio_bos_token_id: int = 21132,
        text_eos_token_id: int = 21133,
        use_text: bool = True,
        streaming: bool = True,
        streaming_text_chunk_size: int = 10,
        streaming_text_reserved_len: int = 300,
        streaming_audio_chunk_size: int = 50,
        attn_implementation: str = "sdpa",
        use_mlp: bool = True,
        aug_loss_weight: bool = True,
        do_sample: bool = True,
        top_p: float = 0.7,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.num_audio_tokens = num_audio_tokens
        self.num_text_tokens = num_text_tokens
        self.num_mel_bins = num_mel_bins
        self.num_vq = num_vq
        self.use_speaker_embedding = use_speaker_embedding
        self.use_llm_hidden_state = use_llm_hidden_state
        self.spk_emb_token_id = spk_emb_token_id
        self.num_spk_embs = num_spk_embs
        self.audio_bos_token_id = audio_bos_token_id
        self.text_eos_token_id = text_eos_token_id
        self.use_text = use_text
        self.streaming = streaming
        self.streaming_text_chunk_size = streaming_text_chunk_size
        self.streaming_text_reserved_len = streaming_text_reserved_len
        self.streaming_audio_chunk_size = streaming_audio_chunk_size
        self.attn_implementation = attn_implementation
        self.use_mlp = use_mlp
        self.aug_loss_weight = aug_loss_weight
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty


# --- Clean-room section (ModelCloud.ai, Apache-2.0) -----------------------------------------
# ConvNeXtBlock, GFSQ, DVAEDecoder and DVAE below are an independent implementation written from
# a behavioral specification (see NOTICE). They keep the published checkpoint's module and
# state_dict layout under `tts.dvae.`.


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-1D residual block: depthwise conv, LayerNorm, pointwise MLP, optional layer scale."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel: int,
        dilation: int,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel,
            padding=dilation * (kernel // 2),
            dilation=dilation,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        if layer_scale_init_value > 0:
            self.coef = nn.Parameter(torch.full((dim,), float(layer_scale_init_value)))
        else:
            self.coef = None

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        # `cond` is accepted for interface compatibility and ignored.
        h = self.dwconv(x).transpose(1, 2)
        h = self.pwconv2(self.act(self.pwconv1(self.norm(h))))
        if self.coef is not None:
            h = h * self.coef
        return x + h.transpose(1, 2)


class GFSQ(nn.Module):
    """Grouped residual FSQ wrapper over `vector_quantize_pytorch.GroupedResidualFSQ`.

    Codes are laid out per frame as (g0r0, g0r1, g1r0, g1r1, ...): channel `c` is group `c // R`,
    residual stage `c % R`.
    """

    def __init__(self, dim: int, levels, G: int, R: int, eps: float = 1e-5, transpose: bool = True):
        super().__init__()
        try:
            from vector_quantize_pytorch import GroupedResidualFSQ
        except ModuleNotFoundError as exc:
            if exc.name == "vector_quantize_pytorch":
                raise ImportError(
                    "MiniCPM-o 2.6 TTS requires vector-quantize-pytorch. "
                    "Install it with pip install 'gptqmodel[minicpm-o]'."
                ) from exc
            raise

        target_device = torch.get_default_device()
        # The dependency computes its constant tables eagerly; build on CPU so that this also
        # works under a `torch.device("meta")` context, then move to the requested device.
        with torch.device("cpu"):
            quantizer = GroupedResidualFSQ(dim=dim, levels=list(levels), num_quantizers=R, groups=G)
        persistent = set(quantizer.state_dict().keys())
        # Reference copies of the derived (non-persistent) constants. They are not checkpointed,
        # so loaders that materialize buffers uninitialized need them restored (see reset_constants).
        self._constant_tables = {
            name: buffer.detach().clone()
            for name, buffer in quantizer.named_buffers()
            if name not in persistent
        }
        self.quantizer = quantizer.to(target_device)
        # On a meta shell the constants stay real CPU tensors: nothing in a checkpoint can
        # materialize them later, and they are tiny.
        self.reset_constants()

        self.dim = dim
        self.levels = tuple(levels)
        self.G = G
        self.R = R
        self.eps = eps
        self.transpose = transpose
        self.n_ind = math.prod(levels)

    @torch.no_grad()
    def reset_constants(self) -> None:
        """Restore the dependency's derived constant buffers (level counts, bases, scales, codebooks)."""
        for name, value in self._constant_tables.items():
            owner_path, _, leaf = name.rpartition(".")
            owner = self.quantizer.get_submodule(owner_path)
            buffer = owner._buffers[leaf]
            if buffer.device.type == "meta":
                owner.register_buffer(leaf, value.to(dtype=buffer.dtype).clone(), persistent=False)
            else:
                buffer.copy_(value.to(device=buffer.device, dtype=buffer.dtype))

    def _embed(self, codes: torch.Tensor) -> torch.Tensor:
        """Integer codes `(B, G*R, T)` (or `(B, T, G*R)` without transpose) to features."""
        if self.transpose:
            codes = codes.transpose(1, 2)
        batch, frames, _ = codes.shape
        indices = codes.reshape(batch, frames, self.G, self.R).permute(2, 0, 1, 3)
        features = self.quantizer.get_output_from_indices(indices)  # (B, T, dim)
        return features.transpose(1, 2) if self.transpose else features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Features `(B, dim, T)` (or `(B, T, dim)` without transpose) to integer codes."""
        if self.transpose:
            x = x.transpose(1, 2)
        _, indices = self.quantizer(x)  # (G, B, T, R)
        groups, batch, frames, stages = indices.shape
        codes = indices.permute(1, 2, 0, 3).reshape(batch, frames, groups * stages)
        return codes.transpose(1, 2) if self.transpose else codes


class DVAEDecoder(nn.Module):
    """Conv1d bottleneck in, a stack of ConvNeXt-1D blocks, then a 1x1 projection out."""

    def __init__(
        self,
        idim: int,
        odim: int,
        n_layer: int = 12,
        bn_dim: int = 64,
        hidden: int = 256,
        kernel: int = 7,
        dilation: int = 2,
        up: bool = False,
    ):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(
            nn.Conv1d(idim, bn_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(bn_dim, hidden, 3, 1, 1),
        )
        self.decoder_block = nn.ModuleList(
            [ConvNeXtBlock(hidden, hidden * 4, kernel, dilation) for _ in range(n_layer)]
        )
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        # `conditioning` is accepted for interface compatibility and ignored.
        x = self.conv_in(x)
        for block in self.decoder_block:
            x = block(x)
        return self.conv_out(x)


class DVAE(nn.Module):
    """Mel-spectrogram codec with fixed geometry: 100 mel bins, 4 codes of 625 values per frame pair."""

    def __init__(self):
        super().__init__()
        self.coef = nn.Parameter(torch.rand(100).reshape(1, 100, 1))
        self.downsample_conv = nn.Sequential(
            nn.Conv1d(100, 512, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(512, 512, 4, 2, 1),
            nn.GELU(),
        )
        self.encoder = DVAEDecoder(idim=512, odim=1024, n_layer=12, bn_dim=128, hidden=256)
        self.decoder = DVAEDecoder(idim=512, odim=512, n_layer=12, bn_dim=128, hidden=256)
        self.out_conv = nn.Conv1d(512, 100, 3, 1, 1, bias=False)
        self.vq_layer = GFSQ(dim=1024, levels=(5, 5, 5, 5), G=2, R=2)

    @torch.inference_mode()
    def forward(self, inp: torch.Tensor, mode: Literal["encode", "decode"] = "decode") -> torch.Tensor:
        if mode == "encode":
            mel = inp / self.coef.reshape(-1, 1)  # (100, T); the caller's tensor is untouched
            hidden = self.downsample_conv(mel).unsqueeze(0)
            hidden = self.encoder(hidden)
            return self.vq_layer(hidden)
        if mode != "decode":
            raise ValueError(f"unknown DVAE mode {mode!r}")

        features = self.vq_layer._embed(inp) if self.vq_layer is not None else inp  # (B, 1024, T)
        batch, channels, frames = features.shape
        # Group 0 reconstructs even frames and group 1 odd frames.
        interleaved = features.reshape(batch, 2, channels // 2, frames).permute(0, 2, 3, 1)
        interleaved = interleaved.reshape(batch, channels // 2, frames * 2)
        mel = self.out_conv(self.decoder(interleaved))
        return mel * self.coef


# --- End of clean-room section ---------------------------------------------------------------


def apply_spk_emb(
    input_ids: torch.Tensor = None,
    spk_emb: torch.Tensor = None,
    input_embeds: torch.Tensor = None,
    spk_emb_token_id: int = 0,
    num_spk_embs: int = 1,
):
    """
    Replace consecutive `num_spk_embs` speaker embedding placeholders in input_embeds with pre-prepared speaker embeddings. This is an in-place replacement, no new tensor is created, so no value is returned.

    Args:
        input_ids (torch.Tensor): Input ID tensor, shape [batch_size, seq_len_max]
        spk_emb (torch.Tensor): Speaker embedding tensor, shape [batch_size, num_spk_emb, hidden_dim]
        input_embeds (torch.Tensor): Input embedding tensor, shape [batch_size, seq_len_max, hidden_dim]
        spk_emb_token_id (int): ID of the speaker embedding token
        num_spk_embs (int): Number of speaker embeddings

    Returns:
        None
    """

    batch_size = input_ids.shape[0]

    for idx in range(batch_size):
        input_ids_ = input_ids[idx]  # [seq_len_max]
        spk_emb_ = spk_emb[idx]  # [num_spk_emb]
        mask_ = input_ids_ == spk_emb_token_id  # [batch_size, seq_len_max]
        nonzero_position_idx = mask_.nonzero(as_tuple=False)  # [num_spk_emb, 1]
        assert nonzero_position_idx.shape[0] == num_spk_embs
        begin_idx = nonzero_position_idx.min()
        end_idx = nonzero_position_idx.max()
        input_embeds[idx, begin_idx : end_idx + 1, :] = spk_emb_

    return


def make_streaming_chunk_mask_generation(
    inputs_embeds: torch.Tensor,
    past_seen_tokens: int,
    streaming_tts_text_mask: torch.Tensor,
    streaming_reserved_length: int = 300,
    streaming_audio_chunk_size: int = 50,
    streaming_text_chunk_size: int = 10,
    num_spk_emb: int = 1,
    use_spk_emb: bool = True,
) -> torch.Tensor:
    """
    In streaming audio generation, determine which `text` positions the TTS model can attend to when generating each chunk of `audio` tokens.

    This function creates a mask that allows the model to attend to a specific chunk of text
    tokens when generating each chunk of audio tokens, enabling streaming TTS generation.

    Args:
        inputs_embeds (torch.Tensor): Input embeddings tensor.
        past_seen_tokens (int): Number of tokens already seen by the model.
        streaming_tts_text_mask (torch.Tensor): Mask for the text tokens.
        streaming_reserved_length (int, optional): Number of reserved tokens for streaming. Defaults to 300.
        streaming_chunk_length (int, optional): Length of each streaming chunk. Defaults to 50.
        streaming_text_chunk_size (int, optional): Size of each text chunk. Defaults to 7.

    Returns:
        torch.Tensor: Causal mask for streaming TTS generation, shape is [batch_size=1, 1, seq_len=1, past_seen_tokens+1]

    Raises:
        AssertionError: If the batch size is not 1 (only supports batch size of 1 for inference).
    """
    assert inputs_embeds.shape[0] == 1

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    min_dtype = torch.finfo(dtype).min

    # Add `1` to the past seen tokens to account for new `tokens` during `generate`
    causal_mask = torch.full((1, past_seen_tokens + inputs_embeds.shape[1]), fill_value=0, dtype=dtype, device=device)

    # Calculate the start of invisible text tokens
    invisible_text_tokens_start = (
        min(
            math.ceil((past_seen_tokens - streaming_reserved_length) / streaming_audio_chunk_size)
            * streaming_text_chunk_size,
            streaming_reserved_length,
        )
        + 1
        + num_spk_emb * use_spk_emb
    )  # Add 1 for [Stts] and N for [spk_emb] tokens if `use_spk_emb` is True

    invisible_text_tokens_end = (
        streaming_reserved_length + 1 + num_spk_emb * use_spk_emb + 1
    )  # Add 1 for [Ptts] (aka `audio_bos_token_id`)

    # Set invisible text tokens to min_dtype (effectively -inf)
    causal_mask[0, invisible_text_tokens_start:invisible_text_tokens_end] = min_dtype

    # Mask padding positions in the text mask
    causal_mask[0, 0 : 1 + num_spk_emb * use_spk_emb + streaming_reserved_length + 1].masked_fill_(
        streaming_tts_text_mask == 0, min_dtype
    )

    # Add extra dimensions for batch and heads
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask




# Clean-room implementation (ModelCloud.ai, Apache-2.0) written from a behavioral specification; see NOTICE.
class CustomRepetitionPenaltyLogitsProcessorRepeat(LogitsProcessor):
    """Windowed repetition penalty over multi-codebook audio-code histories.

    Each logit is divided (if non-negative) or multiplied (if negative) by `penalty ** count`, where
    `count` is how often that id occurs in the last `past_window` ids of the row. Rows with index
    `>= max_input_ids` are left unpenalized.
    """

    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")
        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids[:, -self.past_window :]
        counts = F.one_hot(input_ids, num_classes=scores.size(1)).sum(dim=1)
        if counts.size(0) > self.max_input_ids:
            counts[self.max_input_ids :] = 0
        alpha = torch.pow(self.penalty, counts.to(torch.get_default_dtype()))
        penalized = torch.where(scores < 0, scores * alpha, scores / alpha)
        return penalized.to(scores.dtype).contiguous()


@dataclass
class ConditionalChatTTSGenerationOutput(ModelOutput):
    """
    Output class for ConditionalChatTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.

    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    finished: bool = None


class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features):
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class ConditionalChatTTS(PreTrainedModel):
    """A conditional text-to-speech model that can generate speech from text with speaker conditioning.

    This model extends PreTrainedModel to provide text-to-speech capabilities with:
    - LLM hidden state conditioning
    - Streaming generation

    The model uses a transformer architecture with LLM hidden states and can operate in both
    streaming and non-streaming modes for flexible deployment.

    The model process sequence in the following format:
    | text bos token | LLM embedding projected to tts embedding space | text tokens (fixed length, reserved for future tokens) | audio bos token | audio tokens (audio token length is not fixed)| audio eos token |

    The format is designed to support LLM-conditioned streaming audio generation.

    Usage:
    To support streaming generation, two global variables should be maintained outside of the model.
        1. `audio_input_ids`: stores *discrete* audio codes. It is a tensor with shape [1, sequence length+1, num_vq].
        2. `past_key_values`: a `DynamicCache` storing the KV cache for both text tokens and audio codes. Each layer holds keys/values with shape [1, num_attention_heads, sequence length, hidden_size // num_attention_heads]

    where `num_vq` is the number of audio codebooks, in default setting, it is `4`.

    1. Create an empty `past_key_values` with
    ```python
    initial_kv_cache_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len # where `1` denotes the `bos` token
    dtype = model.emb_text.weight.dtype
    device = model.emb_text.weight.device
    past_key_values = cache_from_pairs(
        (
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device),
            torch.zeros(1, model.config.num_attention_heads, initial_kv_cache_length, model.config.hidden_size // model.config.num_attention_heads, dtype=dtype, device=device)
        )
        for _ in range(model.config.num_hidden_layers)
    )

    2. At the same time, create an empty `audio_input_ids` with shape [1, sequence length, num_vq], `num_vq` denotes multiple layer audio codebooks. But here we also include text tokens in the sequence, but they will be zeros, and will not be used, just a placeholder.

    ```python
    initial_audio_input_ids_length = 1 + model.num_spk_embs + model.streaming_text_reserved_len + 1
    # [bos token, speaker embeddings, text tokens, audio bos token]
    audio_input_ids = torch.zeros(batch_size=1, initial_audio_input_ids_length, model.num_vq)
    ```

    2. Prefill some text tokens to TTS model (for example, 10 tokens) using `prefill_text` method.

    ```python
    outputs = llm.generate(**kwargs)
    llm_tokens = some_function_to_extract_llm_tokens(outputs)
    lm_spk_emb_last_hidden_states = some_function_to_extract_lm_spk_emb_last_hidden_states(outputs)
    tts_text_input_ids = tts_tokenizer.encode(llm_tokenizer.decode(llm_tokens))
    # here assume we are prefilling text token 0 to text token 9 (included), totally 10 tokens.
    begin = 0
    end = 9+1
    position_ids = torch.arange(begin, end, dtype=torch.long, device=device)

    past_key_values = model.prefill_text(
        input_ids=tts_text_input_ids,
        position_ids=position_ids,
        past_key_values=past_key_values,
        lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
    )
    ```

    3. Make a `streaming_tts_text_mask` to denote which position contains valid text tokens, similar to `attention_mask` in standard causal attention.

    ```python
    streaming_tts_text_mask = torch.zeros(model.streaming_reserved_length)
    streaming_tts_text_mask[0:end] = 1 # denotes these post
    ```

    3. Generate audio codes using `generate` method.

    ```python
    outputs = model.generate(
        input_ids=audio_input_ids,
        past_key_values=past_key_values,
        streaming_tts_text_mask=streaming_tts_text_mask,
        max_new_token=50,
    )

    # update past_key_values and input_ids
    past_key_values = outputs.past_key_values
    audio_input_ids = outputs.input_ids
    ```

    The `past_key_values` is extended by `max_new_token=50`, and `audio_input_ids` is also extended by `max_new_token=50` after `generate` calling.

    4. Notice that after prefilling `10` text tokens, the model can generate up to `50` audio tokens, if you want to generate more audio tokens, you need to prefill next `10` text tokens. And it is okay to only generate `25` audio tokens for faster initial response.

    5. Repeat steps `2,3,4` as needed in your streaming audio generation cases, but ensure usage complies with the following guidelines discussed above.
    """

    config_class = ConditionalChatTTSConfig
    _no_split_modules = []

    def __init__(self, config: ConditionalChatTTSConfig):
        super().__init__(config)

        self.use_speaker_embedding = config.use_speaker_embedding
        self.use_llm_hidden_state = config.use_llm_hidden_state
        self.num_spk_embs = config.num_spk_embs
        self.spk_emb_token_id = config.spk_emb_token_id

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_size = config.streaming_text_chunk_size
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        if self.config.use_mlp:
            self.projector = MultiModalProjector(config.llm_dim, config.hidden_size)
        else:
            self.projector = nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        self.emb_code = nn.ModuleList(
            [nn.Embedding(config.num_audio_tokens, config.hidden_size) for _ in range(config.num_vq)]
        )
        self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)
        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )
        dvae = DVAE()
        self.dvae = dvae

        model_config = LlamaConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=config.num_hidden_layers,
            max_position_embeddings=config.max_position_embeddings,
            attn_implementation=config.attn_implementation,
            num_key_value_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
        )

        model = LlamaModel(model_config)
        self.model = model

    @torch.inference_mode()
    def merge_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Merge `input_ids` and `lm_spk_emb_last_hidden_states` to `inputs_embeds`.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            lm_spk_emb_last_hidden_states (Optional[torch.Tensor], optional): Last hidden states of speaker embeddings from the language model. Defaults to None.

        Raises:
            NotImplementedError: If speaker embedding is not used and language model hidden states are not implemented.

        Returns:
            torch.Tensor: Prepared input embeddings for the model.
        """
        assert input_ids.shape[0] == 1

        # Embed input_ids to input_embeds
        inputs_embeds = self.emb_text(input_ids)

        # Inject speaker embedding to input_embeds if it exists
        if self.use_speaker_embedding:
            spk_emb_mask = input_ids == self.spk_emb_token_id
            if spk_emb_mask.any():
                assert lm_spk_emb_last_hidden_states is not None
                # Project spk emb to tts hidden size first, [batch_size, num_spk_emb, llm_dim] -> [batch_size, num_spk_emb, self.hidden_size]
                lm_spk_emb_last_hidden_states = lm_spk_emb_last_hidden_states.to(self.projector.linear1.weight.dtype)
                projected_spk_emb = self.projector(lm_spk_emb_last_hidden_states)
                projected_spk_emb = F.normalize(projected_spk_emb, p=2, dim=-1)
                apply_spk_emb(
                    input_ids=input_ids,
                    spk_emb=projected_spk_emb,
                    input_embeds=inputs_embeds,
                    spk_emb_token_id=self.spk_emb_token_id,
                    num_spk_embs=self.num_spk_embs,
                )
        else:
            raise NotImplementedError

        return inputs_embeds

    @torch.inference_mode()
    def prefill_text(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: DynamicCache,
        lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
    ):
        """Prefill a chunk of new text tokens in streaming setting.
        Specifically speaking, update `past_key_values` using new text tokens, then the model will read the new text tokens.

        Args:
            input_ids (Tensor): Tensor of shape [batch_size, seq_len]
            position_ids (LongTensor): Tensor of shape [batch_size, seq_len]
            past_key_values (DynamicCache): KV Cache of all layers; each layer holds keys and values. Each tensor is of seq_len = `self.streaming_text_reserved_len`. `past_key_values` will be updated.
            lm_spk_emb_last_hidden_states (Tensor, optional): Tensor of shape [batch_size, num_spk_emb, llm_dim]. Defaults to None.
            lm_last_hidden_states (Tensor, optional): _description_. Defaults to None.

        Note that all `batch_size` should be `1`.
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        # Merge text and LLM embeddings
        inputs_embeds = self.merge_inputs_embeds(
            input_ids=input_ids,
            lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
        )

        # Clone KV Cache
        past_key_values_for_prefill = cache_from_pairs(
            (keys[:, :, : position_ids[:, 0], :].clone(), values[:, :, : position_ids[:, 0], :].clone())
            for keys, values in cache_pairs(past_key_values)
        )

        # Model forward
        outputs_prefill: BaseModelOutputWithPast = self.model(
            attention_mask=None,  # because for text, it is standard causal attention mask, do nothing
            position_ids=position_ids,  # position_ids denotes the position of new text tokens in the sequence
            past_key_values=past_key_values_for_prefill,  # `past_key_values` will be updated by the model
            inputs_embeds=inputs_embeds,  # contains text and language model embedding
            use_cache=True,
        )

        # Get model updated KV Cache
        past_key_values_for_prefill_updated = outputs_prefill.past_key_values

        # Update generated KV Cache to input `past_key_values` (in place, inside the reserved text window)
        start, end = position_ids[:, 0], position_ids[:, -1] + 1
        for (keys, values), (new_keys, new_values) in zip(
            cache_pairs(past_key_values), cache_pairs(past_key_values_for_prefill_updated)
        ):
            keys[:, :, start:end, :] = new_keys[:, :, start:end].clone()
            values[:, :, start:end, :] = new_values[:, :, start:end].clone()

        # TODO: del past_key_values_for_prefill_updated recursively
        # TODO: del outputs_prefill recursively

        return past_key_values

    @torch.inference_mode()
    def prefill_audio_ids(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache,
        streaming_tts_text_mask=None,
        add_audio_bos: bool = True,
    ):
        """Prefill a chunk of audio ids to the model. Used in sliding-window long audio generation.
        Specifically, prefill many audio ids (typically from last window) to the model in the new window.

        Args:
            input_ids (torch.Tensor): (1, seq_len, num_vq) Audio input token ids.
            past_key_values (DynamicCache): Past key values for attention mechanism.
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        code_emb = [self.emb_code[i](input_ids[:, :, i]) for i in range(self.num_vq)]
        inputs_embeds = torch.stack(code_emb, 3).sum(3)  # [1,seq_len,768]
        input_len = input_ids.shape[1]

        if add_audio_bos:
            narrowed_input_ids = torch.tensor([[self.audio_bos_token_id]], dtype=torch.long, device=self.device)
            bos_inputs_embeds = self.emb_text(narrowed_input_ids)
            inputs_embeds = torch.cat([bos_inputs_embeds, inputs_embeds], dim=1)
            input_len += 1

        past_key_values_length = past_key_values.get_seq_length()
        position_ids = torch.arange(
            past_key_values_length, past_key_values_length + input_len, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        causal_mask = make_streaming_chunk_mask_generation(
            inputs_embeds=inputs_embeds,
            past_seen_tokens=past_key_values_length,
            streaming_tts_text_mask=streaming_tts_text_mask,
            streaming_reserved_length=self.streaming_text_reserved_len,
            streaming_text_chunk_size=self.streaming_text_chunk_size,
        )  # [1, 1, 1, past_key_values_length + input_len]

        # Model forward
        outputs: BaseModelOutputWithPast = self.model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        return past_key_values

    # Clean-room implementation (ModelCloud.ai, Apache-2.0) written from a behavioral specification;
    # see NOTICE.
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache,
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        streaming_tts_text_mask=None,
        force_no_stop=False,
        min_new_token=10,
        max_new_token=50,
        logits_warpers: List[LogitsProcessor] = [],
        logits_processors: List[CustomRepetitionPenaltyLogitsProcessorRepeat] = [],
        show_tqdm=False,
    ):
        """Sample audio codes step by step, continuing the streaming state in `past_key_values`.

        Args:
            input_ids: `(1, P, num_vq)` codes; rows `[0, C)` are the condition placeholders, rows
                `[C, P)` are audio codes generated so far.
            past_key_values: cache holding `P - 1` positions; extended in place.
            temperature: per-codebook temperature `(num_vq,)` (or a single value).
            eos_token: audio eos id.
            streaming_tts_text_mask: validity mask over the condition positions.
            force_no_stop: never sample eos.
            min_new_token: number of initial steps on which eos is disallowed.
            max_new_token: step budget of this call.
            logits_warpers / logits_processors: applied (processors first) on each step after the
                audio-bos step, as `f(history, logits)`.
            show_tqdm: show a progress bar.

        Returns:
            ConditionalChatTTSGenerationOutput
        """
        assert input_ids.shape[0] == 1
        assert past_key_values is not None

        condition_length = 1 + self.num_spk_embs * self.use_speaker_embedding + self.streaming_text_reserved_len + 1
        batch_size, start_length, num_vq = input_ids.shape
        device = input_ids.device
        eos_id = int(eos_token.reshape(-1)[0]) if isinstance(eos_token, torch.Tensor) else int(eos_token)
        temperature = temperature.to(device=device, dtype=torch.float32).reshape(1, -1, 1)

        codes = torch.zeros(
            batch_size, start_length + max_new_token, num_vq, dtype=input_ids.dtype, device=device
        )
        codes[:, :start_length] = input_ids
        cur = start_length
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        progress = tqdm(total=max_new_token, desc="code") if show_tqdm else None
        for step in range(max_new_token):
            past_length = past_key_values.get_seq_length()
            if cur != past_length + 1:
                raise AssertionError(
                    f"generate expects a cache of {cur - 1} positions for {cur} code rows, got {past_length}"
                )

            audio_bos_step = cur == condition_length
            if audio_bos_step:
                bos = torch.full((batch_size, 1), self.audio_bos_token_id, dtype=torch.long, device=device)
                inputs_embeds = self.emb_text(bos)
            else:
                last = codes[:, cur - 1 : cur]
                code_emb = [self.emb_code[k](last[:, :, k]) for k in range(num_vq)]
                inputs_embeds = torch.stack(code_emb, 3).sum(3)

            position_ids = torch.full((batch_size, 1), past_length, dtype=torch.long, device=device)
            causal_mask = make_streaming_chunk_mask_generation(
                inputs_embeds=inputs_embeds,
                past_seen_tokens=past_length,
                streaming_tts_text_mask=streaming_tts_text_mask,
                streaming_reserved_length=self.streaming_text_reserved_len,
                streaming_text_chunk_size=self.streaming_text_chunk_size,
            )
            outputs: BaseModelOutputWithPast = self.model(
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            hidden = outputs.last_hidden_state[:, -1]  # (B, H)

            logits = torch.empty(batch_size, num_vq, self.num_audio_tokens, dtype=torch.float32, device=device)
            for k in range(num_vq):
                logits[:, k] = self.head_code[k](hidden)
            logits = (logits / temperature).reshape(batch_size * num_vq, self.num_audio_tokens)

            if not audio_bos_step:
                history = codes[:, condition_length:cur].permute(0, 2, 1).reshape(batch_size * num_vq, -1)
                for processor in logits_processors:
                    logits = processor(history, logits)
                for warper in logits_warpers:
                    logits = warper(history, logits)

            if step < min_new_token:
                logits[:, eos_id] = -torch.inf
            if force_no_stop:
                logits[:, eos_id] = -torch.inf

            probs = F.softmax(logits, dim=-1)
            next_codes = torch.multinomial(probs, 1).reshape(batch_size, num_vq)
            finished |= (next_codes == eos_id).any(dim=1)
            codes[:, cur] = next_codes

            if progress is not None:
                progress.update(1)
            if step == 0 and finished.any():
                # The sampled row is not kept, although the cache already holds its position.
                break
            cur += 1
            if finished.all():
                break
        else:
            if show_tqdm:
                logger.info("incomplete result: hit max_new_token %d", max_new_token)
        if progress is not None:
            progress.close()

        audio_input_ids = codes[:, :cur]
        end = cur - 1 if finished.all() else cur
        new_ids = audio_input_ids[:, condition_length:end]

        return ConditionalChatTTSGenerationOutput(
            new_ids=new_ids,
            audio_input_ids=audio_input_ids,
            past_key_values=past_key_values,
            finished=finished.all(),
        )

    # Clean-room implementation (ModelCloud.ai, Apache-2.0) written from a behavioral specification;
    # see NOTICE.
    @torch.inference_mode()
    def decode_to_mel_specs(
        self,
        result_list: List[torch.Tensor],
    ):
        """Decode a list of `(L_i, num_vq)` code tensors to mel spectrograms `(N, 100, 2 * max L_i)`.

        Shorter elements are right-padded with code 0 (not trimmed).
        """
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)

        first = result_list[0]
        max_length = max(int(item.size(0)) for item in result_list)
        batch = torch.zeros(len(result_list), first.size(1), max_length, dtype=first.dtype, device=first.device)
        for index, item in enumerate(result_list):
            batch[index, :, : item.size(0)] = item.transpose(0, 1)

        return self.dvae(batch, mode="decode")

    def _init_weights(self, module):
        super()._init_weights(module)
        # The DVAE quantizer's derived constants are non-persistent buffers: they are not in
        # checkpoints, so loading materializes them uninitialized. Restore them here.
        vq_layer = getattr(getattr(self, "dvae", None), "vq_layer", None)
        if isinstance(vq_layer, GFSQ) and any(module is sub for sub in vq_layer.modules()):
            vq_layer.reset_constants()


# Clean-room implementation (ModelCloud.ai, Apache-2.0) written from a behavioral specification; see NOTICE.
def gen_logits(
    num_code: int,
    top_P=0.7,
    top_K=20,
    repetition_penalty=1.0,
):
    """Build the `(logits_warpers, logits_processors)` pair used for audio-code sampling."""
    logits_warpers = []
    if top_P is not None:
        logits_warpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        logits_warpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        logits_processors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16))

    return logits_warpers, logits_processors


# Copy and modified from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
