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

# Inherited ChatTTS/Vocos code has an unresolved licensing boundary; see NOTICE.

import logging
import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.utils.parametrize as P
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


# [clean-room: removed; implement from cleanroom/minicpm_o/SPEC.md]


# [clean-room: removed; implement from cleanroom/minicpm_o/SPEC.md]


# [clean-room: removed; implement from cleanroom/minicpm_o/SPEC.md]


# [clean-room: removed; implement from cleanroom/minicpm_o/SPEC.md]


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




# [clean-room: removed; implement from cleanroom/minicpm_o/SPEC.md]


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
        raise NotImplementedError("clean-room: implement ConditionalChatTTS.generate from cleanroom/minicpm_o/SPEC.md")

    @torch.inference_mode()
    def decode_to_mel_specs(
        self,
        result_list: List[torch.Tensor],
    ):
        raise NotImplementedError("clean-room: implement ConditionalChatTTS.decode_to_mel_specs from cleanroom/minicpm_o/SPEC.md")


# [clean-room: removed; implement from cleanroom/minicpm_o/SPEC.md]


# Copy and modified from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
