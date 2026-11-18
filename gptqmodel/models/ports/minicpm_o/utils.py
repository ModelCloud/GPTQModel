#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 The OpenBMB Team. All rights reserved.
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

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from ....utils.image import load_bytes_from_url, load_image_from_url
from .cache import DynamicCache


logger = logging.getLogger(__name__)


def get_video_frame_audio_segments(*args, **kwargs):
    raise NotImplementedError(
        "Container video decoding is not qualified; pass decoded PIL frames and waveform arrays."
    )


# text
@dataclass
class GenerateChunkOutput:
    chunk_token_ids: torch.Tensor
    current_inputs_embeds: torch.Tensor
    input_last_hidden_states: Optional[torch.Tensor]  # for tts use_speaker_embedding
    last_hidden_states: Optional[torch.Tensor]  # for tts input feature (projector_semantic)
    past_key_values: Optional[torch.Tensor]
    finished: bool


class ChunkPrefillChunkGenerate:
    def __init__(self, model, tokenizer, terminators):
        self.tokenizer = tokenizer
        self.model = model
        self.terminators = terminators
        self.terminators_ids = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        self.embedding_layer = self.model.get_input_embeddings()

        self.forbidden_tokens = [
            ":",
            "：",
            "；",
            "#",
            "“",
            "”",
            "‘",
            "’",
            "@",
            "*",
            "【",
            "】",
            "「",
            "」",
            "(",
            ")",
            "（",
            "）",
            "[",
            "]",
            "&",
            "/",
            "$",
        ]

        self.forbidden_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in self.forbidden_tokens]
        bad_token_ids = getattr(tokenizer, "bad_token_ids", [])
        if bad_token_ids:
            self.forbidden_token_ids.extend(bad_token_ids)

    @staticmethod
    def prepare_generation_config(do_sample, max_new_tokens=50, min_new_tokens=0, **kwargs):
        num_beams = kwargs.get("num_beams", 3)
        generation_config = {
            "num_beams": num_beams,
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.05,
        }

        if do_sample:
            generation_config.update(
                {
                    "top_p": 0.8,
                    "top_k": 100,
                    "temperature": 0.7,
                    "do_sample": True,
                    "repetition_penalty": 1.05,
                }
            )
        elif num_beams > 1:
            generation_config.update({"num_beams": num_beams, "repetition_penalty": 1.2, "do_sample": False})
        else:
            generation_config.update({"do_sample": False, "repetition_penalty": 1.05})

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config["max_new_tokens"] = max_new_tokens

        return generation_config

    def chunk_generate(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values,
        is_first_generate_chunk: bool,
        chunk_size: int,
        return_hidden_states: bool,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float = 1.05,
        length_penalty: float = 1.0,
        all_input_ids: Optional[torch.Tensor] = None,
    ) -> GenerateChunkOutput:
        """
        Args:
            inputs_embeds: [1, seq_len, hidden_dim], Input embeddings of current chunk.
            past_key_values: [num_layers, 2, batch_size, num_heads, seq_len, head_dim], Past key values for llm.
            is_first_generate_chunk: bool, Whether this is the first generate chunk.
            chunk_size: int, The size of the current chunk, default is 10, and it is fixed during training.
            return_hidden_states: bool Whether to return the hidden states, default is True.
            do_sample: bool Whether to sample from the model, default is True.
            temperature: float The temperature for the model, default is 0.7.
            top_p: float The top-p for the model, default is 0.8.
            top_k: int The top-k for the model, default is 100.
            repetition_penalty: float, The repetition penalty for the model, default is 1.05.
            length_penalty: float, The length penalty for the model, default is 1.0. Higher value means more detailed generation.
            all_input_ids: Optional[torch.Tensor], The input ids for the current chunk.
        """

        finished = False
        current_inputs_embeds = inputs_embeds.clone()
        input_last_hidden_states = []
        last_hidden_states = []
        generated_tokens = []

        for token_idx in range(chunk_size):
            if is_first_generate_chunk and token_idx == 0:
                # first generate chunk, prefill inputs_embeds
                model_inputs = {
                    "inputs_embeds": current_inputs_embeds,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "output_hidden_states": return_hidden_states,
                }
            else:  # for all other cases: prefill the latest generated token
                model_inputs = {
                    "inputs_embeds": current_inputs_embeds[:, -1:, :],
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "output_hidden_states": return_hidden_states,
                }

            with torch.no_grad():
                outputs = self.model(**model_inputs)

            # last token's logits
            logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=inputs_embeds.device)

            # forbid specific tokens decoding = model.generate@suppress_tokens
            if self.forbidden_token_ids:
                logits[:, self.forbidden_token_ids] = float("-inf")

            past_key_values = outputs.past_key_values

            PENALTY_WINDOW_SIZE = 128

            # apply repetition penalty
            if repetition_penalty != 1.0:
                # get token ids for repetition penalty
                if all_input_ids is not None:
                    # use global input ids (including original input and generated part)
                    if len(generated_tokens) > 0:
                        generated_token_ids = torch.cat(generated_tokens, dim=1)
                        current_sequence = torch.cat(
                            [
                                all_input_ids[:, -PENALTY_WINDOW_SIZE:],
                                generated_token_ids,
                            ],
                            dim=1,
                        )
                    else:
                        current_sequence = all_input_ids[:, -PENALTY_WINDOW_SIZE:]
                    unique_token_ids = torch.unique(current_sequence.squeeze(0))
                elif len(generated_tokens) > 0:
                    # revert to original logic: only use generated tokens
                    generated_token_ids = torch.cat(generated_tokens, dim=1).squeeze(0)
                    unique_token_ids = torch.unique(generated_token_ids)
                else:
                    unique_token_ids = torch.tensor([], dtype=torch.long, device=logits.device)

                # apply repetition penalty
                for token_id in unique_token_ids:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] = logits[0, token_id] / repetition_penalty
                    else:
                        logits[0, token_id] = logits[0, token_id] * repetition_penalty

            # apply length penalty, higher value means more detailed generation
            if length_penalty != 1.0:
                for eos_token_id in self.terminators_ids:
                    if logits[0, eos_token_id] > 0:
                        logits[0, eos_token_id] = logits[0, eos_token_id] / length_penalty
                    else:
                        logits[0, eos_token_id] = logits[0, eos_token_id] * length_penalty

            # apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits_filtered = torch.full_like(logits, float("-inf"))
                    logits_filtered.scatter_(1, top_k_indices, top_k_logits)
                    logits = logits_filtered

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # remove tokens with cumulative probability greater than top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                # sampling
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if return_hidden_states:
                if is_first_generate_chunk and token_idx == 0:
                    input_last_hidden_states.append(outputs.hidden_states[-1])
                else:
                    last_hidden_states.append(outputs.hidden_states[-1])

            # if terminator token, stop generating
            if next_token.item() in self.terminators_ids:
                finished = True
                break

            generated_tokens.append(next_token)

            # convert new token to embeddings and concatenate
            next_token_embed = self.embedding_layer(next_token)

            # update inputs_embeds, add one
            current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embed], dim=1)

        if len(generated_tokens) > 0:
            chunk_token_ids = torch.cat(generated_tokens, dim=1)
        else:
            # special case: if last chunk and first predict is eos token, return last token of previous chunk. return a tensor with shape (1, 0)
            if finished:
                chunk_token_ids = torch.zeros((1, 0), dtype=torch.long, device=current_inputs_embeds.device)
            else:
                raise Exception("this should not happen")

        if len(last_hidden_states) > 0:
            last_hidden_states = torch.cat(last_hidden_states, dim=1)
        else:
            # special case: if last chunk, return last token of previous chunk.
            if finished:
                last_hidden_states = torch.cat(last_hidden_states, dim=1)
            else:
                raise Exception("this should not happen")

        if len(input_last_hidden_states) > 0:
            input_last_hidden_states = torch.cat(input_last_hidden_states, dim=1)
        else:
            input_last_hidden_states = None

        return GenerateChunkOutput(
            chunk_token_ids=chunk_token_ids,
            current_inputs_embeds=current_inputs_embeds,
            input_last_hidden_states=input_last_hidden_states,
            last_hidden_states=last_hidden_states,
            past_key_values=past_key_values,
            finished=finished,
        )


def streaming_token_decoder(token_iterator, tokenizer, skip_special_tokens=False):
    """
    Incrementally decode tokens from an iterator, handling partial multi-byte characters.

    When streaming tokens, multi-byte characters (like Chinese) may be split across multiple
    tokens. Decoding partial tokens results in replacement characters (U+FFFD). This function
    buffers tokens and only yields complete characters.

    Args:
        token_iterator: An iterator yielding (token_ids, is_finished) tuples.
                       token_ids can be torch.Tensor or any iterable of integers.
        tokenizer: The tokenizer to use for decoding.
        skip_special_tokens: Whether to skip special tokens during decoding.

    Yields:
        (decoded_text, is_finished) tuples where decoded_text is the new text since last yield.
    """
    accumulated_token_ids = []
    yielded_text_len = 0

    for token_ids, is_finished in token_iterator:
        # Accumulate token IDs
        if torch.is_tensor(token_ids):
            accumulated_token_ids.extend(token_ids.reshape(-1).tolist())
        else:
            accumulated_token_ids.extend(list(token_ids) if hasattr(token_ids, "__iter__") else [token_ids])

        # Decode all accumulated tokens
        full_decoded = tokenizer.decode(accumulated_token_ids, skip_special_tokens=skip_special_tokens)

        if is_finished:
            # Final chunk - yield all remaining text
            new_text = full_decoded[yielded_text_len:]
            yield new_text, is_finished
        else:
            # Find safe prefix without incomplete multi-byte characters
            # The replacement character '�' (U+FFFD) indicates incomplete decoding
            new_text = full_decoded[yielded_text_len:]

            # Hold back text ending with replacement character (incomplete UTF-8 sequence)
            safe_end = len(new_text)
            while safe_end > 0 and new_text[safe_end - 1] == "\ufffd":
                safe_end -= 1

            safe_text = new_text[:safe_end] if safe_end > 0 else ""
            yielded_text_len += len(safe_text)
            yield safe_text, is_finished


def torch_clone_recursive(obj):
    """Recursively clone nested containers of torch.Tensors.

    Supported container types: dict, list, tuple. Non-container non-Tensor
    objects are returned as-is.
    """
    if torch.is_tensor(obj):
        return obj.clone()
    elif isinstance(obj, dict):
        return {k: torch_clone_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [torch_clone_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(torch_clone_recursive(v) for v in obj)
    else:
        raise ValueError(f"Unsupported type: {type(obj)}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class SpeculativeSnapshot:
    """Speculative snapshot for VAD speculative rollback.

    Used in VAD speculative execution: creates a snapshot after streaming_prefill
    and before streaming_generate. If speculation fails (user continues speaking),
    the state can be restored to continue streaming_prefill.

    Implementation:
    - LLM KV Cache: only record length, restore by truncation (zero extra VRAM)
    - Audio KV Cache: requires cloning, as generate sets it to None
    - Mel processor: save full state snapshot (including buffer)
    """

    # KV Cache length (for truncation recovery)
    llm_cache_length: int
    audio_cache_length: int

    # session state
    new_user_msg: bool
    llm_generated: bool
    llm_generate_completed: bool

    # Round management
    next_round_id: int
    pending_round_id: Optional[int]
    omni_chunk_history_length: int

    # TTS state (requires cloning, but usually small)
    tts_last_turn_tokens: Optional[torch.Tensor]

    # Streaming processor state
    audio_chunk_idx: int

    # Mel processor state snapshot (including buffer)
    mel_processor_snapshot: Optional[dict] = None

    # Audio encoder KV cache (requires cloning to ensure determinism after recovery)
    audio_past_key_values: Optional[tuple] = None

    # timestamp (for debugging)
    timestamp: float = 0.0

    # debug field: for verifying correctness of recovery
    llm_cache_checksum: Optional[float] = None  # LLM KV Cache first layer K sum
    audio_cache_checksum: Optional[float] = None  # Audio KV Cache first layer K sum
    mel_buffer_checksum: Optional[float] = None  # Mel buffer sum

    # RNG state (key: for ensuring determinism of dithering etc. after recovery)
    rng_state_cpu: Optional[torch.Tensor] = None  # torch CPU RNG state
    rng_state_cuda: Optional[torch.Tensor] = None  # torch CUDA RNG state (if on GPU)

    def summary(self) -> str:
        mel_buf_len = 0
        if self.mel_processor_snapshot:
            buf = self.mel_processor_snapshot.get("buffer")
            if buf is not None:
                mel_buf_len = len(buf)
        return (
            f"llm_cache={self.llm_cache_length}, "
            f"audio_cache={self.audio_cache_length}, "
            f"audio_chunk_idx={self.audio_chunk_idx}, "
            f"mel_buffer={mel_buf_len}, "
            f"history_len={self.omni_chunk_history_length}, "
            f"new_user_msg={self.new_user_msg}, "
            f"llm_generated={self.llm_generated}"
        )


# tts
@dataclass
class TTSSamplingParams:
    top_p: float = 0.85
    min_p: float = 0.01
    top_k: int = 25
    repetition_penalty: float = 1.05
    temperature: float = 0.8
    win_size: int = 16
    tau_r: float = 0.1


class TTSStreamingGenerator:
    """
    Streaming generator for TTS that processes chunks and yields audio tokens in real-time.

    Supported attention types:
    - full_attention: Full attention, all tokens can attend to each other
    - sliding_window: Sliding window attention, KV cache is truncated to fixed size (token_window_size)
    - sliding_recompute: Sliding recompute, only keep previous chunk and recompute with current chunk
    - reindex: Keep first chunk as sink, reindex sliding window positions via RoPE rotation
    """

    def __init__(
        self,
        model,
        temperature: float,
        eos_token: Union[int, torch.Tensor],
        chunk_size: int = 25,  # s3tokenizer 1s = 25token
        tts_last_turn_tokens: torch.Tensor = None,
        logits_processors=None,
        logits_warpers=None,
    ):
        self.tts = model
        self.device = model.device
        self.temperature = torch.tensor([temperature], dtype=torch.float, device=self.device)
        self.eos_token = (
            torch.tensor(eos_token, device=self.device) if isinstance(eos_token, int) else eos_token.to(self.device)
        )

        self.num_vq = model.num_vq
        self.num_audio_tokens = model.num_audio_tokens
        self.recomputed_chunks = model.recomputed_chunks
        self.emb_code = model.emb_code
        self.head_code = model.head_code

        # Attention type and window sizes
        self.attention_type = (
            model.attention_type
        )  # "full_attention", "sliding_window", "sliding_recompute", "reindex"
        self.chunk_window_size = model.chunk_window_size  # chunk-level window for sliding_recompute (default 2)
        self.token_window_size = model.token_window_size  # token-level window for sliding_window/reindex (default 300)

        # RoPE config (for reindex mode)
        self.rope_theta = model.model.config.rope_theta
        self.head_dim = model.model.config.hidden_size // model.model.config.num_attention_heads

        # Logits processors
        self.logits_processors = logits_processors if logits_processors is not None else []
        # Logits warpers (like TopP/TopK), separate from processors
        self.logits_warpers = logits_warpers if logits_warpers is not None else []

        # initialize state
        self.past_key_values = None
        self.text_start_pos = 0
        self.idx = -1  # start from -1, become 0 when first called
        self.all_conditions = []
        self.all_generated_tokens = []
        self.tts_last_turn_tokens = tts_last_turn_tokens
        self.spk_emb = None

        audio_bos = [self.tts.audio_bos_token_id]
        audio_bos = torch.Tensor(audio_bos).to(self.tts.emb_text.weight.device, dtype=torch.long)

        self.audio_bos_embeds = self.tts.emb_text(audio_bos).unsqueeze(0)
        self.text_eos_embed = self.tts.emb_text(
            torch.tensor(
                [self.tts.config.text_eos_token_id],
                device=self.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        ).unsqueeze(0)

        # buffer related, used to fill up chunk_size and yield to outside
        self.chunk_size = chunk_size
        self._token_buffer: List[torch.Tensor] = []

        # Chunk info tracking for sliding_recompute and reindex
        self._chunk_info: List[dict] = []
        self._total_seq_len = 0

        # Reindex mode: track sink (first chunk) length
        self._sink_kv_len = 0

    def _build_recompute_inputs(self, current_condition: torch.Tensor) -> torch.Tensor:
        """Build recompute inputs for sliding_recompute mode."""
        if len(self._chunk_info) == 0:
            return current_condition

        prev_chunk = self._chunk_info[-1]
        prev_condition = prev_chunk["condition"]
        prev_audio_tokens = prev_chunk["audio_tokens"]

        recompute_list = [prev_condition]
        if len(prev_audio_tokens) > 0:
            prev_audio_embeds = torch.cat([self.emb_code[0](tok) for tok in prev_audio_tokens], dim=1)
            recompute_list.append(prev_audio_embeds)

        recompute_list.append(current_condition)
        return torch.cat(recompute_list, dim=1)

    def _truncate_kv_cache_sliding_window(self):
        """Truncate KV cache for sliding_window mode."""
        if self.past_key_values is None:
            return

        if hasattr(self.past_key_values, "get_seq_length"):
            current_kv_len = self.past_key_values.get_seq_length()
        else:
            current_kv_len = self.past_key_values[0][0].shape[2]

        if current_kv_len <= self.token_window_size:
            return

        new_cache = DynamicCache()
        num_layers = (
            len(self.past_key_values.key_cache)
            if hasattr(self.past_key_values, "key_cache")
            else len(self.past_key_values)
        )

        for layer_idx in range(num_layers):
            if hasattr(self.past_key_values, "key_cache"):
                key = self.past_key_values.key_cache[layer_idx][:, :, -self.token_window_size :, :]
                value = self.past_key_values.value_cache[layer_idx][:, :, -self.token_window_size :, :]
            else:
                key = self.past_key_values[layer_idx][0][:, :, -self.token_window_size :, :]
                value = self.past_key_values[layer_idx][1][:, :, -self.token_window_size :, :]
            new_cache.update(key, value, layer_idx)

        self.past_key_values = new_cache

    @staticmethod
    def _apply_rope_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE rotation to tensor."""
        return x * cos + rotate_half(x) * sin

    def _compute_rope_cos_sin(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype):
        """Compute RoPE cos and sin for given positions."""
        dim_half = self.head_dim // 2
        freq_seq = torch.arange(0, dim_half, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.rope_theta ** (freq_seq / dim_half))

        # positions: [seq_len]
        angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [seq_len, dim_half]
        angles = torch.cat([angles, angles], dim=-1)  # [seq_len, head_dim]

        cos = angles.cos().to(dtype)
        sin = angles.sin().to(dtype)
        return cos, sin

    def _reindex_kv_cache(self):
        """
        Reindex KV cache for reindex mode:
        1. Keep first chunk as attention sink
        2. Keep last chunk
        3. Discard middle chunks
        4. Reindex the last chunk's key positions to be right after sink via RoPE rotation
        """
        if self.past_key_values is None or len(self._chunk_info) < 2:
            return

        # Get current KV cache length
        if hasattr(self.past_key_values, "get_seq_length"):
            current_kv_len = self.past_key_values.get_seq_length()
        else:
            current_kv_len = self.past_key_values[0][0].shape[2]

        # Calculate sink length (first chunk)
        sink_len = self._chunk_info[0]["condition_len"] + self._chunk_info[0]["audio_token_count"]

        # Last chunk length
        last_chunk = self._chunk_info[-1]
        last_chunk_len = last_chunk["condition_len"] + last_chunk["audio_token_count"]

        keep_len = sink_len + last_chunk_len

        # Get device and dtype
        device = self.past_key_values.key_cache[0].device
        dtype = self.past_key_values.key_cache[0].dtype

        if current_kv_len <= keep_len:
            last_chunk_kv_len = current_kv_len - sink_len
            if last_chunk_kv_len <= 0:
                return
            self.text_start_pos = current_kv_len
            return

        # Step 1: Truncate KV cache - keep sink and last chunk
        new_cache = DynamicCache()
        num_layers = len(self.past_key_values.key_cache)

        original_start_pos = current_kv_len - last_chunk_len
        new_start_pos = sink_len
        delta = new_start_pos - original_start_pos  # This is a scalar constant
        delta_positions = torch.full((last_chunk_len,), delta, dtype=torch.float32, device=device)

        # Compute rotation cos/sin
        cos, sin = self._compute_rope_cos_sin(delta_positions, device, dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

        for layer_idx in range(num_layers):
            key_full = self.past_key_values.key_cache[layer_idx]
            value_full = self.past_key_values.value_cache[layer_idx]

            # Extract sink and last chunk
            key_sink = key_full[:, :, :sink_len, :]
            value_sink = value_full[:, :, :sink_len, :]
            key_last = key_full[:, :, -last_chunk_len:, :]
            value_last = value_full[:, :, -last_chunk_len:, :]

            # Apply RoPE rotation to reindex key positions
            key_last_reindexed = self._apply_rope_rotation(key_last, cos, sin)

            # Concatenate sink and reindexed last chunk
            key = torch.cat([key_sink, key_last_reindexed], dim=2)
            value = torch.cat([value_sink, value_last], dim=2)

            new_cache.update(key, value, layer_idx)

        self.past_key_values = new_cache

        # Update text_start_pos to reflect new positions
        self.text_start_pos = sink_len + last_chunk_len

    @torch.inference_mode()
    def generate_with_buffer(
        self,
        condition: torch.Tensor,
        text_finished: bool = False,
        max_new_token: int = 500,
    ):
        """input a condition embedding chunk, generate audio token each time,
        and accumulate to buffer, only yield when buffer satisfies chunk_size.

        Yields:
            torch.Tensor of shape [chunk_size] (2D: [1, chunk_size])
        """
        self.idx += 1
        self.device = self.tts.device

        # if text finished, first concatenate Text EOS
        if text_finished:
            condition = torch.cat([condition, self.text_eos_embed], dim=1)

        # always concatenate Audio BOS
        condition = torch.cat([condition, self.audio_bos_embeds], dim=1).to(self.device)

        self.all_conditions.append(condition)

        # Initialize current chunk info
        current_chunk_info = {
            "condition_len": condition.shape[1],
            "audio_token_count": 0,
            "condition": condition.clone(),
            "audio_tokens": [],
        }

        # Handle different attention types
        if self.attention_type == "sliding_recompute" and self.idx >= 1:
            # sliding_recompute: discard KV cache, recompute with previous + current chunk
            self.past_key_values = None
            current_condition = self._build_recompute_inputs(condition)
            self.text_start_pos = 0
        elif self.attention_type == "reindex" and self.idx >= 1:
            # reindex: truncate KV cache keeping sink + last chunk, reindex positions via RoPE
            self._reindex_kv_cache()
            current_condition = condition
            # Always update text_start_pos based on actual KV cache length (like reference code)
            if self.past_key_values is not None:
                if hasattr(self.past_key_values, "get_seq_length"):
                    kv_len = self.past_key_values.get_seq_length()
                else:
                    kv_len = self.past_key_values[0][0].shape[2]
                self.text_start_pos = kv_len
        else:
            current_condition = condition

        condition_length = current_condition.shape[1]
        prefill_len = condition_length
        finished = torch.zeros(1, dtype=torch.bool, device=self.device)
        chunk_generated_tokens = []

        for t in range(max_new_token):
            if t == 0:
                inputs_embeds = current_condition
                pos_ids = torch.arange(
                    self.text_start_pos,
                    self.text_start_pos + condition_length,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
            else:
                last = self.all_generated_tokens[-1]
                # last: [1,1], directly as code id
                inputs_embeds = self.emb_code[0](last)
                pos_ids = torch.tensor(
                    [self.text_start_pos + prefill_len + t - 1],
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

            outputs = self.tts.model(
                position_ids=pos_ids,
                past_key_values=self.past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
            hidden_states = outputs.last_hidden_state

            # Handle KV cache based on attention type
            if self.attention_type == "sliding_window":
                self.past_key_values = outputs.past_key_values
                self._truncate_kv_cache_sliding_window()
            else:
                self.past_key_values = outputs.past_key_values

            with P.cached():
                logits = torch.empty(
                    hidden_states.size(0),
                    hidden_states.size(1),
                    self.num_audio_tokens,
                    self.num_vq,
                    dtype=torch.float,
                    device=self.device,
                )
                for num_vq_iter in range(self.num_vq):
                    x: torch.Tensor = self.head_code[num_vq_iter](hidden_states)
                    logits[..., num_vq_iter] = x
                    del x

            del hidden_states

            logits = logits[:, -1].float()

            logits = logits.permute(0, 2, 1)
            logits = logits.reshape(-1, logits.size(2))

            logits /= self.temperature

            audio_bos = len(self.all_generated_tokens) == 0 and t == 0

            if not audio_bos:
                # use generated tokens (current chunk) as input for processor/warper (align with modeling_minicpmo)
                all_generated_tokens = torch.cat(self.all_generated_tokens, dim=1).to(self.device)  # [1, T]
                for processor in self.logits_processors:
                    logits = processor(all_generated_tokens, logits)

                for warper in self.logits_warpers:
                    logits = warper(all_generated_tokens, logits)
                del all_generated_tokens

            # sample next token (only use first codebook, same as generate)
            scores = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(scores, num_samples=1)  # [(B*num_vq), 1]
            next_id = idx_next.view(-1, self.num_vq)[:, 0:1]  # only take first codebook → [B, 1]
            del scores

            if next_id.eq(
                self.eos_token
            ).any():  # generated audio eos token, means this chunk is finished, no longer generate new tokens
                finished[:] = True
            else:  # eos token cannot be added to buffer, he does not speak.
                # convert next_id to correct shape [1, 1], no num_vq dimension
                if next_id.dim() == 0:  # if scalar
                    next_tok = next_id.unsqueeze(0).unsqueeze(0)  # [1, 1]
                elif next_id.dim() == 1:  # if 1D [1]
                    next_tok = next_id.unsqueeze(0)  # [1, 1]
                else:
                    next_tok = next_id

                self.all_generated_tokens.append(next_tok)
                chunk_generated_tokens.append(next_tok)

                # Update chunk info for sliding_recompute
                current_chunk_info["audio_tokens"].append(next_tok.clone())
                current_chunk_info["audio_token_count"] += 1

                self._token_buffer.append(next_tok)

            if len(self._token_buffer) == 0:
                # case 1: if last text chunk, yield None
                if text_finished:
                    yield torch.empty(1, 0, dtype=torch.long, device=self.device), True
                    break
                # case 2: if not last text chunk, break directly
                else:
                    break
            else:  # buffer has something
                # case 1: if buffer is larger/equal to chunk_size, yield out
                if len(self._token_buffer) >= self.chunk_size:
                    batch = torch.cat(self._token_buffer[: self.chunk_size], dim=1)  # [1, chunk_size]
                    yield batch, False  # → [1, chunk_size]
                    # discard yielded part
                    self._token_buffer = self._token_buffer[self.chunk_size :]

                # case 2: if buffer is smaller than chunk_size
                else:
                    # if generation finished, and is the last text chunk, yield all remaining tokens, then break
                    if finished.all():
                        if text_finished:
                            batch = torch.cat(self._token_buffer, dim=1)  # [1, chunk_size]
                            yield batch, True  # → [1, chunk_size]
                            self._token_buffer = []
                            break
                        else:
                            # not the last text chunk, need to wait for next text chunk to fill up buffer, then this call ends
                            break
                    else:  # generation of this audio chunk is not finished, continue generating
                        continue

        # Save current chunk info for sliding_recompute and reindex
        self._chunk_info.append(current_chunk_info)
        self._total_seq_len += condition.shape[1] + len(chunk_generated_tokens)

        # Update text_start_pos based on attention type
        if self.attention_type == "sliding_recompute":
            # sliding_recompute: will be reset at next chunk start, update normally here
            self.text_start_pos += prefill_len + len(chunk_generated_tokens)
        elif self.attention_type == "reindex":
            # reindex: position based on actual KV cache length (positions have been reindexed to be continuous)
            if self.past_key_values is not None:
                if hasattr(self.past_key_values, "get_seq_length"):
                    self.text_start_pos = self.past_key_values.get_seq_length()
                else:
                    self.text_start_pos = self.past_key_values[0][0].shape[2]
            else:
                self.text_start_pos += condition.shape[1] + len(chunk_generated_tokens)
        else:
            self.text_start_pos += condition.shape[1] + len(chunk_generated_tokens)
        # note: remaining tokens in buffer will be kept, and accumulated next time


# sliding window
@dataclass
class StreamingWindowConfig:
    text_window_high_tokens: int = 8000
    text_window_low_tokens: int = 6000


@dataclass
class DuplexWindowConfig:
    """duplex sliding window configuration

    sliding window mode:
    - "off": disable sliding window
    - "basic": basic sliding window (trigger by cache length)
    - "context": sliding window with context (trigger by unit number, preserve generated text to previous)
    """

    # sliding window mode
    sliding_window_mode: str = "off"  # "off" / "basic" / "context"

    # basic sliding window parameters
    basic_window_high_tokens: int = 8000  # high watermark: trigger sliding window when exceeded
    basic_window_low_tokens: int = 6000  # low watermark: keep to this value after sliding window

    # context sliding window parameters
    context_previous_max_tokens: int = 500  # previous maximum token number
    context_max_units: int = 24  # maximum unit number (trigger sliding window when exceeded)

    # verification mode (for comparison test)
    verify_mode: bool = False  # whether to enable verification log


def as_dynamic_cache(past_key_values):
    """Convert legacy tuple cache to DynamicCache if needed."""
    if isinstance(past_key_values, DynamicCache):
        return past_key_values

    if isinstance(past_key_values, tuple):
        return DynamicCache.from_legacy_cache(past_key_values)

    return past_key_values


def get_kv_cache_length(cache) -> int:
    """Get the sequence length of a KV cache.

    Args:
        cache: DynamicCache or tuple-based cache

    Returns:
        The number of tokens in the cache
    """
    if cache is None:
        return 0

    if isinstance(cache, DynamicCache):
        if not cache.key_cache or not cache.key_cache[0].numel():
            return 0
        return cache.key_cache[0].shape[-2]

    if isinstance(cache, tuple):
        return cache[0][0].shape[2]

    return 0


def get_rotary_cos_sin(
    head_dim: int,
    positions: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    rope_theta: float = 10000.0,
    inv_freq_cache: Optional[Dict[Tuple, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE cos and sin components for given positions.

    Args:
        head_dim: Dimension of each attention head
        positions: Position indices tensor
        device: Target device
        dtype: Target dtype
        rope_theta: RoPE base frequency (default 10000.0)
        inv_freq_cache: Optional cache dict for inverse frequencies

    Returns:
        Tuple of (cos, sin) tensors with shape [1, 1, seq_len, head_dim]
    """
    cache_key = (head_dim, device)

    inv_freq = inv_freq_cache.get(cache_key) if inv_freq_cache is not None else None
    if inv_freq is None or inv_freq.device != device or inv_freq.shape[0] != head_dim // 2:
        exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
        inv_freq = 1.0 / (rope_theta**exponent)
        if inv_freq_cache is not None:
            inv_freq_cache[cache_key] = inv_freq

    positions = positions.to(device=device, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Use cat instead of repeat_interleave, consistent with model's original RotaryEmbedding
    # Original: emb = torch.cat((freqs, freqs), dim=-1) -> [f0, f1, ..., f_{d/2}, f0, f1, ..., f_{d/2}]
    cos_full = torch.cat([cos, cos], dim=-1).to(dtype=dtype)
    sin_full = torch.cat([sin, sin], dim=-1).to(dtype=dtype)
    cos_full = cos_full.unsqueeze(0).unsqueeze(0)
    sin_full = sin_full.unsqueeze(0).unsqueeze(0)
    return cos_full, sin_full


def realign_rotary_suffix(
    suffix_keys: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rope_theta: float = 10000.0,
    inv_freq_cache: Optional[Dict[Tuple, torch.Tensor]] = None,
) -> torch.Tensor:
    """Realign RoPE position encoding after cache eviction.

    When tokens are dropped from the middle of a cache, the suffix tokens
    need their RoPE embeddings recalculated with new position indices.

    Args:
        suffix_keys: Key tensor to realign, shape [batch, heads, seq_len, head_dim]
        old_positions: Original position indices
        new_positions: New position indices after eviction
        rope_theta: RoPE base frequency
        inv_freq_cache: Optional cache dict for inverse frequencies

    Returns:
        Realigned key tensor with same shape as input
    """
    if suffix_keys.numel() == 0:
        return suffix_keys

    head_dim = suffix_keys.shape[-1]
    device = suffix_keys.device
    dtype = suffix_keys.dtype

    # Compute old position cos/sin
    cos_old, sin_old = get_rotary_cos_sin(head_dim, old_positions, device, dtype, rope_theta, inv_freq_cache)

    # Inverse transform: recover original key
    base = cos_old * suffix_keys - sin_old * rotate_half(suffix_keys)

    # Compute new position cos/sin
    cos_new, sin_new = get_rotary_cos_sin(head_dim, new_positions, device, dtype, rope_theta, inv_freq_cache)

    # Forward transform: re-encode with new positions
    return cos_new * base + sin_new * rotate_half(base)


def drop_tokens_from_cache(
    cache: Optional[DynamicCache | Tuple],
    length: int,
    preserve: int,
    position_offset: int,
    rope_theta: float = 10000.0,
    inv_freq_cache: Optional[Dict[Tuple, torch.Tensor]] = None,
) -> Tuple[Optional[DynamicCache], int, bool]:
    """Drop tokens from a KV cache while preserving system prompt.

    Removes tokens in the range [preserve, preserve + length) from the cache,
    realigning RoPE embeddings for the suffix.

    Args:
        cache: DynamicCache or tuple-based cache (will be converted to DynamicCache)
        length: Number of tokens to drop
        preserve: Number of tokens to preserve at the start (system prompt)
        position_offset: Current position offset for RoPE calculation
        rope_theta: RoPE base frequency
        inv_freq_cache: Optional cache dict for inverse frequencies

    Returns:
        Tuple of (cache, new_position_offset, success)
        Note: Tuple cache will be converted to DynamicCache. Modification is in-place.
    """
    if cache is None or length <= 0:
        return cache, position_offset, False

    cache = as_dynamic_cache(cache)

    total_len = get_kv_cache_length(cache)
    if total_len <= 0:
        return cache, position_offset, False

    preserve = min(preserve, total_len)
    available = total_len - preserve

    if available < length:
        logger.warning(
            "Cannot drop %d tokens: only %d available (total=%d, preserve=%d)",
            length,
            available,
            total_len,
            preserve,
        )
        return cache, position_offset, False

    suffix_len = total_len - preserve - length
    # note: after RoPE reindex, the position of cache has been compressed (from preserve start)
    # so here should not add position_offset, but use the actual layout of current cache
    suffix_offset = preserve + length  # suffix current position in cache
    prefix_offset = preserve  # suffix new position (follow preserve)

    # Prepare position tensors for RoPE realignment
    old_positions = None
    new_positions = None
    if suffix_len > 0:
        device = cache.key_cache[0].device
        old_positions = torch.arange(
            suffix_offset,
            suffix_offset + suffix_len,
            device=device,
            dtype=torch.long,
        )
        new_positions = torch.arange(
            prefix_offset,
            prefix_offset + suffix_len,
            device=device,
            dtype=torch.long,
        )

    keep_len = total_len - length

    # Process each layer (in-place modification)
    for layer_idx in range(len(cache.key_cache)):
        key_tensor = cache.key_cache[layer_idx]
        value_tensor = cache.value_cache[layer_idx]

        if not key_tensor.numel():
            continue

        # Preserve prefix (system prompt)
        prefix_keys = key_tensor[:, :, :preserve, :]
        prefix_values = value_tensor[:, :, :preserve, :]

        if suffix_len > 0:
            # Keep and realign suffix
            suffix_keys = key_tensor[:, :, preserve + length :, :]
            suffix_values = value_tensor[:, :, preserve + length :, :]

            if old_positions is not None and new_positions is not None and suffix_keys.numel():
                suffix_keys = realign_rotary_suffix(
                    suffix_keys,
                    old_positions,
                    new_positions,
                    rope_theta,
                    inv_freq_cache,
                )

            cache.key_cache[layer_idx] = torch.cat([prefix_keys, suffix_keys], dim=-2).contiguous()
            cache.value_cache[layer_idx] = torch.cat([prefix_values, suffix_values], dim=-2).contiguous()
        else:
            cache.key_cache[layer_idx] = prefix_keys.contiguous()
            cache.value_cache[layer_idx] = prefix_values.contiguous()

    cache.crop(keep_len)
    cache._seen_tokens = max(keep_len, 0)

    new_offset = position_offset + length
    logger.debug("Dropped %d tokens from cache, new length=%d", length, keep_len)

    return cache, new_offset, True


# stream decoder
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("inf")):
    logits = logits.clone()

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # keep the first token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value

    return logits


class StreamDecoder:
    def __init__(self, llm, tokenizer, special_token_ids=None, forbidden_token_ids=None):
        self.m = llm
        self.tokenizer = tokenizer
        self.listen_id = self.tokenizer.eos_token_id

        self.chunk_eos_id = self.tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        self.chunk_tts_eos_id = self.tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        self.turn_eos_id = self.tokenizer.convert_tokens_to_ids("<|turn_eos|>")
        self.speak_id = self.tokenizer.convert_tokens_to_ids("<|speak|>")

        self.special_token_ids = special_token_ids if special_token_ids is not None else []

        # cache special tokens (used for context sliding window filtering)
        self._all_special_ids = set()
        self._all_special_tokens_text = set()
        if self.tokenizer:
            if hasattr(self.tokenizer, "all_special_ids"):
                self._all_special_ids = set(self.tokenizer.all_special_ids)
            if hasattr(self.tokenizer, "all_special_tokens"):
                self._all_special_tokens_text = set(self.tokenizer.all_special_tokens)

        custom_special_tokens = [
            "<unit>",
            "</unit>",
            "<image>",
            "</image>",
            "<slice>",
            "</slice>",
            "<|listen|>",
            "<|speak|>",
            "<|tts_bos|>",
            "<|tts_eos|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<|chunk_eos|>",
            "<|chunk_tts_eos|>",
            "<|turn_eos|>",
            "<|audio_start|>",
            "<|audio_end|>",
        ]
        self._all_special_tokens_text.update(custom_special_tokens)
        for token in custom_special_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                self._all_special_ids.add(token_id)

        if forbidden_token_ids is None:
            self.forbidden_token_ids = []
        elif isinstance(forbidden_token_ids, int):
            self.forbidden_token_ids = [self.forbidden_token_ids]
        else:
            self.forbidden_token_ids = forbidden_token_ids
        self.forbidden_token_ids.append(self.chunk_eos_id)

        assert isinstance(self.forbidden_token_ids, list)

        self.cache = None
        self.context = ""
        self.generated_tokens = []  # track generated tokens
        self.generated_special_tokens = []  # track generated special tokens
        self.reset()
        self.embeds = None
        self.system_embeds = None

        # sliding window related states
        self._unit_history: List[Dict[str, Any]] = []
        self._next_unit_id: int = 0
        self._pending_unit_id: Optional[int] = None
        self._pending_unit_start_cache_len: int = 0
        self._system_preserve_length: int = 0
        self._position_offset: int = 0
        self._window_config = DuplexWindowConfig()
        self._window_enabled: bool = True
        self._rope_inv_freq_cache: Dict[Tuple, torch.Tensor] = {}

        # context preserving sliding window states
        # initial cache layout: [prefix] [suffix] [units...]
        # after first sliding window: [prefix] [previous_marker + content] [suffix] [units...]
        #                              fixed     dynamic sliding region      fixed
        self._preserve_prefix_length: int = 0  # original prefix length (fixed)
        self._previous_content_length: int = 0  # previous content length (dynamic, including marker)
        self._suffix_token_ids: List[int] = []  # suffix token ids (e.g. <|im_end|>)

        # previous marker (added dynamically after first sliding window)
        self._previous_marker: str = "\n\nprevious: "  # fixed prefix marker
        self._previous_marker_token_ids: List[int] = []  # marker token ids (initialized)
        self._has_previous: bool = False  # whether previous marker has been added

        # previous content
        self._previous_text: str = ""  # accumulated generated text (without marker)
        self._previous_token_ids: List[int] = []  # previous full token ids (including marker)

        # validation statistics
        self._sliding_event_count: int = 0  # sliding window trigger count
        self._total_dropped_tokens: int = 0  # total dropped token count
        self._total_dropped_units: int = 0  # total dropped unit count

    def sliding_embeds(self):
        # tmp = system_embeds
        # tmp +-》 embeds after 5s
        # reset
        # feed
        pass

    def reset(self):
        self.context = ""
        self.cache = None
        self.generated_tokens = []
        self.generated_special_tokens = []
        self.embeds = None
        self.system_embeds = None

        # sliding window state reset
        len(self._unit_history) if hasattr(self, "_unit_history") else 0
        self._unit_history = []
        self._next_unit_id = 0
        self._pending_unit_id = None
        self._pending_unit_start_cache_len = 0
        self._system_preserve_length = 0
        self._position_offset = 0
        self._rope_inv_freq_cache = {}

        # context preserving sliding window state reset
        self._preserve_prefix_length = 0
        self._previous_content_length = 0
        self._suffix_token_ids = []
        self._previous_marker = "\n\nprevious: "
        self._previous_marker_token_ids = []
        self._has_previous = False
        self._previous_text = ""
        self._previous_token_ids = []

        # validation statistics
        self._sliding_event_count = 0  # sliding window trigger count
        self._total_dropped_tokens = 0  # total dropped token count
        self._total_dropped_units = 0  # total dropped unit count

    def get_cache_length(self) -> int:
        if self.cache is None:
            return 0
        if isinstance(self.cache, DynamicCache):
            if len(self.cache.key_cache) > 0 and self.cache.key_cache[0].numel() > 0:
                return self.cache.key_cache[0].shape[2]
            return 0
        # Tuple cache format
        return self.cache[0][0].shape[2]

    def get_total_generated_tokens(self) -> int:
        return sum(len(u.get("generated_tokens", [])) for u in self._unit_history)

    def register_unit_start(self) -> int:
        self._pending_unit_id = self._next_unit_id
        self._pending_unit_start_cache_len = self.get_cache_length()
        return self._pending_unit_id

    def register_unit_end(
        self,
        input_type: str,
        generated_tokens: Optional[List[int]] = None,
        is_listen: bool = False,
        generated_text: Optional[str] = None,
    ):
        """Call when unit ends, record unit information

        Should be called after feeding </unit> token

        Args:
            input_type: "audio" / "video" / "omni" / "system"
            generated_tokens: tokens generated by the unit (token ids)
            is_listen: whether the unit is in listen state
            generated_text: text generated by the unit (used for context preserving mode)
        """
        if self._pending_unit_id is None:
            logger.warning("register_unit_end called without register_unit_start")
            return

        # calculate the length of the unit
        current_cache_len = self.get_cache_length()
        unit_len = current_cache_len - self._pending_unit_start_cache_len

        if unit_len > 0:
            entry = {
                "unit_id": self._pending_unit_id,
                "length": unit_len,
                "type": input_type,
                "generated_tokens": generated_tokens or [],
                "generated_text": generated_text or "",  # used for context preserving mode
                "is_listen": is_listen,
            }
            self._unit_history.append(entry)

        self._pending_unit_id = None
        self._pending_unit_start_cache_len = 0
        self._next_unit_id += 1

    def register_system_prompt(self):
        """Call after system prompt prefill, record preserve length"""
        self._system_preserve_length = self.get_cache_length()

    # sliding window core methods

    def _get_rope_theta(self) -> float:
        """get model rope_theta configuration"""
        return float(getattr(self.m.config, "rope_theta", 10000.0))

    def _drop_tokens_from_cache(self, length: int) -> bool:
        """remove specified number of tokens from cache (protect system prompt)

        remove tokens in the range [preserve, preserve + length)
        supports DynamicCache and tuple cache formats
        """
        if self.cache is None or length <= 0:
            return False

        "DynamicCache" if isinstance(self.cache, DynamicCache) else "TupleCache"
        self.get_cache_length()

        new_cache, new_offset, success = drop_tokens_from_cache(
            cache=self.cache,
            length=length,
            preserve=self._system_preserve_length,
            position_offset=self._position_offset,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )
        if success:
            self.cache = new_cache  # For DynamicCache this is the same object (in-place)
            self._position_offset = new_offset

        return success

    def _drop_unit(self, unit_id: int) -> bool:
        """remove specified unit"""
        entries = [u for u in self._unit_history if u["unit_id"] == unit_id]
        if not entries:
            return False

        total_len = sum(e["length"] for e in entries)
        if total_len <= 0:
            for e in entries:
                self._unit_history.remove(e)
            return False

        if not self._drop_tokens_from_cache(total_len):
            return False

        for e in entries:
            self._unit_history.remove(e)

        return True

    def _drop_next_unit(self) -> bool:
        """remove the earliest non-system unit"""
        for entry in self._unit_history:
            unit_id = entry.get("unit_id")
            if unit_id is None:
                continue
            # skip system type
            if entry.get("type") == "system":
                continue
            if self._drop_unit(unit_id):
                return True
        return False

    def enforce_window(self) -> bool:
        """enforce sliding window strategy (same as single-mode, only look at cache length)

        when cache length exceeds high water line, loop to remove the earliest unit,
        until cache length drops below the low water line.
        """
        if not self._window_enabled:
            return False

        cfg = self._window_config
        cache_len_before = self.get_cache_length()

        if cache_len_before <= cfg.basic_window_high_tokens:
            return False  # not above high water line, no trigger

        dropped_count = 0
        cache_len = cache_len_before
        while cache_len > cfg.basic_window_low_tokens:
            if not self._drop_next_unit():
                break
            dropped_count += 1
            cache_len = self.get_cache_length()

        if dropped_count > 0:
            # update statistics counters
            self._sliding_event_count += 1
            self._total_dropped_tokens += cache_len_before - cache_len
            self._total_dropped_units += dropped_count

            # consistency check
            expected = self._system_preserve_length + sum(u["length"] for u in self._unit_history)
            is_consistent = expected == cache_len
            if not is_consistent:
                logger.error(
                    "CONSISTENCY ERROR! preserve=%d + sum(units)=%d != cache=%d, offset=%d",
                    self._system_preserve_length,
                    sum(u["length"] for u in self._unit_history),
                    cache_len,
                    self._position_offset,
                )

        return dropped_count > 0

    # context preserving sliding window methods

    def register_system_prompt_with_context(
        self,
        suffix_token_ids: Optional[List[int]] = None,
        context_previous_marker: str = "\n\nprevious: ",
    ):
        """register system prompt (with context preserving mode)

        initial cache layout: [prefix] [suffix] [units...]
        after first sliding window: [prefix] [context_previous_marker + content] [suffix] [units...]

        when calling this method, cache should only have prefix (without previous marker)
        suffix will be fed in later

        Args:
            suffix_token_ids: suffix token ids (e.g. id of <|im_end|>)
            context_previous_marker: previous marker prefix, e.g. "\\n\\nprevious: "
        """
        # prefix = current cache content (fixed, without previous marker)
        self._preserve_prefix_length = self.get_cache_length()
        self._previous_content_length = 0  # initially no previous content
        self._suffix_token_ids = suffix_token_ids or []
        # total preserve length = prefix + suffix (initially no previous)
        self._system_preserve_length = self._preserve_prefix_length + len(self._suffix_token_ids)

        # initialize previous related states
        self._previous_marker = context_previous_marker
        self._previous_marker_token_ids = (
            self.tokenizer.encode(context_previous_marker, add_special_tokens=False) if self.tokenizer else []
        )
        self._has_previous = False
        self._previous_text = ""
        self._previous_token_ids = []

    def _extract_generated_text(self, units: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
        """extract generated text and token ids from units

        Args:
            units: list of units to extract

        Returns:
            (text, token_ids): concatenated text and token ids (filtered out special tokens)
        """
        text_parts = []
        token_ids = []

        for u in units:
            # only keep generated content of non-listen units
            if u.get("is_listen", False):
                continue
            gen_text = u.get("generated_text", "")
            gen_tokens = u.get("generated_tokens", [])

            # filter out special tokens from text
            if gen_text:
                clean_text = gen_text
                for st in self._all_special_tokens_text:
                    clean_text = clean_text.replace(st, "")
                if clean_text.strip():
                    text_parts.append(clean_text)

            # filter out special tokens
            if gen_tokens:
                filtered_tokens = [t for t in gen_tokens if t not in self._all_special_ids]
                token_ids.extend(filtered_tokens)

        return "".join(text_parts), token_ids

    def _rebuild_cache_with_previous(
        self,
        new_previous_tokens: List[int],
        units_to_keep_len: Optional[int] = None,
    ) -> bool:
        """rebuild cache, insert new previous content between prefix and suffix

        cache layout change:
        [prefix] [old_prev] [suffix] [old_units]  →  [prefix] [new_prev] [suffix] [remaining_units]

        Args:
            new_previous_tokens: new previous token ids
            units_to_keep_len: length of units to keep (from cache end backwards)
                                if None, calculate based on unit_history

        Returns:
            whether successful rebuild
        """
        if self.cache is None:
            return False

        old_previous_len = self._previous_content_length
        new_previous_len = len(new_previous_tokens)
        suffix_len = len(self._suffix_token_ids)
        total_cache_len = self.get_cache_length()

        # calculate length of units to keep
        if units_to_keep_len is None:
            units_to_keep_len = sum(u["length"] for u in self._unit_history)

        # special case: if previous is unchanged (new and old are empty), no need to rebuild prefix+suffix part of cache
        # but still need to reindex units RoPE (because a unit was deleted, position changed)
        if new_previous_len == 0 and old_previous_len == 0:
            # cache layout: [prefix(7)] [suffix(1)] [units...]
            # only keep prefix + suffix + remaining_units
            preserve_len = self._preserve_prefix_length + suffix_len

            # simply slice cache: [prefix+suffix] + [remaining_units]
            # remaining_units in cache end
            if units_to_keep_len > 0:
                # [0:preserve_len] + [total-units_to_keep_len:total]
                prefix_suffix_cache = self._slice_cache(0, preserve_len)
                units_cache = self._slice_cache(total_cache_len - units_to_keep_len, None)

                # calculate number of dropped tokens
                dropped_tokens = total_cache_len - preserve_len - units_to_keep_len

                # reindex units RoPE: position from (preserve_len + dropped_tokens) to preserve_len
                # note: no position_offset, because cache position has been compressed (from 0 start)
                if dropped_tokens > 0:
                    old_start = preserve_len + dropped_tokens
                    new_start = preserve_len
                    units_cache = self._reindex_rope_for_cache(units_cache, old_start, new_start, units_to_keep_len)

                self.cache = self._concat_caches(prefix_suffix_cache, units_cache)
            else:
                self.cache = self._slice_cache(0, preserve_len)

            return True

        # 1. get prefix cache (fixed)
        prefix_end = self._preserve_prefix_length
        prefix_cache = self._slice_cache(0, prefix_end)

        # 2. get units cache to keep (from end)
        units_start_in_old_cache = total_cache_len - units_to_keep_len
        units_cache = None
        if units_to_keep_len > 0:
            units_cache = self._slice_cache(units_start_in_old_cache, None)

        # 3. calculate new previous + suffix cache (needs forward)
        # merge previous tokens and suffix tokens
        prev_suffix_tokens = new_previous_tokens + self._suffix_token_ids
        prev_suffix_len = len(prev_suffix_tokens)

        new_prefix_prev_suffix_cache = prefix_cache
        if prev_suffix_len > 0:
            # Embed tokens
            prev_suffix_embeds = self.embed_tokens(prev_suffix_tokens)
            # calculate start position (after prefix)
            start_pos = self._preserve_prefix_length + self._position_offset

            # forward calculate KV cache
            with torch.no_grad():
                device = prev_suffix_embeds.device
                position_ids = torch.arange(
                    start_pos,
                    start_pos + prev_suffix_len,
                    device=device,
                ).unsqueeze(0)

                # use prefix cache as past_key_values
                outputs = self.m(
                    inputs_embeds=(
                        prev_suffix_embeds.unsqueeze(0) if prev_suffix_embeds.dim() == 2 else prev_suffix_embeds
                    ),
                    position_ids=position_ids,
                    past_key_values=prefix_cache,
                    use_cache=True,
                    return_dict=True,
                )
                # new cache contains prefix + new_previous + suffix
                new_prefix_prev_suffix_cache = outputs.past_key_values

        # 4. adjust units cache RoPE
        # new layout: [prefix] [new_prev] [suffix] [units]
        # note: no position_offset, because cache position has been compressed (from 0 start)
        new_system_total = prefix_end + new_previous_len + suffix_len
        if units_cache is not None and self._get_cache_len(units_cache) > 0:
            old_start = units_start_in_old_cache
            new_start = new_system_total

            if old_start != new_start:
                units_cache = self._reindex_rope_for_cache(units_cache, old_start, new_start, units_to_keep_len)

        # 5. concatenate new cache
        if units_cache is not None and self._get_cache_len(units_cache) > 0:
            self.cache = self._concat_caches(new_prefix_prev_suffix_cache, units_cache)
        else:
            self.cache = new_prefix_prev_suffix_cache

        # 6. update length
        self._previous_content_length = new_previous_len
        # total preserve length = prefix + previous + suffix
        self._system_preserve_length = prefix_end + new_previous_len + suffix_len

        # print detailed cache layout information
        self._previous_text[:50] + "..." if len(self._previous_text) > 50 else self._previous_text
        self.tokenizer.decode(self._suffix_token_ids) if self._suffix_token_ids else ""
        return True

    def _slice_cache(self, start: int, end: Optional[int], clone: bool = True):
        """slice cache

        Args:
            start: start position
            end: end position (None means to end)
            clone: whether to clone (default True, to prevent shared memory issues)
        """
        if self.cache is None:
            return None
        if isinstance(self.cache, DynamicCache):
            # DynamicCache
            new_key_cache = [
                k[:, :, start:end, :].clone() if clone else k[:, :, start:end, :] for k in self.cache.key_cache
            ]
            new_value_cache = [
                v[:, :, start:end, :].clone() if clone else v[:, :, start:end, :] for v in self.cache.value_cache
            ]
            new_cache = DynamicCache()
            new_cache.key_cache = new_key_cache
            new_cache.value_cache = new_value_cache
            return new_cache
        else:
            # Tuple cache
            if clone:
                return tuple(
                    (layer[0][:, :, start:end, :].clone(), layer[1][:, :, start:end, :].clone())
                    for layer in self.cache
                )
            else:
                return tuple((layer[0][:, :, start:end, :], layer[1][:, :, start:end, :]) for layer in self.cache)

    @staticmethod
    def _get_cache_len(cache) -> int:
        if cache is None:
            return 0
        if isinstance(cache, DynamicCache):
            if len(cache.key_cache) > 0 and cache.key_cache[0].numel() > 0:
                return cache.key_cache[0].shape[2]
            return 0

        if cache and cache[0] and cache[0][0] is not None:
            return cache[0][0].shape[2]
        return 0

    @staticmethod
    def _concat_caches(cache1, cache2):
        if cache1 is None:
            return cache2
        if cache2 is None:
            return cache1

        if isinstance(cache1, DynamicCache):
            new_cache = DynamicCache()
            new_cache.key_cache = [torch.cat([k1, k2], dim=2) for k1, k2 in zip(cache1.key_cache, cache2.key_cache)]
            new_cache.value_cache = [
                torch.cat([v1, v2], dim=2) for v1, v2 in zip(cache1.value_cache, cache2.value_cache)
            ]
            return new_cache
        else:
            return tuple(
                (
                    torch.cat([layer1[0], layer2[0]], dim=2),
                    torch.cat([layer1[1], layer2[1]], dim=2),
                )
                for layer1, layer2 in zip(cache1, cache2)
            )

    def _reindex_rope_for_cache(self, cache, old_start: int, new_start: int, length: int):
        """reindex RoPE position for cache"""
        if cache is None or length <= 0:
            return cache

        if isinstance(cache, DynamicCache):
            device = cache.key_cache[0].device if cache.key_cache else None
        else:
            device = cache[0][0].device if cache and cache[0] else None

        if device is None:
            return cache

        old_positions = torch.arange(old_start, old_start + length, device=device, dtype=torch.long)
        new_positions = torch.arange(new_start, new_start + length, device=device, dtype=torch.long)

        rope_theta = self._get_rope_theta()

        if isinstance(cache, DynamicCache):
            new_key_cache = []
            for k in cache.key_cache:
                new_k = realign_rotary_suffix(k, old_positions, new_positions, rope_theta, self._rope_inv_freq_cache)
                new_key_cache.append(new_k)
            cache.key_cache = new_key_cache
            return cache
        else:
            new_cache = []
            for layer in cache:
                new_k = realign_rotary_suffix(
                    layer[0], old_positions, new_positions, rope_theta, self._rope_inv_freq_cache
                )
                new_cache.append((new_k, layer[1]))
            return tuple(new_cache)

    def _update_previous(
        self,
        new_text: str,
        new_tokens: List[int],
        max_tokens: int,
    ) -> None:
        """update previous context (also update cache)

        when first sliding window, dynamically add marker + text, subsequent sliding window append text
        when content exceeds max_tokens, truncate content (keep marker)
        rebuild cache to maintain consistency

        Args:
            new_text: new text
            new_tokens: new token ids
            max_tokens: previous content maximum token count (without marker)
        """
        marker_len = len(self._previous_marker_token_ids)
        tokens_to_drop = 0

        # if no new content, do not add marker, but still need to rebuild cache
        if not new_tokens and not new_text:
            # still need to rebuild cache (because a unit was deleted)
            self._rebuild_cache_with_previous(self._previous_token_ids)
            return

        if not self._has_previous:
            # when first has actual content: add marker + text
            self._previous_text = new_text
            self._previous_token_ids = self._previous_marker_token_ids.copy() + new_tokens
            self._has_previous = True
        else:
            # subsequent sliding window: append text to previous
            self._previous_text += new_text
            self._previous_token_ids.extend(new_tokens)

        # calculate token count of content (without marker)
        content_token_count = len(self._previous_token_ids) - marker_len

        # check if need to truncate content (keep marker)
        if content_token_count > max_tokens:
            # truncate left content, keep marker + latest max_tokens content
            tokens_to_drop = content_token_count - max_tokens
            # keep marker + truncated content
            content_tokens = self._previous_token_ids[marker_len + tokens_to_drop :]
            self._previous_token_ids = self._previous_marker_token_ids.copy() + content_tokens
            # redecode text (only decode content part)
            try:
                self._previous_text = self.tokenizer.decode(
                    content_tokens,
                    skip_special_tokens=True,
                )
            except Exception as e:
                logger.warning("_update_previous: decode failed: %s", e)

        # rebuild cache
        self._rebuild_cache_with_previous(self._previous_token_ids)

    def _drop_unit_with_context(
        self,
        unit_id: int,
        max_previous_tokens: int,
    ) -> Tuple[bool, str, List[int]]:
        """remove specified unit and return its generated content (for context preserving)

        process:
        1. extract generated content of unit
        2. remove unit from cache (without prefix+previous)
        3. append generated content to previous
        4. rebuild cache (in _update_previous)

        Args:
            unit_id: unit ID to remove
            max_previous_tokens: previous maximum token count

        Returns:
            (success, extracted_text, extracted_tokens): whether successful, extracted text and tokens
        """
        entries = [u for u in self._unit_history if u["unit_id"] == unit_id]
        if not entries:
            return False, "", []

        # extract generated content
        extracted_text, extracted_tokens = self._extract_generated_text(entries)

        # calculate total length
        total_len = sum(e["length"] for e in entries)
        if total_len <= 0:
            for e in entries:
                self._unit_history.remove(e)
            return False, extracted_text, extracted_tokens

        self.get_cache_length()

        # remove from unit_history (record for later processing)
        for e in entries:
            self._unit_history.remove(e)

        # note: here no longer call _drop_tokens_from_cache
        # because _update_previous will rebuild the entire cache

        # update previous (also rebuild cache)
        self._update_previous(extracted_text, extracted_tokens, max_previous_tokens)

        return True, extracted_text, extracted_tokens

    def _drop_next_unit_with_context(self, max_previous_tokens: int) -> bool:
        """remove the earliest non-system unit (with context preserving)"""
        for entry in self._unit_history:
            unit_id = entry.get("unit_id")
            if unit_id is None:
                continue
            if entry.get("type") == "system":
                continue
            success, _, _ = self._drop_unit_with_context(unit_id, max_previous_tokens)
            if success:
                return True
        return False

    def enforce_window_with_context(self) -> bool:
        """context preserving sliding window execution

        when unit count exceeds max_units, remove the earliest unit,
        and accumulate its generated content to previous.
        Cache will be automatically rebuilt in _update_previous.

        Returns:
            whether sliding window is executed
        """
        if not self._window_enabled:
            return False

        cfg = self._window_config

        if cfg.sliding_window_mode != "context":
            # if not context mode, fallback to basic sliding window
            return self.enforce_window()

        cache_len_before = self.get_cache_length()
        units_before = len(self._unit_history)

        # context preserving mode: only check if unit count exceeds limit
        # (previous exceeds limit in _update_previous will automatically truncate left)
        if units_before <= cfg.context_max_units:
            return False

        # sliding window loop: remove unit until count ≤ max_units
        dropped_count = 0
        while len(self._unit_history) > cfg.context_max_units:
            if not self._drop_next_unit_with_context(cfg.context_previous_max_tokens):
                break

            dropped_count += 1

        cache_len_after = self.get_cache_length()

        if dropped_count > 0:
            # update statistics counter
            self._sliding_event_count += 1
            self._total_dropped_tokens += cache_len_before - cache_len_after
            self._total_dropped_units += dropped_count

            # consistency check
            self._system_preserve_length + sum(u["length"] for u in self._unit_history)

        return dropped_count > 0

    def get_previous_context(self) -> Tuple[str, List[int]]:
        """get current accumulated previous context

        Returns:
            (previous_text, previous_token_ids): current accumulated text and token ids
        """
        return self._previous_text, self._previous_token_ids.copy()

    def get_window_stats(self) -> Dict[str, Any]:
        """get sliding window statistics"""
        unit_lengths = [u["length"] for u in self._unit_history]
        return {
            "cache_length": self.get_cache_length(),
            "unit_count": len(self._unit_history),
            "unit_lengths": unit_lengths,
            "unit_total_length": sum(unit_lengths),
            "system_preserve_length": self._system_preserve_length,
            "position_offset": self._position_offset,
            "window_enabled": self._window_enabled,
            "total_generated_tokens": self.get_total_generated_tokens(),
            "pending_unit_id": self._pending_unit_id,
            "next_unit_id": self._next_unit_id,
            "config": {
                "sliding_window_mode": self._window_config.sliding_window_mode,
                "basic_window_high_tokens": self._window_config.basic_window_high_tokens,
                "basic_window_low_tokens": self._window_config.basic_window_low_tokens,
                "context_previous_max_tokens": self._window_config.context_previous_max_tokens,
                "context_max_units": self._window_config.context_max_units,
            },
            # context preserving related
            "preserve_prefix_length": self._preserve_prefix_length,
            "previous_content_length": self._previous_content_length,
            "suffix_token_count": len(self._suffix_token_ids),
            "previous_text_length": len(self._previous_text),
            "previous_token_count": len(self._previous_token_ids),
            "has_system_template": self._system_prompt_template is not None,
        }

    def _verify_consistency(self) -> bool:
        """verify unit history and cache length consistency"""
        expected = self._system_preserve_length + sum(u["length"] for u in self._unit_history)
        actual = self.get_cache_length()
        return expected == actual

    def print_verification_summary(self) -> Dict[str, Any]:
        """print verification summary (for comparing off/basic/context mode)

        Returns:
            dictionary containing key verification data
        """
        cfg = self._window_config

        # collect all generated text
        all_generated_text = []
        all_generated_tokens = []
        for u in self._unit_history:
            if not u.get("is_listen", False):
                gen_text = u.get("generated_text", "")
                gen_tokens = u.get("generated_tokens", [])
                if gen_text:
                    all_generated_text.append(gen_text)
                if gen_tokens:
                    all_generated_tokens.extend(gen_tokens)

        combined_text = "".join(all_generated_text)

        summary = {
            "mode": cfg.sliding_window_mode,
            "final_cache_length": self.get_cache_length(),
            "final_unit_count": len(self._unit_history),
            "sliding_event_count": self._sliding_event_count,
            "total_dropped_tokens": self._total_dropped_tokens,
            "total_dropped_units": self._total_dropped_units,
            "total_generated_tokens": len(all_generated_tokens),
            "generated_text": combined_text,
            "previous_text": self._previous_text,
            "previous_token_count": len(self._previous_token_ids),
            "position_offset": self._position_offset,
            "system_preserve_length": self._system_preserve_length,
        }

        return summary

    def set_window_config(self, config: DuplexWindowConfig) -> None:
        """set sliding window configuration"""
        self._window_config = config

    def set_window_enabled(self, enabled: bool) -> None:
        """enable/disable sliding window"""
        self._window_enabled = enabled

    def get_context(self):
        return self.context

    def embed_token(self, tid):
        if isinstance(tid, int):
            tid = torch.tensor([tid], device=self.m.device)
        return self.m.model.embed_tokens(tid)

    def embed_tokens(self, token_ids: List[int]) -> torch.Tensor:
        """batch embed multiple tokens

        Args:
            token_ids: list of token ids

        Returns:
            embeddings tensor [L, H]
        """
        if not token_ids:
            return torch.empty(0, self.m.config.hidden_size, device=self.m.device)
        tids = torch.tensor(token_ids, device=self.m.device)
        return self.m.model.embed_tokens(tids)

    @torch.no_grad()
    def feed(self, embeds: torch.Tensor, return_logits: bool = False):
        """
        embeds : [L, H]   —— new embedding sequence fed into model at once
        """
        L = embeds.size(0)
        device = embeds.device

        past_len = self.get_cache_length()
        pos_ids = torch.arange(past_len, past_len + L, device=device).unsqueeze(0)  # [1, L]

        out = self.m(
            inputs_embeds=embeds.unsqueeze(0),  # [1, L, H]
            position_ids=pos_ids,
            past_key_values=self.cache,
            # use_cache = True,
            return_dict=True,
            output_hidden_states=True,
            # attention_mask=attention_mask
        )
        self.cache = out.past_key_values

        if return_logits:
            logits = self.m.lm_head(out.hidden_states[-1])[:, -1]  # [1, vocab]
            return logits, out.hidden_states[-1]

    @torch.no_grad()
    def decode(
        self,
        logits,
        mode: Literal["sampling", "greedy"] = "sampling",
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        listen_top_k=None,
        listen_prob_scale=1.0,
        text_repetition_penalty=1.05,
        text_repetition_window_size=512,
    ):
        """
        Args:
            logits:
            mode: sampling or greedy
            temperature:
            top_k:
            top_p:
            listen_top_k: force listen_id to be in top-k to keep
            listen_prob_scale: multiply listen_id probability by a weight (<1 means decrease, >1 means increase)
            text_repetition_penalty: repetition penalty coefficient, >1.0 means decrease repetition, <1.0 means increase repetition
            text_repetition_window_size: repetition penalty window size

        Sampling strategy:
            1. first sample all tokens with original logits (apply temperature)
            2. if sampled chunk_eos, return directly (keep the original model's decision of when to stop)
            3. if not sampled chunk_eos, mask it (set logit to -inf), continue sampling text tokens
            4. apply repetition penalty, top-k, top-p, etc. to the text tokens for the final sampling
        """

        logits = logits.clone()

        # 0. independently check chunk_eos before sampling
        eos_id = self.chunk_eos_id

        with torch.no_grad():
            if mode == "greedy":
                sampled_token = torch.argmax(logits[0]).item()
            else:
                original_probs = F.softmax(logits[0], dim=-1)
                sampled_token = torch.multinomial(original_probs, num_samples=1).item()

            # if sampled chunk_eos, return directly
            if sampled_token == eos_id:
                next_token_id = torch.tensor([eos_id], device=logits.device)
                next_token_str = self.tokenizer.decode(next_token_id)

                return next_token_id

        # if not sampled chunk_eos, set its logit to -inf
        if self.forbidden_token_ids:
            logits[:, self.forbidden_token_ids] = float("-inf")

        # 1. apply repetition penalty
        if text_repetition_penalty != 1.0 and len(self.generated_tokens) > 0:
            # get recent tokens (within window size) considering special tokens and normal tokens
            recent_tokens = self.generated_tokens[-text_repetition_window_size:]

            # make it unique
            recent_tokens = list(set(recent_tokens))

            # apply penalty to repeated tokens
            for token_id in recent_tokens:
                if token_id < logits.size(-1):  # ensure token_id is in vocabulary range
                    if text_repetition_penalty > 1.0:
                        # penalize repetition: decrease logits
                        logits[0, token_id] /= text_repetition_penalty
                    else:
                        # encourage repetition: increase logits
                        logits[0, token_id] *= 1.0 / text_repetition_penalty

        if listen_prob_scale != 1.0:  # modify listen token logit separately
            logits[0, self.listen_id] *= listen_prob_scale

        listen_rank = (logits[0] > logits[0, self.listen_id]).sum().item()

        if listen_top_k is not None and listen_rank < listen_top_k:  # listen_id is in top-k, return directly
            next_token_id = torch.tensor([self.listen_id], device=logits.device)
            next_token_str = self.tokenizer.decode(next_token_id)

            if next_token_str == "<|listen|>":
                self.context += " "
            else:
                self.context += next_token_str

            return next_token_id

        if mode == "greedy":
            next_token_id = torch.argmax(logits, dim=-1)
        elif mode == "sampling":
            logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            raise ValueError(f"Unsupported decode mode: {mode}")

        if next_token_id.item() not in self.special_token_ids:
            self.generated_tokens.append(next_token_id.item())
        else:
            self.generated_special_tokens.append(next_token_id.item())

        return next_token_id


@contextmanager
def _local_media_path(path: str, suffix: str, timeout: int):
    """Keep a downloaded media file available only for the decoder call."""
    if not _is_url(path):
        yield path
        return
    data = load_bytes_from_url(path, total_timeout=timeout)
    with TemporaryDirectory() as directory:
        temp_path = Path(directory) / f"media{suffix}"
        temp_path.write_bytes(data)
        yield str(temp_path)


def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def normalize_content_item(item) -> Union[str, Any, List[Any]]:
    """Normalize structured content item to native format.

    Supports:
    - Native format: str, PIL.Image, np.ndarray (pass through)
    - OpenAI structured format:
        - {"type": "text", "text": "..."} -> str
        - {"type": "image_url", "image_url": {"url": "..."}} -> PIL.Image
        - {"type": "audio_url", "audio_url": {"url": "..."}} -> np.ndarray
        - {"type": "video_url", "video_url": {"url": "...", ...}} -> List[Image, ndarray, ...]

    URL formats supported:
        - Local file path: "/path/to/file.jpg"
        - HTTP/HTTPS URL: "https://example.com/image.jpg"

    Args:
        item: Content item to normalize

    Returns:
        Normalized item. For video_url, returns a tuple ("__video_contents__", list)
        that will be flattened by normalize_content().

    Raises:
        ValueError: If content type is unknown or unsupported
    """
    import numpy as np
    from PIL import Image

    if isinstance(item, str):
        return item
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, np.ndarray):
        return item

    if isinstance(item, dict):
        item_type = item.get("type")

        if item_type == "text":
            return item.get("text", "")

        elif item_type == "image_url":
            image_url_obj = item.get("image_url", {})
            url = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else image_url_obj

            return load_image_from_url(url) if _is_url(url) else Image.open(url)
        elif item_type == "audio_url":
            import librosa

            audio_url_obj = item.get("audio_url", {})
            url = audio_url_obj.get("url", "") if isinstance(audio_url_obj, dict) else audio_url_obj

            with _local_media_path(url, ".wav", timeout=60) as audio_path:
                audio_np, _ = librosa.load(audio_path, sr=16000, mono=True)
            return audio_np
        elif item_type == "video_url":
            # Video processing - returns a LIST of items (frames + audio segments)
            # Note: Unlike image_url/audio_url which return single items,
            # video_url returns a list that will be flattened into the content
            video_url_obj = item.get("video_url", {})
            if isinstance(video_url_obj, dict):
                video_url = video_url_obj.get("url", "")
                # Get optional parameters from video_url object (OpenAI style)
                stack_frames = video_url_obj.get("stack_frames", 1)
                use_ffmpeg = video_url_obj.get("use_ffmpeg", False)
                use_audio = video_url_obj.get("use_audio", True)
            else:
                video_url = video_url_obj
                stack_frames = 1
                use_ffmpeg = False
                use_audio = True

            # Extract frames and audio segments
            with _local_media_path(video_url, ".mp4", timeout=120) as video_path:
                video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
                    video_path, stack_frames=stack_frames, use_ffmpeg=use_ffmpeg, use_audio=use_audio
                )

            # Build omni_contents (interleaved frames and audio, or frames only)
            omni_contents = []
            for i in range(len(video_frames)):
                omni_contents.append(video_frames[i])
                if use_audio and audio_segments is not None:
                    omni_contents.append(audio_segments[i])
                if stacked_frames is not None and i < len(stacked_frames) and stacked_frames[i] is not None:
                    omni_contents.append(stacked_frames[i])

            # Return as a special marker to be flattened later
            return "__video_contents__", omni_contents
        else:
            raise ValueError(f"Unknown content type: {item_type}")

    raise ValueError(f"Cannot normalize content item of type: {type(item)}")


def normalize_content(content) -> list:
    """Normalize message content to list of native items.

    Input formats:
    - str: "hello" -> ["hello"]
    - list of native items: [str, Image, np.ndarray] -> pass through with normalization
    - list of structured items: [{"type": "text", ...}] -> normalize each
    - video type: automatically expanded to omni_contents
    - mixed: works too

    Args:
        content: Message content in any supported format

    Returns:
        List of native items (str, PIL.Image, np.ndarray)

    Examples:
        >>> normalize_content("hello")
        ["hello"]

        >>> normalize_content([{"type": "text", "text": "hi"}])
        ["hi"]

        >>> normalize_content([{"type": "video", "video": "/path/to/video.mp4"}])
        [<PIL.Image>, <np.ndarray>, <PIL.Image>, <np.ndarray>, ...]
    """
    import numpy as np
    from PIL import Image

    if isinstance(content, str):
        return [content]

    if isinstance(content, list):
        result = []
        for item in content:
            normalized = normalize_content_item(item)
            # Handle video content (returns tuple with marker)
            if isinstance(normalized, tuple) and len(normalized) == 2 and normalized[0] == "__video_contents__":
                # Flatten video contents into result
                result.extend(normalized[1])
            else:
                result.append(normalized)
        return result

    # Single non-list item (Image or np.ndarray)
    if isinstance(content, (Image.Image, np.ndarray)):
        return [content]

    normalized = normalize_content_item(content)
    if isinstance(normalized, tuple) and len(normalized) == 2 and normalized[0] == "__video_contents__":
        return normalized[1]
    return [normalized]
