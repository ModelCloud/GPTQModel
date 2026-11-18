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

# Inherited ChatTTS-derived TTS helpers have an unresolved licensing boundary; see NOTICE.

import json
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ..native.llama.configuration_llama import LlamaConfig
from ..native.qwen2.configuration_qwen2 import Qwen2Config
from ..native.qwen3.configuration_qwen3 import Qwen3Config
from ..native.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
from ..native.whisper.configuration_whisper import WhisperConfig
from ..native.whisper.modeling_whisper import WhisperAttention, WhisperEncoder
from .cache import cache_pairs, set_layer_states
from .configuration_minicpmo import MiniCPMOConfig, MiniCPMTTSConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .native import LlamaModel, Qwen2ForCausalLM, Qwen3ForCausalLM
from .processing_minicpmo import MiniCPMOProcessor
from .utils import (
    ChunkPrefillChunkGenerate,
    DuplexWindowConfig,
    SpeculativeSnapshot,
    StreamingWindowConfig,
    TTSSamplingParams,
    TTSStreamingGenerator,
    config_rope_theta,
    drop_tokens_from_cache,
    get_kv_cache_length,
    normalize_content,
    streaming_token_decoder,
    torch_clone_recursive,
)


logger = logging.getLogger(__name__)


@contextmanager
def _temporary_wav(audio, sample_rate=16000):
    """Write a 16 kHz prompt for the external codec; always removed afterwards."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "prompt.wav")
        sf.write(path, audio, sample_rate)
        yield path


class MiniCPMOPreTrainedModel(Qwen3PreTrainedModel):
    config_class = MiniCPMOConfig
    base_model_prefix = ""
    _no_split_modules = ["Qwen3DecoderLayer", "SiglipEncoderLayer", "MiniCPMWhisperEncoderLayer"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Resampler):
            module._pos_cache_ready = False


class MiniCPMO(MiniCPMOPreTrainedModel):
    def __new__(cls, config, *args, **kwargs):
        if cls is MiniCPMO and config.version == "2.6":
            from .modeling_minicpmo_26 import MiniCPMO26

            return object.__new__(MiniCPMO26)
        return super().__new__(cls)

    def __init__(self, config):
        super().__init__(config)

        text_values = config.to_dict()
        for key in (
            "vision_config",
            "audio_config",
            "tts_config",
            "slice_config",
            "model_type",
            "auto_map",
            "architectures",
        ):
            text_values.pop(key, None)
        text_values["head_dim"] = text_values.get("head_dim") or config.hidden_size // config.num_attention_heads
        text_values["attn_implementation"] = config._attn_implementation
        if config.version == "2.6":
            self.llm = Qwen2ForCausalLM(Qwen2Config(**text_values))
        else:
            self.llm = Qwen3ForCausalLM(Qwen3Config(**text_values))
        self.embed_dim = self.llm.config.hidden_size

        # init vision module
        if self.config.init_vision:
            self.vpm = self.init_vision_module()
            self.vision_dim = self.vpm.embed_dim
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # init audio module
        if self.config.init_audio:
            self.apm = self.init_audio_module()
            audio_output_dim = self.apm.config.d_model
            self.audio_avg_pooler = nn.AvgPool1d(self.config.audio_pool_step, stride=self.config.audio_pool_step)
            self.audio_projection_layer = MultiModalProjector(in_dim=audio_output_dim, out_dim=self.embed_dim)
            self.audio_encoder_layer = -1

        # init tts module
        if self.config.init_tts:
            self.tts = self.init_tts_module()

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

        self.think_str = ""
        if self.llm.__class__.__name__ == "Qwen3ForCausalLM":
            self.think_str = "<think>\\n\\n</think>\\n\\n"

        # for streaming
        self.reset_session(reset_token2wav_cache=False)

        # streaming audio processing constants
        self.SAMPLE_RATE = 16000
        self.CHUNK_MS = 1000  # regular chunk length (ms)
        self.FIRST_CHUNK_MS = 1035  # first chunk length (ms)
        self.CNN_REDUNDANCY_MS = 0  # CNN redundancy (ms)

        # for sliding window
        self.streaming_window_config = StreamingWindowConfig()
        self.streaming_require_system_prompt = True
        self.streaming_window_enabled = True
        self.force_rope_reindex = False  # RoPE reindex testing switch

        # Transformers 5 derives `all_tied_weights_keys` and the other
        # load-time static properties here.
        self.post_init()

    def init_streaming_processor(self):
        self.prepare_processor(processor=None, tokenizer=None)

        if hasattr(self.processor, "set_streaming_mode"):
            self.processor.set_streaming_mode(
                mode="exact",
                chunk_ms=self.CHUNK_MS,
                first_chunk_ms=self.FIRST_CHUNK_MS,
                cnn_redundancy_ms=self.CNN_REDUNDANCY_MS,
                enable_sliding_window=True,
                slide_trigger_seconds=30.0,
                slide_stride_seconds=10.0,
            )
            self.processor.reset_streaming()
            self.audio_chunk_idx = 0

    def reset_session(self, reset_token2wav_cache=True):
        self.llm_past_key_values = None
        self.audio_past_key_values = None
        self.tts_last_turn_tokens = None
        self.llm_generated = False  # last turn generated by llm or not
        self.llm_generate_completed = False
        self.new_user_msg = True

        self.session_id = None

        if reset_token2wav_cache:
            self.token2wav_cache = None

        # for sliding window
        self.streaming_text_preserve = 0
        self.streaming_position_offset = 0

        self._rope_inv_freq_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}

        self._next_round_id = 0
        self._pending_round_id = None

        self._omni_chunk_history: List[Dict[str, Union[str, int]]] = []
        self._round_history: List[Dict[str, Union[int, str, torch.Tensor, Optional[int]]]] = []

    def init_vision_module(self):
        if self.config._attn_implementation == "flash_attention_2":
            raise ValueError("FlashAttention vision is not qualified for local MiniCPM-o; use eager or sdpa")
        else:
            self.config.vision_config._attn_implementation = "eager"
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, "embed_dim", model.embeddings.embed_dim)
        setattr(model, "patch_size", model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=max(1, embed_dim // 128),
            kv_dim=vision_dim,
            adaptive=True,
        )

    def init_audio_module(self):
        if self.config._attn_implementation == "eager":
            self.config.audio_config._attn_implementation = "eager"
        else:
            # using flash_attention_2 will cause: RuntimeError: cu_seqlens_q must have shape (batch_size + 1)
            self.config.audio_config._attn_implementation = "sdpa"

        return MiniCPMWhisperEncoder(self.config.audio_config)

    def init_tts_module(self):
        if self.config._attn_implementation == "flash_attention_2":
            self.config.tts_config.attn_implementation = "flash_attention_2"
        else:
            self.config.tts_config.attn_implementation = "eager"

        if self.config.version == "2.6":
            from .tts_26 import ConditionalChatTTS

            return ConditionalChatTTS(self.config.tts_config)
        return MiniCPMTTS(config=self.config.tts_config, audio_tokenizer=None)

    def init_tts(self, model_dir=None, enable_float16=False, n_timesteps=10, *, trust_remote_code=False):
        """Initialize the external StepAudio codec from caller-selected, trusted assets."""
        from ....utils.security import BYPASS_ENV, RemoteCodeSecurityError, require_remote_code_permission

        if not require_remote_code_permission(trust_remote_code):
            raise RemoteCodeSecurityError(
                "StepAudio codec loading requires trust_remote_code=True and "
                f"{BYPASS_ENV}=1 because its assets are interpreted by an external loader."
            )
        if not model_dir:
            raise ValueError("Set model_dir to the local directory containing trusted StepAudio codec assets.")
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Install MiniCPM-o codec assets locally: {model_dir}")

        try:
            from stepaudio2 import Token2wav
        except ImportError:
            raise ImportError("Please install Token2wav via: pip install minicpmo-utils[all]")

        audio_tokenizer = Token2wav(os.fspath(model_dir), float16=enable_float16, n_timesteps=n_timesteps)
        if self.config.tts_config.audio_tokenizer_type != "s3tokenizer_step_audio":
            logger.warning("audio tokenizer type is set to s3tokenizer_step_audio")
            self.tts.config.audio_tokenizer_type = "s3tokenizer_step_audio"
        self.tts.audio_tokenizer = audio_tokenizer
        return audio_tokenizer

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    @staticmethod
    def get_sys_prompt(ref_audio=None, mode="default", language="en", ref_audio_max_ms=None):
        if ref_audio is not None:
            if isinstance(ref_audio, str):
                import os

                import librosa

                if os.path.isfile(ref_audio):
                    duration = ref_audio_max_ms / 1000.0 if ref_audio_max_ms else None
                    ref_audio, _ = librosa.load(ref_audio, sr=16000, mono=True, duration=duration)
                else:
                    logger.error(f"Could not find {ref_audio}")
                    ref_audio = None

            assert isinstance(ref_audio, np.ndarray), "ref_audio error"

        if mode == "omni":
            if language == "zh":
                sys_prompt = ""
                vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
                vc_prompt_suffix = "请用这种声音风格来为用户提供帮助。 请认真、高质量地回复用户的问题。 请用高自然度的方式和用户聊天。"
            else:
                sys_prompt = ""
                vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                sys_msgs = {"role": "system", "content": [sys_prompt]}

            return sys_msgs
        elif mode == "audio_assistant":
            if language == "zh":
                vc_prompt_prefix = "模仿音频样本的音色并生成新的内容。"
                vc_prompt_suffix = "你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Please assist users while maintaining this voice style. Please answer the user's questions seriously and in a high quality. Please chat with the user in a highly human-like and oral style. You are a helpful assistant developed by ModelBest: MiniCPM-Omni."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                logger.warning(
                    "Warning: ref_audio is None, speech generation will be performed based on the default voice."
                )
                sys_msgs = {"role": "system", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "audio_roleplay":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                sys_msgs = {"role": "system", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "voice_cloning":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."

            if ref_audio is not None:
                sys_msgs = {"role": "system", "content": [vc_prompt_prefix, ref_audio]}
            else:
                raise ValueError("ref_audio con't be None in voice_cloning mode.")

            return sys_msgs
        else:
            sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
            sys_msgs = {"role": "system", "content": [sys_prompt]}

            return sys_msgs

    @staticmethod
    def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        num_lookhead: int = 0,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size) with chunk size,
        this is for streaming encoder

        Args:
            size (int): size of mask
            chunk_size (int): size of chunk
            num_left_chunks (int): number of left chunks
                <0: use full chunk
                >=0: use num_left_chunks
            device (torch.device): "cpu" or "cuda" or torch.Tensor.device
            num_lookhead:

        Returns:
            torch.Tensor: mask
        """
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        for i in range(size):
            if num_left_chunks < 0:
                start = 0
            else:
                start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
            ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
            ret[i, start:ending] = True
        return ret

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """Computes the output length of the convolutional layers and the output length of the audio encoder"""
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_vision_embedding(self, data):
        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values,
                        patch_attention_mask=patch_attn_mask,
                        tgt_sizes=tgt_sizes,
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [
                            [
                                (224 // self.config.patch_size),
                                math.ceil(224 / self.config.patch_size),
                            ]
                        ]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)
        else:
            vision_hidden_states = data["vision_hidden_states"]

        return vision_hidden_states

    def get_vllm_embedding(self, data):
        vision_hidden_states = self.get_vision_embedding(data)

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )
                elif self.training:
                    cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def get_audio_embedding_streaming(
        self,
        data,
        use_extra_context=False,
        prefix_extra_frames=1,
        suffix_extra_frames=1,
        cnn_min_length=None,
    ):
        """Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. It only supports batch_size=1 and is intended
        for streaming scenarios.

        Args:
            data (dict):
                - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                - **"audio_feature_lens"** (List[List[int]]): Lengths of each audio segment for each item in the batch.
            use_extra_context (bool): If True, assumes input contains extra frames for CNN context.
            prefix_extra_frames (int): Number of prefix extra frames.
            suffix_extra_frames (int): Number of suffix extra frames.
            cnn_min_length (int): Minimum length for CNN input padding.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """
        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            assert batch_size == 1
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # whisper's past_key_values management (core)
            if self.audio_past_key_values is not None:
                cache_length = self.audio_past_key_values.get_seq_length()
                apm_max_len = self.apm.embed_positions.weight.shape[0]
                if cache_length + max_seq_len >= apm_max_len:
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )
                    self.audio_past_key_values = None

            # build attention mask (bidirectional attention, same as offline mode)
            batch_size, _, max_mel_seq_len = wavforms.shape
            current_seq_len = (max_mel_seq_len - 1) // 2 + 1
            # if use extra context, need to adjust sequence length
            if use_extra_context:
                # calculate actual sequence length after removing redundancy
                # conv2's stride=2, so the mapping from mel frames to output frames is ceil(x/2)
                prefix_to_remove = (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0
                suffix_to_remove = (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0
                current_seq_len = current_seq_len - prefix_to_remove - suffix_to_remove
            # calculate history length (if there is KV cache)
            if self.audio_past_key_values is not None:
                past_len = self.audio_past_key_values.get_seq_length()  # get history sequence length
                total_seq_len = past_len + current_seq_len
            else:
                past_len = 0
                total_seq_len = current_seq_len
            # create bidirectional attention mask (full attention)
            audio_attention_mask = torch.zeros(
                (batch_size, 1, current_seq_len, total_seq_len),
                dtype=self.apm.conv1.weight.dtype,
                device=wavforms.device,
            )

            # Step 1: APM processing
            audio_outputs = self.apm(
                wavforms,
                past_key_values=self.audio_past_key_values,
                use_cache=True,
                output_hidden_states=True,
                attention_mask=audio_attention_mask,
                use_extra_context=use_extra_context,
                prefix_extra_frames=prefix_extra_frames,
                suffix_extra_frames=suffix_extra_frames,
                cnn_min_length=cnn_min_length,
            )

            if hasattr(self, "audio_encoder_layer"):
                audio_states = audio_outputs.hidden_states[self.audio_encoder_layer]
            else:
                audio_states = audio_outputs.last_hidden_state

            self.audio_past_key_values = audio_outputs.past_key_values

            # Step 2: Projection
            audio_embeds = self.audio_projection_layer(audio_states)

            # Step 3: Pooling
            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)

                return final_audio_embeds
            else:
                return final_audio_embeds
        else:
            return []

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(
                    0,
                    max_seq_len,
                    dtype=audio_feature_lens.dtype,
                    device=audio_feature_lens.device,
                )
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(
                dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
            )

            if chunk_length > 0:
                chunk_num_frame = int(chunk_length * 50)
                chunk_mask = self.subsequent_chunk_mask(
                    size=max_seq_len,
                    chunk_size=chunk_num_frame,
                    num_left_chunks=-1,
                    device=audio_attention_mask_.device,
                )
                audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

            audio_attention_mask[audio_attention_mask_] = float("-inf")
            audio_states = self.apm(
                wavforms, output_hidden_states=True, attention_mask=audio_attention_mask
            ).hidden_states[self.audio_encoder_layer]
            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)

            _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        elif self.training and dummy:
            dtype = self.apm.embed_positions.weight.dtype
            device = self.apm.embed_positions.weight.device

            dummy_wavs = torch.zeros((1, 80, 100), device=device, dtype=dtype)
            audio_states = self.apm(dummy_wavs, output_hidden_states=True).hidden_states[self.audio_encoder_layer]

            audio_embeds = self.audio_projection_layer(audio_states)

            audio_embeds = audio_embeds.transpose(1, 2)
            audio_embeds = self.audio_avg_pooler(audio_embeds)
            audio_embeds = audio_embeds.transpose(1, 2)
            return [audio_embeds]
        else:
            return []

    def get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        """
        Args:
            data:
            input_embeddings:
            chunk_length: whisper use full attention or chunk attention
            stream_input: use streaming audio embedding or not

        Returns:
            final embeddings with audio feature
        """
        if not self.config.init_audio:
            if len(data.get("audio_features", [])):
                raise ValueError("Audio inputs require init_audio=True")
            return input_embeddings
        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(data)
        else:
            audio_embeddings = self.get_audio_embedding(data, chunk_length)

        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]

                if self.config.stream_input:
                    assert bs == 1, "audio stream_input mode only support batch size 1"
                    for i in range(bs):
                        audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                            device=input_embeddings.device, dtype=input_embeddings.dtype
                        )
                        audio_start_pos = 0
                        for bound in audio_bounds[i]:
                            audio_len = bound[1] - bound[0]
                            input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                                audio_start_pos : audio_start_pos + audio_len, :
                            ]
                            audio_start_pos += audio_len
                else:
                    for i in range(bs):
                        audio_embs = audio_embeddings[i]
                        bounds = audio_bounds[i]

                    one_to_one_match = len(audio_embs) == len(bounds) and all(
                        embs.shape[0] == int(bound[1] - bound[0]) for embs, bound in zip(audio_embs, bounds)
                    )

                    if one_to_one_match:
                        for embs, bound in zip(audio_embs, bounds):
                            input_embeddings[i, bound[0] : bound[1]] = embs.to(
                                device=input_embeddings.device,
                                dtype=input_embeddings.dtype,
                            )
                    else:
                        flat_audio_embs = torch.cat(audio_embs, dim=0).to(
                            device=input_embeddings.device,
                            dtype=input_embeddings.dtype,
                        )

                        total_bound_len = sum(int(bound[1] - bound[0]) for bound in bounds)

                        if flat_audio_embs.shape[0] != total_bound_len:
                            raise ValueError(
                                f"Audio total length mismatch: {flat_audio_embs.shape[0]} != {total_bound_len}"
                            )

                        offset = 0
                        for bound in bounds:
                            audio_len = int(bound[1] - bound[0])
                            input_embeddings[i, bound[0] : bound[1]] = flat_audio_embs[offset : offset + audio_len]
                            offset += audio_len

        elif self.training:
            for i in range(bs):
                # dummy audio_embedings
                input_embeddings += audio_embeddings[0].mean() * 0

        return input_embeddings

    def forward(self, data=None, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if data is None:
            data = {
                "input_ids": input_ids,
                "pixel_values": [[] for _ in input_ids],
                "tgt_sizes": [],
                "image_bound": [[] for _ in input_ids],
                "audio_features": [],
                "audio_feature_lens": [],
            }
        if not self.config.init_vision:
            data = dict(data, vision_hidden_states=[[] for _ in data["input_ids"]])
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        vllm_embedding = self.get_omni_embedding(
            data,
            input_embeddings=vllm_embedding,
            chunk_length=self.config.audio_chunk_length,
        )

        position_ids = data.get("position_ids", position_ids)
        if position_ids is not None and position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            attention_mask=data.get("attention_mask", attention_mask),
            **kwargs,
        )

    def _decode(self, inputs_embeds, tokenizer, attention_mask, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )
        return outputs

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        raise NotImplementedError("Threaded text streaming is not qualified for the local MiniCPM-o port.")

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result))
        return result_text

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=None,
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        **kwargs,
    ):
        assert input_ids is not None
        pixel_values = pixel_values if pixel_values is not None else [[] for _ in input_ids]
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if not self.config.init_vision:
            vision_hidden_states = [[] for _ in input_ids]
        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)
            model_inputs["inputs_embeds"] = self.get_omni_embedding(
                model_inputs,
                input_embeddings=model_inputs["inputs_embeds"],
                chunk_length=self.config.audio_chunk_length,
            )

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
                outputs = {}  # if stream return TextIteratorStreamer and output is empty
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)
                result = self._decode_text(outputs.sequences, tokenizer)

        return result, outputs

    def _build_streaming_mask(self, tts_tokens_len):
        tts_sequence_full_length = 1 + self.tts.streaming_text_reserved_len + 1
        streaming_attention_mask = torch.zeros(tts_sequence_full_length, dtype=torch.int8)
        streaming_attention_mask[0 : 1 + 1 + tts_tokens_len + 1] = 1
        streaming_attention_mask[-1] = 1
        return streaming_attention_mask

    def _generate_mel_spec(self, inputs, outputs, text, output_chunk_size=25, tts_max_new_tokens=2048):
        spk_embeds = self._get_last_spk_embeds(inputs, outputs)

        text = text.split("<|tts_bos|>")[-1]
        gen_text = text.split("<|tts_eos|>")[0]
        tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
        tts_inputs = self.tts_processor.text_tokenizer.encode(tts_text, add_special_tokens=False)
        tts_input_ids = torch.Tensor(tts_inputs).unsqueeze(0).to(self.device, dtype=torch.long)
        streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=626,
            top_p=self.tts.top_p,
            top_k=self.tts.top_k,
            repetition_penalty=self.tts.repetition_penalty,
        )

        condition_length = 1 + self.tts.streaming_text_reserved_len + 1

        dtype = self.tts.emb_text.weight.dtype
        emb = torch.zeros(1, condition_length, self.tts.num_vq, dtype=dtype, device=self.tts.device)
        past_key_values = [
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=emb.dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        ]

        audio_input_ids = torch.zeros(
            1,
            condition_length,
            self.tts.num_vq,
            dtype=torch.long,
            device=self.tts.device,
        )

        eos_lab = False
        for chunk_idx in range(math.ceil(emb.shape[1] / self.tts.streaming_text_chunk_size)):
            if chunk_idx == 0:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 0
                end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + 1
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 1
                end = min(
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size + 1,
                    condition_length - 1,
                )

            if end - begin > 0:
                text_input_ids = tts_input_ids[:, begin:end]
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                if begin == 0:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        lm_spk_emb_last_hidden_states=spk_embeds,
                    )
                else:
                    past_key_values = self.tts.prefill_text(
                        input_ids=text_input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                    )

            outputs = self.tts.generate(
                input_ids=audio_input_ids,
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
                max_new_token=output_chunk_size,
                force_no_stop=self.force_no_stop,
                temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                logits_warpers=logits_warpers,
                logits_processors=logits_processors,
            )
            audio_input_ids = outputs.audio_input_ids
            past_key_values = outputs.past_key_values

            if outputs.finished:
                eos_lab = True
                break

        if not eos_lab:
            while True:
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )

                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values

                if outputs.finished:
                    break
                if outputs.new_ids.shape[1] > tts_max_new_tokens:
                    break

    @staticmethod
    def prepare_generation_config(do_sample, max_new_tokens=50, min_new_tokens=0, **kwargs):
        num_beams = kwargs.get("num_beams", 3)
        generation_config = {
            "num_beams": num_beams,
            "top_p": 0.8,
            "top_k": 100,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.02,
        }

        if do_sample:
            generation_config.update(
                {
                    "top_p": 0.8,
                    "top_k": 100,
                    "temperature": 0.7,
                    "do_sample": True,
                    "repetition_penalty": 1.02,
                }
            )
        elif num_beams > 1:
            generation_config.update({"num_beams": num_beams, "repetition_penalty": 1.2, "do_sample": False})
        else:
            generation_config.update({"do_sample": False, "repetition_penalty": 1.02})

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config["max_new_tokens"] = max_new_tokens

        return generation_config

    def prepare_processor(self, processor=None, tokenizer=None):
        if processor is not None:
            self.processor = processor
        if not hasattr(self, "processor") or self.processor is None:
            self.processor = MiniCPMOProcessor.from_pretrained(
                self.config._name_or_path,
                config=self.config,
                tokenizer=tokenizer,
            )
        if tokenizer is not None:
            self.processor.tokenizer = tokenizer

    @torch.inference_mode()
    def chat(
        self,
        image=None,
        msgs=None,
        vision_hidden_states=None,
        max_new_tokens=4096,
        min_new_tokens=0,
        do_sample=True,
        max_inp_length=8192,
        max_slice_nums=None,
        use_image_id=None,
        enable_thinking=False,
        use_tts_template=False,
        generate_audio=False,
        output_audio_path=None,
        output_tts_inputs_embeds_path=None,
        omni_mode=False,
        teacher_forcing=False,
        return_prompt=False,
        tts_proj_layer=-1,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        merge_audio_from_same_content=True,
        stream=False,
        stream_input=False,
        tokenizer=None,
        processor=None,
        **kwargs,
    ):
        from PIL import Image

        batched = isinstance(msgs[0], list)
        msgs_list = msgs
        images_list = image

        if not batched:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        self.prepare_processor(processor=processor, tokenizer=tokenizer)

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert do_sample or not stream, "if use stream mode, make sure do_sample=True"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            audios = []
            audio_parts = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["system", "user", "assistant"]
                if i == 0:
                    assert role in ["user", "system"], "The role of first msg should be user"
                # Normalize structured content (OpenAI format) to native format
                content = normalize_content(content)
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("<image>./</image>")
                    elif isinstance(c, np.ndarray):  # audio
                        audios.append(c)
                        audio_parts.append(i)
                        cur_msgs.append("<audio>./</audio>")
                        use_tts_template = True
                    elif isinstance(c, str):
                        cur_msgs.append(c)

                if omni_mode or stream_input:
                    msg["content"] = "".join(cur_msgs)
                else:
                    msg["content"] = "\n".join(cur_msgs)

            prompts_lists.append(
                self.processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False if teacher_forcing else True,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        if not merge_audio_from_same_content:
            audio_parts_list = None

        inputs = self.processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            stream_input=stream_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        if stream:
            kwargs["num_beams"] = 1
        generation_config = self.prepare_generation_config(
            do_sample=do_sample, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens, **kwargs
        )
        generation_config.pop("max_new_tokens", None)

        inputs.pop("image_sizes")

        # teacher_forcing = True => generate audio with given text
        with torch.inference_mode():
            res, outputs = self.generate(
                **inputs,
                tokenizer=self.processor.tokenizer,
                max_new_tokens=1 if teacher_forcing else max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config,
            )

        if stream:
            return res

        # spk bound and tts bound
        tts_bos_token = self.processor.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        tts_eos_token = self.processor.tokenizer.convert_tokens_to_ids("<|tts_eos|>")

        # Combine input_ids and generated sequences to get complete sequence
        input_ids = inputs["input_ids"][0]
        generated_ids = outputs.sequences[0]
        # Combine by concatenating input_ids with the new tokens from generated sequence
        full_sequence = torch.cat([input_ids, generated_ids])
        # Update the sequences in outputs
        full_sequences = full_sequence.unsqueeze(0)

        outputs["full_sequences"] = full_sequences

        tts_bos_indices = []
        tts_eos_indices = []
        for i, x in enumerate(full_sequences[0]):
            if x == tts_bos_token:
                # tts_bos + 1 is the position of the first tts, so that it is convenient to slice hidden states for tts
                tts_bos_indices.append(i + 1)
            elif x == tts_eos_token:
                if teacher_forcing and i == len(full_sequences[0]) - 1:
                    continue
                tts_eos_indices.append(i)

        tts_bos_idx = tts_bos_indices[-1] if tts_bos_indices else -1
        # Use None instead of -1 when no EOS token found, so that slice [start:None]
        # means "to the end" rather than [start:-1] which excludes the last element
        tts_eos_idx = tts_eos_indices[-1] if tts_eos_indices else None

        tts_bound = (tts_bos_idx, tts_eos_idx)

        answer = res[0]
        if answer is not None:
            answer = answer.split("<|tts_eos|>")[0]

        if use_tts_template and generate_audio and output_audio_path:
            import soundfile as sf

            try:
                generated_waveform = self._generate_speech_non_streaming(
                    outputs=outputs,
                    tts_bound=tts_bound,
                    tts_proj_layer=tts_proj_layer,
                    audio_prompt=(
                        input_audios_list[0][0]
                        if len(input_audios_list) > 0 and len(input_audios_list[0]) > 0
                        else None
                    ),
                    output_tts_inputs_embeds_path=output_tts_inputs_embeds_path,
                    tts_sampling_params=tts_sampling_params,
                )
                if isinstance(generated_waveform, torch.Tensor):
                    sf.write(output_audio_path, generated_waveform.cpu().numpy(), samplerate=24000)
                elif isinstance(generated_waveform, np.ndarray):
                    sf.write(output_audio_path, generated_waveform, samplerate=24000)
                logger.debug(f"audio saved to {output_audio_path}")
            except:
                import traceback

                traceback.print_exc()

        if return_prompt:
            return answer, prompts_lists[0]
        else:
            return answer

    @torch.inference_mode()
    def _generate_speech_non_streaming(
        self,
        outputs,
        tts_bound,
        tts_proj_layer,
        audio_prompt,
        output_tts_inputs_embeds_path=None,
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
    ):
        last_hidden_states = [hs[tts_proj_layer] for hs in outputs.hidden_states]
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])

        spk_embeds = (
            torch.ones([0, self.tts.config.hidden_size]).to(last_hidden_states.device).to(last_hidden_states.dtype)
        )

        if self.tts.condition_type == "hidden_text_merge":
            llm_tokens = outputs["full_sequences"][0][tts_bound[0] : tts_bound[1]]
            llm_tokens = torch.tensor(llm_tokens, device=self.tts.emb_text.weight.device, dtype=torch.long)
            llm_embeds = self.tts.emb_text(llm_tokens)  # make sure emb_text is compatible with llm vocab size

            hidden_embeds = last_hidden_states[tts_bound[0] : tts_bound[1]]
            hidden_embeds = self.tts.projector_semantic(hidden_embeds)

            if self.tts.config.normalize_projected_hidden:
                hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

            tts_embeds = llm_embeds + hidden_embeds
        else:
            raise NotImplementedError

        audio_bos = [self.tts.audio_bos_token_id]
        audio_bos = torch.tensor(audio_bos, device=self.tts.emb_text.weight.device, dtype=torch.long)

        audio_bos_embeds = self.tts.emb_text(audio_bos)

        text_eos_embed = self.tts.emb_text(
            torch.tensor(
                [self.tts.config.text_eos_token_id],
                device=self.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        )

        inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds], dim=0).unsqueeze(0)

        # save inputs_embeds to file
        if output_tts_inputs_embeds_path:
            from safetensors.torch import save_file

            save_file({"inputs_embeds": inputs_embeds.contiguous()}, output_tts_inputs_embeds_path)

        outputs = self.tts.generate(
            inputs_embeds=inputs_embeds,
            sampling_params=tts_sampling_params,
            eos_token=torch.tensor(
                [self.tts.config.num_audio_tokens - 1],
                dtype=torch.long,
                device=self.tts.device,
            ),
        )

        import io

        import soundfile as sf

        generated_tokens = outputs.new_ids.squeeze(-1)
        reference_audio = audio_prompt
        if reference_audio is not None:
            logger.debug("use reference audio in data to generate waveform")
        with _temporary_wav(reference_audio) if reference_audio is not None else nullcontext() as prompt_wav_path:
            wav_bytes = self.tts.audio_tokenizer(
                generated_tokens.squeeze(0).tolist(),
                prompt_wav_path,
            )
        # convert wav bytes back to tensor for caller compatibility
        waveform, sr = sf.read(io.BytesIO(wav_bytes))
        return torch.tensor(waveform, dtype=torch.float32)

    @torch.inference_mode()
    def init_token2wav_cache(self, prompt_speech_16k):
        if hasattr(self.tts.audio_tokenizer, "set_stream_cache"):
            self.tts.audio_tokenizer.cache = None
            with _temporary_wav(prompt_speech_16k) as prompt_wav_path:
                flow_cache_base, hift_cache_base = self.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)

            self.token2wav_cache = {
                "flow_cache_base": torch_clone_recursive(flow_cache_base),
                "hift_cache_base": torch_clone_recursive(hift_cache_base),
            }
        else:
            model_input = self.tts.audio_tokenizer.frontend.frontend_token2wav(
                speech_tokens=torch.zeros(1, 1, dtype=torch.long, device=self.tts.device),
                speech_16k=None,
                prompt_speech_16k=prompt_speech_16k,
                resample_rate=self.tts.audio_tokenizer.sample_rate,
                prompt_speech=None,
            )

            prompt_token = model_input["flow_prompt_speech_token"]
            prompt_feat = model_input["prompt_speech_feat"]
            embedding = model_input["flow_embedding"]

            if self.tts.audio_tokenizer.fp16:
                prompt_feat = prompt_feat.to(torch.half)
                embedding = embedding.to(torch.half)

            prepared_cache = self.tts.audio_tokenizer.model.prepare_cache_from_prompt(
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
                n_timesteps=self.tts.config.s3_stream_n_timesteps,
                code_chunk_size=self.tts.config.s3_stream_chunk_size,
                chunk_prelook_size=self.tts.config.s3_stream_prelook_size,
                use_attn_idx=False,
            )

            self.token2wav_cache = prepared_cache

    # for sliding window
    def _ensure_dynamic_cache(self):
        cache = self.llm_past_key_values
        if cache is None:
            return None

        return cache if isinstance(cache, DynamicCache) else None

    def _get_kv_cache_length(self, cache=None):
        cache = cache if cache is not None else self.llm_past_key_values
        return get_kv_cache_length(cache)

    def _get_rope_theta(self) -> float:
        return config_rope_theta(self.llm.config)

    def _encode_text(self, tokenizer, text) -> Optional[torch.Tensor]:
        if tokenizer is None or not text:
            return None
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        return ids.to(self.device)

    @staticmethod
    def _safe_decode(tokenizer, input_ids):
        if tokenizer is None or input_ids is None:
            return None
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.cpu().tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            ids = input_ids
        try:
            return tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            return None

    def _finalize_round(
        self, round_id: Optional[int], cache_before: int, assistant_input_ids: Optional[torch.Tensor] = None
    ):
        if round_id is None:
            self._pending_round_id = None
            return
        cache_after = self._get_kv_cache_length()
        if assistant_input_ids is not None:
            assistant_len = assistant_input_ids.shape[1]
        else:
            assistant_len = max(cache_after - cache_before, 0)
        if assistant_len > 0:
            self._register_chunk(
                assistant_len,
                "assistant",
                round_id=round_id,
                input_ids=assistant_input_ids,
                tokenizer=self.processor.tokenizer if hasattr(self, "processor") else None,
            )

        self._pending_round_id = None
        self._next_round_id += 1

    def _register_chunk(
        self,
        seq_len: int,
        chunk_type: str,
        *,
        round_id: int,
        input_ids=None,
        tokenizer=None,
    ) -> None:
        if seq_len <= 0:
            return
        entry = {"length": int(seq_len), "type": chunk_type, "round": round_id}
        if input_ids is not None:
            entry["input_ids"] = input_ids.clone().detach()
            entry["decoded"] = self._safe_decode(tokenizer, entry["input_ids"])
        else:
            entry["input_ids"] = None
            entry["decoded"] = None
        self._omni_chunk_history.append(entry)

        if chunk_type == "system":
            self.streaming_text_preserve = max(self.streaming_text_preserve, entry["length"])

    def _drop_tokens_from_cache(self, length: int, cache: DynamicCache) -> bool:
        """Drop tokens from cache using the utility function."""
        _, new_offset, success = drop_tokens_from_cache(
            cache=cache,
            length=length,
            preserve=self.streaming_text_preserve,
            position_offset=self.streaming_position_offset,
            rope_theta=self._get_rope_theta(),
            inv_freq_cache=self._rope_inv_freq_cache,
        )
        if success:
            self.streaming_position_offset = new_offset
        return success

    def _drop_next_round(self, cache: DynamicCache) -> bool:
        seen_rounds = set()
        for entry in self._omni_chunk_history:
            round_id = entry.get("round")
            if round_id is None or round_id in seen_rounds:
                continue
            seen_rounds.add(round_id)
            round_entries = [e for e in self._omni_chunk_history if e.get("round") == round_id]
            if any(e.get("type") == "system" for e in round_entries):
                continue
            if self._drop_round(round_id, cache):
                return True
        return False

    def _drop_round(self, round_id: int, cache: DynamicCache) -> bool:
        entries = [e for e in self._omni_chunk_history if e.get("round") == round_id]
        if not entries:
            return False
        total_len = sum(e["length"] for e in entries)
        if total_len <= 0:
            for e in entries:
                self._omni_chunk_history.remove(e)
            return False
        if not self._drop_tokens_from_cache(total_len, cache):
            return False
        for e in entries:
            self._omni_chunk_history.remove(e)
        return True

    def _enforce_text_window(self) -> None:
        if not self.streaming_window_enabled:
            return
        cache = self._ensure_dynamic_cache()
        if cache is None:
            return
        high_limit = max(0, int(self.streaming_window_config.text_window_high_tokens))
        low_limit = max(0, int(self.streaming_window_config.text_window_low_tokens))
        if high_limit <= 0:
            return
        target = max(0, low_limit)
        total_len = self._get_kv_cache_length(cache)
        if total_len <= high_limit:
            return
        while total_len > target:
            if not self._drop_next_round(cache):
                break
            total_len = self._get_kv_cache_length(cache)

    # snapshot, vad
    def save_speculative_snapshot(self) -> SpeculativeSnapshot:
        """Internal method: save speculative snapshot.

        Called at the start of streaming_generate, saves to self._speculative_snapshot.

        Save strategy:
        - LLM KV Cache: only record length (restore by truncation, zero extra VRAM)
        - Audio KV Cache: deep clone (as generate sets it to None)
        - Mel processor: full state snapshot (including buffer)
        """
        # get LLM cache information
        llm_cache_length = self._get_kv_cache_length()
        llm_cache_checksum = None
        if self.llm_past_key_values is not None and cache_pairs(self.llm_past_key_values):
            llm_cache_checksum = cache_pairs(self.llm_past_key_values)[0][0].sum().item()

        # get audio cache length and clone audio_past_key_values
        audio_cache_length = 0
        audio_cache_checksum = None
        audio_past_key_values_clone = None
        if self.audio_past_key_values is not None:
            # The streaming Whisper encoder returns an EncoderDecoderCache (self-attention part used only).
            audio_cache = self.audio_past_key_values
            self_attn_cache = audio_cache.self_attention_cache if isinstance(audio_cache, EncoderDecoderCache) else audio_cache
            self_attn_pairs = cache_pairs(self_attn_cache)
            if self_attn_pairs:
                audio_cache_length = self_attn_pairs[0][0].shape[2]
                audio_cache_checksum = self_attn_pairs[0][0].sum().item()
            audio_past_key_values_clone = deepcopy(audio_cache)

        # get mel processor snapshot
        mel_processor_snapshot = None
        mel_buffer_checksum = None
        if hasattr(self, "processor") and self.processor is not None:
            mel_processor_snapshot = self.processor.get_streaming_snapshot()
            if mel_processor_snapshot:
                buf = mel_processor_snapshot.get("buffer")
                if buf is not None and len(buf) > 0:
                    mel_buffer_checksum = float(buf.sum())

        # save RNG state (important: for deterministic dithering and other random operations after restoration)
        rng_state_cpu = torch.get_rng_state()
        rng_state_cuda = None
        if torch.cuda.is_available() and self.device.type == "cuda":
            rng_state_cuda = torch.cuda.get_rng_state(self.device)

        # create snapshot
        snapshot = SpeculativeSnapshot(
            llm_cache_length=llm_cache_length,
            audio_cache_length=audio_cache_length,
            new_user_msg=self.new_user_msg,
            llm_generated=self.llm_generated,
            llm_generate_completed=self.llm_generate_completed,
            next_round_id=self._next_round_id,
            pending_round_id=self._pending_round_id,
            omni_chunk_history_length=len(self._omni_chunk_history),
            tts_last_turn_tokens=self.tts_last_turn_tokens.clone() if self.tts_last_turn_tokens is not None else None,
            audio_chunk_idx=self.audio_chunk_idx,
            mel_processor_snapshot=mel_processor_snapshot,
            audio_past_key_values=audio_past_key_values_clone,
            timestamp=time.time(),
            # debug fields
            llm_cache_checksum=llm_cache_checksum,
            audio_cache_checksum=audio_cache_checksum,
            mel_buffer_checksum=mel_buffer_checksum,
            # RNG state
            rng_state_cpu=rng_state_cpu,
            rng_state_cuda=rng_state_cuda,
        )

        return snapshot

    def restore_speculative_snapshot(self, snapshot=None) -> bool:
        """Restore speculative snapshot - called when VAD speculation fails.

        Restores model state to before streaming_generate was called,
        allowing continued streaming_prefill for newly arrived audio.

        Notes:
        - Snapshot is saved when streaming_generate is called with enable_speculative_snapshot=True
        - This method uses the most recent snapshot for restoration
        - Snapshot is cleared after restore, cannot be called repeatedly

        Returns:
            bool: Whether restoration was successful
        """
        snapshot = snapshot or getattr(self, "_speculative_snapshot", None)

        if snapshot is None:
            return False

        try:
            current_cache_length = self._get_kv_cache_length()
            current_history_length = len(self._omni_chunk_history)

            # 1. truncate LLM KV Cache
            if current_cache_length > snapshot.llm_cache_length:
                self._truncate_llm_cache(snapshot.llm_cache_length)

            # 2. restore Audio KV Cache (important: restore from cloned copy)
            # because streaming_generate will set audio_past_key_values to None
            self.audio_past_key_values = snapshot.audio_past_key_values

            # 3. restore session state
            self.new_user_msg = snapshot.new_user_msg
            self.llm_generated = snapshot.llm_generated
            self.llm_generate_completed = snapshot.llm_generate_completed

            # 4. restore Round management
            self._next_round_id = snapshot.next_round_id
            self._pending_round_id = snapshot.pending_round_id

            # 5. truncate chunk history
            if current_history_length > snapshot.omni_chunk_history_length:
                self._omni_chunk_history = self._omni_chunk_history[: snapshot.omni_chunk_history_length]

            # 6. restore TTS state
            self.tts_last_turn_tokens = snapshot.tts_last_turn_tokens

            # 7. restore streaming processor state
            self.audio_chunk_idx = snapshot.audio_chunk_idx

            # 8. restore mel processor state (important: otherwise subsequent prefill will fail due to frame number mismatch)
            if (
                snapshot.mel_processor_snapshot is not None
                and hasattr(self, "processor")
                and self.processor is not None
            ):
                self.processor.restore_streaming_snapshot(snapshot.mel_processor_snapshot)

            # 9. restore RNG state (important: ensure determinism of dithering and other random operations after restoration)
            if snapshot.rng_state_cpu is not None:
                torch.set_rng_state(snapshot.rng_state_cpu)
            if snapshot.rng_state_cuda is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(snapshot.rng_state_cuda, self.device)

            # 10. clean up temporary states generated during generation
            if hasattr(self, "_streaming_generated_token_ids"):
                del self._streaming_generated_token_ids
            if hasattr(self, "_last_streaming_text"):
                del self._last_streaming_text

            # 11. clear snapshot (can only be restored once)
            self._speculative_snapshot = None

            return True
        except Exception:
            import traceback

            logger.error(traceback.format_exc())
            return False

    def has_speculative_snapshot(self) -> bool:
        return getattr(self, "_speculative_snapshot", None) is not None

    def clear_speculative_snapshot(self) -> None:
        if hasattr(self, "_speculative_snapshot"):
            self._speculative_snapshot = None

    def _truncate_llm_cache(self, target_length: int) -> None:
        if self.llm_past_key_values is None:
            return

        cache = self._ensure_dynamic_cache()
        if cache is None:
            return

        current_length = self._get_kv_cache_length(cache)
        if current_length <= target_length:
            return

        # truncate each layer of cache
        for layer_idx, (keys, values) in enumerate(cache_pairs(cache)):
            if keys.numel() > 0:
                set_layer_states(
                    cache,
                    layer_idx,
                    keys[:, :, :target_length, :].contiguous(),
                    values[:, :, :target_length, :].contiguous(),
                )

    @torch.inference_mode()
    def streaming_prefill(
        self,
        session_id,
        msgs,
        omni_mode=True,
        max_slice_nums=None,
        use_tts_template=True,
        enable_thinking=False,
        is_last_chunk=False,  # for audio chunk, if is the last chunk, set to True
        tokenizer=None,
        processor=None,
        **kwargs,
    ):
        from PIL import Image

        assert session_id is not None, "session_id cannot be None"
        self.is_first = self.session_id is None or session_id != self.session_id

        self.prepare_processor(processor=processor, tokenizer=tokenizer)

        images = []
        audios = []

        assert len(msgs) == 1
        copy_msgs = deepcopy(msgs)
        msg = copy_msgs[0]

        assert msg["role"] in ["system", "user", "assistant"]
        is_not_system_prefill = msg["role"] != "system"

        content = msg["content"]
        cur_msgs = []
        for j, c in enumerate(content):
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("<image>./</image>")
            elif isinstance(c, np.ndarray):
                audios.append(c)
                cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur_msgs.append(c)
            else:
                logger.error(f"Invalid content type: {c}, ignore it.")

        cur_contents = "".join(cur_msgs) if omni_mode else "\n".join(cur_msgs)

        if msg["role"] in ["system", "assistant"]:
            self.new_user_msg = True
            self.audio_past_key_values = None

        if self.is_first:
            self.reset_session(reset_token2wav_cache=False)
            self.session_id = session_id

            self.init_streaming_processor()

            if msg["role"] == "user":
                # no system prefill, the first segment of the first user turn
                # do not use apply_chat_template, manually build prompt to avoid automatic addition of <|im_end|>
                prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False  # mark subsequent segments do not need to add user prefix anymore
            else:
                # system or assistant prefill, use apply_chat_template
                msg["content"] = cur_contents
                prompt = self.processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    use_tts_template=use_tts_template,
                    enable_thinking=enable_thinking,
                )
            add_special_tokens = True  # add bos
        else:
            # non-first prefill
            if self.new_user_msg and msg["role"] == "user":
                # the first segment of the new user turn
                if self.llm_generated:
                    if self.llm_generate_completed:
                        prompt = "<|im_end|>\n<|im_start|>user\n" + cur_contents
                    else:
                        prompt = "<|tts_eos|><|im_end|>\n<|im_start|>user\n" + cur_contents
                else:
                    prompt = "<|im_start|>user\n" + cur_contents
                self.new_user_msg = False
            else:
                # subsequent segments of the same turn, directly use content
                prompt = cur_contents
            add_special_tokens = False

        # when first user audio prefill, ensure audio length satisfies FIRST_CHUNK_MS requirements
        if is_not_system_prefill and len(audios) > 0 and self.audio_chunk_idx == 0:
            assert len(audios) == 1, f"streaming mode only supports single audio, currently {len(audios)}"
            first_chunk_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
            if len(audios[0]) < first_chunk_samples:
                pad_len = first_chunk_samples - len(audios[0])
                audios[0] = np.concatenate([np.zeros(pad_len, dtype=audios[0].dtype), audios[0]])

        model_inputs = self.processor(
            [prompt],
            [images],
            [audios],
            max_slice_nums=1 if max_slice_nums is None else max_slice_nums,
            use_image_id=False,
            chunk_input=True,
            return_tensors="pt",
            max_length=None,
            sampling_rate=16000,
            add_special_tokens=add_special_tokens,
            online_streaming=is_not_system_prefill,
            audio_chunk_idx=self.audio_chunk_idx,
            is_last_chunk=is_last_chunk,
        ).to(self.device)

        if len(audios) > 0 and is_not_system_prefill:
            self.audio_chunk_idx += 1

        # 1. prepare input embeddings
        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        # get audio embedding with audio_past_key_values
        inputs_embeds = self.get_omni_embedding(
            model_inputs, input_embeddings=model_inputs["inputs_embeds"], stream_input=is_not_system_prefill
        )

        if self.is_first:
            self.audio_past_key_values = None

        round_id = self._next_round_id
        self._pending_round_id = round_id
        chunk_type = "system" if msg["role"] == "system" else ("user" if msg["role"] == "user" else "assistant")
        seq_len = inputs_embeds.shape[1]
        self._enforce_text_window()
        cache_length = self._get_kv_cache_length()

        attention_mask = torch.ones((1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)

        # 2. do prefill
        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            use_cache=True,
            return_dict=True,
        )

        self.llm_past_key_values = outputs["past_key_values"]
        self._register_chunk(
            seq_len,
            chunk_type,
            round_id=round_id,
            input_ids=model_inputs["input_ids"],
            tokenizer=self.processor.tokenizer,
        )
        self._enforce_text_window()
        if self.force_rope_reindex:
            self._force_reindex_all_cache()

        return prompt

    @torch.inference_mode()
    def streaming_generate(
        self,
        session_id,
        bos_input=None,
        generate_audio=True,
        audio_token_chunk_size=25,  # 25 token/s
        tts_sampling_params: TTSSamplingParams = TTSSamplingParams(),
        max_new_tokens=256,
        enable_thinking=False,
        use_tts_template=True,
        do_sample=True,
        enable_speculative_snapshot=False,
        tokenizer=None,
        processor=None,
        # Teacher forcing (only for the "text → hidden → TTS condition" pipeline in streaming_generate)
        # When enabled: instead of letting the LLM auto-regressively generate the text to be spoken,
        # it forces the tokens from teacher_forcing_text to be fed in, using the hidden states
        # corresponding to these tokens to construct the TTS condition, ensuring the output audio matches the input text.
        teacher_forcing: bool = False,
        teacher_forcing_text: str = "",
        **kwargs,
    ):
        # save speculative snapshot (before modifying any state)
        # for VAD speculative snapshot: if speculative snapshot fails, can call restore_speculative_snapshot() to restore
        # enable_speculative_snapshot=True when enabled, skip (save some overhead) when disabled
        if enable_speculative_snapshot:
            self._speculative_snapshot = self.save_speculative_snapshot()

        # reset buf
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = False
        self.audio_past_key_values = None

        self.prepare_processor(processor=processor, tokenizer=tokenizer)

        # reset current turn generated token IDs
        if hasattr(self, "_streaming_generated_token_ids"):
            del self._streaming_generated_token_ids
        # reset full generated text
        if hasattr(self, "_last_streaming_text"):
            del self._last_streaming_text

        cache = self._ensure_dynamic_cache()
        cache_length = self._get_kv_cache_length(cache)
        host_round_id = self._pending_round_id

        ## in single-turn streaming, each call to streaming_generate needs to reinitialize the streaming_processor, enter the next turn
        self.init_streaming_processor()

        # 1) llm generate token and hidden states per chunk=10, 2) tts generate audio token chunk per chunk=25, 3) yield 1 chunk audio token
        def audio_chunk_generator(
            bos_input,
            tokenizer,
            generate_audio,
            tts_sampling_params,
            max_new_tokens,
            do_sample,
            teacher_forcing=False,
            teacher_forcing_text="",
            **kwargs,
        ):
            generate_chunk_size = 10

            if bos_input is None:
                bos_input = "".join(
                    [
                        "<|im_end|>\n<|im_start|>assistant\n",
                        "" if enable_thinking else self.think_str.replace("\\n", "\n"),
                        "<|tts_bos|>" if use_tts_template else "",
                    ]
                )

            bos_input_ids = tokenizer.encode(bos_input)
            bos_input_ids = torch.tensor(bos_input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            bos_input_embeds = self.llm.get_input_embeddings()(bos_input_ids)

            generation_inputs_embeds = bos_input_embeds
            generated_ids = torch.empty((1, 0), dtype=torch.long, device=self.device)

            num_chunks_decode = (max_new_tokens + generate_chunk_size - 1) // generate_chunk_size

            conditions = []

            # generate chunk by chunk, each chunk has 10 tokens, each chunk takes last hidden states, and pass tokens to tts
            llm_streaming_generator = ChunkPrefillChunkGenerate(
                model=self.llm,
                tokenizer=tokenizer,
                terminators=["<|tts_eos|>", "<|im_end|>", "</s>"],
            )

            if generate_audio:
                logits_warpers, logits_processors = gen_logits(
                    num_code=self.tts.config.num_audio_tokens,
                    repetition_penalty=tts_sampling_params.repetition_penalty,
                    top_p=tts_sampling_params.top_p,
                    top_k=tts_sampling_params.top_k,
                )

                tts_streaming_generator = TTSStreamingGenerator(
                    model=self.tts,
                    temperature=tts_sampling_params.temperature,
                    eos_token=torch.tensor(
                        [self.tts.config.num_audio_tokens - 1],
                        dtype=torch.long,
                        device=self.tts.device,
                    ),
                    chunk_size=audio_token_chunk_size,  # s3tokenizer 1s = 25token
                    tts_last_turn_tokens=self.tts_last_turn_tokens,
                    logits_processors=logits_processors,
                    logits_warpers=logits_warpers,
                )

            # Teacher forcing branch
            # This branch does not rely on ChunkPrefillChunkGenerate's sampling logic, instead:
            # 1) First prefill bos_input (assistant + tts_bos) into llm_past_key_values
            # 2) Tokenize teacher_forcing_text into token ids
            # 3) Feed tokens one by one into the LLM (teacher forcing), obtaining the last_hidden_states for each token
            # 4) Use (token_ids, hidden_states) to construct tts condition, then feed it to TTSStreamingGenerator
            if teacher_forcing:
                # --- 1) prefill bos_input，延续 streaming_prefill 的 KV cache ---
                bos_outputs = self.llm(
                    inputs_embeds=generation_inputs_embeds,
                    past_key_values=self.llm_past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                self.llm_past_key_values = bos_outputs.past_key_values

                if generate_audio:
                    # Give a length-0 tensor as speaker embedding (no speaker embedding)
                    spk_emb = torch.empty(
                        (bos_input_embeds.shape[0], 0, bos_input_embeds.shape[2]),
                        dtype=bos_input_embeds.dtype,
                        device=bos_input_embeds.device,
                    )
                    tts_streaming_generator.spk_emb = spk_emb

                # --- 2) tokenize teacher_forcing_text ---
                tf_text = teacher_forcing_text or ""
                try:
                    forced_input_ids = tokenizer(tf_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
                except Exception:
                    # Compatible with rare tokenizer return object attributes
                    forced_input_ids = tokenizer(tf_text, add_special_tokens=False, return_tensors="pt").input_ids
                forced_input_ids = forced_input_ids.to(self.device)

                total_len = int(forced_input_ids.shape[1])
                ptr = 0

                # Special case: empty text should also let TTS finish (text_finished=True will automatically concatenate text_eos_embed)
                if total_len == 0:
                    if not generate_audio:
                        yield forced_input_ids, True
                        return
                    empty_tts_embeds = torch.empty(
                        (1, 0, self.tts.config.hidden_size),
                        dtype=bos_input_embeds.dtype,
                        device=self.device,
                    )
                    if not hasattr(self, "_streaming_generated_token_ids"):
                        self._streaming_generated_token_ids = []
                    tts_generator = tts_streaming_generator.generate_with_buffer(
                        condition=empty_tts_embeds,
                        text_finished=True,
                    )
                    for audio_token_chunk, is_last_audio_chunk in tts_generator:
                        yield audio_token_chunk, is_last_audio_chunk
                    self.tts_last_turn_tokens = tts_streaming_generator.tts_last_turn_tokens
                    self._last_streaming_text = ""
                    yield None, None
                    return

                # --- 3) chunk-by-chunk teacher forcing ---
                while ptr < total_len:
                    end = min(ptr + generate_chunk_size, total_len)
                    chunk_ids = forced_input_ids[:, ptr:end]  # [1, chunk_len]
                    chunk_hidden_list = []

                    for j in range(chunk_ids.shape[1]):
                        tok = chunk_ids[:, j : j + 1]  # [1, 1]
                        tok_emb = self.llm.get_input_embeddings()(tok)
                        out = self.llm(
                            inputs_embeds=tok_emb,
                            past_key_values=self.llm_past_key_values,
                            use_cache=True,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        self.llm_past_key_values = out.past_key_values
                        chunk_hidden_list.append(out.hidden_states[-1])  # [1, 1, hidden]

                    chunk_hidden = torch.cat(chunk_hidden_list, dim=1)  # [1, chunk_len, hidden]
                    text_finished = end >= total_len

                    # Save token IDs cache (external eval script will use _last_streaming_text to write generated_text)
                    if not hasattr(self, "_streaming_generated_token_ids"):
                        self._streaming_generated_token_ids = []
                    self._streaming_generated_token_ids.extend(chunk_ids[0].tolist())

                    if not generate_audio:
                        yield chunk_ids, text_finished
                    else:
                        llm_embeds = self.tts.emb_text(chunk_ids)
                        hidden_embeds = self.tts.projector_semantic(chunk_hidden)
                        if self.tts.config.normalize_projected_hidden:
                            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
                        tts_embeds = llm_embeds + hidden_embeds

                        tts_generator = tts_streaming_generator.generate_with_buffer(
                            condition=tts_embeds,
                            text_finished=text_finished,
                        )
                        for audio_token_chunk, is_last_audio_chunk in tts_generator:
                            yield audio_token_chunk, is_last_audio_chunk

                    ptr = end
                    if text_finished:
                        if generate_audio:
                            self.tts_last_turn_tokens = tts_streaming_generator.tts_last_turn_tokens
                        break

                # Finish: decode this round of text
                if hasattr(self, "_streaming_generated_token_ids"):
                    try:
                        self._last_streaming_text = tokenizer.decode(self._streaming_generated_token_ids)
                        assistant_input_ids = self._encode_text(tokenizer=tokenizer, text=self._last_streaming_text)
                        self._finalize_round(
                            round_id=host_round_id, cache_before=cache_length, assistant_input_ids=assistant_input_ids
                        )
                    except Exception:
                        self._last_streaming_text = None
                else:
                    self._last_streaming_text = None

                # Finally send the end signal
                if generate_audio:
                    yield None, None
                else:
                    return
                return

            # LLM chunk generate outer loop
            for chunk_idx in range(num_chunks_decode):
                is_first_generate_chunk = chunk_idx == 0

                output = llm_streaming_generator.chunk_generate(
                    inputs_embeds=generation_inputs_embeds,
                    past_key_values=self.llm_past_key_values,
                    is_first_generate_chunk=is_first_generate_chunk,
                    return_hidden_states=True,
                    chunk_size=generate_chunk_size + 1 * is_first_generate_chunk,
                    do_sample=do_sample,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.8),
                    top_k=kwargs.get("top_k", 100),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.02),
                    length_penalty=kwargs.get("length_penalty", 1.0),
                    all_input_ids=generated_ids,
                )

                if output.chunk_token_ids is None:
                    break

                if is_first_generate_chunk:
                    if generate_audio:
                        spk_emb = torch.empty(
                            (bos_input_embeds.shape[0], 0, bos_input_embeds.shape[2]),
                            dtype=bos_input_embeds.dtype,
                            device=bos_input_embeds.device,
                        )
                        tts_streaming_generator.spk_emb = spk_emb

                    if output.finished:
                        yield_chunk_token_ids = output.chunk_token_ids
                    else:
                        # the first chunk generated chunk_size + 1 tokens, we only take the first chunk_size tokens,
                        # the last token is not prefilled, and last hidden states is not obtained
                        yield_chunk_token_ids = output.chunk_token_ids[:, :-1]

                elif output.finished:
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], output.chunk_token_ids], dim=1)
                else:
                    # in the chunk that is not the first chunk, we need to add the token at the end of the previous chunk,
                    # it is not prefilled into the model to get last hidden states
                    # similarly, the last generated token of subsequent chunks is not prefilled, and last hidden states is not obtained,
                    # so it is not passed out
                    yield_chunk_token_ids = torch.cat([generated_ids[:, -1:], output.chunk_token_ids[:, :-1]], dim=1)

                if not generate_audio:
                    tokenizer.decode(yield_chunk_token_ids[0])
                    yield yield_chunk_token_ids, output.finished
                else:
                    # TTS inner loop
                    # dense connection here is hardcoded to use text-hidden merged as condition
                    llm_embeds = self.tts.emb_text(yield_chunk_token_ids)
                    hidden_embeds = output.last_hidden_states
                    hidden_embeds = self.tts.projector_semantic(hidden_embeds)
                    if self.tts.config.normalize_projected_hidden:  # default should be opened
                        hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

                    tts_embeds = llm_embeds + hidden_embeds
                    conditions.append(tts_embeds)

                    # Store token IDs instead of decoded text to avoid UTF-8 multi-byte character truncation
                    if not hasattr(self, "_streaming_generated_token_ids"):
                        self._streaming_generated_token_ids = []
                    self._streaming_generated_token_ids.extend(yield_chunk_token_ids[0].tolist())

                    # there is buffer generated, each time exactly returns 25 audio tokens,
                    # the last audio chunk returns audio tokens of variable length, length [0, 25]
                    tts_generator = tts_streaming_generator.generate_with_buffer(
                        condition=tts_embeds, text_finished=output.finished
                    )

                    for audio_token_chunk, is_last_audio_chunk in tts_generator:
                        yield audio_token_chunk, is_last_audio_chunk

                generated_ids = torch.cat([generated_ids, output.chunk_token_ids], dim=1)
                generation_inputs_embeds = output.current_inputs_embeds
                self.llm_past_key_values = output.past_key_values

                if output.finished:
                    if generate_audio:
                        self.tts_last_turn_tokens = tts_streaming_generator.tts_last_turn_tokens
                    break

            # IMPORTANT: Flush remaining TTS buffer when LLM generation ends
            # This handles BOTH cases:
            # 1. LLM finished with terminator (output.finished=True) - buffer may still have tokens
            # 2. LLM hit max chunks limit (output.finished=False) - buffer definitely has tokens
            if generate_audio:
                if len(tts_streaming_generator._token_buffer) > 0:
                    batch = torch.cat(tts_streaming_generator._token_buffer, dim=1)
                    yield batch, True
                    tts_streaming_generator._token_buffer = []

            if generate_audio:
                if hasattr(self, "_streaming_generated_token_ids"):
                    try:
                        self._last_streaming_text = tokenizer.decode(self._streaming_generated_token_ids)
                        assistant_input_ids = self._encode_text(tokenizer=tokenizer, text=self._last_streaming_text)
                        self._finalize_round(
                            round_id=host_round_id, cache_before=cache_length, assistant_input_ids=assistant_input_ids
                        )
                    except Exception:
                        self._last_streaming_text = None
                else:
                    self._last_streaming_text = None

                yield None, None
            else:
                return

        # iter for generating text chunk and audio chunk
        audio_chunk_generator_iter = audio_chunk_generator(
            bos_input=bos_input,
            tokenizer=self.processor.tokenizer,
            generate_audio=generate_audio,
            tts_sampling_params=tts_sampling_params,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            teacher_forcing=teacher_forcing,
            teacher_forcing_text=teacher_forcing_text,
            **kwargs,
        )

        if generate_audio:
            if self.tts.config.audio_tokenizer_type == "s3tokenizer_step_audio":
                self.tts.audio_tokenizer.stream_cache = torch_clone_recursive(self.token2wav_cache["flow_cache_base"])
                self.tts.audio_tokenizer.hift_cache_dict = torch_clone_recursive(
                    self.token2wav_cache["hift_cache_base"]
                )

                # pre-insert 3-5 prefix 4218 silence tokens, each token corresponds to 0.04s,
                # adding 5 tokens means introducing 0.2s of silence
                buffer = [4218] * 3
                pre_lookahead = 3
                CHUNK_SIZE = 25
                chunk_idx = 0
                prev_text_len = 0  # track text position for streaming text output
                for audio_token_chunk, is_last_audio_chunk in audio_chunk_generator_iter:
                    if audio_token_chunk is None:
                        break

                    buffer += audio_token_chunk.reshape(-1).tolist()

                    if len(buffer) >= CHUNK_SIZE + pre_lookahead:
                        waveform_chunk = self.tts.audio_tokenizer.stream(
                            buffer[: CHUNK_SIZE + pre_lookahead],
                            prompt_wav=None,
                            last_chunk=is_last_audio_chunk,
                            return_waveform=True,
                        )

                        waveform_chunk = torch.from_numpy(waveform_chunk)

                        # get new text chunk corresponding to this waveform
                        # Decode from accumulated token IDs to avoid UTF-8 multi-byte truncation
                        new_text = ""
                        if hasattr(self, "_streaming_generated_token_ids"):
                            current_text = self.processor.tokenizer.decode(self._streaming_generated_token_ids)
                            # Filter out trailing replacement characters (incomplete UTF-8 sequences)
                            safe_end = len(current_text)
                            while safe_end > 0 and current_text[safe_end - 1] == "\ufffd":
                                safe_end -= 1
                            safe_text = current_text[:safe_end]
                            new_text = safe_text[prev_text_len:]
                            prev_text_len = len(safe_text)

                        yield waveform_chunk, new_text

                        buffer = buffer[CHUNK_SIZE:]
                        chunk_idx += 1

                # flush rest
                if len(buffer) > 0:
                    waveform_chunk = self.tts.audio_tokenizer.stream(
                        buffer,
                        prompt_wav=None,
                        last_chunk=True,
                        return_waveform=True,
                    )

                    waveform_chunk = torch.from_numpy(waveform_chunk)

                    # get remaining new text for the final chunk
                    # Final chunk: decode all remaining text without filtering
                    new_text = ""
                    if hasattr(self, "_streaming_generated_token_ids"):
                        current_text = self.processor.tokenizer.decode(self._streaming_generated_token_ids)
                        new_text = current_text[prev_text_len:]
                        prev_text_len = len(current_text)

                    yield waveform_chunk, new_text

                # maybe the buffer is empty, and text is not empty, should we flush text without wave?
            else:
                raise NotImplementedError(f"not supported audio tokenizer: {self.tts.config.audio_tokenizer_type}")
        else:
            # For text-only generation, decode tokens and handle partial multi-byte characters
            yield from streaming_token_decoder(
                audio_chunk_generator_iter,
                self.processor.tokenizer,
                skip_special_tokens=False,
            )

    def as_duplex(self, device: Optional[str] = None, **kwargs) -> "MiniCPMODuplex":
        """Convert this MiniCPMO instance to MiniCPMODuplex for full-duplex streaming."""
        return MiniCPMODuplex.from_existing_model(
            model=self,
            device=device,
            **kwargs,
        )


class MiniCPMODuplex:
    """MiniCPMODuplex model with full-duplex streaming capabilities.

    This is a wrapper class that provides duplex streaming functionality.
    Use MiniCPMO.as_duplex() to create from an existing model without reloading.
    """

    # Default duplex parameters
    _default_duplex_params = {
        "generate_audio": True,
        "ls_mode": "explicit",
        "max_new_speak_tokens_per_chunk": 20,
        "text_repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_k": 100,
        "top_p": 0.8,
        "text_repetition_window_size": 512,
        "listen_prob_scale": 1.0,
        "force_listen_count": 0,
        "tts_temperature": 0.8,
        "tts_repetition_penalty": 1.05,
        "enable_float16": False,
        "n_timesteps": 10,
        "chunk_ms": 1000,
        "first_chunk_ms": 1035,
        "cnn_redundancy_ms": 20,
        "sample_rate": 16000,
        "sliding_window_mode": "off",
        "basic_window_high_tokens": 8000,
        "basic_window_low_tokens": 6000,
        "context_previous_max_tokens": 500,
        "context_max_units": 24,
    }

    @classmethod
    def from_existing_model(
        cls,
        model: "MiniCPMO",
        device: Optional[str] = None,
        *,
        model_dir=None,
        trust_remote_code=False,
        **kwargs,
    ) -> "MiniCPMODuplex":
        """Create MiniCPMODuplex from an existing MiniCPMO instance."""
        # Create instance without calling __init__
        instance = cls.__new__(cls)

        instance.name_or_path = getattr(model.config, "_name_or_path", "")

        # Get default params helper
        def get_param(name):
            if name in kwargs:
                return kwargs[name]
            return cls._default_duplex_params.get(name)

        instance.generate_audio = get_param("generate_audio")
        instance.ls_mode = get_param("ls_mode")

        # Determine device
        if device is not None:
            instance.device = device
        else:
            try:
                instance.device = str(next(model.parameters()).device)
            except StopIteration:
                instance.device = "cuda"

        # Reuse the existing model - THIS IS THE KEY: no reloading!
        instance.model = model
        instance.processor = getattr(model, "processor", None)
        instance.tokenizer = getattr(instance.processor, "tokenizer", None) if instance.processor else None

        if instance.tokenizer is None:
            from ....utils.hf import load_hf_tokenizer

            instance.tokenizer = load_hf_tokenizer(
                instance.name_or_path,
                config=model.config,
                model_config=model.config,
                trust_remote_code=False,
            )

        if instance.processor is None:
            from .processing_minicpmo import MiniCPMOProcessor

            instance.processor = MiniCPMOProcessor.from_pretrained(
                instance.name_or_path,
                config=model.config,
                tokenizer=instance.tokenizer,
            )
            instance.processor.tokenizer = instance.tokenizer

        # Ensure model has processor reference (same as __init__)
        instance.model.processor = instance.processor

        codec = getattr(getattr(model, "tts", None), "audio_tokenizer", None)
        if instance.generate_audio and (model_dir is not None or codec is None):
            model.init_tts(
                model_dir=model_dir,
                enable_float16=get_param("enable_float16"),
                n_timesteps=get_param("n_timesteps"),
                trust_remote_code=trust_remote_code,
            )

        instance.break_event = threading.Event()
        instance.session_stop_event = threading.Event()

        # LLM generation config
        instance.max_new_speak_tokens_per_chunk = get_param("max_new_speak_tokens_per_chunk")
        instance.text_repetition_penalty = get_param("text_repetition_penalty")
        instance.temperature = get_param("temperature")
        instance.top_k = get_param("top_k")
        instance.top_p = get_param("top_p")
        instance.text_repetition_window_size = get_param("text_repetition_window_size")
        instance.listen_prob_scale = get_param("listen_prob_scale")
        instance.force_listen_count = get_param("force_listen_count")

        # TTS generation config
        tts_temp_value = get_param("tts_temperature")
        instance.tts_temperature = torch.tensor([tts_temp_value], dtype=torch.float, device=instance.device)
        instance.tts_repetition_penalty = get_param("tts_repetition_penalty")

        # Stream config
        instance.CHUNK_MS = get_param("chunk_ms")
        instance.FIRST_CHUNK_MS = get_param("first_chunk_ms")
        instance.CNN_REDUNDANCY_MS = get_param("cnn_redundancy_ms")
        instance.SAMPLE_RATE = get_param("sample_rate")

        instance.model.CHUNK_MS = instance.CHUNK_MS
        instance.model.FIRST_CHUNK_MS = instance.FIRST_CHUNK_MS
        instance.model.CNN_REDUNDANCY_MS = instance.CNN_REDUNDANCY_MS
        instance.model.SAMPLE_RATE = instance.SAMPLE_RATE

        # Special tokens
        instance.unit_token_id = instance.tokenizer.convert_tokens_to_ids("<unit>")
        instance.image_start_token_id = instance.tokenizer.convert_tokens_to_ids("<image>")
        instance.image_end_token_id = instance.tokenizer.convert_tokens_to_ids("</image>")
        instance.slice_start_token_id = instance.tokenizer.convert_tokens_to_ids("<slice>")
        instance.slice_end_token_id = instance.tokenizer.convert_tokens_to_ids("</slice>")

        instance.listen_token_id = instance.tokenizer.convert_tokens_to_ids("<|listen|>")
        instance.speak_token_id = instance.tokenizer.convert_tokens_to_ids("<|speak|>")
        instance.tts_bos_token_id = instance.tokenizer.convert_tokens_to_ids("<|tts_bos|>")
        instance.tts_eos_token_id = instance.tokenizer.convert_tokens_to_ids("<|tts_eos|>")

        instance.chunk_eos_token_id = instance.tokenizer.convert_tokens_to_ids("<|chunk_eos|>")
        instance.chunk_tts_eos_token_id = instance.tokenizer.convert_tokens_to_ids("<|chunk_tts_eos|>")
        instance.turn_eos_token_id = instance.tokenizer.convert_tokens_to_ids("<|turn_eos|>")

        instance.chunk_terminator_token_ids = [
            instance.listen_token_id,
            instance.chunk_eos_token_id,
            instance.chunk_tts_eos_token_id,
        ]
        instance.turn_terminator_token_ids = [instance.turn_eos_token_id]
        instance.chunk_speak_token_ids = [instance.speak_token_id]

        instance.tts_pad_id = instance.tokenizer.convert_tokens_to_ids("<|tts_pad|>")
        bad_token_ids = getattr(instance.tokenizer, "bad_token_ids", [])
        instance.forbidden_token_ids = [instance.tts_pad_id] + list(bad_token_ids)

        from .utils import StreamDecoder

        instance.decoder = StreamDecoder(
            llm=instance.model.llm, tokenizer=instance.tokenizer, forbidden_token_ids=instance.forbidden_token_ids
        )

        # Sliding window config
        sliding_window_mode = get_param("sliding_window_mode")
        basic_window_high_tokens = get_param("basic_window_high_tokens")
        basic_window_low_tokens = get_param("basic_window_low_tokens")
        context_previous_max_tokens = get_param("context_previous_max_tokens")
        context_max_units = get_param("context_max_units")

        instance.decoder.set_window_config(
            DuplexWindowConfig(
                sliding_window_mode=sliding_window_mode,
                basic_window_high_tokens=basic_window_high_tokens,
                basic_window_low_tokens=basic_window_low_tokens,
                context_previous_max_tokens=context_previous_max_tokens,
                context_max_units=context_max_units,
            )
        )
        window_enabled = sliding_window_mode != "off"
        instance.decoder.set_window_enabled(window_enabled)

        instance.tts_logits_processors = None
        instance.tts_eos_token = None
        if instance.generate_audio:
            instance.tts_logits_processors = gen_logits(
                num_code=instance.model.tts.config.num_audio_tokens,
                repetition_penalty=instance.tts_repetition_penalty,
            )
            instance.tts_eos_token = torch.tensor(
                [instance.model.tts.config.num_audio_tokens - 1],
                dtype=torch.long,
                device=instance.device,
            )

        instance._reset_streaming_state()

        return instance

    def set_break_event(self):
        self.break_event.set()

    def clear_break_event(self):
        self.break_event.clear()

    def set_session_stop(self):
        self.session_stop_event.set()
        self.break_event.set()

    def clear_session_stop(self):
        self.session_stop_event.clear()

    def is_break_set(self) -> bool:
        return self.break_event.is_set()

    def is_session_stop_set(self) -> bool:
        return self.session_stop_event.is_set()

    def _init_token2wav_cache(self, prompt_wav_path: str):
        self.model.tts.audio_tokenizer.cache = None
        flow_cache, hift_cache = self.model.tts.audio_tokenizer.set_stream_cache(prompt_wav_path)
        self.flow_cache_base = torch_clone_recursive(flow_cache)
        self.hift_cache_base = torch_clone_recursive(hift_cache)
        self.pre_lookahead = int(self.model.tts.audio_tokenizer.flow.pre_lookahead_len)
        self.token2wav_initialized = True

    def _reset_token2wav_for_new_turn(self):
        if self.token2wav_initialized:
            self.model.tts.audio_tokenizer.stream_cache = torch_clone_recursive(self.flow_cache_base)
            self.model.tts.audio_tokenizer.hift_cache_dict = torch_clone_recursive(self.hift_cache_base)
            self.token2wav_buffer = [4218] * 3  # silence token prefix

    def _reset_streaming_state(self):
        self.audio_chunk_idx = 0
        self.current_turn_ended = True
        self.speak_count = 0
        self.res_ids = []
        self.total_ids = []
        self.total_hidden = []

        # TTS state
        self.tts_text_start_pos = 0
        self.tts_past_key_values = None
        self.tts_current_turn_start_time = None

        # token2wav state
        self.token2wav_initialized = False
        self.token2wav_buffer = []
        self.flow_cache_base = None
        self.hift_cache_base = None

        # Audio prefill state
        self.audio_buffer = np.array([], dtype=np.float32)
        self.pending_logits: Optional[torch.Tensor] = None
        self.current_mode: Optional[str] = None

        # Force listen state
        self._streaming_generate_count = 0

        # Schema tracking: record the complete prefill + generate token sequence
        # prefill_schema_tokens: each element is a list of prefill tokens for a unit
        # format: [[unit0_prefill_tokens], [unit1_prefill_tokens], ...]
        self.prefill_schema_tokens = []
        self._current_unit_prefill_tokens = []

    def prepare(
        self,
        prefix_system_prompt: Optional[str] = None,
        ref_audio: Optional[np.ndarray] = None,
        prompt_wav_path: Optional[str] = None,
        context_previous_marker: str = "\n\nprevious: ",
        **kwargs,
    ):
        prefix_system_prompt = prefix_system_prompt or "Streaming Omni Conversation."

        prefix_system_prompt = "<|im_start|>system\n" + prefix_system_prompt
        suffix_system_prompt = "<|im_end|>"
        if isinstance(ref_audio, np.ndarray):
            prefix_system_prompt += "\n<|audio_start|>"
            suffix_system_prompt = "<|audio_end|>" + suffix_system_prompt

        self.clear_break_event()
        self.clear_session_stop()

        self._reset_streaming_state()
        self.decoder.reset()

        self.model.init_streaming_processor()

        if prompt_wav_path is not None and prompt_wav_path and self.generate_audio:
            self._init_token2wav_cache(prompt_wav_path)
            self._reset_token2wav_for_new_turn()

        # Prefill system prompt prefix
        if prefix_system_prompt:
            tokens = self.tokenizer.encode(prefix_system_prompt, add_special_tokens=False)
            for token_id in tokens:
                self.decoder.feed(self.decoder.embed_token(token_id))

        # Prefill reference audio
        if ref_audio is not None:
            data = self.processor.process_audio([ref_audio])
            embeds_nested = self.model.get_audio_embedding(data, chunk_length=self.model.config.audio_chunk_length)
            embeds = torch.cat([t for g in embeds_nested for t in g], dim=0) if embeds_nested else None
            if embeds is not None:
                self.decoder.feed(embeds)

        # register system prompt protection length (protect this part from being removed when sliding window is enabled)
        if prefix_system_prompt or suffix_system_prompt or ref_audio is not None:
            if self.decoder._window_config.sliding_window_mode == "context":
                # Context preserve mode:
                # initial layout: [prefix] [suffix] [units...]
                # after the first sliding window: [prefix] [context_previous_marker + content] [suffix] [units...]
                # register prefix length first, then feed suffix
                self._prefix_system_prompt = prefix_system_prompt
                self._suffix_system_prompt = suffix_system_prompt
                self._ref_audio = ref_audio

                suffix_token_ids = []
                if suffix_system_prompt:
                    suffix_token_ids = self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)

                # register (when cache only has prefix, no suffix, no previous)
                self.decoder.register_system_prompt_with_context(
                    suffix_token_ids=suffix_token_ids,
                    context_previous_marker=context_previous_marker,  # dynamically added after the first sliding window
                )

                # now feed suffix
                for token_id in suffix_token_ids:
                    self.decoder.feed(self.decoder.embed_token(token_id))
            else:
                # non-context preserve mode: first feed suffix, then register total length
                if suffix_system_prompt:
                    tokens = self.tokenizer.encode(suffix_system_prompt, add_special_tokens=False)
                    for token_id in tokens:
                        self.decoder.feed(self.decoder.embed_token(token_id))
                self.decoder.register_system_prompt()

        if prefix_system_prompt or suffix_system_prompt:
            if ref_audio is not None:
                full_prompt = (prefix_system_prompt or "") + "[audio embedding]" + (suffix_system_prompt or "")
            else:
                full_prompt = (prefix_system_prompt or "") + (suffix_system_prompt or "")

            return full_prompt

        return ""

    @torch.no_grad()
    def streaming_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[list] = None,
        text_list: Optional[list] = None,
        max_slice_nums: Union[int, List[int]] = 1,
        batch_vision_feed: bool = False,
    ):
        """Streaming prefill - called once per second, processing audio/video data

        Args:
            audio_waveform: audio waveform data
            frame_list: image frame list
            text_list: text
            max_slice_nums: maximum number of slices for HD image encoding (default 1, no slicing)
                           Can be an int (same for all images) or a list matching frame_list length
            batch_vision_feed: if True, batch all vision embeddings into a single feed call for better performance.
                              if False (default), feed each embedding individually (original behavior).

        Process:
            0. determine mode based on input: AUDIO / VISION / OMNI
            1. feed <unit> token
            2. get and feed image embed (if frame_list) - return pending logits in VISION MODE
            3. get and feed audio embed (if audio_waveform) - return pending logits in AUDIO/OMNI MODE

        Returns:
            dict with keys:
                - success: bool
                - cost_vision_process: float (image processing time)
                - cost_vision_embed: float (vision embedding time)
                - cost_vision_feed: float (vision feed time)
                - cost_audio_process: float (audio processing time)
                - cost_audio_embed: float (audio embedding time)
                - cost_audio_feed: float (audio feed time)
                - cost_all: float (total time)
        """
        start_time = time.time()
        cost_vision_process = 0.0
        cost_vision_embed = 0.0
        cost_vision_feed = 0.0
        cost_audio_process = 0.0
        cost_audio_embed = 0.0
        cost_audio_feed = 0.0

        def _make_result(success, reasons=""):
            reason = reasons
            if isinstance(reasons, list):
                reason = "; ".join(reasons)

            return {
                "success": success,
                "reason": reason,
                "cost_vision_process": cost_vision_process,
                "cost_vision_embed": cost_vision_embed,
                "cost_vision_feed": cost_vision_feed,
                "cost_audio_process": cost_audio_process,
                "cost_audio_embed": cost_audio_embed,
                "cost_audio_feed": cost_audio_feed,
                "cost_all": time.time() - start_time,
            }

        if self.is_session_stop_set() or self.is_break_set():
            return _make_result(False)

        has_frames = frame_list is not None and len(frame_list) > 0
        has_audio = audio_waveform is not None and len(audio_waveform) > 0
        has_text = text_list is not None and len(text_list) > 0

        if has_frames and has_audio:
            mode = "OMNI"
        elif has_frames:
            mode = "VISION"
        elif has_audio:
            mode = "AUDIO"
        elif has_text:
            mode = "TEXT"
        else:
            return _make_result(False)

        self.pending_logits = None

        # sliding window: record unit start position
        self.decoder.register_unit_start()

        # Schema tracking: start new unit, record prefill tokens
        self._current_unit_prefill_tokens = []

        # Step 1: Feed <unit> token
        self.decoder.feed(self.decoder.embed_token(self.unit_token_id))
        self._current_unit_prefill_tokens.append(self.unit_token_id)

        # Step 2: process image
        if has_frames:
            t0 = time.time()

            # normalize max_slice_nums to a list matching frame_list length
            if isinstance(max_slice_nums, int):
                max_slice_nums_list = [max_slice_nums] * len(frame_list)
            else:
                max_slice_nums_list = list(max_slice_nums)
                if len(max_slice_nums_list) != len(frame_list):
                    raise ValueError(
                        f"max_slice_nums list length ({len(max_slice_nums_list)}) "
                        f"must match frame_list length ({len(frame_list)})"
                    )

            # check if all max_slice_nums are the same (can use batch processing)
            all_same = len(set(max_slice_nums_list)) == 1

            if all_same:
                # all images use the same max_slice_nums, use batch processing
                processed_frames = self.processor.process_image(frame_list, max_slice_nums=max_slice_nums_list[0])
                if self.device:
                    processed_frames = processed_frames.to(self.device)
            else:
                # different max_slice_nums per image, process individually and merge
                all_pixel_values = []
                all_tgt_sizes = []
                for frame, max_slices in zip(frame_list, max_slice_nums_list):
                    pf = self.processor.process_image([frame], max_slice_nums=max_slices)
                    if self.device:
                        pf = pf.to(self.device)
                    # pf["pixel_values"][0] is the list of slices for this image
                    all_pixel_values.extend(pf["pixel_values"][0])
                    # pf["tgt_sizes"][0] is the array of target sizes for this image's slices
                    if hasattr(pf["tgt_sizes"][0], "tolist"):
                        all_tgt_sizes.extend(pf["tgt_sizes"][0].tolist())
                    else:
                        all_tgt_sizes.extend(list(pf["tgt_sizes"][0]))

                # reconstruct processed_frames with merged data
                processed_frames = {
                    "pixel_values": [all_pixel_values],
                    "tgt_sizes": [torch.tensor(all_tgt_sizes) if all_tgt_sizes else []],
                }

            cost_vision_process = time.time() - t0

            t0 = time.time()
            # get vision embeddings for all images (each may have multiple slices)
            # vision_hidden_states is a list, one entry per input image
            # each entry contains embeddings for [source_image, slice_1, slice_2, ...]
            vision_hidden_states = self.model.get_vision_embedding(processed_frames)
            cost_vision_embed = time.time() - t0

            if vision_hidden_states is not None and len(vision_hidden_states) > 0:
                t0 = time.time()

                # vision_hidden_states[0] contains ALL slices from ALL images (flattened)
                # shape: [total_slices, 64, D] where total_slices = sum of slices across all images
                # we need to know how many slices each image has to correctly group them

                # calculate slice counts for each image using get_sliced_grid (lightweight, no actual slicing)
                slice_counts = []  # e.g., [5, 9] means img1 has 5 slices (1 source + 4 HD), img2 has 9 slices
                for frame_idx, frame in enumerate(frame_list):
                    max_slices = max_slice_nums_list[frame_idx]
                    if hasattr(frame, "size"):
                        # get_sliced_grid returns [M, N] grid or None if no slicing needed
                        # total images = 1 (source) + M * N (HD slices)
                        grid = self.processor.image_processor.get_sliced_grid(
                            frame.size, max_slices, nerver_split=False
                        )
                        if grid is not None:
                            slice_counts.append(1 + grid[0] * grid[1])  # 1 source + M*N slices
                        else:
                            slice_counts.append(1)  # no slicing, only source image
                    else:
                        slice_counts.append(1)  # default: single image, no slicing

                # get the flattened embeddings tensor
                # vision_hidden_states is a list with one element (the batch)
                # vision_hidden_states[0] shape: [total_slices, 64, D]
                all_embeds = vision_hidden_states[0]

                # collect all feed operations first, then execute
                # this allows us to identify the last token for VISION mode logits
                feed_operations = []  # List of (embed, is_last_for_vision_mode, token_id_or_none)

                embed_idx = 0  # current index in all_embeds
                for img_idx, num_slices in enumerate(slice_counts):
                    if num_slices == 0:
                        continue

                    # the first embedding is always the source image (downsampled overview)
                    # Feed <image> token
                    feed_operations.append(
                        (self.decoder.embed_token(self.image_start_token_id), False, self.image_start_token_id)
                    )
                    # Feed source image embedding (shape: [64, D]) - use None to indicate embedding
                    feed_operations.append((all_embeds[embed_idx], False, None))
                    # Feed </image> token
                    feed_operations.append(
                        (self.decoder.embed_token(self.image_end_token_id), False, self.image_end_token_id)
                    )
                    embed_idx += 1

                    # remaining embeddings are HD slices (if num_slices > 1)
                    if num_slices > 1:
                        for slice_i in range(1, num_slices):
                            # Feed <slice> token
                            feed_operations.append(
                                (self.decoder.embed_token(self.slice_start_token_id), False, self.slice_start_token_id)
                            )
                            # Feed slice embedding (shape: [64, D])
                            feed_operations.append((all_embeds[embed_idx], False, None))
                            # Feed </slice> token
                            feed_operations.append(
                                (self.decoder.embed_token(self.slice_end_token_id), False, self.slice_end_token_id)
                            )
                            embed_idx += 1

                # mark the last operation for VISION mode logits
                if feed_operations:
                    feed_operations[-1] = (feed_operations[-1][0], True, feed_operations[-1][2])

                # execute feed operations
                if batch_vision_feed and feed_operations:
                    # batch mode: concatenate all embeddings and feed at once
                    # this reduces LLM forward passes from N to 1
                    #
                    # NOTE: batch mode may have slight numerical differences compared to for-loop mode
                    # due to floating-point precision in attention computation. This is expected behavior
                    # for causal attention with incremental vs batch computation.

                    all_embeds_list = []
                    for embed, is_last, token_id in feed_operations:
                        # ensure all embeddings have shape [L, H]
                        if embed.dim() == 1:
                            embed = embed.unsqueeze(0)
                        all_embeds_list.append(embed)

                    # concatenate all embeddings
                    # torch.cat requires consistent dtype; embeddings should already be same dtype
                    all_embeds_to_feed = torch.cat(all_embeds_list, dim=0)  # [total_L, H]

                    if mode == "VISION":
                        # vision mode needs logits from the last token
                        self.pending_logits, _ = self.decoder.feed(all_embeds_to_feed, return_logits=True)
                    else:
                        # omni mode: just feed, wait for audio to get logits
                        self.decoder.feed(all_embeds_to_feed)

                    # schema tracking: record all token IDs and embedding markers
                    for embed, is_last, token_id in feed_operations:
                        if token_id is not None:
                            self._current_unit_prefill_tokens.append(token_id)
                        else:
                            embed_dim = embed.shape[0] if len(embed.shape) > 1 else 1
                            self._current_unit_prefill_tokens.append(("img", embed_dim))
                else:
                    for embed, is_last, token_id in feed_operations:
                        if mode == "VISION" and is_last:
                            # get logits from the last token
                            self.pending_logits, _ = self.decoder.feed(embed, return_logits=True)
                        else:
                            self.decoder.feed(embed)
                        # schema tracking: record token ID or embedding marker
                        if token_id is not None:
                            self._current_unit_prefill_tokens.append(token_id)
                        else:
                            # use tuple to mark image embedding: ("img", dim)
                            embed_dim = embed.shape[0] if len(embed.shape) > 1 else 1
                            self._current_unit_prefill_tokens.append(("img", embed_dim))
                # for omni mode, no pending logits needed here (wait for audio)

                cost_vision_feed = time.time() - t0

        # Step 3: process audio (if any)
        if has_audio:
            # accumulate audio to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_waveform])

            # calculate required audio length
            if self.audio_chunk_idx == 0:
                required_samples = int(self.FIRST_CHUNK_MS * self.SAMPLE_RATE / 1000)
                if len(self.audio_buffer) < required_samples:
                    padding_samples = required_samples - len(self.audio_buffer)
                    padding = np.zeros(padding_samples, dtype=np.float32)
                    self.audio_buffer = np.concatenate([padding, self.audio_buffer])
            else:
                required_samples = int(self.CHUNK_MS * self.SAMPLE_RATE / 1000)

            need_samples = self.processor.get_streaming_chunk_size()
            if len(self.audio_buffer) < need_samples:
                return _make_result(
                    False, f"audio not enough: need {need_samples} samples, only {len(self.audio_buffer)}"
                )

            audio_chunk = self.audio_buffer[:need_samples]

            t0 = time.time()
            batch_feature = self.processor.process_audio_streaming(
                audio_chunk,
                reset=False,
                return_batch_feature=True,
            )

            if batch_feature is None or batch_feature.audio_features.shape[-1] == 0:
                return _make_result(False, "streaming audio processing returned empty")

            # metadata
            batch_feature.chunk_idx = self.audio_chunk_idx
            batch_feature.use_extra_context = True
            batch_feature.prefix_extra_frames = 0 if self.audio_chunk_idx == 0 else 2
            batch_feature.suffix_extra_frames = 2

            batch_feature = batch_feature.to(self.device)
            cost_audio_process = time.time() - t0

            t0 = time.time()
            embeds_nested = self.model.get_audio_embedding_streaming(
                batch_feature,
                use_extra_context=batch_feature.use_extra_context,
                prefix_extra_frames=batch_feature.prefix_extra_frames,
                suffix_extra_frames=batch_feature.suffix_extra_frames,
            )
            audio_embeds = torch.cat([t for g in embeds_nested for t in g], dim=0)
            cost_audio_embed = time.time() - t0

            t0 = time.time()
            self.pending_logits, _ = self.decoder.feed(audio_embeds, return_logits=True)
            cost_audio_feed = time.time() - t0

            # schema tracking: use tuple to mark audio embedding: ("audio", dim)
            embed_dim = audio_embeds.shape[0] if len(audio_embeds.shape) > 1 else 1
            self._current_unit_prefill_tokens.append(("audio", embed_dim))

            if self.audio_chunk_idx == 0:
                cfg = self.processor._streaming_mel_processor.get_config()
                consumed_ms = int(cfg.get("effective_first_chunk_ms", self.FIRST_CHUNK_MS))
                consumed_samples = int(consumed_ms * self.SAMPLE_RATE / 1000)
            else:
                consumed_samples = int(self.CHUNK_MS * self.SAMPLE_RATE / 1000)

            self.audio_buffer = self.audio_buffer[consumed_samples:]

            self.audio_chunk_idx += 1

        # Step 4: process text
        if has_text:
            # concatenate all text items
            text_content = "".join(text_list) if isinstance(text_list, list) else str(text_list)

            # tokenize text
            text_token_ids = self.tokenizer.encode(text_content, add_special_tokens=False)

            if len(text_token_ids) > 0:
                # get token embeddings
                text_token_ids_tensor = torch.tensor(text_token_ids, dtype=torch.long, device=self.device)
                text_embeds = self.decoder.embed_token(text_token_ids_tensor)

                # feed to decoder
                if mode == "TEXT":
                    # text-only mode: get logits from the last token
                    self.pending_logits, _ = self.decoder.feed(text_embeds, return_logits=True)
                else:
                    # mixed mode: just feed, let other modality get logits
                    self.decoder.feed(text_embeds)

                # schema tracking: record text token IDs
                for token_id in text_token_ids:
                    self._current_unit_prefill_tokens.append(token_id)

        self.current_mode = mode

        if mode == "VISION":
            self.audio_chunk_idx += 1

        # schema tracking: save current unit's prefill tokens
        self.prefill_schema_tokens.append(self._current_unit_prefill_tokens)

        return _make_result(True)

    @torch.no_grad()
    def streaming_generate(
        self,
        prompt_wav_path=None,
        max_new_speak_tokens_per_chunk=20,
        decode_mode: str = "sampling",
        temperature=0.7,
        top_k=100,
        top_p=0.8,
        listen_prob_scale=1.0,
        listen_top_k=None,
        text_repetition_penalty=1.05,
        text_repetition_window_size=512,
    ):
        start_time = time.time()

        if self.is_session_stop_set() or self.is_break_set():
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": True,
                "current_time": self.audio_chunk_idx,
                "cost_llm": 0.0,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": 0,
                "n_tts_tokens": 0,
            }

        # check if there are pending logits to process
        if not hasattr(self, "pending_logits") or self.pending_logits is None:
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": False,
                "current_time": self.audio_chunk_idx,
                "cost_llm": 0.0,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": 0,
                "n_tts_tokens": 0,
            }

        # use pending logits generated in streaming_prefill
        logits = self.pending_logits
        self.pending_logits = None

        # Force listen: check if we should force listen for first N calls
        force_listen = self._streaming_generate_count < self.force_listen_count
        self._streaming_generate_count += 1

        total_hidden_in_unit = []
        total_ids_in_unit = []
        current_time = self.audio_chunk_idx
        is_listen = False
        end_of_turn = False

        llm_start_time = time.time()

        for j in range(max_new_speak_tokens_per_chunk):
            if j == max_new_speak_tokens_per_chunk - 1:
                if self.ls_mode == "explicit":
                    self.decoder.feed(self.decoder.embed_token(self.chunk_eos_token_id))
                    self.total_ids.append(self.chunk_eos_token_id)
                    break

            if force_listen:
                last_id = torch.tensor([self.listen_token_id], dtype=torch.long, device=self.device)
            else:
                last_id = self.decoder.decode(
                    logits=logits,
                    mode=decode_mode,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    listen_top_k=listen_top_k,
                    listen_prob_scale=listen_prob_scale,
                    text_repetition_penalty=text_repetition_penalty,
                    text_repetition_window_size=text_repetition_window_size,
                )

                # if current turn not ended, not allowed to listen (only check when not force_listen)
                if last_id.item() == self.listen_token_id and (not self.current_turn_ended):
                    last_id = torch.tensor([self.tts_bos_token_id], dtype=torch.long, device=self.device)

            self.total_ids.append(last_id.item())

            is_listen = last_id.item() == self.listen_token_id

            # termination condition detection
            if last_id.item() in self.chunk_terminator_token_ids:
                if self.ls_mode == "explicit":
                    logits, _ = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)
                break
            else:
                # normal speak
                self.current_turn_ended = False

                if last_id.item() in self.chunk_speak_token_ids:
                    pass
                else:
                    self.res_ids.append(last_id.item())
                    self.speak_count += 1

                logits, hidden = self.decoder.feed(self.decoder.embed_token(last_id.item()), return_logits=True)

                assert len(hidden.shape) == 3
                assert hidden.shape[0] == 1
                assert hidden.shape[1] == 1

                end_of_turn = last_id.item() in self.turn_terminator_token_ids

                if end_of_turn:
                    self.current_turn_ended = True

                if j != 0:
                    total_hidden_in_unit.append([last_id.item(), hidden, end_of_turn])
                    total_ids_in_unit.append(last_id.item())

        # Prefill </unit> token
        unit_end_id = self.tokenizer.convert_tokens_to_ids("</unit>")
        self.decoder.feed(self.decoder.embed_token(unit_end_id))
        self.total_ids.append(unit_end_id)

        # calculate generated text (for sliding window context preserve, filter out special tokens)
        generated_text = (
            self.tokenizer.decode(total_ids_in_unit, skip_special_tokens=True) if total_ids_in_unit else ""
        )

        # sliding window: register unit end, and check if sliding window is needed
        input_type = self.current_mode.lower() if self.current_mode else "audio"

        self.decoder.register_unit_end(
            input_type=input_type,
            generated_tokens=total_ids_in_unit,
            is_listen=is_listen,
            generated_text=generated_text,
        )
        # select sliding window method based on sliding window mode
        if self.decoder._window_config.sliding_window_mode == "context":
            self.decoder.enforce_window_with_context()
        elif self.decoder._window_config.sliding_window_mode == "basic":
            self.decoder.enforce_window()

        llm_end_time = time.time()

        if is_listen:
            self.total_hidden.append([])
            return {
                "is_listen": True,
                "text": "",
                "audio_waveform": self._generate_silence_waveform(),
                "end_of_turn": False,
                "current_time": current_time,
                "cost_llm": llm_end_time - llm_start_time,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": len(total_ids_in_unit),
                "n_tts_tokens": 0,
            }

        self.total_hidden.append(total_hidden_in_unit)
        text = generated_text  # reuse already calculated text

        if not self.generate_audio:
            return {
                "is_listen": False,
                "text": text,
                "audio_waveform": None,
                "end_of_turn": end_of_turn,
                "current_time": current_time,
                "cost_llm": llm_end_time - llm_start_time,
                "cost_tts_prep": 0.0,
                "cost_tts": 0.0,
                "cost_token2wav": 0.0,
                "cost_all": time.time() - start_time,
                "n_tokens": len(total_ids_in_unit),
                "n_tts_tokens": 0,
            }

        # TTS generate
        tts_start_time = time.time()
        tts_prep_start_time = time.time()
        tts_condition = self._convert_results_to_tts_input(total_hidden_in_unit)
        tts_prep_end_time = time.time()

        max_token_per_chunk = 25 + 1
        min_token_per_chunk = 25 + 1

        if end_of_turn:
            min_token_per_chunk = 0
        force_flush = False
        if self.tts_text_start_pos == 0:  # this is the start of the turn
            min_token_per_chunk = 0  # allow decoding <1s audio
            force_flush = True

        if self.tts_current_turn_start_time is None:
            self.tts_current_turn_start_time = current_time

        new_tokens, old_kv = self.model.tts.generate_chunk(
            inputs_embeds=tts_condition,
            temperature=self.tts_temperature,
            repetition_penalty=self.tts_repetition_penalty,
            eos_token=self.tts_eos_token,
            force_no_stop=False,
            max_new_token=max_token_per_chunk,
            min_new_tokens=min_token_per_chunk,
            past_key_values=self.tts_past_key_values,
            logits_processors=self.tts_logits_processors,
            text_start_pos=self.tts_text_start_pos,
        )

        tts_end_time = time.time()

        # update TTS state (note: token2wav reset must be after audio generation, otherwise tokens in buffer will be lost)
        if end_of_turn:
            self.tts_text_start_pos = 0
            self.tts_past_key_values = None
            self.tts_current_turn_start_time = None
        else:
            self.tts_past_key_values = old_kv
            self.tts_text_start_pos += tts_condition.shape[1] + new_tokens.shape[1]

        # token2wav generation (must be before reset, otherwise tokens in the last but second chunk will be lost)
        token2wav_start_time = time.time()
        audio_waveform = self._generate_waveform_from_tokens(
            new_tokens, prompt_wav_path, end_of_turn, force_flush=force_flush
        )
        token2wav_end_time = time.time()

        # reset token2wav state after audio generation, ensure all tokens in buffer are processed
        if end_of_turn:
            self._reset_token2wav_for_new_turn()

        end_time = time.time()

        return {
            "is_listen": False,
            "text": text,
            "audio_waveform": audio_waveform,
            "end_of_turn": end_of_turn,
            "current_time": current_time,
            "cost_llm": llm_end_time - llm_start_time,
            "cost_tts_prep": tts_prep_end_time - tts_prep_start_time,
            "cost_tts": tts_end_time - tts_start_time,
            "cost_token2wav": token2wav_end_time - token2wav_start_time,
            "cost_all": end_time - start_time,
            "n_tokens": len(total_ids_in_unit),
            "n_tts_tokens": new_tokens.numel(),
        }

    def get_session_schema(self, include_embeddings: bool = True) -> str:
        """get complete schema for current session (includes prefill and generate stages)

        Args:
            include_embeddings: whether to include embedding placeholders (e.g. [img_embed_64], [audio_embed_50])

        Returns:
            complete schema string, each unit format:
            <unit><image>[img_embed_64]</image>[audio_embed_50]<|listen|or|speak|>generated_content</unit>
        """
        if not hasattr(self, "prefill_schema_tokens") or not hasattr(self, "total_ids"):
            return ""

        # get </unit> token id for splitting generate tokens
        unit_end_token_id = self.tokenizer.convert_tokens_to_ids("</unit>")

        # split generate tokens into each unit
        generate_units = []
        current_unit = []
        for tid in self.total_ids:
            current_unit.append(tid)
            if tid == unit_end_token_id:
                generate_units.append(current_unit)
                current_unit = []

        # build complete schema
        full_schema_parts = []
        num_units = max(len(self.prefill_schema_tokens), len(generate_units))

        for unit_idx in range(num_units):
            unit_schema = ""

            # prefill part
            if unit_idx < len(self.prefill_schema_tokens):
                prefill_tokens = self.prefill_schema_tokens[unit_idx]
                for item in prefill_tokens:
                    if isinstance(item, tuple):
                        # tuple represents embedding: ("img", dim) or ("audio", dim)
                        embed_type, embed_dim = item
                        if include_embeddings:
                            unit_schema += f"[{embed_type}_embed_{embed_dim}]"
                    else:
                        # normal token ID
                        unit_schema += self.tokenizer.decode([item], skip_special_tokens=False)

            # generate part
            if unit_idx < len(generate_units):
                unit_schema += self.tokenizer.decode(generate_units[unit_idx], skip_special_tokens=False)

            full_schema_parts.append(unit_schema)

        return "".join(full_schema_parts)

    def get_unit_schemas(self, include_embeddings: bool = True) -> list:
        """get list of schema for each unit

        Returns:
            list of schema strings for each unit
        """
        if not hasattr(self, "prefill_schema_tokens") or not hasattr(self, "total_ids"):
            return []

        unit_end_token_id = self.tokenizer.convert_tokens_to_ids("</unit>")

        # split generate tokens into each unit
        generate_units = []
        current_unit = []
        for tid in self.total_ids:
            current_unit.append(tid)
            if tid == unit_end_token_id:
                generate_units.append(current_unit)
                current_unit = []

        # build schema for each unit
        unit_schemas = []
        num_units = max(len(self.prefill_schema_tokens), len(generate_units))

        for unit_idx in range(num_units):
            unit_schema = ""

            # prefill part
            if unit_idx < len(self.prefill_schema_tokens):
                prefill_tokens = self.prefill_schema_tokens[unit_idx]
                for item in prefill_tokens:
                    if isinstance(item, tuple):
                        # tuple represents embedding: ("img", dim) or ("audio", dim)
                        embed_type, embed_dim = item
                        if include_embeddings:
                            unit_schema += f"[{embed_type}_embed_{embed_dim}]"
                    else:
                        # normal token ID
                        unit_schema += self.tokenizer.decode([item], skip_special_tokens=False)

            # generate part
            if unit_idx < len(generate_units):
                unit_schema += self.tokenizer.decode(generate_units[unit_idx], skip_special_tokens=False)

            unit_schemas.append(unit_schema)

        return unit_schemas

    def _convert_results_to_tts_input(self, results):
        """convert LLM hidden states to TTS input"""
        if len(results) == 0:
            audio_bos = self.model.tts.emb_text(
                torch.tensor(
                    [self.model.tts.audio_bos_token_id],
                    device=self.model.tts.emb_text.weight.device,
                    dtype=torch.long,
                )
            )
            return audio_bos.unsqueeze(0)

        llm_tokens = []
        llm_hidden = []
        for hidden in results:
            llm_tokens.append(hidden[0])
            llm_hidden.append(hidden[1].squeeze(0))

        llm_tokens_tensor = torch.Tensor(llm_tokens).to(self.device, dtype=torch.long)
        llm_embeds = self.model.tts.emb_text(llm_tokens_tensor)

        llm_hidden_tensor = torch.cat(llm_hidden, dim=0)
        llm_hidden_tensor = self.model.tts.projector_semantic(llm_hidden_tensor)
        llm_hidden_tensor = torch.nn.functional.normalize(llm_hidden_tensor, p=2, dim=-1)

        tts_embeds = llm_embeds + llm_hidden_tensor

        audio_bos = self.model.tts.emb_text(
            torch.tensor(
                [self.model.tts.audio_bos_token_id],
                device=self.model.tts.emb_text.weight.device,
                dtype=torch.long,
            )
        )

        tts_embeds = torch.cat([tts_embeds, audio_bos], dim=0)
        return tts_embeds.unsqueeze(0)

    def _generate_waveform_from_tokens(
        self,
        new_tokens: torch.Tensor,
        prompt_wav_path: Optional[str],
        is_last_chunk: bool = False,
        force_flush: bool = False,
    ) -> Optional[np.ndarray]:
        if not self.token2wav_initialized:
            logger.warning("token2wav_initialized is uninitialized")
            return None

        CHUNK_SIZE = 25

        token_ids = torch.reshape(new_tokens, (-1,)).tolist()
        self.token2wav_buffer += token_ids

        has_chunk_eos = any(tid in self.chunk_terminator_token_ids for tid in token_ids)

        pcm_bytes_list = []

        # process enough tokens
        # if there is chunk_eos, try to flush more content
        if has_chunk_eos or force_flush:
            # when there is chunk_eos, try to flush more content
            while len(self.token2wav_buffer) >= self.pre_lookahead + 5:  # at least keep some lookahead
                chunk_to_process = min(CHUNK_SIZE + self.pre_lookahead, len(self.token2wav_buffer))
                pcm_bytes = self.model.tts.audio_tokenizer.stream(
                    self.token2wav_buffer[:chunk_to_process],
                    prompt_wav=prompt_wav_path,
                )
                pcm_bytes_list.append(pcm_bytes)
                self.token2wav_buffer = self.token2wav_buffer[min(CHUNK_SIZE, chunk_to_process - self.pre_lookahead) :]
        else:
            while len(self.token2wav_buffer) >= CHUNK_SIZE + self.pre_lookahead:
                pcm_bytes = self.model.tts.audio_tokenizer.stream(
                    self.token2wav_buffer[: CHUNK_SIZE + self.pre_lookahead],
                    prompt_wav=prompt_wav_path,
                )
                pcm_bytes_list.append(pcm_bytes)
                self.token2wav_buffer = self.token2wav_buffer[CHUNK_SIZE:]

        # if is the last chunk, flush remaining tokens
        if is_last_chunk and len(self.token2wav_buffer) > 0:
            pcm_bytes = self.model.tts.audio_tokenizer.stream(
                self.token2wav_buffer,
                prompt_wav=prompt_wav_path,
                last_chunk=True,
            )
            pcm_bytes_list.append(pcm_bytes)
            self.token2wav_buffer = []

        if not pcm_bytes_list:
            return None

        # merge PCM and convert to numpy array (24kHz, int16 -> float32)
        all_pcm = b"".join(pcm_bytes_list)
        if len(all_pcm) == 0:
            return None

        pcm_np = np.frombuffer(all_pcm, dtype="<i2")
        audio_waveform = pcm_np.astype(np.float32) / 32768.0

        # left pad with zeros if audio is less than 1 second (24kHz), skip for last chunk
        min_samples = 24000  # 1 second at 24kHz
        if not is_last_chunk and len(audio_waveform) < min_samples:
            pad_length = min_samples - len(audio_waveform)
            audio_waveform = np.pad(audio_waveform, (pad_length, 0), mode="constant", constant_values=0)

        return audio_waveform

    @staticmethod
    def _generate_silence_waveform(duration_sec: float = 1.0) -> np.ndarray:
        """generate silence waveform (24kHz)"""
        sample_rate = 24000
        num_samples = int(duration_sec * sample_rate)
        return np.zeros(num_samples, dtype=np.float32)

    def get_generated_text(self) -> str:
        return self.tokenizer.decode(self.res_ids)

    def get_current_time(self) -> int:
        return self.audio_chunk_idx

    def as_simplex(self, reset_session: bool = True, reset_token2wav_cache: bool = False) -> "MiniCPMO":
        """Convert this MiniCPMODuplex instance back to MiniCPMO for simplex mode.

        Args:
            reset_session: If True, reset streaming session state (KV cache, etc.).
                          Recommended when switching from duplex to simplex mode.

        Returns the underlying MiniCPMO model instance without reloading.
        """
        if reset_session:
            self.model.reset_session(reset_token2wav_cache=reset_token2wav_cache)
        return self.model


def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim=None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        adaptive=False,
        max_size=(70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size, device="cpu"):
        if is_deepspeed_zero3_enabled():
            device = "cuda"
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)
        self._pos_cache_ready = not self.pos_embed.is_meta

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._pos_cache_ready = not self.pos_embed.is_meta
        if self._pos_cache_ready:
            self._set_2d_pos_cache(self.max_size, self.pos_embed.device)
        return result

    def _adjust_pos_cache(self, tgt_sizes, device):
        if not self._pos_cache_ready:
            self._set_2d_pos_cache(self.max_size, device)
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_sizes=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D +  L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


class MiniCPMWhisperEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: WhisperConfig, layer_idx: int = None):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


# Copied from from transformers.models.whisper.modeling_whisper.WhisperEncoder and add use_cache for streaming inference
class MiniCPMWhisperEncoder(WhisperEncoder):
    # Streaming MiniCPM-o consumes encoder hidden states only; attention probabilities are not recorded.
    _can_record_outputs = {"hidden_states": MiniCPMWhisperEncoderLayer}

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MiniCPMWhisperEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)]
        )

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_features,
        attention_mask=None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        use_extra_context: Optional[bool] = False,
        prefix_extra_frames: Optional[int] = 1,
        suffix_extra_frames: Optional[int] = 1,
        cnn_min_length: Optional[int] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        # Ignore copy
        input_features = input_features.to(dtype=self.conv1.weight.dtype, device=self.conv1.weight.device)

        # Optional: pad short input to minimum length for CNN computation consistency
        original_length = input_features.shape[2]
        padded_for_cnn = False
        if cnn_min_length is not None and original_length < cnn_min_length:
            padded_features = torch.zeros(
                input_features.shape[0],
                input_features.shape[1],
                cnn_min_length,
                dtype=input_features.dtype,
                device=input_features.device,
            )
            padded_features[:, :, :original_length] = input_features
            input_features = padded_features
            padded_for_cnn = True

        conv1_output = self.conv1(input_features)
        inputs_embeds = nn.functional.gelu(conv1_output)
        conv2_output = self.conv2(inputs_embeds)
        inputs_embeds = nn.functional.gelu(conv2_output)
        # If padding was done before, now need to remove the effect of padding
        if padded_for_cnn:
            # Conv1: stride=1, output length=input length
            # Conv2: stride=2, output length=(input length+1)//2
            actual_cnn_output_length = (original_length + 1) // 2
            inputs_embeds = inputs_embeds[:, :, :actual_cnn_output_length]

        # If extra context is used, CNN operations need to remove redundant frames
        # conv2 stride=2, so the redundant frames in the input will be halved (upward rounding)
        if use_extra_context:
            # Input has prefix_extra_frames prefix frames and suffix_extra_frames suffix frames
            # conv2 stride=2, output length = ceil(input length / 2)
            # For 2 redundant frames, the output is 1 frame (ceil(2/2) = 1)
            prefix_to_remove = (prefix_extra_frames + 1) // 2 if prefix_extra_frames > 0 else 0
            suffix_to_remove = (suffix_extra_frames + 1) // 2 if suffix_extra_frames > 0 else 0

            # Remove redundant frames before and after (batch, channels, time)
            if prefix_to_remove > 0:
                inputs_embeds = inputs_embeds[:, :, prefix_to_remove:]
            if 0 < suffix_to_remove < inputs_embeds.shape[2]:
                inputs_embeds = inputs_embeds[:, :, :-suffix_to_remove]

        inputs_embeds = inputs_embeds.permute(0, 2, 1)

        embed_pos = self.embed_positions.weight
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif isinstance(past_key_values, DynamicCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            past_key_values_length = past_key_values.self_attention_cache.get_seq_length()
            if inputs_embeds.shape[1] + past_key_values_length > embed_pos.shape[0]:
                logger.warning("seems the audio is longer than 30s. repeating the last part of the audio")
                embed_pos_front = embed_pos[past_key_values_length:, :]
                embed_pos = torch.cat(
                    (
                        embed_pos_front,
                        torch.repeat_interleave(
                            embed_pos[-1, :].unsqueeze(0),
                            inputs_embeds.shape[1] - embed_pos.shape[0] + past_key_values_length,
                            dim=0,
                        ),
                    )
                )
            else:
                embed_pos = embed_pos[past_key_values_length : inputs_embeds.shape[1] + past_key_values_length, :]
        else:
            embed_pos = embed_pos[: inputs_embeds.shape[1], :]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for encoder_layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.training and self.layerdrop > 0.0 and torch.rand([]) < self.layerdrop:
                continue
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.layer_norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


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


class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.in_dim = config.llm_hidden_size
        self.out_dim = config.hidden_size
        self.intermediate_size = config.llm_intermediate_size
        self.gate_proj = nn.Linear(self.in_dim, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.in_dim, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.out_dim, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


@dataclass
class MiniCPMTTSGenerationOutput(ModelOutput):
    """
    Output class for MiniCPMTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.
    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_input_ids: Optional[torch.LongTensor] = None
    finished: bool = None


def make_streaming_chunk_mask_inference(
    tts_text_scope: List[int],
    tts_text_mask: torch.Tensor,
    streaming_audio_chunk_size: int = 50,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
    max_sequence_length: int = 4096,
):
    """
    Example:
    Input sequence:
    [t1, t2, t3, t4, t5, [Ptts], a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, ...]
    Output 4D causal mask:
    ------- text positions -------
    [0] <- here is [Stts]
    [0,   0] <- here is [spk_emb] * N
    [0,   0,   0]
    [0,   0,   0,    0]
    [0,   0,   0,    0,    0]
    ------- audio positions --------
    [0,    0, -inf, -inf, -inf, 0] <- here is [Ptts], [Ptts]'s last hidden state should predict the first audio token
                                v- here is [Ptts]
    [0,    0, -inf, -inf, -inf, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0, 0]
    [0,    0, -inf, -inf, -inf, 0, 0, 0, 0, 0, 0] # end of first 1s audio chunk
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0,    0, 0   , -inf, -inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """

    # Create a complete attention mask for input embeds [batch_size, seq_len], without considering audio mask as audio is always at the end

    assert tts_text_mask.dtype == torch.int8

    padding_mask = torch.ones(max_sequence_length, dtype=torch.int8, device=device)
    padding_mask[tts_text_scope[0] : tts_text_scope[1]] = tts_text_mask

    # Initialize a standard upper triangular causal mask
    min_dtype = torch.finfo(dtype).min

    causal_mask = torch.full(
        (max_sequence_length, max_sequence_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if max_sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    else:
        raise ValueError("max_sequence_length of tts could not be 1.")

    # For each data sample
    audio_token_start = tts_text_scope[1]
    audio_duration = max_sequence_length - tts_text_scope[1]

    # Record which text chunk the current audio chunk can see up to
    text_pivot = 0
    num_valid_text_tokens = torch.sum(tts_text_mask).item() - 1  # [Ptts] excluded
    # How many audio chunks are in total, the num of buckets should be smaller as possible

    num_text_tokens_per_audio_chunk = 10

    # For each chunk of audio
    for chunk_idx in range(math.ceil(audio_duration / streaming_audio_chunk_size)):
        audio_chunk_start = audio_token_start + chunk_idx * streaming_audio_chunk_size
        audio_chunk_end = audio_token_start + (chunk_idx + 1) * streaming_audio_chunk_size
        # New text seen by this new audio chunk
        new_text_this_chunk = num_text_tokens_per_audio_chunk
        # The right bound of visible text tokens
        text_pivot = min(new_text_this_chunk + text_pivot, num_valid_text_tokens)
        # Mask all text chunks after the visible ones
        # -> [text_pivot, len(tts_text_scope)-1] excluding [Ptts]
        causal_mask[
            audio_chunk_start - 1 : audio_chunk_end - 1,
            # tts_text_scope[0] + text_pivot: tts_text_scope[1],
            tts_text_scope[0] + text_pivot : tts_text_scope[1] - 1,
        ] = min_dtype

    # Mask the padding parts in tts_text_masks (no position will attend to it)
    causal_mask[:, padding_mask == 0] = min_dtype

    # Add extra dimensions, [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask


class MiniCPMTTS(PreTrainedModel):
    config_class = MiniCPMTTSConfig

    def __init__(self, config: MiniCPMTTSConfig, audio_tokenizer: None):
        super().__init__(config)

        self.use_llm_hidden_state = config.use_llm_hidden_state

        self.use_text = config.use_text
        self.streaming = config.streaming
        self.streaming_text_chunk_min = config.streaming_text_chunk_min
        self.streaming_text_chunk_max = config.streaming_text_chunk_max
        self.streaming_audio_chunk_size = config.streaming_audio_chunk_size
        self.streaming_text_reserved_len = config.streaming_text_reserved_len
        # streaming tts
        self.streaming_text_chunk_size = config.streaming_text_chunk_max
        self.audio_bos_token_id = config.audio_bos_token_id
        self.num_mel_bins = config.num_mel_bins
        self.num_vq = config.num_vq
        self.num_audio_tokens = config.num_audio_tokens

        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        self.interleaved = config.interleaved
        self.attention_type = config.attention_type
        self.recomputed_chunks = config.recomputed_chunks

        # Two different window size concepts:
        # 1. chunk_window_size: number of chunks for sliding_recompute mode (default 2)
        # 2. token_window_size: number of tokens for sliding_window mode (default 300)
        self.chunk_window_size = config.window_size  # chunk-level window for sliding_recompute
        self.token_window_size = (
            config.streaming_sliding_window_audio_window_size
        )  # token-level window for sliding_window

        # Legacy aliases (for backward compatibility with existing code)
        self.window_size = self.chunk_window_size  # used in generate_streaming for sliding_recompute
        self.sliding_window_size = self.token_window_size  # used in TTSStreamingGenerator for sliding_window

        if self.attention_type == "sliding_recompute" and self.chunk_window_size <= self.recomputed_chunks:
            raise ValueError(
                f"sliding_recompute requires chunk_window_size > recomputed_chunks, "
                f"but got chunk_window_size={self.chunk_window_size} and recomputed_chunks={self.recomputed_chunks}"
            )

        if config.backbone_model == "llama":
            model_config = LlamaConfig(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_hidden_layers=config.num_hidden_layers,
                num_key_value_heads=config.num_key_value_heads,
                max_position_embeddings=config.max_position_embeddings,
                attn_implementation=config.attn_implementation,
                head_dim=config.hidden_size // config.num_attention_heads,
            )

            self.emb_text = nn.Embedding(config.num_text_tokens, config.hidden_size)

            model = LlamaModel(model_config)
            self.model = model
        else:
            raise ValueError(f"Unsupported backbone model: {config.backbone_model}")

        self.projector_spk = self.create_projector(config)
        self.projector_semantic = self.create_projector(config)

        self.audio_tokenizer = audio_tokenizer

        self.emb_code = nn.ModuleList(
            [nn.Embedding(config.num_audio_tokens, config.hidden_size) for _ in range(config.num_vq)]
        )

        self.head_code = nn.ModuleList(
            [
                weight_norm(
                    nn.Linear(config.hidden_size, config.num_audio_tokens, bias=False),
                    name="weight",
                )
                for _ in range(config.num_vq)
            ]
        )

        self.condition_type = config.condition_type

        return

    @staticmethod
    def create_projector(config):
        if config.projector_type == "mlp":
            return MultiModalProjector(config.llm_dim, config.hidden_size)
        elif config.projector_type == "minicpm":
            return MiniCPMMLP(config)
        elif config.projector_type == "default":
            return nn.Linear(config.llm_dim, config.hidden_size, bias=False)
        else:
            raise ValueError(f"Unsupported projector type: {config.projector_type}")

    # non-streaming
    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        min_new_token=50,
        max_new_token=2048,
        show_tqdm=True,
        streaming=False,
        text_lengths=None,
        sampling_params: TTSSamplingParams = TTSSamplingParams(),
    ):
        temperature = torch.tensor(
            [sampling_params.temperature] * self.config.num_vq,
            dtype=torch.float,
            device=self.device,
        )
        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        condition_length = inputs_embeds.shape[1]
        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        if streaming:
            raise NotImplementedError("this kind of streaming is not supported yet")

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        past_key_values = None

        for t in range(max_new_token):
            audio_bos = False
            # If this is the first audio token, the case is special
            if t == 0:
                audio_bos = True
                inputs_embeds = inputs_embeds
                position_ids = torch.tensor(
                    list(range(0, condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                if streaming:
                    raise NotImplementedError("this kind of streaming is not supported yet")
                else:
                    causal_mask_4d = None

            else:
                code_emb = []
                for q in range(self.num_vq):
                    x = self.emb_code[q](new_tokens[:, t - 1 : t, q])
                    code_emb.append(x)

                inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor(
                    [condition_length + t - 1], dtype=torch.long, device=self.device
                ).unsqueeze(0)

                if streaming:
                    raise NotImplementedError("this kind of streaming is not supported yet")
                else:
                    causal_mask_4d = None

            if self.config.backbone_model == "llama":
                outputs: BaseModelOutputWithPast = self.model(
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask_4d,
                    use_cache=True,
                )
            else:
                raise ValueError(f"Unsupported backbone model: {self.config.backbone_model}")

            del position_ids
            del inputs_embeds

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

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

            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens
                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

                del logits_token

            if t < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits

            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

            del scores

            idx_next = idx_next.view(-1, self.num_vq)

            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        genrated_input_ids = new_tokens[:, 0:t, :]

        return MiniCPMTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=None,  # for update purpose
            past_key_values=None,  # for update purpose
            past_input_ids=None,  # for update purpose
            finished=finish.all(),
        )

    # fake streaming
    @torch.inference_mode()
    def generate_mock_legacy_streaming(
        self,
        inputs_embeds: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        min_new_token=50,
        max_new_token=2048,
        show_tqdm=True,
        streaming=False,
        text_lengths=None,
        sampling_params: TTSSamplingParams = TTSSamplingParams(),
        valid_text_length=None,
    ):
        assert valid_text_length is not None, "valid_text_length should be not None"

        tts_text_scope = [0, inputs_embeds.shape[1]]
        tts_text_mask = torch.zeros(inputs_embeds.shape[1], dtype=torch.int8, device=inputs_embeds.device)
        tts_text_mask[0:valid_text_length] = 1
        tts_text_mask[-1] = 1  # [Ptts]

        streaming_mask_4d_full = make_streaming_chunk_mask_inference(
            tts_text_scope=tts_text_scope,
            tts_text_mask=tts_text_mask,
            dtype=torch.bfloat16,
            device=self.device,
            streaming_audio_chunk_size=50,
            max_sequence_length=4096,
        )

        temperature = torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.device)
        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        condition_length = inputs_embeds.shape[1]
        pbar: Optional[tqdm] = None
        if show_tqdm:
            pbar = tqdm(
                total=max_new_token,
                desc="code",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
            )

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        past_key_values = None

        for t in range(max_new_token):
            audio_bos = False
            if t == 0:
                audio_bos = True
                inputs_embeds = inputs_embeds
                position_ids = torch.tensor(
                    list(range(0, condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

                causal_mask_4d = streaming_mask_4d_full[:, :, :condition_length, :condition_length]
            else:
                code_emb = []
                for q in range(self.num_vq):
                    x = self.emb_code[q](new_tokens[:, t - 1 : t, q])
                    code_emb.append(x)

                inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor(
                    [condition_length + t - 1], dtype=torch.long, device=self.device
                ).unsqueeze(0)

                causal_mask_4d = streaming_mask_4d_full[
                    :,
                    :,
                    condition_length + t : condition_length + t + 1,
                    : condition_length + t,
                ]

                # get length of past_key_values
                past_key_values_length = past_key_values.get_seq_length()

                assert causal_mask_4d.shape[-1] == (past_key_values_length + 1)

            if self.config.backbone_model == "llama":
                outputs: BaseModelOutputWithPast = self.model(
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    attention_mask=causal_mask_4d,
                    use_cache=True,
                )
            else:
                raise ValueError(f"Unsupported backbone model: {self.config.backbone_model}")

            del position_ids
            del inputs_embeds

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

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
            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                for logitsWarpers in logits_warpers:
                    logits = logitsWarpers(logits_token, logits)

                del logits_token

            if t < min_new_token:
                logits[:, eos_token] = -torch.inf

            if force_no_stop:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)

            del logits
            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)

            del scores

            idx_next = idx_next.view(-1, self.num_vq)
            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        if not finish.all():
            logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

        genrated_input_ids = new_tokens[:, 0:t, :]

        return MiniCPMTTSGenerationOutput(
            new_ids=genrated_input_ids,
            audio_input_ids=None,  # for update purpose
            past_key_values=None,  # for update purpose
            past_input_ids=None,  # for update purpose
            finished=finish.all(),
        )

    # non-streaming, interleave
    @torch.inference_mode()
    def generate_chunk(
        self,
        inputs_embeds: torch.Tensor,
        temperature: torch.Tensor,
        repetition_penalty: float,
        eos_token: Union[int, torch.Tensor],
        force_no_stop=False,
        max_new_token=500,
        min_new_tokens=0,
        past_key_values=None,
        logits_processors=None,
        text_start_pos=None,
    ):
        """For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
        |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
        where the last position is the audio BOS token.
        So, the first iteration in generation directly forward the model with inputs_embeds, and
        the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
        """
        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens, repetition_penalty=repetition_penalty
        )

        # We only support batch size `1` for now
        assert inputs_embeds.shape[0] == 1
        eos_token = eos_token.to(inputs_embeds.device)
        finish = torch.zeros(inputs_embeds.shape[0], device=inputs_embeds.device).bool()

        temperature = (temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)).to(
            inputs_embeds.device
        )

        condition_length = inputs_embeds.shape[1]

        new_tokens = torch.zeros(
            inputs_embeds.shape[0],
            max_new_token,
            self.num_vq,
            device=inputs_embeds.device,
            dtype=torch.long,
        )

        for t in range(max_new_token):
            audio_bos = False

            # If this is the first audio token, the case is special
            if t == 0:
                audio_bos = True
                inputs_embeds_ = inputs_embeds
                position_ids = torch.tensor(
                    list(range(text_start_pos, text_start_pos + condition_length)),
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
            else:
                # Generate the following audio tokens, it is applicable to all other cases, including second and the following calling of `generate`
                inputs_embeds_ = self.emb_code[0](new_tokens[:, t - 1 : t, 0])

                position_ids = torch.tensor(
                    [text_start_pos + condition_length + t - 1],  # prefill the previous token
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)

            outputs: BaseModelOutputWithPast = self.model(
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds_,
                use_cache=True,
            )

            del position_ids
            del inputs_embeds_

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

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

            logits /= temperature

            if not audio_bos:
                input_ids_sliced = new_tokens[:, 0:t].permute(0, 2, 1)  # get previous t new tokens

                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)

                del input_ids_sliced

                for logitsProcessors in logits_processors:
                    logits = logitsProcessors(logits_token, logits)

                del logits_token

            if force_no_stop or t < min_new_tokens:
                logits[:, eos_token] = -torch.inf

            scores = F.softmax(logits, dim=-1)
            del logits

            idx_next = torch.multinomial(scores, num_samples=1).to(finish.device)
            del scores

            idx_next = idx_next.view(-1, self.num_vq)

            finish_or = idx_next.eq(eos_token).any(1)
            finish.logical_or_(finish_or)

            del finish_or
            new_tokens[:, t] = idx_next

            if t == 0 and finish.any():
                break

            del idx_next

            if finish.all():
                break

        # The latest generated token is not in the range returned this time. If it is an eos token, it is not returned. If it is a normal token, it is not returned.
        genrated_input_ids = new_tokens[:, 0:t, :]

        return genrated_input_ids, past_key_values

    @torch.inference_mode()
    def interleaved_generate(
        self,
        spk_embeds: torch.Tensor,
        conditions: List[torch.Tensor],
        temperature: torch.Tensor,
        repetition_penalty: float,
        eos_token: Union[int, torch.Tensor],
        **kwargs,
    ):
        """
        For inputs_embeds, it should be like [bs=1, seq_len, hidden_dim], its content is like:
        |Text BOS|Spk embeds|Text-Hidden states Interleave (if applicable)|Audio BOS|
        where the last position is the audio BOS token.
        So, the first iteration in generation directly forward the model with inputs_embeds, and the last hidden states of the last position (Audio BOS) will be decoded to get the first audio token.
        """
        temperature = torch.tensor([temperature], dtype=torch.float, device=self.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=self.config.num_audio_tokens,
            repetition_penalty=repetition_penalty,
        )

        eos_token = eos_token.to(conditions[0].device)

        num_chunks = len(conditions)
        generated_tokens = []
        text_start_pos = 0
        last_window_size = 0
        past_key_values = None

        for idx in range(num_chunks):
            condition = conditions[idx].to(conditions[0].device)
            if self.attention_type == "sliding_recompute":
                recomputed_conditions = []

                if (
                    idx >= self.window_size
                    and (idx - self.recomputed_chunks) % (self.window_size - self.recomputed_chunks) == 0
                ):
                    for i in range(self.recomputed_chunks):
                        recomputed_conditions.append(conditions[idx - self.recomputed_chunks + i])
                        recomputed_conditions.append(
                            self.emb_code[0](generated_tokens[-self.recomputed_chunks + i][:, :, 0])
                        )
                    recomputed_conditions.append(condition)
                    condition = torch.cat(recomputed_conditions, dim=1)

                    text_start_pos = 0
                    new_tokens, old_kv = self.generate_chunk(
                        inputs_embeds=condition,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        eos_token=eos_token,
                        force_no_stop=False,
                        max_new_token=500,
                        past_key_values=None,
                        logits_processors=logits_processors,
                        text_start_pos=text_start_pos,
                    )

                else:
                    new_tokens, old_kv = self.generate_chunk(
                        inputs_embeds=condition,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        eos_token=eos_token,
                        force_no_stop=False,
                        max_new_token=500,
                        past_key_values=past_key_values,
                        logits_processors=logits_processors,
                        text_start_pos=text_start_pos,
                    )
            else:
                new_tokens, old_kv = self.generate_chunk(
                    inputs_embeds=condition,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    eos_token=eos_token,
                    force_no_stop=False,
                    max_new_token=500,
                    past_key_values=past_key_values,
                    logits_processors=logits_processors,
                    text_start_pos=text_start_pos,
                )

            past_key_values = []
            if self.attention_type == "sliding_window" and idx >= 1:
                for layer_idx in range(len(old_kv)):
                    past_key_values.append(
                        (
                            old_kv[layer_idx][0][:, :, last_window_size:, :],
                            old_kv[layer_idx][1][:, :, last_window_size:, :],
                        )
                    )
            else:
                past_key_values = old_kv

            last_window_size = condition.shape[1] + new_tokens.shape[1]
            text_start_pos += last_window_size

            if idx == 0:
                generated_tokens = [new_tokens]
            else:
                generated_tokens.append(new_tokens)

        return MiniCPMTTSGenerationOutput(new_ids=torch.cat(generated_tokens, dim=1), finished=True)


class CustomRepetitionPenaltyLogitsProcessorRepeat:
    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = F.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(0, self.max_input_ids, freq.size(0) - self.max_input_ids).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        inp = scores.multiply(alpha)
        oth = scores.divide(alpha)
        con = scores < 0
        out = torch.where(con, inp, oth)
        del inp, oth, scores, con, alpha
        return out


def gen_logits(num_code: int, top_p=0.7, top_k=20, repetition_penalty=1.0):
    logits_warpers = []

    if top_p is not None:
        logits_warpers.append(TopPLogitsWarper(top_p, min_tokens_to_keep=3))

    if top_k is not None:
        logits_warpers.append(TopKLogitsWarper(top_k, min_tokens_to_keep=3))

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        logits_processors.append(CustomRepetitionPenaltyLogitsProcessorRepeat(repetition_penalty, num_code, 16))

    return logits_warpers, logits_processors
