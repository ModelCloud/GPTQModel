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

import json
import logging
import math
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers.modeling_outputs import ModelOutput

from ..native.qwen2.configuration_qwen2 import Qwen2Config
from .cache import cache_from_pairs, cache_pairs
from .modeling_minicpmo import MiniCPMO, MiniCPMOPreTrainedModel, MultiModalProjector
from .native import Qwen2ForCausalLM
from .processing_minicpmo import MiniCPMOProcessor
from .tts_26 import gen_logits
from .utils_26 import NumberToTextConverter, VoiceChecker, sentence_end


logger = logging.getLogger(__name__)


@dataclass
class OmniOutput(ModelOutput):
    text: Optional[Union[str, List[str], Iterator]] = None
    spk_embeds: Optional[torch.FloatTensor] = None
    audio_wav: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None


class MiniCPMO26(MiniCPMO):
    def __init__(self, config):
        MiniCPMOPreTrainedModel.__init__(self, config)
        values = config.to_dict()
        values.pop("model_type", None)
        values["attn_implementation"] = config._attn_implementation
        self.llm = Qwen2ForCausalLM(Qwen2Config(**values))

        self.embed_dim = self.llm.config.hidden_size

        # init vision module
        if self.config.init_vision:
            self.vpm = self.init_vision_module()
            self.vision_dim = self.vpm.embed_dim
            self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # init audio module
        if self.config.init_audio:
            self.apm = self.init_audio_module()
            audio_output_dim = int(self.apm.config.encoder_ffn_dim // 4)
            self.audio_avg_pooler = nn.AvgPool1d(self.config.audio_pool_step, stride=self.config.audio_pool_step)
            self.audio_projection_layer = MultiModalProjector(in_dim=audio_output_dim, out_dim=self.embed_dim)
            self.audio_encoder_layer = -1

        # init tts module
        if self.config.init_tts:
            self.tts = self.init_tts_module()

        self.processor = None

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

        self.default_tts_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}{% endif %}"
        self.force_no_stop = False

        # for stream api
        self.reset_session()

        # Transformers 5 derives `all_tied_weights_keys` and the other
        # load-time static properties here.
        self.post_init()

    def reset_session(self):
        self.session_id = None
        self.new_user_msg = True
        self.llm_generated = False
        self.llm_generate_completed = False
        self.llm_past_key_values = None
        self.audio_past_key_values = None  # apm kv cache

    def init_tts(self, **kwargs):
        raise NotImplementedError(
            "Legacy 2.6 Vocos codec loading is not qualified. The checkpoint-owned TTS/DVAE remains available."
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def subsequent_chunk_mask(
        self,
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

        Returns:
            torch.Tensor: mask

        Examples:
            >>> subsequent_chunk_mask(4, 2)
            [[1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]
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
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (
            input_lengths_after_cnn - self.config.audio_pool_step
        ) // self.config.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_vllm_embedding(self, data):
        """
        Compute all visual embeddings, and set into llm embeddings.
        Args:
            data: Dict
                tgt_sizes: image size after patch embedding
                pixel_values: image features
                image_bound: position of each picture corresponding to input_ids
                input_ids: full input_ids, include placeholder
        Returns:
                embedding with vision, vision_hidden_states
        """
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
                        all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
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
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data["vision_hidden_states"]

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        new_vllm_embedding = vllm_embedding.clone()

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

                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )

                elif self.training:
                    new_vllm_embedding[i] += cur_vs_hs[0].mean() * 0

        return new_vllm_embedding, vision_hidden_states

    def get_audio_embedding_streaming(self, data):
        r"""
        Extract audio embeddings in a streaming manner using cached key-value pairs.

        This method processes incoming audio features incrementally and stores/updates `past_key_values`
        for faster inference on subsequent audio frames. It only supports batch_size=1 and is intended
        for streaming scenarios.

        Args:
            data (dict):
                - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                - **"audio_feature_lens"** (List[List[int]]): Lengths of each audio segment for each item in the batch.

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

            if self.audio_past_key_values is not None:
                cache_length = self.audio_past_key_values.get_seq_length()
                apm_max_len = self.apm.embed_positions.weight.shape[0]
                if cache_length + max_seq_len >= apm_max_len:
                    logger.warning(
                        f"audio_past_key_values length {cache_length + max_seq_len} exceed {apm_max_len}, reset."
                    )
                    self.audio_past_key_values = None

            audio_outputs = self.apm(wavforms, past_key_values=self.audio_past_key_values, use_cache=True)
            audio_states = audio_outputs.last_hidden_state  # [:, :audio_feat_lengths, :]
            self.audio_past_key_values = audio_outputs.past_key_values

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
        else:
            return []

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True):
        r"""
        Extract full audio embeddings with optional chunk-based attention.

        This method computes embeddings for all audio frames at once, either using full attention (when
        `chunk_length` is -1) or chunk-based attention (when `chunk_length` is a positive number). It does
        not use key-value caching and is suitable for non-streaming inference.

        Args:
            data (dict):
                - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                - **"audio_feature_lens"** (List[List[int]]): Lengths of each audio segment for each item in the batch.
            chunk_length (int, optional): Determines whether to use full attention (-1) or chunk-based
                attention (>0) during embedding computation.

        Returns:
            List[List[torch.Tensor]]: audio embeddings
        """

        wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
        audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feature_lens.dtype, device=audio_feature_lens.device)
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
            stream_input: use streaming audio embedding
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

                if self.config.chunk_input:
                    for i in range(bs):
                        if not audio_embeddings[i]:
                            continue
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
                        for embs, bound in zip(audio_embs, bounds):
                            audio_indices = torch.arange(bound[0], bound[1], dtype=torch.long).to(
                                input_embeddings.device
                            )

                            if embs.shape[0] != len(audio_indices):
                                raise ValueError(
                                    f"Shape mismatch: Trying to assign embeddings of shape {embs.shape} "
                                    f"to input indices of length {len(audio_indices)}"
                                )
                            input_embeddings[i, audio_indices] = embs.to(input_embeddings.dtype)
        elif self.training:
            for i in range(bs):
                # dummy audio_embeddings
                input_embeddings = input_embeddings + audio_embeddings[0].mean() * 0

        return input_embeddings

    def _decode(self, inputs_embeds, tokenizer, attention_mask, **kwargs):
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict_in_generate", None)
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

    def _decode_stream(self, *args, **kwargs):
        raise NotImplementedError("Threaded text streaming is not qualified for local MiniCPM-o.")

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

    def get_sys_prompt(self, ref_audio=None, mode="default", language="zh"):
        """
        Choose different system prompts according to different tasks
        Args:
            ref_audio: if ref_audio is not None, will use the voice cloning prompts, and the voice
                       generated by the model will refer to the timbre of ref audio
            mode:
                "default": default system prompt and not refer to any task
                "omni": input video and audio simultaneously
                "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
                "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
                "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
            language: prompts language, the model has the ability to automatically select the response language
                    based on the question language
        Returns:

        """
        if ref_audio is not None:
            assert isinstance(ref_audio, np.ndarray), "ref_audio error"
        if mode == "omni":
            if language == "zh":
                sys_prompt = "你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。"
                vc_prompt_prefix = sys_prompt + "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
            else:
                sys_prompt = "You are a helpful assistant. You can accept video, audio and text input and output voice and text. "
                vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "As an assistant, you will speak using this voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                sys_msgs = {"role": "user", "content": [sys_prompt]}

            return sys_msgs
        elif mode == "audio_assistant":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
            else:
                vc_prompt_prefix = "Use the voice in the audio prompt to synthesize new content."
                vc_prompt_suffix = "You are a helpful assistant with the above voice style."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

            else:
                logger.warning(
                    "Warning: ref_audio is None, speech generation will be performed based on the default voice."
                )
                sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "audio_roleplay":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
                vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."
                vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
            else:
                print("Warning: ref_audio is None, speech generation will be performed based on the default voice.")
                sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

            return sys_msgs
        elif mode == "voice_cloning":
            if language == "zh":
                vc_prompt_prefix = "模仿输入音频中的声音特征。"
            else:
                vc_prompt_prefix = "Clone the voice in the provided audio prompt."

            if ref_audio is not None:
                sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio]}
            else:
                raise ValueError("ref_audio con't be None in voice_cloning mode.")

            return sys_msgs
        else:
            sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
            sys_msgs = {"role": "user", "content": [sys_prompt]}

            return sys_msgs

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        audio_features=[],
        audio_feature_lens=None,
        image_bound=None,
        audio_bounds=None,
        spk_bounds=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        stream=False,
        decode_text=True,
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
                # if stream return TextIteratorStreamer and output is empty
                outputs = {}
            else:
                outputs = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, **kwargs)

                result = self._decode_text(outputs.sequences, tokenizer)

        if decode_text is False:
            return outputs

        return result, outputs

    def chat(
        self,
        image=None,
        msgs=None,
        tokenizer=None,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=32768,
        stream=False,
        chunk_input=True,
        omni_input=False,
        max_slice_nums=None,
        use_image_id=None,
        use_tts_template=False,
        generate_audio=False,
        return_spk_embed=False,
        return_dict=False,
        output_audio_path=None,
        **kwargs,
    ):
        """
        Unified chat function

        Args:
            image: use for batch_size=1 vqa, It is not recommended to continue to use this parameter
            msgs: the input chat msgs, support text: (string)  / image: (PIL.Image) / audio (numpy.ndarray)
            tokenizer: tokenizer for llm
            processor: if None, use the default processor
            max_new_tokens: the maximum length of the generation
            min_new_tokens: the minimum length of the generation
            sampling: whether to use sampling decoding or beam search decoding
            max_inp_length: the maximum length of input
            stream: whether to return generator, only used when tts is not required
            chunk_input: whether to split audio into 1s chunks
            omni_input: determine whether it is omni mode
            max_slice_nums: control the maximum number of image slices
            use_image_id: for video understanding or omni understanding, use_image_id should be False
            use_tts_template: if the msgs contain audio, use_tts_template should be True
            generate_audio: whether to generate audio output, only used when return_dict=True
            return_spk_embed: whether to return spk embedding, only used when return_dict=True
            return_dict: whether to return dict
            output_audio_path: audio save path when generate_audio
            **kwargs:
        """
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False

        if generate_audio or return_spk_embed:
            return_dict = True

        msgs_list = msgs
        images_list = image

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if self.processor is None:
                self.processor = MiniCPMOProcessor.from_pretrained(
                    self.config._name_or_path, config=self.config, tokenizer=tokenizer
                )
            processor = self.processor

        assert self.config.query_num == processor.image_processor.image_feature_size, (
            "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        )
        assert self.config.patch_size == processor.image_processor.patch_size, (
            "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        )
        assert self.config.use_image_id == processor.image_processor.use_image_id, (
            "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        )
        assert self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums, (
            "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        )
        assert self.config.slice_mode == processor.image_processor.slice_mode, (
            "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        )

        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []

        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

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
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, np.ndarray):  # audio
                        audios.append(c)
                        audio_parts.append(i)
                        cur_msgs.append("(<audio>./</audio>)")
                        use_tts_template = True
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                if omni_input:
                    msg["content"] = "".join(cur_msgs)
                else:
                    msg["content"] = "\n".join(cur_msgs)

            prompts_lists.append(
                processor.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=self.default_tts_chat_template if use_tts_template else None,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)

        inputs = processor(
            prompts_lists,
            input_images_list,
            input_audios_list,
            audio_parts_list,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            chunk_input=chunk_input,
            return_tensors="pt",
            max_length=max_inp_length,
        ).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config["min_new_tokens"] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        inputs.pop("image_sizes")
        with torch.inference_mode():
            res, outputs = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                **generation_config,
            )

        if stream:

            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, "")
                    yield text

            if return_dict:
                return OmniOutput(text=stream_gen())
            else:
                return stream_gen()

        else:
            spk_embeds = wav_numpy = sr = None

            if batched:
                answer = res
            else:
                answer = res[0]

                if use_tts_template and generate_audio:
                    mel_spec = self._generate_mel_spec(inputs, outputs, answer)
                    wav_numpy, sr = self.decode_mel_to_audio(mel_spec, output_audio_path)

            if return_spk_embed:
                spk_embeds = self._get_last_spk_embeds(inputs, outputs)

            if isinstance(answer, list):
                answer = [i.replace(tokenizer.tts_end, "") for i in answer]
            else:
                answer = answer.replace(tokenizer.tts_end, "")

            if return_dict:
                return OmniOutput(text=answer, spk_embeds=spk_embeds, audio_wav=wav_numpy, sampling_rate=sr)
            else:
                return answer

    @torch.inference_mode()
    def streaming_prefill(
        self,
        session_id,
        msgs,
        tokenizer,
        omni_input=True,
        max_slice_nums=None,
        ls_temperature=1.0,
        **kwargs,
    ):
        """
        Streaming video/audio input and output audio stream, Only support batch_size=1
        Args:
            session_id: Note: new connection should use a new session_id
        """
        assert session_id is not None
        if self.session_id is None or session_id != self.session_id:  # new session
            self.is_first = True
        else:
            self.is_first = False

        images = []
        audios = []

        assert len(msgs) == 1
        copy_msgs = deepcopy(msgs)
        msg = copy_msgs[0]

        assert msg["role"] in ["system", "user", "assistant"]

        content = msg["content"]
        cur_msgs = []
        for j, c in enumerate(content):
            if isinstance(c, Image.Image):
                images.append(c)
                cur_msgs.append("(<image>./</image>)")
            elif isinstance(c, np.ndarray):  # audio
                audios.append(c)
                cur_msgs.append("(<audio>./</audio>)")
            elif isinstance(c, str):
                cur_msgs.append(c)
            else:
                logger.error("Invalid content type:", c)

        cur_contents = "".join(cur_msgs) if omni_input else "\n".join(cur_msgs)
        if not self.is_first and self.new_user_msg and msg["role"] == "user":  # new user add im_start
            if self.llm_generated:
                if self.llm_generate_completed:
                    msg["content"] = "<|im_end|>\n<|im_start|>user\n" + cur_contents
                else:  # break llm gen, add tts_eos
                    msg["content"] = "<|tts_eos|><|im_end|>\n<|im_start|>user\n" + cur_contents
            else:
                msg["content"] = "<|im_start|>user\n" + cur_contents
            self.new_user_msg = False
        else:
            msg["content"] = cur_contents

        if msg["role"] in ["system", "assistant"]:
            self.new_user_msg = True
            self.audio_past_key_values = None  # apm kv cache

        if self.is_first:
            # init pask_key_values
            logger.info(f"new session_id: {session_id}, reset kv cache")
            self.reset_session()
            self.session_id = session_id

            prompt = tokenizer.apply_chat_template(
                copy_msgs, tokenize=False, add_generation_prompt=False, chat_template=self.default_tts_chat_template
            )
            add_special_tokens = True  # add bos
        else:
            prompt = copy_msgs[0]["content"]
            add_special_tokens = False

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
        ).to(self.device)

        # 1. prepare input embeddings
        model_inputs["inputs_embeds"], _ = self.get_vllm_embedding(model_inputs)
        # get audio embedding with audio_past_key_values
        inputs_embeds = self.get_omni_embedding(
            model_inputs, input_embeddings=model_inputs["inputs_embeds"], stream_input=True
        )

        if self.is_first:
            # clean audio_past_key_values after first prefill
            self.audio_past_key_values = None

        if self.llm_past_key_values is not None:
            cache_length = self.llm_past_key_values.get_seq_length()
        else:
            cache_length = 0

        attention_mask = torch.ones((1, cache_length + inputs_embeds.shape[1]), dtype=torch.bool, device=self.device)

        # 2. do prefill and predict listen/speak label
        outputs = self.llm(
            past_key_values=self.llm_past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,  # position_ids,
            use_cache=True,
            return_dict=True,
        )
        self.llm_past_key_values = outputs["past_key_values"]
        return

    @torch.inference_mode()
    def streaming_generate(
        self,
        session_id,
        tokenizer,
        max_new_tokens=512,
        min_new_tokens=0,
        sampling=True,
        generate_audio=True,
        enable_regenerate=False,
        **kwargs,
    ):
        """
        Streaming video/audio input and output audio stream
        Args:
        """
        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05,
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }
        generation_config["min_new_tokens"] = min_new_tokens
        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        # do generate
        # reset buffer
        self.new_user_msg = True
        self.llm_generated = True
        self.llm_generate_completed = False
        self.audio_past_key_values = None  # apm kv cache

        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        generate_prompt = "<|im_end|>\n<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>"
        input_ids = tokenizer(generate_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            self.device
        )

        spk_start_idx = torch.where(input_ids[0] == tokenizer.spk_start_id)[0]
        spk_end_idx = torch.where(input_ids[0] == tokenizer.spk_end_id)[0]
        spk_bounds = [
            torch.hstack([(spk_start_idx + 1).unsqueeze(-1), spk_end_idx.unsqueeze(-1)])
        ]  # List[Tensor], (1,2)

        cache_length = self.llm_past_key_values.get_seq_length()
        attention_mask = torch.ones((1, cache_length + input_ids.shape[1]), dtype=torch.bool, device=self.device)

        generation_config["max_new_tokens"] = max_new_tokens
        streamer = self.llm_generate_chunk(input_ids, attention_mask, tokenizer, terminators, generation_config)

        if generate_audio:
            result = self._generate_mel_spec_audio_streaming(
                spk_bounds, streamer, output_chunk_size=25, enable_regenerate=enable_regenerate
            )
            return result
        else:
            return streamer

    def llm_generate_chunk(self, input_ids, attention_mask, tokenizer, terminators, generation_config):
        def check_uncompleted_token(ids):
            cur_text = tokenizer.decode(ids)
            end = len(ids)
            while cur_text[-1] == "�":
                end -= 1
                if end == 0:
                    break
                cur_text = tokenizer.decode(ids[:end])
            return end

        max_new_tokens = int(generation_config.pop("max_new_tokens", 2048))
        new_len = 0
        first_chunk = True
        eos = False
        left_ids = None

        while True:
            outputs = self.llm.generate(
                input_ids=input_ids,
                past_key_values=self.llm_past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                max_new_tokens=3,  # reduce first token delay
                pad_token_id=0,
                output_hidden_states=True if first_chunk else False,
                return_dict_in_generate=True,
                eos_token_id=terminators,
                **generation_config,
            )
            if outputs.sequences[0, -1] in terminators:
                eos = True
            input_len = input_ids.shape[1]
            cur_ids = outputs.sequences[:, input_len:]
            new_len += cur_ids.shape[1]

            if left_ids is not None and left_ids.shape[1] > 0:
                cur_ids = torch.cat([left_ids, cur_ids], dim=1)
            end = check_uncompleted_token(cur_ids[0])
            left_ids = cur_ids[:, end:]
            cur_ids = cur_ids[:, :end]
            text = self._decode_text(cur_ids, tokenizer)[0] if end > 0 else ""

            self.llm_past_key_values = outputs.past_key_values
            input_ids = outputs.sequences[:, -1:]
            cache_length = self.llm_past_key_values.get_seq_length()
            attention_mask = torch.ones((1, cache_length + input_ids.shape[1]), dtype=torch.bool, device=self.device)

            res = {"text": text}
            if first_chunk:
                res["hidden_states"] = outputs.hidden_states
                first_chunk = False
            yield res

            if eos:
                self.llm_generate_completed = True
                break
            if new_len >= max_new_tokens:
                logger.debug(f"LLM generation {new_len} exceeds max_new_tokens({max_new_tokens}), break.")
                break

    def prepare_tts_text(self, text):
        tts_tokens = self.tts_processor.text_tokenizer.encode(text, add_special_tokens=False)
        tts_tokens_len = len(tts_tokens)
        if tts_tokens_len < self.tts.streaming_text_reserved_len:
            num_pad_tokens = self.tts.streaming_text_reserved_len - tts_tokens_len

            pad_str = "[Etts]" + "[PAD]" * (num_pad_tokens - 1)
        else:
            tts_tokens = tts_tokens[0 : self.tts.streaming_text_reserved_len]
            tts_tokens_len = len(tts_tokens)
            text = self.tts_processor.text_tokenizer.decode(tts_tokens, add_special_tokens=False)
            pad_str = ""
        spk_emb_placeholder_tts = "[spk_emb]" * self.tts.num_spk_embs

        new_text_tts = f"[Stts]{spk_emb_placeholder_tts}{text}{pad_str}[Ptts]"
        return new_text_tts, tts_tokens_len

    def get_tts_text_start_token_ids(self):
        text = "[Stts]" + "[spk_emb]" * self.tts.num_spk_embs
        tts_input_ids = self.tts_processor.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(self.device)
        return tts_input_ids

    def _build_streaming_mask(self, tts_tokens_len):
        tts_sequence_full_length = (
            1 + self.tts.num_spk_embs * self.tts.use_speaker_embedding + self.tts.streaming_text_reserved_len + 1
        )
        streaming_attention_mask = torch.zeros(tts_sequence_full_length, dtype=torch.int8)
        streaming_attention_mask[0 : 1 + 1 + tts_tokens_len + 1] = 1
        streaming_attention_mask[-1] = 1
        return streaming_attention_mask

    def _get_last_spk_embeds(self, inputs, outputs):
        last_hidden_states = [hs[-1] for hs in outputs.hidden_states]

        # batch = 1
        last_hidden_states = torch.vstack([i[0] for i in last_hidden_states])

        # last spk
        spk_bound = inputs["spk_bounds"][0][-1]

        spk_embeds = last_hidden_states[spk_bound[0] : spk_bound[1]]
        return spk_embeds

    def _generate_mel_spec(self, inputs, outputs, text, output_chunk_size=25, tts_max_new_tokens=2048):
        spk_embeds = self._get_last_spk_embeds(inputs, outputs)

        text = text.split("<|tts_bos|>")[-1]
        gen_text = text.split("<|tts_eos|>")[0]
        tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
        tts_inputs = self.tts_processor.text_tokenizer.encode(tts_text, add_special_tokens=False)
        tts_input_ids = torch.Tensor(tts_inputs).unsqueeze(0).to(self.device, dtype=torch.long)
        streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)

        logits_warpers, logits_processors = gen_logits(
            num_code=626, top_P=self.tts.top_p, top_K=self.tts.top_k, repetition_penalty=self.tts.repetition_penalty
        )

        condition_length = (
            1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs + self.tts.streaming_text_reserved_len + 1
        )

        dtype = self.tts.emb_text.weight.dtype
        emb = torch.zeros(1, condition_length, self.tts.num_vq, dtype=dtype, device=self.tts.device)
        past_key_values = cache_from_pairs(
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
        )

        audio_input_ids = torch.zeros(1, condition_length, self.tts.num_vq, dtype=torch.long, device=self.tts.device)

        eos_lab = False
        for chunk_idx in range(math.ceil(emb.shape[1] / self.tts.streaming_text_chunk_size)):
            if chunk_idx == 0:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + 0
                end = (
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs
                )
            else:
                begin = (
                    chunk_idx * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs
                )
                end = min(
                    (chunk_idx + 1) * self.tts.streaming_text_chunk_size
                    + 1
                    + self.tts.use_speaker_embedding * self.tts.num_spk_embs,
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
                        input_ids=text_input_ids, position_ids=position_ids, past_key_values=past_key_values
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
                logger.debug("Generation finished.")
                eos_lab = True
                break

        if not eos_lab:
            logger.debug("eos_lab False, Generation continue.")
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
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > tts_max_new_tokens:
                    logger.debug(f"Generation length > {tts_max_new_tokens}, stopped.")
                    break

        mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids)
        return mel_spec

    def _linear_overlap_add2_wav(self, frames: List[torch.Tensor], overlap: int):
        """
        Merge two audio waveforms with smooth in streaming audio generation.
        Borrowed some codes from `https://github.com/huggingface/transformers/blob/main/src/transformers/models/encodec/modeling_encodec.py`
        """
        assert len(frames) == 2
        device = frames[0].device
        dtype = frames[0].dtype
        # shape = frames[0].shape[:-1]

        frame0_length = frames[0].shape[-1]
        frame1_length = frames[1].shape[-1]
        total_size = frame0_length + frame1_length - overlap
        weight_len = max(frame0_length, frame1_length) + overlap
        t = torch.linspace(0, 1, weight_len + 2, device=device, dtype=dtype)[1:-1]
        weight = 0.5 - (t - 0.5).abs()

        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(total_size, device=device, dtype=dtype)
        offset: int = 0

        out[offset : offset + frame0_length] += weight[-frame0_length:] * frames[0]
        sum_weight[offset : offset + frame0_length] += weight[-frame0_length:]
        offset += frame0_length - overlap
        out[offset : offset + frame1_length] += weight[:frame1_length] * frames[1]
        sum_weight[offset : offset + frame1_length] += weight[:frame1_length]

        assert sum_weight.min() > 0
        out = out / sum_weight
        return out[:frame0_length], out[frame0_length:]

    def _generate_mel_spec_audio_streaming(
        self,
        spk_bounds,
        streamer,
        output_chunk_size=25,
        spk_embeds=None,
        prev_seg_text_ids=None,
        prev_seg_text_left="",
        prev_seg_audio_ids=None,
        enable_regenerate=False,
    ):
        # get spk_embedding
        gen_text = ""
        tts_text = ""
        new_segment_gen = False
        if spk_embeds is None:
            spk_bound = spk_bounds[0][-1]
            r = next(streamer)
            txt = r["text"]
            gen_text += txt.split("<|tts_eos|>")[0]
            tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
            last_hidden_states = r["hidden_states"][0][-1][0]  # output: (input_seq_len, dim)
            spk_embeds = last_hidden_states[spk_bound[0] : spk_bound[1]]

        # init past_key_values
        logits_warpers, logits_processors = gen_logits(
            num_code=626, top_P=self.tts.top_p, top_K=self.tts.top_k, repetition_penalty=self.tts.repetition_penalty
        )
        condition_length = (
            1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs + self.tts.streaming_text_reserved_len + 1
        )
        tts_start_token_len = 1 + self.tts.use_speaker_embedding * self.tts.num_spk_embs
        dtype = self.tts.emb_text.weight.dtype
        past_key_values = cache_from_pairs(
            (
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
                torch.zeros(
                    1,
                    self.tts.config.num_attention_heads,
                    condition_length - 1,
                    self.tts.config.hidden_size // self.tts.config.num_attention_heads,
                    dtype=dtype,
                    device=self.tts.device,
                ),
            )
            for _ in range(self.tts.config.num_hidden_layers)
        )
        audio_input_ids = torch.zeros(1, condition_length, self.tts.num_vq, dtype=torch.long, device=self.tts.device)

        # prefill prev segment for smooth
        chunk_idx = 0
        new_ids_len = 0
        prev_text_len = 0
        if prev_seg_text_ids is not None and prev_seg_audio_ids is not None:
            tts_token_lens = prev_seg_text_ids.shape[1]
            # assert tts_token_lens % self.tts.streaming_text_chunk_size == 0
            streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
            position_ids = torch.arange(
                0, tts_token_lens + tts_start_token_len, dtype=torch.long, device=self.tts.device
            ).unsqueeze(0)

            text_input_ids = self.get_tts_text_start_token_ids()
            text_input_ids = torch.cat([text_input_ids, prev_seg_text_ids], dim=1)
            past_key_values = self.tts.prefill_text(
                input_ids=text_input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                lm_spk_emb_last_hidden_states=spk_embeds,
            )
            past_key_values = self.tts.prefill_audio_ids(
                input_ids=prev_seg_audio_ids[:, :-1, :],
                # not prefill last id, which will be input_id of next generation
                past_key_values=past_key_values,
                streaming_tts_text_mask=streaming_tts_text_mask,
            )

            # update init
            chunk_idx += int(tts_token_lens / self.tts.streaming_text_chunk_size)
            audio_input_ids = torch.cat([audio_input_ids, prev_seg_audio_ids], dim=1)
            text = self.tts_processor.text_tokenizer.decode(prev_seg_text_ids[0].tolist(), add_special_tokens=False)

            gen_text += text
            gen_text += prev_seg_text_left
            prev_text_len = len(gen_text)  # takecare the position
            new_ids_len += prev_seg_audio_ids.shape[1]

        prev_wav = None
        eos_lab = False
        stop = False
        shift_len = 180
        voice_checker = VoiceChecker()
        number_converter = NumberToTextConverter()
        lang = None
        gen_text_raw = gen_text
        for t, r in enumerate(streamer):
            t += 1
            txt = r["text"]
            txt = txt.split("<|tts_eos|>")[0]
            gen_text_raw += txt
            if t == 1 and txt == "" and prev_seg_text_ids is not None:
                logger.warning("New segment is empty, generation finished.")
                return
            if t <= 2:  # do just one time, more token greater certainty
                lang = number_converter.detect_language(gen_text_raw)
            gen_text += number_converter.replace_numbers_with_text(txt, lang).replace("*", "")  # markdown **

            # TODO speed up
            tts_text, tts_token_lens = self.prepare_tts_text(gen_text)

            if tts_token_lens >= self.tts.streaming_text_reserved_len - shift_len:
                end_c = sentence_end(txt)
                if end_c:
                    end_c_idx = gen_text.rfind(end_c)
                    assert end_c_idx != -1
                    text_left = gen_text[end_c_idx + 1 :]
                    gen_text = gen_text[: end_c_idx + 1]
                    tts_text, tts_token_lens = self.prepare_tts_text(gen_text)
                    new_segment_gen = True
                    logger.debug(
                        f"tts_text tokens {tts_token_lens} exceed {self.tts.streaming_text_reserved_len - shift_len}, starting a new segment generation"
                    )
                    break

            if tts_token_lens >= (chunk_idx + 1) * self.tts.streaming_text_chunk_size:
                # do prefill and generate
                if chunk_idx == 0:
                    begin = 0
                    end = (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len
                else:
                    begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
                    end = min(
                        (chunk_idx + 1) * self.tts.streaming_text_chunk_size + tts_start_token_len,
                        condition_length - 1,
                    )

                tts_input_ids = self.tts_processor.text_tokenizer(
                    tts_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].to(self.device)
                text_input_ids = tts_input_ids[:, begin:end]
                streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds if chunk_idx == 0 else None,
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
                audio_input_ids = (
                    outputs.audio_input_ids
                )  # [1,seq_len,4] seq_len=tts.streaming_text_reserved_len + 3 + len(new_ids)
                past_key_values = outputs.past_key_values
                chunk_idx += 1

                mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, max(new_ids_len - 4, 0) :, :])
                new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]

                wav_np, sr = self.decode_mel_to_audio(mel_spec)  # [1,100,50] -> [50*256]

                if enable_regenerate:
                    if prev_wav is not None:
                        check_wav_np = wav_np[2048:].cpu().numpy()  # 2*4*256(hop)
                        check_mel = mel_spec[0, :, 8:].cpu().numpy()  # 2*4
                    else:
                        check_wav_np = wav_np.cpu().numpy()
                        check_mel = mel_spec[0].cpu().numpy()
                if enable_regenerate and voice_checker.is_bad(check_wav_np, check_mel, chunk_size=2560):
                    voice_checker.reset()
                    # regenerate
                    N = output_chunk_size if prev_wav is None else output_chunk_size * 2
                    past_kv = cache_from_pairs(
                        (keys[:, :, :-N, :], values[:, :, :-N, :]) for keys, values in cache_pairs(past_key_values)
                    )
                    outputs = self.tts.generate(
                        input_ids=audio_input_ids[:, :-N, :],
                        past_key_values=past_kv,
                        streaming_tts_text_mask=streaming_tts_text_mask,
                        max_new_token=N,
                        force_no_stop=self.force_no_stop,
                        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                        eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                        logits_warpers=logits_warpers,
                        logits_processors=logits_processors,
                    )
                    audio_input_ids = outputs.audio_input_ids
                    past_key_values = outputs.past_key_values

                    new_ids_len -= N
                    mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, new_ids_len:, :])
                    new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]
                    wav_np, sr = self.decode_mel_to_audio(mel_spec)

                    if prev_wav is not None:
                        wav_y = wav_np[: len(prev_wav)]
                        prev_wav = wav_np[len(prev_wav) :]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_y, sampling_rate=sr)

                    else:
                        prev_wav = wav_np
                else:
                    # smooth wav
                    if prev_wav is not None:
                        wav_np, prev_wav = self._linear_overlap_add2_wav(
                            [prev_wav, wav_np], overlap=512 * 4
                        )  # tts_hop256*2
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)

                    else:
                        prev_wav = wav_np

                if outputs.finished:
                    logger.debug("Generation finished.")
                    eos_lab = True
                    break

        if not eos_lab and tts_text:
            logger.debug("eos_lab False, Generation continue.")

            if chunk_idx == 0:
                begin = 0
            else:
                begin = chunk_idx * self.tts.streaming_text_chunk_size + tts_start_token_len
            end = tts_token_lens + tts_start_token_len + 1  # 1 for [Etts]
            if end > begin:
                tts_input_ids = self.tts_processor.text_tokenizer(
                    tts_text, return_tensors="pt", add_special_tokens=False
                )["input_ids"].to(self.device)
                text_input_ids = tts_input_ids[:, begin:end]
                streaming_tts_text_mask = self._build_streaming_mask(tts_token_lens).to(device=self.tts.device)
                position_ids = torch.arange(begin, end, dtype=torch.long, device=self.tts.device).unsqueeze(0)

                past_key_values = self.tts.prefill_text(
                    input_ids=text_input_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    lm_spk_emb_last_hidden_states=spk_embeds if chunk_idx == 0 else None,
                )

            while True:
                # temp = [0.1, 0.3, 0.1, 0.3] if chunk_idx < 21 else [0.1] * self.tts.num_vq
                outputs = self.tts.generate(
                    input_ids=audio_input_ids,
                    past_key_values=past_key_values,
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    max_new_token=output_chunk_size,
                    force_no_stop=self.force_no_stop,
                    # temperature=torch.tensor([0.1] * self.tts.num_vq, dtype=torch.float, device=self.tts.device),
                    temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                    eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                    logits_warpers=logits_warpers,
                    logits_processors=logits_processors,
                )
                audio_input_ids = outputs.audio_input_ids
                past_key_values = outputs.past_key_values
                chunk_idx += 1

                mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, max(new_ids_len - 4, 0) :, :])
                new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]

                wav_np, sr = self.decode_mel_to_audio(mel_spec)

                if enable_regenerate:
                    if prev_wav is not None:
                        check_wav_np = wav_np[2048:].cpu().numpy()  # 2*4*256(hop)
                        check_mel = mel_spec[0, :, 8:].cpu().numpy()  # 2*4
                    else:
                        check_wav_np = wav_np.cpu().numpy()
                        check_mel = mel_spec[0].cpu().numpy()
                if enable_regenerate and voice_checker.is_bad(check_wav_np, check_mel, chunk_size=2560):
                    voice_checker.reset()
                    # regenerate
                    N = output_chunk_size if prev_wav is None else output_chunk_size * 2
                    past_kv = cache_from_pairs(
                        (keys[:, :, :-N, :], values[:, :, :-N, :]) for keys, values in cache_pairs(past_key_values)
                    )
                    outputs = self.tts.generate(
                        input_ids=audio_input_ids[:, :-N, :],
                        past_key_values=past_kv,
                        streaming_tts_text_mask=streaming_tts_text_mask,
                        max_new_token=N,
                        force_no_stop=self.force_no_stop,
                        temperature=torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.tts.device),
                        eos_token=torch.tensor([625], dtype=torch.long, device=self.tts.device),
                        logits_warpers=logits_warpers,
                        logits_processors=logits_processors,
                    )
                    audio_input_ids = outputs.audio_input_ids
                    past_key_values = outputs.past_key_values

                    new_ids_len -= N
                    mel_spec = self.tts.decode_to_mel_specs(outputs.new_ids[:, new_ids_len:, :])
                    new_ids_len = outputs.new_ids.shape[1]  # [1, seq_len, 4]
                    wav_np, sr = self.decode_mel_to_audio(mel_spec)

                    if prev_wav is not None:
                        wav_y = wav_np[: len(prev_wav)]
                        prev_wav = wav_np[len(prev_wav) :]
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_y, sampling_rate=sr)
                    else:
                        prev_wav = wav_np
                else:
                    # smooth wav
                    if prev_wav is not None:
                        wav_np, prev_wav = self._linear_overlap_add2_wav(
                            [prev_wav, wav_np], overlap=512 * 4
                        )  # tts_hop256*2
                        cur_text = gen_text_raw[prev_text_len:]
                        prev_text_len = len(gen_text_raw)
                        yield OmniOutput(text=cur_text, audio_wav=wav_np, sampling_rate=sr)
                    else:
                        prev_wav = wav_np

                if outputs.finished:
                    logger.debug("Generation finished.")
                    break
                if outputs.new_ids.shape[1] > 2048:
                    stop = True
                    logger.debug("Generation length > 2048, stopped.")
                    break

        if prev_wav is not None:
            cur_text = gen_text_raw[prev_text_len:]
            yield OmniOutput(
                text=cur_text, audio_wav=prev_wav, sampling_rate=sr
            )  # yield last chunk wav without smooth

        if new_segment_gen and not stop:
            logger.debug(
                f"tts_text tokens {tts_token_lens} exceed {self.tts.streaming_text_reserved_len - shift_len}, start a new segment generation"
            )
            tid_len = 5  # self.tts.streaming_text_chunk_size
            prev_seg_text_ids = tts_input_ids[:, end - 1 - tid_len : end - 1]  # exclude last Etts
            aid_len = 50  # int(tid_len * new_ids_len / tts_token_lens)
            prev_seg_audio_ids = outputs.new_ids[:, -aid_len:, :]

            result = self._generate_mel_spec_audio_streaming(
                spk_bounds,
                streamer,
                output_chunk_size,
                spk_embeds,
                prev_seg_text_ids,
                text_left,
                prev_seg_audio_ids,
                enable_regenerate=enable_regenerate,
            )
            for res in result:
                yield res

    def decode_mel_to_audio(self, mel_spec, output_path=""):
        import soundfile as sf

        with torch.inference_mode():
            wav_numpy = self.vocos.decode(mel_spec.float()).cpu().squeeze()
            sr = 24000
        if output_path:
            sf.write(output_path, wav_numpy.numpy(), samplerate=sr)
            logger.info(f"Audio saved to {output_path}")
        return wav_numpy, sr


# Copied from transformers.models.whisper.modeling_whisper.WhisperEncoderLayer and add use_cache for streaming inference
