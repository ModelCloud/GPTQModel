# Copyright 2026 The HuggingFace Team. All rights reserved.
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
# GPT-QModel adaptation: package-owned loading and Transformers 5 execution.
# Pinned source and modifications: NOTICE.

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import PreTrainedAudioTokenizerBase
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, torch_compilable_check
from transformers.utils.import_utils import is_tracing

from .configuration_liquid_codec import LiquidCodecConfig
from .modeling_lfm3 import Lfm3Model


@auto_docstring
@dataclass
class LiquidCodecEncoderOutput(ModelOutput):
    r"""
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
        The code of every codebook of every frame.
    """

    audio_codes: torch.LongTensor = None


@auto_docstring
@dataclass
class LiquidCodecDecoderOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        The decoded 24 kHz audio in float32.
    """

    audio_values: torch.FloatTensor = None


@auto_docstring
@dataclass
class LiquidCodecOutput(ModelOutput):
    r"""
    audio_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
        The reconstructed audio, cut to the input length.
    audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
        The code of every codebook of every frame.
    """

    audio_values: torch.FloatTensor = None
    audio_codes: torch.LongTensor = None


class LiquidCodecStreamingCache:
    """Carries the decoder state between chunks: the LFM3 cache and the unfinished overlap-add tail."""

    def __init__(self, config: LiquidCodecConfig):
        self.past_key_values = DynamicCache(config=config.lfm_config(config.decoder_window))
        self.audio: torch.Tensor | None = None
        self.envelope: torch.Tensor | None = None
        self.started = False
        self.finished = False


class LiquidCodecSTFT(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.n_fft = config.stft_n_fft
        self.hop_length = config.stft_hop_length
        self.win_length = config.stft_win_length
        self.window = nn.Buffer(torch.hann_window(self.win_length, dtype=torch.float32), persistent=False)
        self.num_features = 3 * (self.n_fft // 2 + 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        total_pad = self.win_length - self.hop_length
        audio = F.pad(audio.float(), (total_pad // 2, total_pad - total_pad // 2), mode="reflect")
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.float(),
            center=False,
            return_complex=True,
        )
        magnitude = spec.abs()
        phase_denominator = magnitude.clamp_min(1e-5)
        return torch.cat((torch.log1p(magnitude), spec.real / phase_denominator, spec.imag / phase_denominator), dim=1)


class LiquidCodecFrontEnd(nn.Module):
    def __init__(self, config: LiquidCodecConfig, num_features: int):
        super().__init__()
        self.frame_stack = config.stft_frame_stack
        self.in_proj = nn.Conv1d(num_features, config.encoder_conv_dim, kernel_size=1)
        self.conv = nn.Conv1d(config.encoder_conv_dim, config.encoder_conv_dim, kernel_size=3, padding=2)
        self.out = nn.Conv1d(
            config.encoder_conv_dim, config.encoder_dim, kernel_size=self.frame_stack, stride=self.frame_stack
        )
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(self.in_proj(features.to(self.in_proj.weight.dtype)))
        hidden_states = self.act(self.conv(hidden_states)[:, :, : hidden_states.shape[-1]])
        return self.out(hidden_states)


class LiquidCodecLfm(nn.Module):
    def __init__(self, config: LiquidCodecConfig, sliding_window: int | None = None):
        super().__init__()
        self.lfm = Lfm3Model(config.lfm_config(sliding_window))
        self.lfm.register_module("embed_tokens", None)
        # the projections run in the model dtype but the rotary tables stay in float32; a named hook keeps the
        # module picklable
        self.lfm.rotary_emb.register_forward_pre_hook(self._rotary_in_float32)

    @staticmethod
    def _rotary_in_float32(module, args):
        return (args[0].float(), *args[1:])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lfm(inputs_embeds=hidden_states, use_cache=False).last_hidden_state


class LiquidCodecQuantizer(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        levels = list(config.fsq_levels)
        self.num_codebooks = config.n_q
        self.codebook_dim = len(levels)
        self.levels = nn.Buffer(torch.tensor(levels, dtype=torch.int32), persistent=False)
        self.basis = nn.Buffer(
            torch.tensor([math.prod(levels[:index]) for index in range(self.codebook_dim)], dtype=torch.int32),
            persistent=False,
        )
        effective_dim = self.codebook_dim * self.num_codebooks
        if config.quantizer_dim == effective_dim:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
        else:
            self.project_in = nn.Linear(config.quantizer_dim, effective_dim)
            self.project_out = nn.Linear(effective_dim, config.quantizer_dim)

    def _bound(self, latents: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half_levels = (self.levels - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_levels)
        return torch.tanh(latents + shift) * half_levels - offset

    def encode(self, latents: torch.Tensor) -> torch.Tensor:
        # the quantizer runs outside `forward`, so no `device_map` hook moves its inputs (here and in `decode`) to the
        # device of its buffers
        latents = self.project_in(latents.movedim(1, -1).contiguous()).float().to(self.levels.device)
        if not is_tracing(latents):
            # finite audio far above full scale overflows the float32 STFT, and NaN latents round to INT64_MIN codes
            torch_compilable_check(
                torch.isfinite(latents),
                "The audio overflows the codec (non-finite latents); expected samples in [-1, 1]",
            )
        latents = latents.reshape(*latents.shape[:-1], self.num_codebooks, self.codebook_dim)
        half_width = self.levels // 2
        codes = self._bound(latents).round() / half_width
        indices = ((codes * half_width + half_width) * self.basis).sum(-1).round().long()
        return indices.transpose(1, 2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        indices = codes.to(self.basis.device).transpose(1, 2).unsqueeze(-1)
        half_width = self.levels // 2
        latents = ((indices // self.basis) % self.levels - half_width) / half_width
        latents = latents.reshape(*latents.shape[:-2], -1)
        if isinstance(self.project_out, nn.Linear):
            latents = latents.to(self.project_out.weight.dtype)
        return self.project_out(latents).movedim(-1, 1)


class LiquidCodecISTFT(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.n_fft = config.stft_n_fft
        self.hop_length = config.stft_hop_length
        self.win_length = config.stft_win_length
        self.window = nn.Buffer(torch.hann_window(self.win_length, dtype=torch.float32))

    def forward(self, spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_frames = spec.shape[-1]
        frames = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward") * self.window[None, :, None]
        output_size = (num_frames - 1) * self.hop_length + self.win_length
        fold = {"output_size": (1, output_size), "kernel_size": (1, self.win_length), "stride": (1, self.hop_length)}
        audio = F.fold(frames, **fold)[:, 0, 0]
        envelope = F.fold(self.window.square().expand(1, num_frames, -1).transpose(1, 2), **fold)[0, 0, 0]
        return audio, envelope


class LiquidCodecDecoder(nn.Module):
    def __init__(self, config: LiquidCodecConfig):
        super().__init__()
        self.frame_stack = config.stft_frame_stack
        self.lfm = LiquidCodecLfm(config, sliding_window=config.decoder_window).lfm
        self.lin = nn.Linear(config.encoder_dim, 2 * (config.stft_n_fft // 2 + 1))
        self.istft = LiquidCodecISTFT(config)

    def forward(
        self, embeddings: torch.Tensor, past_key_values: DynamicCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.lfm(
            inputs_embeds=F.interpolate(
                embeddings.mT, embeddings.shape[1] * self.frame_stack, mode="nearest-exact"
            ).mT,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        ).last_hidden_state
        log_magnitude, angle = torch.chunk(self.lin(hidden_states).mT.contiguous(), 2, 1)
        return self.istft(torch.polar(log_magnitude.float().exp(), angle.float()))


@auto_docstring(
    custom_intro="The Liquid Codec neural audio codec: 24 kHz audio to 12 codebooks at 12.5 frames per second and back."
)
class LiquidCodecModel(PreTrainedAudioTokenizerBase):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Bundled legacy codec keys are renamed only for this load, without a global registry mutation.
        kwargs.setdefault('key_mapping', {r'^_frontend\.': 'frontend.', r'^quantizer\.fsq\.': 'quantizer.'})
        kwargs.pop('trust_remote_code', None)
        kwargs.pop('code_revision', None)
        return super().from_pretrained(*args, **kwargs)

    config: LiquidCodecConfig
    base_model_prefix = "codec"
    main_input_name = "input_values"
    # the analysis window is a non-persistent buffer cast in the forward; only the synthesis window is a weight
    _keep_in_fp32_modules_strict = ["istft"]
    _supports_flash_attn = False
    _supports_sdpa = True

    def __init__(self, config: LiquidCodecConfig):
        super().__init__(config)
        self.stft = LiquidCodecSTFT(config)
        self.frontend = LiquidCodecFrontEnd(config, self.stft.num_features)
        self.encoder = LiquidCodecLfm(config)
        self.to_quantizer = nn.Linear(config.encoder_dim, config.quantizer_dim)
        self.quantizer = LiquidCodecQuantizer(config)
        self.decoder = LiquidCodecDecoder(config)
        self.from_quantizer = nn.Linear(config.quantizer_dim, config.encoder_dim)
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, (LiquidCodecSTFT, LiquidCodecISTFT)):
            init.copy_(module.window, torch.hann_window(module.win_length, dtype=torch.float32))
        elif isinstance(module, LiquidCodecQuantizer):
            levels = list(self.config.fsq_levels)
            init.copy_(module.levels, torch.tensor(levels, dtype=torch.int32))
            init.copy_(
                module.basis,
                torch.tensor([math.prod(levels[:index]) for index in range(len(levels))], dtype=torch.int32),
            )

    @can_return_tuple
    def encode(self, input_values: torch.FloatTensor, *, return_dict: bool | None = None) -> LiquidCodecEncoderOutput:
        r"""
        Non-finite audio, or audio that overflows the codec, is refused eagerly; a compiled `encode` skips the check.

        input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            24 kHz audio. The tail is zero-padded to a whole codec frame of 1920 samples.
        return_dict (`bool`, *optional*):
            Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if input_values.ndim != 2 or input_values.shape[-1] == 0:
            raise ValueError(
                f"Expected non-empty audio of shape (batch_size, num_samples), got {tuple(input_values.shape)}"
            )
        if not is_tracing(input_values):
            # a non-finite sample would round to an INT64_MIN code that only fails far downstream; the tensor-valued
            # check would break a `fullgraph` compile or a CUDA-graph capture
            torch_compilable_check(torch.isfinite(input_values), "`input_values` must be finite")
        frame_size = self.config.frame_size
        num_samples = -(input_values.shape[-1] // -frame_size) * frame_size
        input_values = F.pad(input_values, (0, num_samples - input_values.shape[-1]))
        hidden_states = self.frontend(self.stft(input_values)).mT
        latents = self.to_quantizer(self.encoder(hidden_states)).mT
        return LiquidCodecEncoderOutput(audio_codes=self.quantizer.encode(latents))

    @can_return_tuple
    def decode(
        self,
        audio_codes: torch.LongTensor,
        streaming_cache: LiquidCodecStreamingCache | None = None,
        finalize: bool = True,
        *,
        return_dict: bool | None = None,
    ) -> LiquidCodecDecoderOutput:
        r"""
        audio_codes (`torch.LongTensor` of shape `(batch_size, num_codebooks, num_frames)`):
            The codes to decode; every frame becomes 1920 samples.
        streaming_cache ([`LiquidCodecStreamingCache`], *optional*):
            The state carried between consecutive chunks of one stream. The chunks decode to the same audio as the
            concatenated codes.
        finalize (`bool`, *optional*, defaults to `True`):
            Whether this is the last chunk of the stream. Until then the trailing samples that later frames still
            contribute to are held back in the cache.
        return_dict (`bool`, *optional*):
            Whether to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        if audio_codes.ndim != 3 or audio_codes.shape[1] != self.config.n_q:
            raise ValueError(
                f"Expected codes of shape (batch_size, {self.config.n_q}, num_frames), got {tuple(audio_codes.shape)}"
            )
        torch_compilable_check(
            (audio_codes >= 0) & (audio_codes < self.config.codebook_size),
            f"Codes must be in [0, {self.config.codebook_size})",
        )
        if streaming_cache is not None and streaming_cache.finished:
            raise ValueError("This stream is finalized; start with a new cache")
        trim = (self.config.stft_win_length - self.config.stft_hop_length) // 2
        tail = 2 * trim
        batch_size = audio_codes.shape[0]
        if not audio_codes.shape[-1] and not (finalize and streaming_cache is not None and streaming_cache.started):
            # no frames and no held-back tail to flush: the overlap-add state must stay untouched
            if streaming_cache is not None and finalize:
                streaming_cache.finished = True
            return LiquidCodecDecoderOutput(audio_values=audio_codes.new_zeros((batch_size, 0), dtype=torch.float32))
        if audio_codes.shape[-1]:
            # `quantizer.decode` returns float32 latents when its projections are `nn.Identity`
            latents = self.quantizer.decode(audio_codes).mT.to(self.from_quantizer.weight.dtype)
            embeddings = self.from_quantizer(latents)
            past_key_values = None if streaming_cache is None else streaming_cache.past_key_values
            audio, envelope = self.decoder(embeddings, past_key_values)
        else:
            audio = self.decoder.istft.window.new_zeros(batch_size, tail)
            envelope = self.decoder.istft.window.new_zeros(tail)
        if streaming_cache is None:
            audio, envelope = audio[:, trim:-trim], envelope[trim:-trim]
        else:
            if streaming_cache.audio is not None:
                audio = torch.cat([audio[:, :tail] + streaming_cache.audio, audio[:, tail:]], dim=1)
                envelope = torch.cat([envelope[:tail] + streaming_cache.envelope, envelope[tail:]])
            if not streaming_cache.started:
                audio, envelope = audio[:, trim:], envelope[trim:]
                streaming_cache.started = True
            if finalize:
                audio, envelope = audio[:, :-trim], envelope[:-trim]
                streaming_cache.audio, streaming_cache.envelope = None, None
                streaming_cache.finished = True
            else:
                streaming_cache.audio, streaming_cache.envelope = audio[:, -tail:], envelope[-tail:]
                audio, envelope = audio[:, :-tail], envelope[:-tail]
        return LiquidCodecDecoderOutput(audio_values=(audio / envelope).float())

    @can_return_tuple
    @auto_docstring
    def forward(self, input_values: torch.FloatTensor) -> LiquidCodecOutput:
        r"""
        input_values (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            24 kHz audio.
        """
        audio_codes = self.encode(input_values, return_dict=True).audio_codes
        audio = self.decode(audio_codes, return_dict=True).audio_values
        return LiquidCodecOutput(audio_values=audio[..., : input_values.shape[-1]], audio_codes=audio_codes)


__all__ = ["LiquidCodecModel", "LiquidCodecStreamingCache"]
