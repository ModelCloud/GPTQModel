# Copyright (C) 2025 AIDC-AI
# SPDX-License-Identifier: Apache-2.0
# Derived from pinned Ovis2.5 sources; attribution and changes: NOTICE.
"""Ovis2.5 SigLIP2 NaViT: merged patch order, 2D RoPE and soft visual tokens."""

import math
from typing import List, Optional, Tuple

import numpy as np
import PIL.Image
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import can_return_tuple

from ..native.siglip.image_processing_pil_siglip import SiglipImageProcessorPil
from .configuration_ovis2_5 import Siglip2NavitConfig


IMAGE_PLACEHOLDER = "<image>"
IMAGE_PLACEHOLDER_ID = -200
VIDEO_PLACEHOLDER = "<video>"
VIDEO_PLACEHOLDER_ID = -201

VISUAL_ATOM_ID = -300
INDICATOR_IDS = [-301, -302, -303, -304]


# copied from qwen2.5-vl
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Siglip2VisionEmbeddings(nn.Module):
    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = config.num_patches
        self.preserve_original_pe = config.preserve_original_pe
        self.hidden_stride = config.hidden_stride


        # siglip2 naflex
        if self.num_patches > 0:
            self.patch_embedding = nn.Linear(
                    in_features=config.num_channels * self.patch_size * self.patch_size,
                    out_features=self.embed_dim,
                )
            if self.preserve_original_pe:
                self.position_embedding_size = int(self.num_patches**0.5)
                self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

        else:
            self.patch_embedding = nn.Conv2d(
                    in_channels=config.num_channels,
                    out_channels=self.embed_dim,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                    padding="valid",
                )
            if self.preserve_original_pe:
                self.num_patches = (self.image_size // self.patch_size) ** 2
                self.position_embedding_size = self.image_size // self.patch_size
                self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(self, pixel_values: torch.FloatTensor,
                grid_thws: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (num_patches, num_channels * temporal_patch_size * patch_size * patch_size)
            grid_thws: (`torch.LongTensor`):
                grid shape (num_patches, 3)
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        if isinstance(self.patch_embedding, nn.Linear):
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        elif isinstance(self.patch_embedding, nn.Conv2d):
            pixel_values = pixel_values.view(-1, self.config.num_channels * self.config.temporal_patch_size, self.patch_size,
                   self.patch_size)
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            patch_embeds = patch_embeds.reshape(-1, self.embed_dim)


        if self.preserve_original_pe:
            assert grid_thws is not None
            pos_embed_new = torch.zeros_like(patch_embeds)
            positional_embeddings = self.position_embedding.weight.reshape(
                                self.position_embedding_size, self.position_embedding_size, -1
                            ).unsqueeze(0).permute(0,3,1,2)
            # pos_embed = self.pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2)
            cnt = 0
            for t, h, w in grid_thws:
                thw = t * h * w
                pe = F.interpolate(positional_embeddings, size=(h, w), mode='bicubic', align_corners=False)
                pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
                pe = pe[0].repeat(t, 1)
                pe = pe.reshape(t, h // self.hidden_stride, self.hidden_stride, w // self.hidden_stride,
                                self.hidden_stride, -1)
                pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(thw, -1)
                pos_embed_new[cnt:cnt + thw] = pe
                cnt += thw
            patch_embeds = patch_embeds + pos_embed_new

        return patch_embeds


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.use_rope = config.use_rope

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        seq_length, embed_dim = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(seq_length, self.num_heads, self.head_dim)
        keys = keys.view(seq_length, self.num_heads, self.head_dim)
        values = values.view(seq_length, self.num_heads, self.head_dim)

        if self.use_rope:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb_vision(queries.unsqueeze(0), keys.unsqueeze(0), cos, sin)
            queries = queries.squeeze(0)
            keys = keys.squeeze(0)

        outputs = []
        cu = cu_seqlens.tolist()
        for start_idx, end_idx in zip(cu[:-1], cu[1:]):
            q_i, k_i, v_i = [x[start_idx:end_idx].transpose(0, 1).unsqueeze(0)
                             for x in (queries, keys, values)]
            if self.config._attn_implementation == "eager":
                scores = (q_i @ k_i.transpose(-1, -2)) * self.scale
                output_i = scores.float().softmax(-1).to(v_i.dtype) @ v_i
            else:
                output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
            outputs.append(output_i.transpose(1, 2).reshape(-1, self.embed_dim))
        attn_output = torch.cat(outputs, dim=0)

        attn_output = self.out_proj(attn_output)
        return attn_output

class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Siglip2Encoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`Siglip2EncoderLayer`].

    Args:
        config: Siglip2NavitConfig
    """

    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.rotary_pos_emb = VisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)
        self.patch_size = config.patch_size
        self.hidden_stride = config.hidden_stride
        self.window_size = config.window_size
        self.spatial_merge_unit = config.hidden_stride * config.hidden_stride
        self.fullatt_block_indexes = None if config.fullatt_block_indexes is None else ([int(i) for i in config.fullatt_block_indexes.split('|')] if isinstance(config.fullatt_block_indexes, str) else list(config.fullatt_block_indexes))


    # copied from qwen2.5_vl
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.hidden_stride,
                self.hidden_stride,
                w // self.hidden_stride,
                self.hidden_stride,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.hidden_stride // self.patch_size  # patch (after merge) number in each window

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.hidden_stride,  # number of patch after merge
                grid_w // self.hidden_stride,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(
        self,
        inputs_embeds,
        grid_thws: torch.Tensor,
    ) -> torch.Tensor:
        rotary_pos_emb = self.rot_pos_emb(grid_thws)
        window_index, cu_window_seqlens = self.get_window_index(grid_thws)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=inputs_embeds.device,
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = inputs_embeds.size()
        inputs_embeds = inputs_embeds.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        inputs_embeds = inputs_embeds[window_index, :, :]
        inputs_embeds = inputs_embeds.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thws[:, 1] * grid_thws[:, 2], grid_thws[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thws.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        reverse_indices = torch.argsort(window_index)

        hidden_states = inputs_embeds
        for index, block in enumerate(self.layers):
            if self.fullatt_block_indexes is None or index in self.fullatt_block_indexes:
                cu_seqlens_tmp = cu_seqlens
            else:
                cu_seqlens_tmp = cu_window_seqlens
            hidden_states = block(hidden_states, cu_seqlens_tmp, position_embeddings)
        # tokens = self.post_trunk_norm(tokens)
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :].reshape(seq_len, -1)

        return hidden_states

class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor,
    ) -> BaseModelOutputWithNoAttention:
        # ``hidden_states`` holds only the final, window-unpermuted encoder state before
        # ``post_layernorm``; that is the feature the Ovis2.5 visual head consumes.
        encoder_state = self.encoder(self.embeddings(pixel_values, grid_thws), grid_thws)
        return BaseModelOutputWithNoAttention(
            last_hidden_state=self.post_layernorm(encoder_state),
            hidden_states=(encoder_state,),
        )

class Siglip2PreTrainedModel(PreTrainedModel):
    config_class = Siglip2NavitConfig
    base_model_prefix = "siglip2_navit"
    supports_gradient_checkpointing = True

    _no_split_modules = [
        "Siglip2VisionEmbeddings",
        "Siglip2EncoderLayer",
    ]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_flex_attn = False
    _supports_attention_backend = False

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, VisionRotaryEmbedding):
            init.copy_(module.inv_freq, 1.0 / (module.theta ** (
                torch.arange(0, module.dim, 2, dtype=torch.float32, device=module.inv_freq.device) / module.dim)))


class Siglip2NavitModel(Siglip2PreTrainedModel):
    config_class = Siglip2NavitConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2NavitConfig):
        super().__init__(config)

        self.vision_model = Siglip2VisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @can_return_tuple
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        grid_thws: torch.LongTensor,
        **kwargs,
    ) -> BaseModelOutputWithNoAttention:
        return self.vision_model(pixel_values=pixel_values, grid_thws=grid_thws)

class VisualEmbedding(torch.nn.Embedding):
    """
    A visual embedding layer that can handle both discrete token IDs (long) and continuous
    soft-token probabilities (float).
    """

    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        # Handle soft tokens (probabilities) by matrix multiplication with the embedding weight
        return torch.matmul(visual_tokens, self.weight)


class VisualTokenizer(torch.nn.Module):
    """
    Tokenizes images or videos into a sequence of continuous visual tokens.
    """

    def __init__(self, vit, visual_vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit = vit
        self.image_processor = SiglipImageProcessorPil(do_center_crop=False)
        head_dim = visual_vocab_size - len(INDICATOR_IDS)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.vit.config.hidden_size * self.vit.config.hidden_stride ** 2, head_dim, bias=False),
            torch.nn.LayerNorm(head_dim)
        )

    def _encode(self, pixel_values, grid_thws):
        output = self.vit(pixel_values, grid_thws)
        features = output.hidden_states[-1]
        seq_len, _ = features.shape
        features = features.reshape(seq_len // (self.vit.config.hidden_stride ** 2), -1)
        return features

    def preprocess(self, image=None, video=None, min_pixels=448 * 448, max_pixels=1344 * 1792):
        return preprocess_media(self.image_processor, self.vit.config, image, video, min_pixels, max_pixels)

    def forward(
        self, pixel_values, grid_thws
    ) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self._encode(pixel_values, grid_thws)
        logits = self.head(features)
        tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

        token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(token_len, len(INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=1)
        return tokens




def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 448 * 448, max_pixels: int = 1344 * 1792
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions are divisible by 'factor'.
    2. The total number of pixels is within ['min_pixels', 'max_pixels'].
    3. The aspect ratio is maintained as closely as possible.
    """
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor

    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_media(
    image_processor, vision_config,
    image: Optional[PIL.Image.Image] = None,
    video: Optional[List[PIL.Image.Image]] = None,
    min_pixels: Optional[int] = 448 * 448,
    max_pixels: Optional[int] = 1344 * 1792
):
    patch_size = vision_config.patch_size
    temporal_patch_size = vision_config.temporal_patch_size
    hidden_stride = vision_config.hidden_stride
    assert (image is None) ^ (video is None), "Invalid input: expect either image or video"
    if image is not None:
        images = [image]
    else:
        images = video
    images = [image.convert("RGB") if image.mode != 'RGB' else image for image in images]
    width, height = images[0].size
    processed_images = []
    for image in images:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * hidden_stride,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        new_size = {"height": resized_height, "width": resized_width}
        new_image = image_processor.preprocess(image, size=new_size, return_tensors="np")['pixel_values'][0]
        processed_images.append(new_image)

    patches = np.array(processed_images)
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = np.repeat(patches[-1][np.newaxis], temporal_patch_size - 1, axis=0)
        patches = np.concatenate([patches, repeats], axis=0)
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]])

    patches = patches.reshape(
        grid_t, temporal_patch_size, channel,
        grid_h // hidden_stride, hidden_stride, patch_size,
        grid_w // hidden_stride, hidden_stride, patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    flatten_patches = torch.tensor(flatten_patches)

    return flatten_patches, grid_thw
