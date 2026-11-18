# SPDX-FileCopyrightText: 2024 AIDC-AI
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""NaViT/soft visual vocabulary adapted from AIDC-AI Ovis2.6 (see NOTICE)."""

import math
from typing import List, Optional

import numpy as np
import PIL.Image
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import SiglipImageProcessor
from transformers import initialization as init
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
from transformers.models.siglip2.modeling_siglip2 import Siglip2MLP, eager_attention_forward

from .configuration_ovis2_6 import Siglip2NavitConfig


IMAGE_PLACEHOLDER = "<image>"
IMAGE_PLACEHOLDER_ID = -200
VIDEO_PLACEHOLDER = "<video>"
VIDEO_PLACEHOLDER_ID = -201
VISUAL_ATOM_ID = -300
INDICATOR_IDS = [-301, -302, -303, -304]


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

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.
        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to
        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, grid_thws: Optional[torch.LongTensor] = None) -> torch.Tensor:
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
            pixel_values = pixel_values.view(
                -1, self.config.num_channels * self.config.temporal_patch_size, self.patch_size, self.patch_size
            )
            patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
            patch_embeds = patch_embeds.reshape(-1, self.embed_dim)

        if self.preserve_original_pe:
            assert grid_thws is not None
            pos_embed_new = torch.zeros_like(patch_embeds)
            positional_embeddings = (
                self.position_embedding.weight.reshape(self.position_embedding_size, self.position_embedding_size, -1)
                .unsqueeze(0)
                .permute(0, 3, 1, 2)
            )
            # pos_embed = self.pos_embed.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2)
            cnt = 0
            for t, h, w in grid_thws:
                thw = t * h * w
                pe = F.interpolate(positional_embeddings, size=(h, w), mode="bicubic", align_corners=False)
                pe = pe.permute(0, 2, 3, 1).reshape(1, h * w, -1)
                pe = pe[0].repeat(t, 1)
                pe = pe.reshape(
                    t, h // self.hidden_stride, self.hidden_stride, w // self.hidden_stride, self.hidden_stride, -1
                )
                pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(thw, -1)
                pos_embed_new[cnt : cnt + thw] = pe
                cnt += thw
            patch_embeds = patch_embeds + pos_embed_new

        return patch_embeds


class Siglip2Attention(nn.Module):
    """SigLIP attention over packed NaViT sequences (one call per ``cu_seqlens`` window)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.use_rope = config.use_rope

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        shape = (seq_length, self.num_heads, self.head_dim)
        queries = self.q_proj(hidden_states).view(shape)
        keys = self.k_proj(hidden_states).view(shape)
        values = self.v_proj(hidden_states).view(shape)
        if self.use_rope:
            cos, sin = position_embeddings
            queries, keys = apply_rotary_pos_emb_vision(queries, keys, cos, sin)
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outputs = [
            attention_interface(
                self,
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                v.transpose(0, 1)[None],
                None,
                is_causal=False,
                scaling=self.scale,
                dropout=0.0 if not self.training else self.dropout,
                **kwargs,
            )[0].reshape(-1, self.embed_dim)
            for q, k, v in zip(*(x.split(lengths) for x in (queries, keys, values)))
        ]
        return self.out_proj(torch.cat(outputs))


class Siglip2EncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(self, hidden_states, cu_seqlens, position_embeddings, **kwargs):
        residual = hidden_states
        hidden_states = self.self_attn(self.layer_norm1(hidden_states), cu_seqlens, position_embeddings, **kwargs)
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class Siglip2Encoder(nn.Module):

    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        self.rotary_pos_emb = VisionRotaryEmbedding(config.hidden_size // config.num_attention_heads // 2)
        self.patch_size = config.patch_size
        self.hidden_stride = config.hidden_stride
        self.window_size = config.window_size
        self.spatial_merge_unit = config.hidden_stride * config.hidden_stride
        self.fullatt_block_indexes = (
            None
            if config.fullatt_block_indexes is None
            else [int(i) for i in config.fullatt_block_indexes.split("|")]
            if isinstance(config.fullatt_block_indexes, str)
            else list(config.fullatt_block_indexes)
        )

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
        vit_merger_window_size = (
            self.window_size // self.hidden_stride // self.patch_size
        )  # patch (after merge) number in each window

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

    def forward(self, inputs_embeds, grid_thws, output_hidden_states=False):

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
        encoder_states = () if output_hidden_states else None

        hidden_states = inputs_embeds
        for index, block in enumerate(self.layers):
            if self.fullatt_block_indexes is None or index in self.fullatt_block_indexes:
                cu_seqlens_tmp = cu_seqlens
            else:
                cu_seqlens_tmp = cu_window_seqlens
            hidden_states = block(hidden_states, cu_seqlens_tmp, position_embeddings)
            if output_hidden_states:
                hidden_states_ = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
                encoder_states += (hidden_states_[reverse_indices, :].reshape(seq_len, -1),)
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[reverse_indices, :].reshape(seq_len, -1)

        return hidden_states, encoder_states


class Siglip2VisionTransformer(nn.Module):
    def __init__(self, config: Siglip2NavitConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values, grid_thws, output_hidden_states=True, return_dict=True):

        hidden_states = self.embeddings(pixel_values, grid_thws)

        last_hidden_state, hidden_states = self.encoder(hidden_states, grid_thws, output_hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        if not return_dict:
            output = (last_hidden_state,)
            output += (hidden_states,) if output_hidden_states else ()
            return output

        return BaseModelOutputWithNoAttention(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


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
            frequencies = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float32) / module.dim))
            init.copy_(module.inv_freq, frequencies)


class Siglip2NavitModel(Siglip2PreTrainedModel):
    config_class = Siglip2NavitConfig
    main_input_name = "pixel_values"

    def __init__(self, config: Siglip2NavitConfig):
        super().__init__(config)

        self.vision_model = Siglip2VisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(self, pixel_values, grid_thws, output_hidden_states=None, return_dict=None):
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.return_dict

        return self.vision_model(
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


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
        self.image_processor = SiglipImageProcessor()
        head_dim = visual_vocab_size - len(INDICATOR_IDS)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.vit.config.hidden_size * self.vit.config.hidden_stride**2, head_dim, bias=False),
            torch.nn.LayerNorm(head_dim),
        )

    def _encode(self, pixel_values, grid_thws):
        output = self.vit(pixel_values, grid_thws, output_hidden_states=True, return_dict=True)
        features = output.hidden_states[-1]
        seq_len, _ = features.shape
        features = features.reshape(seq_len // (self.vit.config.hidden_stride**2), -1)
        return features

    # Adapted from qwen2_vl
    @staticmethod
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

    def preprocess(
        self,
        image: Optional[PIL.Image.Image] = None,
        video: Optional[List[PIL.Image.Image]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        patch_size = self.vit.config.patch_size
        temporal_patch_size = self.vit.config.temporal_patch_size
        hidden_stride = self.vit.config.hidden_stride
        assert (image is None) ^ (video is None), "Invalid input: expect either image or video"
        if image is not None:
            images = [image]
        else:
            images = video
        images = [image.convert("RGB") if image.mode != "RGB" else image for image in images]
        width, height = images[0].size
        processed_images = []
        for image in images:
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=patch_size * hidden_stride,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            new_size = {"height": resized_height, "width": resized_width}
            new_image = self.image_processor.preprocess(image, size=new_size, return_tensors="np")["pixel_values"][0]
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
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // hidden_stride,
            hidden_stride,
            patch_size,
            grid_w // hidden_stride,
            hidden_stride,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )
        flatten_patches = torch.tensor(flatten_patches)

        return flatten_patches, grid_thw

    def forward(
        self, pixel_values, grid_thws
    ) -> torch.Tensor:  # [BatchSize, ImageShape] -> [BatchSize, #Token, VocabSize]
        features = self._encode(pixel_values, grid_thws)
        logits = self.head(features)
        tokens = torch.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

        token_len, _ = tokens.shape
        padding_tensor = torch.zeros(
            size=(token_len, len(INDICATOR_IDS)),
            dtype=tokens.dtype,
            device=tokens.device,
            layout=tokens.layout,
            requires_grad=False,
        )
        tokens = torch.cat((tokens, padding_tensor), dim=1)
        return tokens
