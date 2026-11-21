# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Released ZDTaichu5 wrapper with native Transformers 5 Qwen3.5 decoding."""

import itertools
import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.utils import logging

from ..native.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from .configuration import ZDTaichu5_0_Config


logger = logging.get_logger(__name__)
_CACHE_DELTA = "_zdtaichu5_rope_deltas"


class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class ZDTaichu5_0_ForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = ZDTaichu5_0_Config
    main_input_name = "input_ids"
    _supports_sdpa = True
    _is_stateful = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"^mtp\.", r"vision_model\.radio_model\.input_conditioner\."]

    def __init__(self, config):
        super().__init__(config)
        from .cradio_model import RADIOModel

        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.patch_size = config.vision_config.patch_size
        self.tile_h = self.tile_w = int(config.force_image_size // self.patch_size * self.downsample_ratio)
        self.num_image_token = self.tile_h * self.tile_w
        self.img_context_token_id = config.img_context_token_id
        self.video_context_token_id = config.video_context_token_id
        self.language_model = Qwen3_5ForCausalLM(config.llm_config)
        self.vision_model = RADIOModel(config.vision_config)
        self.vision_model.radio_model.external_preprocessing = True
        if config.vision_config.hidden_size != config.vit_hidden_size:
            raise ValueError("vit_hidden_size must match the C-RADIO feature width")
        width = config.vit_hidden_size * round(1 / self.downsample_ratio) ** 2
        self.mlp1 = nn.Sequential(
            RMSNorm(width), nn.Linear(width, config.projector_hidden_size, bias=False),
            SquaredReLU(), nn.Linear(config.projector_hidden_size, config.llm_config.hidden_size, bias=False),
        )
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.llm_config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language_model.set_output_embeddings(value)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.shape
        x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, int(h * scale_factor), int(w * scale_factor), int(c / scale_factor**2))
        return x if self.ps_version == "v1" else x.permute(0, 2, 1, 3).contiguous()

    def extract_feature(self, pixel_values):
        features = self.vision_model(pixel_values).features
        side = math.isqrt(features.shape[1])
        if side**2 != features.shape[1]:
            raise ValueError("ZDTaichu5 visual tiles must have a square patch grid")
        features = self.pixel_shuffle(features.reshape(features.shape[0], side, side, -1), self.downsample_ratio)
        features = features.flatten(1, 2).to(self.mlp1[1].weight.dtype)
        return self.mlp1(features)

    def _positions(self, input_ids, inputs_embeds, attention_mask, past_key_values,
                   mm_token_type_ids, image_grid_thw, video_grid_thw, rope_deltas):
        batch, length = inputs_embeds.shape[:2]
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        if attention_mask is None:
            text = torch.arange(past, past + length, device=inputs_embeds.device).expand(batch, -1)
        else:
            text = (attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)[:, -length:]
        if past == 0 and input_ids is not None and mm_token_type_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            vision, rope_deltas = self.get_rope_index(
                input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask,
            )
            return torch.cat([text.unsqueeze(0), vision]), rope_deltas
        if past and rope_deltas is None:
            rope_deltas = getattr(past_key_values, _CACHE_DELTA, None)
        if rope_deltas is not None and past:
            vision = text.unsqueeze(0) + rope_deltas.view(1, batch, 1)
            return torch.cat([text.unsqueeze(0), vision.expand(3, -1, -1)]), rope_deltas
        return torch.cat([text.unsqueeze(0), text.unsqueeze(0).expand(3, -1, -1)]), rope_deltas

    def forward(
        self, input_ids=None, pixel_values=None, pixel_values_videos=None,
        num_patches=None, image_flags=None, image_grid_thw=None, video_grid_thw=None,
        mm_token_type_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
        labels=None, inputs_embeds=None, use_cache=None, rope_deltas=None, **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.clone()
        for pixels, token_id, flags in (
            (pixel_values, self.img_context_token_id, image_flags),
            (pixel_values_videos, self.video_context_token_id, None),
        ):
            if pixels is None:
                continue
            if input_ids is None:
                raise ValueError("input_ids are required to place visual features")
            vision = self.extract_feature(pixels)
            if flags is not None:
                vision = vision[flags.reshape(-1).bool()]
            selected = input_ids == token_id
            if selected.sum().item() != vision.shape[0] * vision.shape[1]:
                raise ValueError("Visual feature count does not match prompt placeholders")
            inputs_embeds[selected] = vision.reshape(-1, vision.shape[-1]).to(inputs_embeds)
        use_cache = self.config.llm_config.use_cache if use_cache is None else use_cache
        if labels is not None:
            use_cache = False
        if use_cache and "full_attention" not in self.config.llm_config.layer_types:
            raise ValueError("Native Qwen3.5 cache requires at least one full-attention layer")
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config.llm_config)
        if position_ids is None:
            position_ids, rope_deltas = self._positions(
                input_ids, inputs_embeds, attention_mask, past_key_values,
                mm_token_type_ids, image_grid_thw, video_grid_thw, rope_deltas,
            )
        if use_cache and rope_deltas is not None:
            setattr(past_key_values, _CACHE_DELTA, rope_deltas)
        return self.language_model(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids.contiguous(), past_key_values=past_key_values,
            labels=labels, use_cache=use_cache, **kwargs,
        )

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        ids = model_kwargs.get("input_ids", inputs_tensor)
        embeds = torch.empty(*ids.shape[:2], 0, device=ids.device)
        positions, delta = self._positions(
            ids, embeds, model_kwargs.get("attention_mask"), model_kwargs.get("past_key_values"),
            model_kwargs.get("mm_token_type_ids"), model_kwargs.get("image_grid_thw"),
            model_kwargs.get("video_grid_thw"), model_kwargs.get("rope_deltas"),
        )
        if delta is not None:
            model_kwargs["rope_deltas"] = delta
        if positions.shape[-1] != text.shape[-1]:
            delta = delta if delta is not None else 0
            return text.unsqueeze(0) + delta
        return positions

    def prepare_inputs_for_generation(self, input_ids, is_first_iteration=False, use_cache=True, **kwargs):
        result = super().prepare_inputs_for_generation(
            input_ids, is_first_iteration=is_first_iteration, use_cache=use_cache, **kwargs,
        )
        if not is_first_iteration and use_cache:
            result["pixel_values"] = result["pixel_values_videos"] = None
        return result

    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False, input_ids=None, **kwargs):
        if expand_size == 1:
            return input_ids, kwargs
        media = {}
        for key, token_id in (("pixel_values", self.img_context_token_id),
                              ("pixel_values_videos", self.video_context_token_id)):
            pixels = kwargs.pop(key, None)
            if pixels is None:
                continue
            if key == "pixel_values" and kwargs.get("image_flags") is not None:
                pixels = pixels[kwargs.pop("image_flags").reshape(-1).bool()]
            counts = (input_ids == token_id).sum(-1).tolist()
            if any(c % self.num_image_token for c in counts):
                raise ValueError("Visual placeholder counts must cover whole tiles")
            groups = pixels.split([c // self.num_image_token for c in counts])
            media[key] = torch.cat([g for g in groups for _ in range(expand_size)])
        for key, modality in (("image_grid_thw", 1), ("video_grid_thw", 2)):
            grids = kwargs.pop(key, None)
            if grids is None:
                continue
            types = kwargs.get("mm_token_type_ids")
            if types is None:
                raise ValueError("Multimodal beam expansion requires mm_token_type_ids")
            if modality == 1:
                mask = types == 1
                counts = (mask & ~torch.nn.functional.pad(mask[:, :-1], (1, 0))).sum(-1).tolist()
            else:
                counts, offset = [], 0
                for row in types:
                    frames = (row == 2).sum().item() // self.num_image_token
                    count = 0
                    while frames > 0:
                        frames -= grids[offset + count, 0].item()
                        count += 1
                    if frames:
                        raise ValueError("Video grids disagree with frame placeholders")
                    counts.append(count)
                    offset += count
            groups = grids.split(counts)
            media[key] = torch.cat([g for g in groups for _ in range(expand_size)])
        kwargs.pop("num_patches", None)
        ids, kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder, input_ids=input_ids, **kwargs,
        )
        return ids, {**kwargs, **media}


    def get_vision_position_ids(self, start_position: int, tile_rows: int, tile_cols: int, has_thumbnail: bool=True, device: torch.device=None) -> torch.LongTensor:
        tile_h, tile_w = (self.tile_h, self.tile_w)
        npt = tile_h * tile_w
        num_grid_tiles = tile_rows * tile_cols
        tile_idx = torch.arange(num_grid_tiles, device=device)
        tr = tile_idx // tile_cols
        tc = tile_idx % tile_cols
        local_idx = torch.arange(npt, device=device)
        lr = local_idx // tile_w
        lc = local_idx % tile_w
        global_h = (tr[:, None] * tile_h + lr[None, :]).reshape(-1).long()
        global_w = (tc[:, None] * tile_w + lc[None, :]).reshape(-1).long()
        total_grid = num_grid_tiles * npt
        pos_t = torch.full((total_grid,), start_position, device=device, dtype=torch.long)
        pos_h = start_position + global_h
        pos_w = start_position + global_w
        if has_thumbnail:
            thumb_h = (lr * tile_rows).long()
            thumb_w = (lc * tile_cols).long()
            pos_t = torch.cat([pos_t, torch.full((npt,), start_position, device=device, dtype=torch.long)])
            pos_h = torch.cat([pos_h, start_position + thumb_h])
            pos_w = torch.cat([pos_w, start_position + thumb_w])
        return torch.stack([pos_t, pos_h, pos_w], dim=0)

    def get_rope_index(self, input_ids: torch.LongTensor, mm_token_type_ids: torch.IntTensor, image_grid_thw: Optional[torch.LongTensor]=None, video_grid_thw: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_h, tile_w = (self.tile_h, self.tile_w)
        npt = tile_h * tile_w
        B, S = input_ids.shape
        device = input_ids.device
        position_ids = torch.zeros(3, B, S, dtype=input_ids.dtype, device=device)
        mrope_position_deltas = []
        if mm_token_type_ids is not None and video_grid_thw is None and torch.any(mm_token_type_ids == 2).item():
            mm_token_type_ids = mm_token_type_ids.clone()
            if image_grid_thw is not None:
                total_visual_groups = 0
                for b in range(mm_token_type_ids.shape[0]):
                    cur_types = mm_token_type_ids[b]
                    if attention_mask is not None:
                        cur_types = cur_types[attention_mask[b].bool()]
                    prev_type = None
                    for t in cur_types.tolist():
                        if t in (1, 2) and t != prev_type:
                            total_visual_groups += 1
                        prev_type = t
                num_image_grids = image_grid_thw.shape[0]
                if total_visual_groups <= num_image_grids:
                    mm_token_type_ids[mm_token_type_ids == 2] = 1
                    if 'logger' in globals():
                        logger.warning_once('Converting mm_token_type_ids type 2 to type 1 because video_grid_thw is None and image_grid_thw has enough grids. This matches video-as-image processing.')
                else:
                    mm_token_type_ids[mm_token_type_ids == 2] = 0
                    if 'logger' in globals():
                        logger.warning_once('mm_token_type_ids contains type 2 but video_grid_thw is None, and image_grid_thw does not have enough grids. Treating type 2 as text. This likely means the model generated orphan <|video_pad|> tokens.')
            else:
                mm_token_type_ids[mm_token_type_ids == 2] = 0
                if 'logger' in globals():
                    logger.warning_once('mm_token_type_ids contains type 2, but both video_grid_thw and image_grid_thw are None. Treating type 2 as text.')
        grid_iters = {1: iter(image_grid_thw) if image_grid_thw is not None else None, 2: iter(video_grid_thw) if video_grid_thw is not None else None}
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]
            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))
            current_pos = 0
            llm_pos_ids_list: List[torch.Tensor] = []
            vid_active = False
            vid_num_frames = 0
            vid_frame_idx = 0
            vid_spatial_start = 0
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + current_pos)
                    current_pos += text_len
                elif modality_type == 1:
                    seg_len = end_idx - start_idx
                    grid = next(grid_iters[1])
                    tile_rows = grid[1].item()
                    tile_cols = grid[2].item()
                    grid_tokens = tile_rows * tile_cols * npt
                    has_thumbnail = seg_len > grid_tokens
                    vpos = self.get_vision_position_ids(start_position=current_pos, tile_rows=tile_rows, tile_cols=tile_cols, has_thumbnail=has_thumbnail, device=device)
                    assert vpos.shape[1] == seg_len, f'Position count ({vpos.shape[1]}) ≠ image token count ({seg_len}) for grid=({tile_rows},{tile_cols}), thumbnail={has_thumbnail}'
                    llm_pos_ids_list.append(vpos)
                    current_pos += max(tile_rows * tile_h, tile_cols * tile_w)
                elif modality_type == 2:
                    seg_len = end_idx - start_idx
                    if not vid_active:
                        grid = next(grid_iters[2])
                        vid_num_frames = grid[0].item()
                        vid_active = True
                        vid_frame_idx = 0
                        vid_spatial_start = current_pos
                    if seg_len % npt != 0:
                        raise ValueError(f'Video segment length {seg_len} is not a multiple of npt={npt} (tile_h*tile_w). Check that the processor produced one <|video_pad|> block per frame with exactly npt tokens each.')
                    frames_in_run = seg_len // npt
                    if vid_frame_idx + frames_in_run > vid_num_frames:
                        raise ValueError(f"Video has {vid_num_frames} frames but input_ids contain at least {vid_frame_idx + frames_in_run} frame blocks. Check the processor's video_grid_thw against the actual <|video_pad|> count.")
                    local_idx = torch.arange(npt, device=device)
                    lr = local_idx // tile_w
                    lc = local_idx % tile_w
                    all_t, all_h, all_w = ([], [], [])
                    for _ in range(frames_in_run):
                        all_t.append(torch.full((npt,), vid_spatial_start + vid_frame_idx, device=device, dtype=torch.long))
                        all_h.append((vid_spatial_start + lr).long())
                        all_w.append((vid_spatial_start + lc).long())
                        vid_frame_idx += 1
                        current_pos += max(tile_h, tile_w)
                    vpos = torch.stack([torch.cat(all_t), torch.cat(all_h), torch.cat(all_w)], dim=0)
                    assert vpos.shape[1] == seg_len, f'Position count ({vpos.shape[1]}) ≠ video token count ({seg_len})'
                    llm_pos_ids_list.append(vpos)
                    if vid_frame_idx >= vid_num_frames:
                        vid_active = False
                        vid_num_frames = 0
                        vid_frame_idx = 0
                        vid_spatial_start = 0
            if vid_active:
                raise ValueError(f'Reached end of input with video state still active: consumed {vid_frame_idx}/{vid_num_frames} frames. video_grid_thw declares more frames than the <|video_pad|> blocks contain.')
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=device).unsqueeze(1)
        return (position_ids, mrope_position_deltas)
