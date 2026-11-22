# Copyright (c) 2023 DeepSeek. Distributed under LICENSE-CODE (MIT).
"""DeepSeek-VL2 wrapper. No checkpoint code discovery or global HF registration."""

from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin

from .configuration_deepseek_vl_v2 import DeepseekVLV2Config
from .modeling_deepseek import DeepseekV2ForCausalLM
from .siglip_vit import VisionTransformer


class MlpProjector(nn.Module):
    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.depth
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == "downsample_mlp_gelu":
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(
                x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio, padding=0
            )  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)


class DeepseekVLV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekVLV2Config
    base_model_prefix = "deepseek_vl_v2"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_sdpa = True  # dispatched to the text decoder; the vision tower has its own attention


class DeepseekVLV2ForCausalLM(DeepseekVLV2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"language.lm_head.weight": "language.model.embed_tokens.weight"}

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = VisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.heads,
            mlp_ratio=vision_config.mlp_ratio,
            class_token=vision_config.class_token,
            global_pool=vision_config.global_pool,
            ignore_head=vision_config.ignore_head,
            weight_init=vision_config.weight_init,
            num_classes=0,
            deterministic=vision_config.deterministic,
            num_recomputing_layers=vision_config.num_recomputing_layers,
        )

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config)

        # image token format 形式
        # FIXME 目前tile tag & global_view_pos的默认取值都是之前的实验策略；后续应当去掉默认取值，改为没有取值就raise error
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # 用于format image token sequence的特殊token
        embed_std = 1 / torch.sqrt(torch.tensor(projector_config.n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}"
                )
            self.tile_indicators = nn.Parameter(
                torch.randn(
                    size=(
                        max(w * h // config.vision_config.image_size**2 for w, h in candidate_resolutions) + 1,
                        projector_config.n_embed,
                    )
                )
                * embed_std
            )
        else:
            raise ValueError(f"tile tag should be either 1D or 2D, but got {self.tile_tag}")

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(language_config)
        self.post_init()

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.LongTensor] = None,
        images_spatial_crop: Optional[torch.LongTensor] = None,
        **ignore_kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            images (torch.FloatTensor): [b, max_n_images, 3, height, width]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_spatial_crop (torch.LongTensor): [b, max_n_images, 2]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        if images is not None and (images_spatial_crop is None or images_seq_mask is None):
            raise ValueError("Images require images_spatial_crop and images_seq_mask")
        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += 1 + num_width_tiles * num_height_tiles

            total_tiles.append(images[idx, : batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        if total_tiles.shape[0] == 0:
            return self.language.get_input_embeddings()(input_ids)

        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision(total_tiles)

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        # put image tokens into the input_embeds, [b, T, D]
        input_embeds = self.language.get_input_embeddings()(input_ids)

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):
                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[tile_index + 1 : tile_index + 1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1

                # format global and local features
                if self.tile_tag == "2D":
                    # ----------------- global view add newline -----------------
                    # [hw, D] -> [h, w, D]
                    global_features = global_features.view(h, w, n_dim)
                    # [D]     -> [h, 1, D]
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                    global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                    # [h, w + 1, D] -> [h * (w + 1), D]
                    global_features = global_features.view(-1, n_dim)

                    # ----------------- local view add newline -----------------
                    # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,
                        h=h,
                        w=w,
                    )

                    # [D] -> [num_height_tiles * h, 1, D]
                    new_lines_in_local = repeat(self.image_newline, "d -> (th h) 1 d", th=num_height_tiles, h=h)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                    local_features = local_features.view(-1, n_dim)

                    # ----------------- merge global and local tiles -----------------
                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [global_features, self.view_seperator[None, :], local_features], dim=0
                        )
                    else:
                        global_local_features = torch.cat(
                            [local_features, self.view_seperator[None, :], global_features], dim=0
                        )

                else:
                    # abandoned，实际上不会走这个逻辑
                    global_features = torch.cat([self.tile_indicators[0:1], global_features], dim=0)
                    local_features = torch.cat(
                        [self.tile_indicators[1 : num_tiles_in_image + 1].unsqueeze(1), local_features], dim=1
                    )
                    local_features = rearrange(local_features, "crop_num hw d -> (crop_num hw) d")

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat([global_features, local_features], dim=0)
                    else:
                        global_local_features = torch.cat([local_features, global_features], dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                if int(images_seq_mask[idx].sum()) != images_in_this_batch.shape[0]:
                    raise ValueError("Image token mask does not match the projected tile sequence")
                input_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), images_in_this_batch)

        return input_embeds

    def get_input_embeddings(self):
        return self.language.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.language.set_output_embeddings(value)

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DeepseekVLV2ForCausalLM):
            std = self.config.projector_config.n_embed**-0.5
            for name in ("image_newline", "view_seperator", "tile_indicators"):
                if hasattr(module, name):
                    nn.init.normal_(getattr(module, name), std=std)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        images=None,
        images_seq_mask=None,
        images_spatial_crop=None,
        labels=None,
        use_cache=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )
        return self.language(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        is_first_iteration=False,
        images=None,
        images_seq_mask=None,
        images_spatial_crop=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if past_key_values is None or past_key_values.get_seq_length() == 0 or not kwargs.get("use_cache", True):
            model_inputs.update(
                images=images, images_seq_mask=images_seq_mask, images_spatial_crop=images_spatial_crop
            )
        return model_inputs

    @torch.no_grad()
    def incremental_prefilling(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        images_seq_mask=None,
        images_spatial_crop=None,
        chunk_size=1024,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )
        cache = None
        for start in range(0, inputs_embeds.shape[1] - 1, chunk_size):
            end = min(start + chunk_size, inputs_embeds.shape[1] - 1)
            outputs = self(
                inputs_embeds=inputs_embeds[:, start:end],
                attention_mask=None if attention_mask is None else attention_mask[:, :end],
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
        return inputs_embeds, cache
