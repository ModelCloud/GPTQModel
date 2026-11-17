# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026, ZDTaichu-5.0-9B Contributors.  All rights reserved.
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
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone inference-only C-RADIO ViT vision tower.

This file intentionally contains the small subset of C-RADIO needed by the
ZDTaichu-5.0-9B checkpoint. It does not depend on the cradio_v4 package.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import MethodType
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel


try:
    from timm.models import VisionTransformer, checkpoint_seq
except ImportError as exc:
    raise ImportError(
        "ZDTaichu C-RADIO vision requires timm; install with `pip install 'gptqmodel[vision]'` "
        "using a torchvision build compatible with your PyTorch installation."
    ) from exc

from .cradio_config import RADIOConfig


class Resolution(NamedTuple):
    height: int
    width: int


class RadioOutput(NamedTuple):
    summary: Optional[torch.Tensor]
    features: Optional[torch.Tensor]

    def to(self, *args, **kwargs) -> "RadioOutput":
        return RadioOutput(
            self.summary.to(*args, **kwargs) if self.summary is not None else None,
            self.features.to(*args, **kwargs) if self.features is not None else None,
        )


class InputConditioner(nn.Module):
    def __init__(
        self,
        input_scale: float,
        norm_mean: Union[Tuple[float, float, float], torch.Tensor],
        norm_std: Union[Tuple[float, float, float], torch.Tensor],
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.register_buffer("norm_mean", torch.as_tensor(norm_mean, dtype=torch.float32).view(-1, 1, 1) / input_scale)
        self.register_buffer("norm_std", torch.as_tensor(norm_std, dtype=torch.float32).view(-1, 1, 1) / input_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = (x - self.norm_mean) / self.norm_std
        if self.dtype is not None:
            y = y.to(self.dtype)
        return y


def get_default_conditioner() -> InputConditioner:
    from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

    return InputConditioner(1.0, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)


class ClsToken(nn.Module):
    def __init__(
        self,
        ndim: int,
        num_tokens: int = 1,
        enabled: bool = True,
        register_multiple: Optional[int] = None,
        num_registers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.ndim = ndim
        self.enabled = enabled
        self.num_registers = 0
        self.num_tokens = num_tokens
        if enabled:
            if num_registers:
                self.num_registers = num_registers
            elif register_multiple:
                self.num_registers = register_multiple - (num_tokens % register_multiple)
            scale = ndim ** -0.5
            self.token = nn.Parameter(torch.randn(num_tokens + self.num_registers, ndim) * scale)
        else:
            self.token = None
        self.num_patches = self.num_tokens + self.num_registers

    def disable(self) -> None:
        self.token = None
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.token is None:
            return x
        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token, x], dim=1)

    def no_weight_decay(self) -> List[str]:
        return ["token"]


class Im2Patches(nn.Module):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_size == 1:
            return x.flatten(2).transpose(1, 2)
        return F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)


class ViTPatchLinear(nn.Linear):
    def __init__(self, patch_size: int, embed_dim: int, bias: bool = False, **factory) -> None:
        super().__init__(3 * (patch_size ** 2), embed_dim, bias=bias, **factory)
        self.patch_size = patch_size


class ViTPatchGenerator(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        input_dims: Union[int, Tuple[int, int]],
        abs_pos: bool = True,
        normalize_patches: bool = False,
        cls_token: bool = False,
        max_input_dims: Optional[Union[int, Tuple[int, int]]] = None,
        pos_dropout: float = 0.0,
        return_pos_enc: bool = False,
        num_cls_tokens: int = 1,
        register_multiple: Optional[int] = None,
        num_registers: Optional[int] = None,
        patch_bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(input_dims, int):
            input_dims = (input_dims, input_dims)
        if max_input_dims is None:
            max_input_dims = input_dims
        if isinstance(max_input_dims, int):
            max_input_dims = (max_input_dims, max_input_dims)

        max_input_dims = tuple(int(math.ceil(d / patch_size) * patch_size) for d in max_input_dims)
        factory = {"device": device, "dtype": dtype}

        self.cpe_mode = max_input_dims != input_dims
        self.pos_dropout = pos_dropout
        self.return_pos_enc = return_pos_enc
        self.patch_size = patch_size
        self.abs_pos = abs_pos
        self.embed_dim = embed_dim
        self.num_rows = max_input_dims[0] // patch_size
        self.num_cols = max_input_dims[1] // patch_size
        self.input_dims = tuple(d // patch_size for d in input_dims)
        self.num_patches = self.num_rows * self.num_cols
        self.max_input_dims = max_input_dims
        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(patch_size, embed_dim, bias=patch_bias, **factory)
        if abs_pos:
            scale = embed_dim ** -0.5
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim, **factory) * scale)
        self.cls_token = ClsToken(
            embed_dim,
            num_tokens=num_cls_tokens,
            enabled=cls_token,
            register_multiple=register_multiple,
            num_registers=num_registers,
        )
        self.patch_normalizer = nn.LayerNorm(embed_dim) if normalize_patches else nn.Identity()
        self.num_video_frames = None

    @property
    def apply_cls_token(self) -> bool:
        return self.cls_token.enabled

    @property
    def num_cls_tokens(self) -> int:
        return self.cls_token.num_tokens

    @property
    def num_cls_patches(self) -> int:
        return self.cls_token.num_patches

    @property
    def num_registers(self) -> int:
        return self.cls_token.num_registers

    @property
    def num_skip(self) -> int:
        return self.num_cls_tokens + self.num_registers

    def no_weight_decay(self) -> List[str]:
        return ["pos_embed"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.embedder(self.im_to_patches(x))
        patches, pos_enc = self.apply_pos_enc(patches, input_size=x.shape[2:])
        patches = self.cls_token(patches)
        patches = self.patch_normalizer(patches)
        if self.return_pos_enc:
            return patches, pos_enc
        return patches

    def apply_pos_enc(
        self,
        patches: torch.Tensor,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.abs_pos:
            return patches, torch.empty(0, device=patches.device, dtype=patches.dtype)
        pos_enc = self.get_pos_enc(patches.shape[0], patch_idxs, input_size)
        if self.training and self.pos_dropout > 0:
            keeps = torch.rand(patches.shape[0], 1, 1, dtype=pos_enc.dtype, device=pos_enc.device) > self.pos_dropout
            pos_enc_drop = torch.where(keeps, pos_enc, 0)
        else:
            pos_enc_drop = pos_enc
        return patches + pos_enc_drop, pos_enc

    def get_pos_enc(
        self,
        batch_size: int,
        patch_idxs: Optional[torch.Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        input_dims = self.input_dims if input_size is None else tuple(d // self.patch_size for d in input_size)
        pos_embed = self._get_pos_embeddings(batch_size, input_dims)
        if patch_idxs is None:
            return pos_embed
        exp_patch_idxs = patch_idxs.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])
        return torch.gather(pos_embed.expand(patch_idxs.shape[0], -1, -1), dim=1, index=exp_patch_idxs)

    def _get_pos_embeddings(self, batch_size: int, input_dims: Tuple[int, int]) -> torch.Tensor:
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pos_embed

        pos_embed = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)

        def window_select(pe: torch.Tensor) -> torch.Tensor:
            if input_dims[0] < pe.shape[-2]:
                pe = pe[..., :input_dims[0], :]
            if input_dims[1] < pe.shape[-1]:
                pe = pe[..., :, :input_dims[1]]
            return pe

        if self.cpe_mode:
            if self.training:
                if self.num_video_frames is not None:
                    if batch_size % self.num_video_frames != 0:
                        raise ValueError(
                            f"Batch size {batch_size} must be divisible by num_video_frames "
                            f"{self.num_video_frames} for CPE mode."
                        )
                    batch_size //= self.num_video_frames

                min_scale = math.sqrt(0.1)
                scale = torch.rand(batch_size, 1, 1, device=pos_embed.device) * (1 - min_scale) + min_scale
                aspect_min = math.log(3 / 4)
                aspect = torch.exp(torch.rand(batch_size, 1, 1, device=pos_embed.device) * (-2 * aspect_min) + aspect_min)
                scale_xy = torch.stack([scale * aspect, scale / aspect], dim=-1).clamp_(0, 1)
                pos_xy = torch.rand(batch_size, 1, 1, 2, device=pos_embed.device) * (1 - scale_xy)
                lin_x = torch.linspace(0, 1, steps=input_dims[1], device=pos_embed.device)[None, None].expand(batch_size, input_dims[0], -1)
                lin_y = torch.linspace(0, 1, steps=input_dims[0], device=pos_embed.device)[None, :, None].expand(batch_size, -1, input_dims[1])
                grid_xy = torch.stack([lin_x, lin_y], dim=-1) * scale_xy + pos_xy
                grid_xy.mul_(2).sub_(1)
                pos_embed = F.grid_sample(
                    pos_embed.float().expand(batch_size, -1, -1, -1),
                    grid=grid_xy,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                ).to(pos_embed.dtype)
                if self.num_video_frames is not None:
                    pos_embed = torch.repeat_interleave(pos_embed, self.num_video_frames, dim=0)
            else:
                max_dim = max(input_dims)
                pos_embed = F.interpolate(pos_embed.float(), size=(max_dim, max_dim), align_corners=False, mode="bilinear").to(pos_embed.dtype)
                pos_embed = window_select(pos_embed)
        else:
            pos_embed = window_select(pos_embed)

        if pos_embed.shape[-2:] != input_dims:
            pos_embed = F.interpolate(pos_embed.float(), size=input_dims, align_corners=False, mode="bilinear").to(pos_embed.dtype)
        return pos_embed.flatten(2).permute(0, 2, 1)


def _forward_cpe(self: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    x = self.patch_generator(x)
    if getattr(self, "grad_checkpointing", False) and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    x = self.norm(x)
    return x


@contextmanager
def _video_mode(self: VisionTransformer, t: int):
    original_num_frames = self.patch_generator.num_video_frames
    self.patch_generator.num_video_frames = t
    try:
        yield
    finally:
        self.patch_generator.num_video_frames = original_num_frames


def enable_cpe(
    model: VisionTransformer,
    max_img_size: Union[int, Tuple[int, int]] = 1024,
    num_cls_tokens: int = 1,
    pos_dropout: float = 0.1,
    register_multiple: Optional[int] = None,
    num_registers: Optional[int] = None,
) -> None:
    if not isinstance(model, VisionTransformer):
        raise ValueError(f"CPE only supports timm VisionTransformer models, got {type(model)}")

    patch_size = model.patch_embed.patch_size[0]
    embed_dim = model.embed_dim
    input_dims = model.patch_embed.img_size
    normalize_patches = not isinstance(model.patch_embed.norm, nn.Identity)
    cls_token = model.cls_token is not None
    if isinstance(max_img_size, int):
        max_img_size = int(round(max_img_size / patch_size) * patch_size)
    else:
        max_img_size = tuple(int(round(d / patch_size) * patch_size) for d in max_img_size)

    model.patch_generator = ViTPatchGenerator(
        patch_size=patch_size,
        embed_dim=embed_dim,
        input_dims=input_dims,
        normalize_patches=normalize_patches,
        cls_token=cls_token,
        max_input_dims=max_img_size,
        pos_dropout=pos_dropout,
        num_cls_tokens=num_cls_tokens,
        register_multiple=register_multiple,
        num_registers=num_registers,
    )
    model.patch_embed = None
    model.cls_token = None
    model.pos_embed = None
    model.pos_drop = None
    model.patch_size = patch_size
    model.num_cls_tokens = num_cls_tokens
    model.num_registers = model.patch_generator.num_registers
    model.forward_features = MethodType(_forward_cpe, model)
    model.cpe_video_mode = MethodType(_video_mode, model)


class FeatureNormalizer(nn.Module):
    def __init__(self, embed_dim: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(embed_dim, dtype=dtype))
        self.register_buffer("tx", torch.eye(embed_dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 3:
            return (x - self.mean) @ self.tx.T
        if x.ndim == 4:
            kernel = self.tx.reshape(*self.tx.shape, 1, 1)
            return F.conv2d(x - self.mean.reshape(1, -1, 1, 1), weight=kernel, bias=None, stride=1, padding=0)
        raise ValueError(f"Unsupported input dimension: {x.ndim}, shape: {x.shape}")


class InnerRADIOModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_conditioner: nn.Module,
        patch_size: int,
        max_resolution: int,
        preferred_resolution: Resolution,
        summary_idxs: Optional[torch.Tensor] = None,
        feature_normalizer: Optional[nn.Module] = None,
        window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_conditioner = input_conditioner
        if summary_idxs is not None:
            self.register_buffer("summary_idxs", summary_idxs)
        else:
            self.summary_idxs = None
        self._preferred_resolution = preferred_resolution
        self._patch_size = patch_size
        self._max_resolution = max_resolution
        self._window_size = window_size
        self.feature_normalizer = feature_normalizer if feature_normalizer is not None else nn.Identity()

    @property
    def num_summary_tokens(self) -> int:
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.num_skip
        if getattr(self.model, "global_pool", None) == "avg":
            return 0
        return 1

    @property
    def num_cls_tokens(self) -> int:
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.num_cls_tokens
        if getattr(self.model, "global_pool", None) == "avg":
            return 0
        return 1

    @property
    def patch_size(self) -> int:
        if self._patch_size is not None:
            return self._patch_size
        if hasattr(self.model, "patch_size"):
            return self.model.patch_size
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            return patch_gen.patch_size
        raise AttributeError("Unable to infer patch_size from RADIO vision model")

    @property
    def max_resolution(self) -> int:
        return self._max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self._preferred_resolution

    @property
    def window_size(self) -> Optional[int]:
        return self._window_size

    @property
    def min_resolution_step(self) -> int:
        res = self.patch_size
        if self.window_size is not None:
            res *= self.window_size
        return res

    @property
    def blocks(self) -> Iterable[nn.Module]:
        return getattr(self.model, "blocks", None)

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    @property
    def summary_dim(self) -> int:
        embed_dim = self.embed_dim
        if self.summary_idxs is not None:
            embed_dim *= self.summary_idxs.shape[0]
        return embed_dim

    def make_preprocessor_external(self) -> Callable[[torch.Tensor], torch.Tensor]:
        ret = self.input_conditioner
        self.input_conditioner = nn.Identity()
        return ret

    def get_nearest_supported_resolution(self, height: int, width: int) -> Resolution:
        height = int(round(height / self.min_resolution_step) * self.min_resolution_step)
        width = int(round(width / self.min_resolution_step) * self.min_resolution_step)
        return Resolution(max(height, self.min_resolution_step), max(width, self.min_resolution_step))

    def switch_to_deploy(self) -> None:
        fn = getattr(self.model, "switch_to_deploy", None)
        if fn is not None:
            fn()

    def cpe_video_mode(self, t: int):
        return self.model.cpe_video_mode(t)

    def forward(self, x: torch.Tensor, feature_fmt: str = "NLC") -> RadioOutput:
        res_step = self.min_resolution_step
        if res_step is not None and (x.shape[-2] % res_step != 0 or x.shape[-1] % res_step != 0):
            raise ValueError(
                "The input resolution must be a multiple of self.min_resolution_step. "
                f"Input: {x.shape[-2:]}, Nearest: {self.get_nearest_supported_resolution(*x.shape[-2:])}"
            )
        if not getattr(self, "external_preprocessing", False):
            x = self.input_conditioner(x)
        y = self.model.forward_features(x)
        return self._extract_final(x, y, feature_fmt=feature_fmt)

    def _extract_final(self, x: torch.Tensor, y: torch.Tensor, feature_fmt: str = "NLC") -> RadioOutput:
        patch_gen = getattr(self.model, "patch_generator", None)
        if patch_gen is not None:
            all_summary = y[:, : patch_gen.num_cls_tokens]
            bb_summary = all_summary[:, self.summary_idxs] if self.summary_idxs is not None else all_summary
            all_feat = y[:, patch_gen.num_skip :]
        elif getattr(self.model, "global_pool", None) == "avg":
            all_summary = y[:, self.model.num_prefix_tokens :].mean(dim=1)
            bb_summary = all_summary
            all_feat = y
        else:
            all_summary = y[:, 0]
            bb_summary = all_summary
            all_feat = y[:, 1:]

        all_feat = self.feature_normalizer(all_feat)
        if feature_fmt == "NCHW":
            fmt_feat = all_feat.reshape(
                all_feat.shape[0],
                x.shape[-2] // self.patch_size,
                x.shape[-1] // self.patch_size,
                all_feat.shape[2],
            ).permute(0, 3, 1, 2)
        elif feature_fmt == "NLC":
            fmt_feat = all_feat
        else:
            raise ValueError(f"Unsupported feature_fmt: {feature_fmt}. Must be one of ['NLC', 'NCHW']")
        return RadioOutput(bb_summary.flatten(1), fmt_feat)


def _as_namespace(value):
    if value is None:
        return type("RADIOArgs", (), {})()
    if isinstance(value, dict):
        ns = type("RADIOArgs", (), {})()
        for k, v in value.items():
            setattr(ns, k, v)
        return ns
    return value


def _dtype_from_config(config: RADIOConfig) -> torch.dtype:
    dtype_name = getattr(config, "dtype", None) or getattr(config, "amp_dtype", None)
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    if isinstance(dtype_name, str) and hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    return torch.float32


def create_vit_from_config(config: RADIOConfig) -> VisionTransformer:
    args = _as_namespace(getattr(config, "args", {}))
    model_name = getattr(args, "model", None) or "vit_huge_patch16_224"
    if model_name != "vit_huge_patch16_224":
        raise ValueError(
            "This standalone cradio_model.py keeps only the ZDTaichu ViT-H/16 structure. "
            f"Unsupported RADIO args.model={model_name!r}."
        )

    model = VisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.hidden_size,
        depth=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=True,
        num_classes=0,
        global_pool="",
    )
    for block in model.blocks:
        block.attn.fused_attn = config._attn_implementation == "sdpa"

    # The ZDTaichu checkpoint was exported after RADIO removed the final ViT norm/head
    # and replaced patch embedding, cls token, and absolute pos embedding with CPE.
    if hasattr(model, "norm") and not getattr(args, "model_norm", False):
        model.norm = nn.Identity()
    model.head = nn.Identity()

    cpe_max_size = getattr(args, "cpe_max_size", None) or getattr(config, "max_resolution", None)
    if cpe_max_size is not None:
        teachers = getattr(args, "teachers", []) or []
        teacher_names = {t.get("name") for t in teachers if isinstance(t, dict) and t.get("name")}
        num_cls_tokens = len(teacher_names) if getattr(args, "cls_token_per_teacher", False) and teacher_names else 1
        enable_cpe(
            model,
            cpe_max_size,
            num_cls_tokens=num_cls_tokens,
            register_multiple=getattr(args, "register_multiple", None),
            num_registers=getattr(args, "cpe_num_registers", None),
        )
    return model


class RADIOModel(PreTrainedModel):
    """Inference-only HuggingFace wrapper for the ZDTaichu C-RADIO ViT tower."""

    _supports_sdpa = True
    config_class = RADIOConfig
    base_model_prefix = "radio_model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def __init__(self, config: RADIOConfig) -> None:
        super().__init__(config)
        args = _as_namespace(getattr(config, "args", {}))
        dtype = _dtype_from_config(config)
        vit = create_vit_from_config(config)

        summary_idxs = None
        if getattr(args, "cls_token_per_teacher", False):
            teachers = getattr(args, "teachers", []) or []
            if teachers:
                summary_idxs = torch.tensor(
                    [i for i, t in enumerate(teachers) if not isinstance(t, dict) or t.get("use_summary", True)],
                    dtype=torch.int64,
                )

        feature_normalizer = None
        fn_cfg = getattr(config, "feature_normalizer_config", None)
        if fn_cfg is not None:
            embed_dim = fn_cfg.get("embed_dim", vit.embed_dim) if isinstance(fn_cfg, dict) else vit.embed_dim
            feature_normalizer = FeatureNormalizer(embed_dim, dtype=torch.float32)

        pref = getattr(config, "preferred_resolution", (512, 512))
        self.radio_model = InnerRADIOModel(
            model=vit,
            input_conditioner=get_default_conditioner(),
            patch_size=getattr(config, "patch_size", 16),
            max_resolution=getattr(config, "max_resolution", 2048),
            preferred_resolution=Resolution(int(pref[0]), int(pref[1])),
            summary_idxs=summary_idxs,
            feature_normalizer=feature_normalizer,
            window_size=getattr(config, "vitdet_window_size", None),
        )
        if dtype is not torch.float32:
            self.radio_model = self.radio_model.to(dtype=dtype)

    @property
    def adaptors(self):
        return nn.ModuleDict()

    @property
    def model(self) -> nn.Module:
        return self.radio_model.model

    @property
    def input_conditioner(self) -> nn.Module:
        return self.radio_model.input_conditioner

    @property
    def num_summary_tokens(self) -> int:
        return self.radio_model.num_summary_tokens

    @property
    def patch_size(self) -> int:
        return self.radio_model.patch_size

    @property
    def max_resolution(self) -> int:
        return self.radio_model.max_resolution

    @property
    def preferred_resolution(self) -> Resolution:
        return self.radio_model.preferred_resolution

    @property
    def window_size(self) -> Optional[int]:
        return self.radio_model.window_size

    @property
    def min_resolution_step(self) -> int:
        return self.radio_model.min_resolution_step

    def make_preprocessor_external(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return self.radio_model.make_preprocessor_external()

    def get_nearest_supported_resolution(self, height: int, width: int) -> Resolution:
        return self.radio_model.get_nearest_supported_resolution(height, width)

    def switch_to_deploy(self) -> None:
        self.radio_model.switch_to_deploy()

    def forward(self, pixel_values: torch.Tensor, feature_fmt: str = "NLC", **kwargs) -> RadioOutput:
        return self.radio_model(pixel_values, feature_fmt=feature_fmt)


__all__ = [
    "RADIOModel",
    "RADIOConfig",
    "RadioOutput",
    "Resolution",
    "InputConditioner",
    "ViTPatchGenerator",
]
