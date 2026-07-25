# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import functools

import torch

from ..config import GemmType, LayerConfig
from .base import DeviceHeuristics
from .raster import raster_group_m_for_config
from .sm8x import (
    Sm80Heuristics,
    Sm86Heuristics,
    Sm87Heuristics,
    Sm89Heuristics,
)
from .sm75 import Sm75Heuristics
from .sm90 import Sm90Heuristics
from .sm90_h20 import Sm90H20Heuristics
from .sm100 import Sm100Heuristics
from .sm120 import Sm120Heuristics

heuristics_map: dict[int, type[DeviceHeuristics]] = {
    75: Sm75Heuristics,
    80: Sm80Heuristics,
    86: Sm86Heuristics,
    87: Sm87Heuristics,
    89: Sm89Heuristics,
    90: Sm90Heuristics,
    100: Sm100Heuristics,
    103: Sm100Heuristics,
    120: Sm120Heuristics,
    121: Sm120Heuristics,
}


def get_heuristics_class(
    sm_version: int | tuple[int, int] | None = None,
    device: int | torch.device | None = None,
) -> type[DeviceHeuristics]:
    if sm_version is None:
        sm_version = torch.cuda.get_device_capability(device)
    if isinstance(sm_version, tuple):
        sm_version = sm_version[0] * 10 + sm_version[1]
    assert isinstance(sm_version, int)
    name = torch.cuda.get_device_name(device)
    if "H20" in name and "H200" not in name:
        return Sm90H20Heuristics

    return heuristics_map[sm_version]


def _apply_m_major_input_scale(
    config: dict,
    use_m_major_input_scale: bool,
    layer_config: LayerConfig,
    gemm_type: GemmType,
) -> None:
    if not use_m_major_input_scale:
        return
    use_tma = config.get("use_tma", False)
    if use_tma and layer_config.input_scale_group_size > 0 and gemm_type == GemmType.DENSE:
        config["use_tma_as"] = True


def _apply_raster_group_m(config: dict, layer_config, gemm_type) -> None:
    if gemm_type != GemmType.DENSE:
        return
    if config.get("raster_group_m") is not None or "block_shape" not in config:
        return
    try:
        config["raster_group_m"] = raster_group_m_for_config(
            layer_config,
            config["block_shape"],
            config.get("multi_cast_size_a", 1),
        )
    except Exception:
        pass


@functools.lru_cache(maxsize=1024)
def get_heuristics_config(
    layer_config: LayerConfig | dict,
    shape_m: int | None = None,
    use_f16_accum: bool = False,
    use_batch_invariant: bool = False,
    use_m_major_input_scale: bool = False,
    gemm_type: str | GemmType = "dense",
):
    if isinstance(gemm_type, str):
        gemm_type = GemmType(gemm_type)

    if isinstance(layer_config, dict):
        layer_config = LayerConfig(**layer_config)
    heuristics_cls = get_heuristics_class()
    if isinstance(shape_m, int):
        config = heuristics_cls.get_config(
            layer_config=layer_config,
            shape_m=shape_m,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
        _apply_m_major_input_scale(config, use_m_major_input_scale, layer_config, gemm_type)
        _apply_raster_group_m(config, layer_config, gemm_type)
        return config
    else:
        configs = heuristics_cls.get_configs(
            layer_config=layer_config,
            use_f16_accum=use_f16_accum,
            use_batch_invariant=use_batch_invariant,
            gemm_type=gemm_type,
        )
        for entry in configs:
            _apply_m_major_input_scale(entry[2], use_m_major_input_scale, layer_config, gemm_type)
            _apply_raster_group_m(entry[2], layer_config, gemm_type)
        return configs
