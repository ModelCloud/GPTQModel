# SPDX-License-Identifier: Apache-2.0
# Derived from LongLLaMA; see NOTICE and README.md.
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class LongLlamaMemConfig:
    positionals: bool = True
    cache_dtype: torch.dtype = torch.bfloat16
    attention_grouping: Optional[Tuple[int, int]] = None


@dataclass
class LongLlamaMemCache:
    keys: torch.FloatTensor
    values: torch.FloatTensor
    masks: torch.FloatTensor


def mem_apply_update(
    prev_mem_cache: LongLlamaMemCache, new_mem_content: LongLlamaMemCache, mem_config: LongLlamaMemConfig
):

    def update_one(prev, new):
        if len(prev.shape) != 4 or len(new.shape) != 4:
            raise ValueError(f"Memory cache content should be consistent in shape got {prev.shape} {new.shape}")
        return torch.concat([prev, new], dim=-2)

    insert_size = new_mem_content.keys.shape[-2]
    if new_mem_content.values.shape[-2] != insert_size or new_mem_content.masks.shape[-2] != insert_size:
        raise ValueError("Inconsistent mem_length in new_mem_content")
    return LongLlamaMemCache(
        keys=update_one(prev_mem_cache.keys, new_mem_content.keys),
        values=update_one(prev_mem_cache.values, new_mem_content.values),
        masks=update_one(prev_mem_cache.masks, new_mem_content.masks),
    )
