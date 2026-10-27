# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class InputCache:
    """Stores captured layer inputs and per-batch kwargs for replayed forwards."""

    layer_inputs: List[List[torch.Tensor]]
    layer_input_kwargs: List[Dict[str, torch.Tensor]]
    position_ids: List[torch.Tensor]
    attention_masks: List[torch.Tensor]
    src_inputs: Optional[List[List[torch.Tensor]]] = None

    def __post_init__(self):
        if self.src_inputs is None:
            self.src_inputs = list(self.layer_inputs) if self.layer_inputs is not None else []

    def module_kwargs(self):
        """Returns the replay kwargs that are shared across cached module calls."""

        result = dict()
        result["position_ids"] = self.position_ids
        result["attention_masks"] = self.attention_masks
        return result

    def cleanup(self):
        """Release captured tensors so a per-layer caller can reclaim them.

        Staged GSQ keeps one captured input cache alive per decoder block.
        ``quantize_llama_gsq_model`` releases each cache in a ``finally`` block
        once its block is written, but only for caches that expose
        ``cleanup``. Without this method the guard silently matched nothing and
        GPU-resident captures accumulated for the whole model: at paper scale
        one block's capture is ~69 GB, so the run exhausted a 287 GiB device
        part-way through the blocks even though each block in isolation fits.
        """

        self.layer_inputs = []
        self.layer_input_kwargs = []
        self.position_ids = []
        self.attention_masks = []
        self.src_inputs = []
