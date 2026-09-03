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
