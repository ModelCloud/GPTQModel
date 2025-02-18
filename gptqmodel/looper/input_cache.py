from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class InputCache:
    layer_inputs: List[List[torch.Tensor]]
    layer_input_kwargs: List[Dict[str, torch.Tensor]]
    position_ids: List[torch.Tensor]
    attention_masks: List[torch.Tensor]
