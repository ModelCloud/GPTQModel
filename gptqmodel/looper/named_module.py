# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
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

from typing import Dict

import torch
import transformers
from torch import nn

STAT_GPTQ_FWD_TIME = "stat_fwd_time"
STAT_GPTQ_DAMP_PERCENT = "stat_damp_percent"
STAT_GPTQ_AVG_LOSS = "stat_avg_loss"
STAT_GPTQ_DURATION = "stat_duration"

class NamedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, name: str, full_name:str, layer_index: int) -> None:
        super().__init__()

        self.module = module # wrapped module
        self.name = name # module name
        self.full_name = full_name # module full name (path) within model
        self.layer_index = layer_index # layerid in a repeating layer, if in outside layer, this info may be fake
        self.state = {} # state is dict to store all temp data used in processor

        # store original in/out features since weight.data will changed later on
        if isinstance(module.module, nn.Linear):
            in_features = module.module.in_features
            out_features = module.module.out_features
        elif isinstance(module.module, nn.Conv2d):
            in_features = module.module.in_channels
            out_features = module.module.out_channels
        elif isinstance(module.module, transformers.pytorch_utils.Conv1D):
            in_features = module.module.weight.shape[0]
            out_features = module.module.weight.shape[1]
        else:
            raise NotImplementedError(f"Unsupported module.module type: `{type(module.module)}`")

        self.state.update({
            "in_features": in_features,
            "out_features": out_features,
        })

    # return stats for mo
    def stats(self) -> Dict[str, float]:
        # -1 means no stats have yet to gathered for the stat property
        return {
            STAT_GPTQ_DURATION: self.state.get(STAT_GPTQ_DURATION, -1),
            STAT_GPTQ_AVG_LOSS: self.state.get(STAT_GPTQ_AVG_LOSS, -1),
            STAT_GPTQ_DAMP_PERCENT: self.state.get(STAT_GPTQ_DAMP_PERCENT, -1),
            STAT_GPTQ_FWD_TIME: self.state.get(STAT_GPTQ_FWD_TIME, -1),
        }

    def __getattr__(self, name: str):
        if name in ["stats", "module", "name", "full_name", "layer_index", "state"]:
            return getattr(self, name)

        return getattr(self.module, name)
