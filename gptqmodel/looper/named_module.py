# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any

import torch
import transformers
from torch import nn
from torch.nn.modules.conv import _ConvNd


class NamedModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, name: str, full_name:str, layer_index: int) -> None:
        super().__init__()

        self.module = module # wrapped module
        self.module_dtype = next(module.parameters()).dtype
        self.name = name # module name
        self.full_name = full_name # module full name (path) within model
        self.layer_index = layer_index # layerid in a repeating layer, if in outside layer, this info may be fake

        # some processing will move this module to target_device gptq, eora, etc
        # self.target_device, self.target_device_stream = device_next()
        self.target_device, self.target_device_stream = None, None

        # persistent work state forLoopProcessors
        # store all `processed()` work state/data/result here
        self.state = {}

        # print(f"NamedModule init: name: `{name}, full-name: `{full_name}`")

        # store original in/out features since weight.data will changed later on
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
        elif isinstance(module, _ConvNd):
            in_features = module.in_channels
            out_features = module.out_channels
        # elif isinstance(module, nn.Conv2d):
        #     in_features = module.in_channels
        #     out_features = module.out_channels
        # elif isinstance(module, nn.Conv2d):
        #     in_features = module.in_channels
        #     out_features = module.out_channels
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]
        else:
            raise NotImplementedError(f"Unsupported module.module type: `{type(module)}`")

        self.state.update({
            "in_features": in_features,
            "out_features": out_features,
        })

    # return stats for mo
    # def stats(self) -> Dict[str, float]:
    #     # -1 means no stats have yet to gathered for the stat property
    #     return {
    #         STAT_GPTQ_DURATION: self.state.get(STAT_GPTQ_DURATION, -1),
    #         STAT_GPTQ_AVG_LOSS: self.state.get(STAT_GPTQ_AVG_LOSS, -1),
    #         STAT_GPTQ_DAMP_PERCENT: self.state.get(STAT_GPTQ_DAMP_PERCENT, -1),
    #         STAT_GPTQ_FWD_TIME: self.state.get(STAT_GPTQ_FWD_TIME, -1),
    #     }

    # getattr is only called if python cannot find attr for `self`
    def __getattr__(self, name: str):
        return getattr(self.module, name)

    # setattr is always called by python even if attr exists in `self`
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ["module", "name", "full_name", "layer_index", "state"]:
            self.__dict__[name] = value
        else:
            self.module.__dict__[name] = value
