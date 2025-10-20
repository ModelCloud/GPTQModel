# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
from typing import Any, Dict, Optional

import torch
import transformers
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd

from ..utils.logger import setup_logger
from ..utils.module_locks import get_parent_lock, parent_module_lock
from ..utils.stream import stream_sync as stream_sync_events
from ..utils.stream import stream_tensor_dict_to_cpu

log = setup_logger()

class NamedModule(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, name: str, full_name:str, layer_index: int) -> None:
        super().__init__()

        self.module = module  # wrapped module
        self.module_dtype = next(module.parameters()).dtype
        self.name = name  # module name
        self.full_name = full_name  # full dotted path inside model
        self.layer_index = layer_index  # layer index for repeated blocks
        self._parent_lock = get_parent_lock(full_name)

        # persistent work state for named module (used by some LoopProcessors)
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

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True):
        return self.module.buffers(recurse=recurse)

    def named_buffers(self, prefix: str = "", recurse: bool = True):
        return self.module.named_buffers(prefix=prefix, recurse=recurse)

    def register_buffer(
        self, name: str, tensor: Optional[Tensor], persistent: bool = True
    ) -> None:
        with self._parent_lock:
            return self.module.register_buffer(name, tensor, persistent)

    def unregister_buffer(self, name: str):
        with self._parent_lock:
            if name in self.module._buffers:
                del self.module._buffers[name]
                if hasattr(self.module, name):
                    delattr(self.module, name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        with self._parent_lock:
            return self.module.register_parameter(name, param)

    def unregister_parameter(self, name: str) -> None:
        with self._parent_lock:
            if name in self.module._parameters:
                del self.module._parameters[name]
                if hasattr(self.module, name):
                    delattr(self.module, name)
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
        try:
            lock = object.__getattribute__(self, "_parent_lock")
        except AttributeError:
            lock = None
        module = object.__getattribute__(self, "module")
        if lock is None:
            return getattr(module, name)
        with lock:
            return getattr(module, name)

    # setattr is always called by python even if attr exists in `self`
    def __setattr__(self, name: str, value: Any) -> None:
        if name in [
            "module",
            "module_dtype",
            "name",
            "full_name",
            "layer_index",
            "state",
            "_parent_lock",
            "target_device",
            "register_buffer",
            "unregister_buffer",
            "register_parameter",
            "unregister_parameter",
        ]:
            object.__setattr__(self, name, value)
            return

        try:
            lock = object.__getattribute__(self, "_parent_lock")
        except AttributeError:
            lock = None
        module = object.__getattribute__(self, "module")
        if lock is None:
            setattr(module, name, value)
        else:
            with lock:
                setattr(module, name, value)

    def stream_state_payload_to_cpu(
        self,
        tensors: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        state_lock = self._parent_lock
        return stream_tensor_dict_to_cpu(
            tensors,
            store_callback=lambda host_map: self.state.update(host_map),
            state=self.state,
            state_lock=state_lock,
        )

    def stream_parameters_to_cpu(self) -> Dict[str, torch.Tensor]:
        state_lock = self._parent_lock
        tensor_map = {name: param for name, param in self.module.named_parameters(recurse=False)}
        return stream_tensor_dict_to_cpu(
            tensor_map,
            store_callback=lambda host_map: self.state.setdefault("parameters_cpu", {}).update(host_map),
            state=self.state,
            state_lock=state_lock,
        )

    def stream_buffers_to_cpu(self) -> Dict[str, torch.Tensor]:
        state_lock = self._parent_lock
        tensor_map = {name: buf for name, buf in self.module.named_buffers(recurse=False)}
        return stream_tensor_dict_to_cpu(
            tensor_map,
            store_callback=lambda host_map: self.state.setdefault("buffers_cpu", {}).update(host_map),
            state=self.state,
            state_lock=state_lock,
        )

    def stream_all_to_cpu(self) -> Dict[str, Dict[str, torch.Tensor]]:
        params = self.stream_parameters_to_cpu()
        buffers = self.stream_buffers_to_cpu()
        return {"parameters": params, "buffers": buffers}

    def stream_sync(self) -> None:
        stream_sync_events(self.state, self._parent_lock)
