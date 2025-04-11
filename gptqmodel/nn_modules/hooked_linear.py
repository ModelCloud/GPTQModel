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
import copy
from typing import Dict, List, Tuple, Union

import torch
import transformers
from torch import nn

from ..utils.logger import setup_logger

log = setup_logger()

# Models using conv1d: gpt2
class HookedConv1D(transformers.Conv1D):
    def __init__(self, nf: int, nx: int) -> None:
        torch.nn.Module.__init__(self)
        self.nf = nf
        self.nx = nx
        self.forward_hook = None

    @staticmethod
    def from_conv1d(m: transformers.Conv1D):
        assert isinstance(m, transformers.Conv1D)
        custom = HookedConv1D(m.nf, m.nx)
        custom.weight = m.weight
        custom.bias = m.bias
        return custom

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.forward_hook:
            self.forward_hook(self, (input,), output)
        return output

class HookedConv1d(torch.nn.Conv1d):
    def __init__(
        self,
    ) -> None:
        # in_channels,
        # out_channels,
        # kernel_size,
        # stride,
        # padding,
        # dilation,
        # groups,
        # padding_mode,
        torch.nn.Module.__init__(self)
        # TODO: call super().__init__() is too slow, need to find a better way
        # super().__init__(
        #     in_channels,
        #     out_channels,
        #     kernel_size,
        #     stride,
        #     padding,
        #     dilation,
        #     groups,
        #     padding_mode,
        # )
        self.forward_hook = None

    @staticmethod
    def from_conv1d(m: torch.nn.Conv1d):
        assert isinstance(m, torch.nn.Conv1d)

        custom = HookedConv1d()
        custom.in_channels = m.in_channels
        custom.out_channels = m.out_channels
        custom.kernel_size = m.kernel_size
        custom.stride = m.stride
        custom.padding = m.padding
        custom.dilation = m.dilation
        custom.transposed = m.transposed
        custom.output_padding = m.output_padding
        custom.groups = m.groups
        custom.padding_mode = m.padding_mode

        custom.weight = m.weight
        custom.bias = m.bias
        return custom

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.forward_hook:
            self.forward_hook(self, (input,), output)
        return output

# Models using conv2d: ovis
class HookedConv2d(torch.nn.Conv2d):
    def __init__(
        self,
    ) -> None:
        # in_channels,
        # out_channels,
        # kernel_size,
        # stride,
        # padding,
        # dilation,
        # groups,
        # padding_mode,
        torch.nn.Module.__init__(self)
        # TODO: call super().__init__() is too slow, need to find a better way
        # super().__init__(
        #     in_channels,
        #     out_channels,
        #     kernel_size,
        #     stride,
        #     padding,
        #     dilation,
        #     groups,
        #     padding_mode,
        # )
        self.forward_hook = None

    @staticmethod
    def from_conv2d(m: torch.nn.Conv2d):
        assert isinstance(m, torch.nn.Conv2d)

        custom = HookedConv2d()
        custom.in_channels = m.in_channels
        custom.out_channels = m.out_channels
        custom.kernel_size = m.kernel_size
        custom.stride = m.stride
        custom.padding = m.padding
        custom.dilation = m.dilation
        custom.transposed = m.transposed
        custom.output_padding = m.output_padding
        custom.groups = m.groups
        custom.padding_mode = m.padding_mode

        custom.weight = m.weight
        custom.bias = m.bias
        return custom

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.forward_hook:
            self.forward_hook(self, (input,), output)
        return output

# Models using transformers.conv1d: gpt2
class HookedTransformerConv1D(transformers.Conv1D):
    def __init__(self, nf: int, nx: int) -> None:
        torch.nn.Module.__init__(self)
        self.nf = nf
        self.nx = nx
        self.forward_hook = None

    @staticmethod
    def from_conv1d(conv1d: transformers.Conv1D):
        custom = HookedConv1D(conv1d.nf, conv1d.nx)
        custom.weight = conv1d.weight
        custom.bias = conv1d.bias
        return custom

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.forward_hook:
            self.forward_hook(self, (input,), output)
        return output

class HookedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        # avoid calling super().__init__() as it would allocate memory based on in/out features
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features

        self.forward_hook = None

    @staticmethod
    def from_linear(linear: torch.nn.Linear):
        custom_linear = HookedLinear(linear.in_features, linear.out_features)
        custom_linear.weight = linear.weight
        custom_linear.bias = linear.bias
        return custom_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.forward_hook:
            self.forward_hook(self, (input,), output)
        return output

def replace_module_with_hooked_legacy(module, level: int = 0):
    if level == 0:
        log.info("Hooked Modules: Using legacy based config for targeting of modules")

    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, HookedLinear.from_linear(child))
        elif isinstance(child, nn.Conv1d):
            setattr(module, name, HookedConv1d.from_conv1d(child))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, HookedConv2d.from_conv2d(child))
        elif isinstance(child, transformers.Conv1D):
            setattr(module, name, HookedConv1D.from_conv1d(child))
        # elif isinstance(child, torch.nn.Conv2d):
        #     setattr(module, name, HookedConv2d.from_conv2d(child))
        else:
            replace_module_with_hooked_legacy(child, level=level+1)

def replace_module_with_hooked_tree(module, tree: Union[List,Dict] = [], level: int = 0, debug: bool = False):
    if level == 0:
        log.info("Hooked Modules: Using tree based config for accurate targeting of modules")

    tree = copy.copy(tree) # defensive copy

    # tuple represents targeted modules
    execute_replace = isinstance(tree, Tuple)

    # level indent
    level_indent = "---" * level

    for name, child in module.named_children():
        if debug:
            log.info(f"{level_indent} child name: {name}")
        if execute_replace:
            if name in tree:
                if isinstance(child, torch.nn.Linear):
                    if debug:
                        log.info(f"{level_indent} Hook: nn.Linear: {name}")
                    setattr(module, name, HookedLinear.from_linear(child))
                elif isinstance(child, transformers.Conv1D):
                    if debug:
                        log.info(f"{level_indent} Hook: transformers.Conv1D: {name}")
                    setattr(module, name, HookedTransformerConv1D.from_conv1d(child))
                elif isinstance(child, nn.Conv1d):
                    if debug:
                        log.info(f"{level_indent} Hook: nn.Conv1d: {name}")
                    setattr(module, name, HookedConv1d.from_conv1d(child))
                elif isinstance(child, nn.Conv2d):
                    if debug:
                        log.info(f"{level_indent} Hook: nn.Conv2d: {name}")
                    setattr(module, name, HookedConv2d.from_conv2d(child))
                else:
                    if debug:
                        log.error(f"{level_indent} Hook: execute_replace but layer skipped due to type not supported: {name}")
            else:
                if debug:
                    log.warn(f"{level_indent} Hook: execute_replace but layer skipped due to not targeted: {name}")
        else:
            if isinstance(tree, Dict):
                if name in tree:
                    if debug:
                        log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                    replace_module_with_hooked_tree(child, tree=tree[name],
                                                    level=level+1, debug=debug)
                elif "#" in tree and name.isdigit():
                    if debug:
                        log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                    replace_module_with_hooked_tree(child, tree=tree["#"],
                                                    level=level+1, debug=debug)
                else:
                    if debug:
                        log.warn(f"{level_indent} Hook: skipped unknown tree node dict: {name} vs tree: {tree}")
            elif isinstance(tree, List):
                next_node = tree[0]
                if isinstance(next_node, Dict):
                    if name in next_node:
                        if debug:
                            log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                        replace_module_with_hooked_tree(child, tree=next_node[name],
                                                        level=level + 1, debug=debug)
                    elif name.isdigit and "#" in next_node:
                        if debug:
                            log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                        replace_module_with_hooked_tree(child, tree=next_node["#"],
                                                        level=level + 1, debug=debug)
                    else:
                        if debug:
                            log.warn(
                            f"{level_indent} Hook: skipped unknown tree node dict: {name} vs next_node: {next_node}")
                elif name == next_node or (next_node == "#" and name.isdigit()):
                    if debug:
                        log.info(f"{level_indent} Hook: follow tree node: {name} -> nest into {name}")
                    replace_module_with_hooked_tree(child, tree=tree[1:],
                                                    level=level + 1, debug=debug)
                else:
                    if debug:
                        log.warn(f"{level_indent} Hook: skipped unknown tree node list: {name} vs next_node: {next_node}")

            # if len(tree) > 0:
            #     # list or dict
            #     node = next(tree_iter)
            #     log.info(f"{level_indent} next node: {node}")
            #
            #     # matched user designed tree node/path
            #     if name == node or (node == "#" and name.isdigit()):
            #         log.info(f"{level_indent} Hook: follow tree node: {node} -> nest into {name}")
            #         replace_module_with_hooked(child, tree=tree[1:] if isinstance(tree, List) else tree[name], level=level + 1, debug=debug)
            #     # # list: simple node
            #     # elif isinstance(node, List):
            #     #     if name in node:
            #     #         log.info(f"{level_indent} Hook: follow tree node list: {node} -> {name} -> nest into {name}")
            #     #         replace_module_with_hooked(child, tree=tree[1:], level=level + 1, debug=debug)
            #     #     elif name.isdigit() and "#" in node:
            #     #         log.info(f"{level_indent} Hook: follow tree node list: {node} -> # -> nest into {name}")
            #     #         replace_module_with_hooked(child, tree=tree[1:], level=level + 1, debug=debug)
            #     #     else:
            #     #         log.warn(f"{level_indent} Hook: stopped at unknown tree node list: {node} -> {name}")
            #     # # dict: nested node with final modules larget as value [list]
            #     # elif isinstance(node, Dict):
            #     #     if name in node:
            #     #         log.info(f"{level_indent} Hook: follow tree node dict: {node} -> {name} -> nest into {name}")
            #     #         replace_module_with_hooked(child, tree=node[name], level=level + 1, debug=debug)
            #     #     elif name.isdigit() and "#" in node:
            #     #         log.info(f"{level_indent} Hook: follow tree node dict: {node} -> # -> nest into {name}")
            #     #         replace_module_with_hooked(child, tree=node["#"], level=level + 1, debug=debug)
            #     #     else:
            #     #         log.warn(f"{level_indent} Hook: stopped at unknown tree node dict: {node} -> {name}")
            #     else:
            #         log.warn(f"{level_indent} Hook: skipped at unknown tree node: {name}")
            # else:
            #     log.warn(f"{level_indent} Hook: follow naively -> nest into {name}")
            #     replace_module_with_hooked(child, [], level=level + 1, debug=debug)
