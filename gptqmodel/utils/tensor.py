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

import math

import numpy as np
import torch


# helper method to get accurate parameter count for a gptq model
def tensor_parameters(
        tensor_name: str,  # name of tensor weight in model
        tensor_shape: torch.Size,  # shape of tensor
        bits: int,  # gptq bits
) -> int:
    # only .qweight is relevant for `parameters` in gptq model
    if tensor_name.endswith(".qweight"):
        real_infeatures = math.ceil(tensor_shape[0] / bits * 32)
        real_tensor_shape = (real_infeatures,) + tensor_shape[1:]
        return np.prod(real_tensor_shape)
    # .scales and .qzeros are not model parameters but aux data for .qweight
    elif tensor_name.endswith((".scales", ".qzeros", ".g_idx")):
        return 0
    else:
        return np.prod(tensor_shape)
